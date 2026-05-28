"""
Phase 2b: Poisson Fair Value Model
For each O/U N.5 market, compute theoretical fair price given:
  - Current score (total goals so far)
  - Time remaining (minutes)
  - Estimated scoring rate λ (goals/minute)

Also computes fair price deviation from Polymarket market price,
and labels events by how large the deviation is.

Outputs: outputs/fair_value_paths.parquet, outputs/fair_value_summary.json
"""

from __future__ import annotations
from typing import Optional, List, Dict, Tuple

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob
import json
import os
from pathlib import Path
from scipy.stats import poisson

DATA_ROOT = Path("data")
OUT_DIR = Path("outputs")

# Historical football scoring rate: ~2.6 goals/90min = 0.0289/min
LEAGUE_LAMBDA = {
    "default": 2.6 / 90,
    "Premier League": 2.8 / 90,
    "La Liga": 2.5 / 90,
    "Bundesliga": 3.1 / 90,
    "Ligue 1": 2.6 / 90,
    "Major League Soccer": 2.9 / 90,
}


def poisson_ou_fair_price(total_line: float, goals_so_far: int, time_remaining_min: float,
                           lambda_per_min: float) -> float:
    """
    P(total goals at final whistle > total_line | goals_so_far, time_remaining, λ)

    Uses Poisson distribution: remaining goals ~ Poisson(λ * time_remaining)
    Need at least ceil(total_line + 1 - goals_so_far) more goals to go OVER.
    """
    if time_remaining_min <= 0:
        return 1.0 if goals_so_far > total_line else 0.0

    goals_needed = max(0, int(total_line) + 1 - goals_so_far)
    expected_more = lambda_per_min * time_remaining_min

    if goals_needed <= 0:
        return 1.0  # Already OVER

    # P(X >= goals_needed) where X ~ Poisson(expected_more)
    return 1.0 - poisson.cdf(goals_needed - 1, expected_more)


def btts_fair_price(home_goals: int, away_goals: int, time_remaining_min: float,
                    lambda_home: float, lambda_away: float) -> float:
    """
    P(both teams score | current score, time remaining)
    = P(home scores ≥ 1) * P(away scores ≥ 1)  [if neither has scored yet]
    """
    if home_goals > 0 and away_goals > 0:
        return 1.0  # Already BTTS
    if time_remaining_min <= 0:
        return 1.0 if (home_goals > 0 and away_goals > 0) else 0.0

    mu_h = lambda_home * time_remaining_min
    mu_a = lambda_away * time_remaining_min

    # P(home scores at least 1 more) = 1 - P(Poisson(mu_h) = 0) = 1 - exp(-mu_h)
    p_home_scores = 1.0 if home_goals > 0 else (1.0 - np.exp(-mu_h))
    p_away_scores = 1.0 if away_goals > 0 else (1.0 - np.exp(-mu_a))

    return p_home_scores * p_away_scores


def get_lambda(league_name: str, base_lambda: float = None) -> float:
    """Get scoring rate for a league."""
    if base_lambda is not None:
        return base_lambda
    for key, lam in LEAGUE_LAMBDA.items():
        if key.lower() in (league_name or "").lower():
            return lam
    return LEAGUE_LAMBDA["default"]


def parse_ou_line(market_type: str) -> Optional[float]:
    """Extract N from 'O/U N.5' market type."""
    if not market_type or "O/U" not in market_type:
        return None
    try:
        return float(market_type.replace("O/U", "").strip())
    except ValueError:
        return None


def compute_dynamic_lambda(msgs: pd.DataFrame, t_event: pd.Timestamp,
                            window_min: int = 15) -> float:
    """
    Compute scoring rate from last N minutes of actual game data.
    Falls back to league default if insufficient data.
    """
    t_start = t_event - pd.Timedelta(minutes=window_min)
    recent = msgs[
        (msgs["timestamp_utc"] >= t_start) &
        (msgs["timestamp_utc"] < t_event) &
        (msgs["incident_name"] == "Score") &
        (msgs["confidence_grade"] == 1.0)
    ].copy()

    if len(recent) < 2:
        return None

    recent["h"] = pd.to_numeric(recent["home_value"], errors="coerce").fillna(0)
    recent["a"] = pd.to_numeric(recent["away_value"], errors="coerce").fillna(0)
    recent["total"] = recent["h"] + recent["a"]

    goal_changes = recent["total"].diff().fillna(0)
    goals_in_window = goal_changes[goal_changes > 0].sum()

    if goals_in_window == 0:
        return None

    return float(goals_in_window) / window_min


def load_prices_for_condition(condition_id: str) -> Optional[pd.DataFrame]:
    """Load price file for a specific condition_id."""
    price_dir = DATA_ROOT / "polymarket/prices"
    short = condition_id.replace("0x", "")[:32]

    for f in glob.glob(str(price_dir / "*.parquet")):
        stem = Path(f).stem
        if stem == short or short.startswith(stem):
            df = pd.read_parquet(f)
            df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
            df["mid"] = (df["best_bid"] + df["best_ask"]) / 2
            df = df.sort_values("timestamp").reset_index(drop=True)

            # For files with asset_id, pick YES token (OVER token for O/U markets)
            # Strategy: YES token should have price closest to pre-game Poisson fair price
            # We return both assets; caller selects correct one
            if "asset_id" in df.columns:
                pass  # caller handles asset selection

            return df

    return None


def select_yes_asset(prices: pd.DataFrame, pregame_fair: float) -> pd.DataFrame:
    """
    For price files with two assets (YES/NO tokens), select the YES (OVER) token.
    Strategy: at game start, the YES token price should be close to the Poisson pre-game fair price.
    """
    if "asset_id" not in prices.columns or prices["asset_id"].nunique() <= 1:
        return prices

    asset_means = prices.groupby("asset_id")["mid"].mean()
    # Pick the asset whose mean is closest to pregame_fair
    yes_asset = (asset_means - pregame_fair).abs().idxmin()
    return prices[prices["asset_id"] == yes_asset].reset_index(drop=True)


def get_price_at_time(prices: pd.DataFrame, t: pd.Timestamp) -> Optional[float]:
    """Get market mid price at or just before time t."""
    if prices is None or len(prices) == 0:
        return None
    mask = prices["timestamp"] <= t
    if not mask.any():
        return None
    return float(prices.loc[mask, "mid"].iloc[-1])


def process_fixture_fair_value(fixture_id: int, condition_id: str, market_type: str,
                                league_name: str) -> Optional[pd.DataFrame]:
    """
    For a specific fixture + market, build a fair value path:
    - At each Score/Timer event, compute fair price from Poisson model
    - Compare to market price
    - Compute deviation = fair_price - market_price
    """
    # Load LSports messages
    files = glob.glob(str(DATA_ROOT / f"hyper/football/*/{fixture_id}/messages.parquet"))
    if not files:
        return None

    msgs = pd.read_parquet(files[0])
    msgs["timestamp_utc"] = pd.to_datetime(msgs["timestamp_utc"], utc=True)
    msgs = msgs.sort_values("timestamp_utc").reset_index(drop=True)

    # Load market prices
    prices = load_prices_for_condition(condition_id)
    if prices is None:
        return None

    # Parse market type
    ou_line = parse_ou_line(market_type)
    is_btts = "Both Teams" in (market_type or "")
    if ou_line is None and not is_btts:
        return None

    base_lambda = get_lambda(league_name)

    # Pre-game fair price (at t=0, 0-0, full 90 min)
    if ou_line is not None:
        pregame_fair = poisson_ou_fair_price(ou_line, 0, 90, base_lambda)
    else:
        pregame_fair = btts_fair_price(0, 0, 90, base_lambda / 2, base_lambda / 2)

    # Select correct YES asset
    prices = select_yes_asset(prices, pregame_fair)

    # Build event-by-event fair value path
    # Sample at key events: Score events + every 5 minutes via Timer
    key_events = msgs[msgs["incident_name"].isin(["Score", "Timer", "Period"])].copy()
    key_events = key_events.sort_values("timestamp_utc").drop_duplicates("timestamp_utc")

    # Forward-fill score
    score_events = msgs[msgs["incident_name"] == "Score"].copy()
    score_events["home_score"] = pd.to_numeric(score_events["home_value"], errors="coerce").fillna(0).astype(int)
    score_events["away_score"] = pd.to_numeric(score_events["away_value"], errors="coerce").fillna(0).astype(int)

    # Build running score series
    score_ts = score_events["timestamp_utc"].values.astype("datetime64[ns]").astype("int64")
    home_arr = score_events["home_score"].values
    away_arr = score_events["away_score"].values

    results = []
    for _, ev in key_events.iterrows():
        t = ev["timestamp_utc"]
        elapsed_s = float(pd.to_numeric(ev.get("seconds", 0), errors="coerce") or 0)
        period = ev.get("period_name", "")

        # Skip pre-game and post-game
        if period not in ("1st Half", "2nd Half", "1st half", "2nd half",
                          "Extra Time 1st Half", "Extra Time 2nd Half"):
            continue

        # Compute time remaining
        if "Extra" in (period or ""):
            total_match_min = 120.0
        else:
            total_match_min = 90.0
        elapsed_min = elapsed_s / 60.0
        # Add stoppage buffer (typically 5-8 min stoppage total)
        time_remaining_min = max(0.0, total_match_min - elapsed_min)

        # Get current score
        t_ns = np.datetime64(t, "ns").astype("int64")
        idx = np.searchsorted(score_ts, t_ns, side="right") - 1
        if idx >= 0:
            h_score = int(home_arr[idx])
            a_score = int(away_arr[idx])
        else:
            h_score, a_score = 0, 0
        total_goals = h_score + a_score

        # Market price
        market_mid = get_price_at_time(prices, t)
        if market_mid is None:
            continue

        # Dynamic lambda
        dyn_lambda = compute_dynamic_lambda(msgs, t)
        lambda_used = dyn_lambda if dyn_lambda is not None else base_lambda

        # Compute fair value
        if ou_line is not None:
            fair_price = poisson_ou_fair_price(ou_line, total_goals, time_remaining_min, lambda_used)
        elif is_btts:
            fair_price = btts_fair_price(h_score, a_score, time_remaining_min,
                                         lambda_used / 2, lambda_used / 2)
        else:
            continue

        deviation = fair_price - market_mid
        score_diff = h_score - a_score

        results.append({
            "fixture_id": fixture_id,
            "condition_id": condition_id,
            "market_type": market_type,
            "timestamp_utc": t,
            "elapsed_min": elapsed_min,
            "time_remaining_min": time_remaining_min,
            "period": period,
            "home_score": h_score,
            "away_score": a_score,
            "total_goals": total_goals,
            "score_diff": score_diff,
            "incident_name": ev["incident_name"],
            "confidence_grade": ev.get("confidence_grade", 1.0),
            "lambda_used": lambda_used,
            "fair_price": fair_price,
            "market_mid": market_mid,
            "deviation": deviation,  # positive = market underprices OVER
            "abs_deviation": abs(deviation),
        })

    if not results:
        return None

    return pd.DataFrame(results)


def main():
    fm = pd.read_parquet(DATA_ROOT / "polymarket/fixture_market_matches.parquet")
    fi = pd.read_parquet(DATA_ROOT / "fixture_index.parquet").set_index("fixture_id")

    lsports_ids = {
        int(Path(f).parent.name)
        for f in glob.glob(str(DATA_ROOT / "hyper/football/*/*/messages.parquet"))
    }

    # Focus on O/U markets and BTTS where fair value is calculable
    target_markets = fm[
        fm["market_type"].isin(["O/U 1.5", "O/U 2.5", "O/U 3.5", "O/U 4.5", "O/U 5.5",
                                  "Both Teams to Score"]) &
        fm["fixture_id"].isin(lsports_ids)
    ].copy()

    target_markets["league_name"] = target_markets["fixture_id"].map(
        lambda fid: fi.loc[fid, "league_name"] if fid in fi.index else ""
    )

    print(f"Processing {len(target_markets)} fixture-market pairs...")

    all_dfs = []
    for _, row in target_markets.iterrows():
        fid = row["fixture_id"]
        cid = row["condition_id"]
        mtype = row["market_type"]
        league = row.get("league_name", "")

        df = process_fixture_fair_value(fid, cid, mtype, league)
        if df is not None and len(df) > 0:
            df["home"] = row.get("home", "")
            df["away"] = row.get("away", "")
            df["league_name"] = league
            all_dfs.append(df)
            goals_n = df["total_goals"].max()
            max_dev = df["abs_deviation"].max()
            print(f"  {fid} [{mtype:8s}]: {row.get('home','')} vs {row.get('away','')} "
                  f"— {len(df)} snapshots, max deviation={max_dev:.3f}")

    if not all_dfs:
        print("No results!")
        return

    master = pd.concat(all_dfs, ignore_index=True)
    out_path = OUT_DIR / "fair_value_paths.parquet"
    master.to_parquet(out_path, index=False)

    print(f"\n{'='*60}")
    print(f"FAIR VALUE ANALYSIS ({len(master):,} snapshots, {master['fixture_id'].nunique()} fixtures)")
    print(f"{'='*60}")

    print("\nMean absolute deviation by market type:")
    stats = master.groupby("market_type").agg(
        snapshots=("fair_price", "count"),
        mean_abs_dev=("abs_deviation", "mean"),
        max_abs_dev=("abs_deviation", "max"),
        mean_fair=("fair_price", "mean"),
        mean_market=("market_mid", "mean"),
    ).round(3)
    print(stats.to_string())

    print("\nDeviation > 5¢ (pricing opportunity) by market type:")
    large = master[master["abs_deviation"] > 0.05]
    print(large.groupby("market_type")["abs_deviation"].agg(["count","mean","max"]).round(3).to_string())

    print("\nDeviation by elapsed time bucket:")
    master["time_bucket"] = pd.cut(master["elapsed_min"],
                                    bins=[0, 15, 30, 45, 60, 75, 90, 120],
                                    labels=["0-15", "15-30", "30-45", "45-60", "60-75", "75-90", "90+"])
    print(master.groupby("time_bucket")["abs_deviation"].agg(["mean","count"]).round(3).to_string())

    # Save summary
    summary = {
        "n_snapshots": len(master),
        "n_fixtures": int(master["fixture_id"].nunique()),
        "n_markets": int(master.groupby(["fixture_id","condition_id"]).ngroups),
        "mean_abs_deviation": float(master["abs_deviation"].mean()),
        "pct_deviation_gt_5cents": float((master["abs_deviation"] > 0.05).mean()),
        "pct_deviation_gt_10cents": float((master["abs_deviation"] > 0.10).mean()),
        "by_market_type": {
            k: {"mean_abs_dev": float(v["abs_deviation"].mean()),
                "n": int(len(v))}
            for k, v in master.groupby("market_type")
        }
    }
    with open(OUT_DIR / "fair_value_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved: {out_path}, outputs/fair_value_summary.json")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)
    main()
