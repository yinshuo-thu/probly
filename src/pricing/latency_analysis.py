"""
Phase 2a: Millisecond-Level Latency Analysis
For each confirmed goal event (confidence reaches 1.0), measure:
  - When LSports first reports the score change (confidence ~0.1)
  - When Polymarket price first significantly moves
  - Latency = T_polymarket_move - T_lsports_first_report
Outputs: outputs/latency_analysis.parquet + outputs/latency_summary.json
"""

from __future__ import annotations
from typing import Optional, List, Dict

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob
import json
import os
from pathlib import Path

DATA_ROOT = Path("data")
OUT_DIR = Path("outputs")


def load_lsports(fixture_id: int) -> Optional[pd.DataFrame]:
    files = glob.glob(str(DATA_ROOT / f"hyper/football/*/{fixture_id}/messages.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def load_pm_prices_per_asset(condition_ids: List[str], fixture_id: int) -> Dict[str, pd.DataFrame]:
    """Load Polymarket prices, returns dict of {asset_id: prices_df} for each YES token."""
    price_dir = DATA_ROOT / "polymarket/prices"
    all_files = {Path(f).stem: f for f in glob.glob(str(price_dir / "*.parquet"))}

    all_dfs = []

    fid_file = price_dir / f"{fixture_id}.parquet"
    if fid_file.exists():
        all_dfs.append(pd.read_parquet(fid_file))

    for cid in condition_ids:
        short = cid.replace("0x", "")[:32]
        for stem, fpath in all_files.items():
            if stem == short or short.startswith(stem):
                df = pd.read_parquet(fpath)
                df["condition_id"] = cid
                all_dfs.append(df)

    if not all_dfs:
        return {}

    combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates()
    combined["timestamp"] = pd.to_datetime(combined["timestamp"]).dt.tz_convert("UTC")
    combined["mid"] = (combined["best_bid"] + combined["best_ask"]) / 2
    combined = combined.sort_values("timestamp").reset_index(drop=True)

    # Normalize: if no asset_id column, use BUY side as proxy for YES token
    if "asset_id" not in combined.columns:
        # Files with condition_id + side columns: BUY side = YES token price
        if "side" in combined.columns:
            combined["asset_id"] = combined.apply(
                lambda r: f"buy_{r.get('condition_id','x')[:16]}" if r.get("side") == "BUY"
                else f"sell_{r.get('condition_id','x')[:16]}", axis=1
            )
        else:
            combined["asset_id"] = "unknown"

    # Split by asset_id, pick YES token (higher mean mid)
    result = {}
    for asset_id, grp in combined.groupby("asset_id"):
        grp = grp.sort_values("timestamp").reset_index(drop=True)
        result[str(asset_id)[:32]] = grp

    return result


def extract_goal_sequences(msgs: pd.DataFrame) -> List[Dict]:
    """
    Extract goal events with their confidence escalation trajectory.
    A 'goal' is when Score event shows score_diff change AND eventually reaches confidence=1.0.
    """
    score_events = msgs[msgs["incident_name"] == "Score"].copy()
    score_events["home_score"] = pd.to_numeric(score_events["home_value"], errors="coerce").fillna(0).astype(int)
    score_events["away_score"] = pd.to_numeric(score_events["away_value"], errors="coerce").fillna(0).astype(int)
    score_events = score_events.sort_values("timestamp_utc").reset_index(drop=True)

    goals = []
    prev_home, prev_away = 0, 0
    current_goal_start = None
    current_goal_score = None
    conf_trajectory = []

    for _, row in score_events.iterrows():
        h, a = row["home_score"], row["away_score"]
        t = row["timestamp_utc"]
        conf = row["confidence_grade"]

        if (h + a) > (prev_home + prev_away):
            # New goal detected at this score
            if current_goal_score is not None and conf_trajectory:
                # Close previous goal
                goals.append({
                    "first_report_ts": current_goal_start,
                    "score_before": f"{prev_home}-{prev_away}",
                    "score_after": f"{current_goal_score[0]}-{current_goal_score[1]}",
                    "total_goals_after": current_goal_score[0] + current_goal_score[1],
                    "conf_trajectory": conf_trajectory,
                    "first_conf": conf_trajectory[0]["conf"],
                    "confirmed_ts": conf_trajectory[-1]["ts"] if conf_trajectory[-1]["conf"] == 1.0 else None,
                    "time_to_confirm_s": (conf_trajectory[-1]["ts"] - current_goal_start).total_seconds() if conf_trajectory[-1]["conf"] == 1.0 else None,
                })
            prev_home, prev_away = h, a
            current_goal_start = t
            current_goal_score = (h, a)
            conf_trajectory = [{"ts": t, "conf": conf}]
        elif current_goal_score is not None and (h, a) == current_goal_score:
            # Same goal, confidence update
            conf_trajectory.append({"ts": t, "conf": conf})
        else:
            # Score correction or different state
            prev_home, prev_away = h, a
            current_goal_start = None
            current_goal_score = None
            conf_trajectory = []

    # Close last goal
    if current_goal_score is not None and conf_trajectory:
        goals.append({
            "first_report_ts": current_goal_start,
            "score_before": f"{prev_home}-{prev_away}",
            "score_after": f"{current_goal_score[0]}-{current_goal_score[1]}",
            "total_goals_after": current_goal_score[0] + current_goal_score[1],
            "conf_trajectory": conf_trajectory,
            "first_conf": conf_trajectory[0]["conf"],
            "confirmed_ts": conf_trajectory[-1]["ts"] if conf_trajectory[-1]["conf"] == 1.0 else None,
            "time_to_confirm_s": (conf_trajectory[-1]["ts"] - current_goal_start).total_seconds() if conf_trajectory[-1]["conf"] == 1.0 else None,
        })

    return goals


def find_price_move_time(prices: pd.DataFrame, event_ts: pd.Timestamp,
                          lookback_s: int = 120, threshold: float = 0.01) -> Optional[Dict]:
    """
    Find the first significant price move around an event.
    Returns dict with timing and magnitude info.
    """
    if prices is None or len(prices) == 0:
        return None

    t_start = event_ts - pd.Timedelta(seconds=lookback_s)
    t_end = event_ts + pd.Timedelta(seconds=300)

    window = prices[(prices["timestamp"] >= t_start) & (prices["timestamp"] <= t_end)].copy()
    if len(window) < 2:
        return None

    window = window.sort_values("timestamp").reset_index(drop=True)
    mid_vals = window["mid"].values
    ts_vals = window["timestamp"].values

    # Baseline: median price in the 60s before event
    baseline_mask = window["timestamp"] <= event_ts
    if baseline_mask.sum() < 2:
        return None
    baseline_mid = np.median(mid_vals[baseline_mask])

    # Find first time abs(mid - baseline) > threshold after event
    after_mask = window["timestamp"] >= event_ts
    after_window = window[after_mask].reset_index(drop=True)

    first_move_idx = None
    first_move_ts = None
    first_move_delta = None

    for i, row in after_window.iterrows():
        delta = row["mid"] - baseline_mid
        if abs(delta) >= threshold:
            first_move_idx = i
            first_move_ts = row["timestamp"]
            first_move_delta = delta
            break

    # Max move in 60s window
    after_60s = after_window[after_window["timestamp"] <= event_ts + pd.Timedelta(seconds=60)]
    max_delta = None
    if len(after_60s) > 0:
        pass  # computed below in return dict

    max_delta_60s = float(after_60s["mid"].iloc[-1] - baseline_mid) if len(after_60s) > 0 else None

    return {
        "baseline_mid": baseline_mid,
        "first_move_ts": first_move_ts,
        "first_move_delta": first_move_delta,
        "max_delta_60s": max_delta_60s,
        "latency_s": float((first_move_ts - event_ts).total_seconds()) if first_move_ts is not None else None,
    }


def analyze_fixture(fixture_id: int, condition_ids: List[str], meta: Dict) -> List[Dict]:
    msgs = load_lsports(fixture_id)
    if msgs is None:
        return []

    price_assets = load_pm_prices_per_asset(condition_ids, fixture_id)
    if not price_assets:
        return []

    goals = extract_goal_sequences(msgs)
    if not goals:
        return []

    results = []
    for goal in goals:
        t_lsports = goal["first_report_ts"]

        for asset_key, prices in price_assets.items():
            move_info = find_price_move_time(prices, t_lsports)
            if move_info is None:
                continue

            record = {
                "fixture_id": fixture_id,
                "home": meta.get("home", ""),
                "away": meta.get("away", ""),
                "league_name": meta.get("league_name", ""),
                "score_before": goal["score_before"],
                "score_after": goal["score_after"],
                "total_goals_after": goal["total_goals_after"],
                "lsports_first_ts": t_lsports,
                "lsports_first_conf": goal["first_conf"],
                "lsports_confirm_ts": goal["confirmed_ts"],
                "lsports_confirm_lag_s": goal["time_to_confirm_s"],
                "asset_key": asset_key,
                "baseline_mid": move_info["baseline_mid"],
                "pm_first_move_ts": move_info["first_move_ts"],
                "pm_first_move_delta": move_info["first_move_delta"],
                "pm_max_delta_60s": move_info["max_delta_60s"],
                "pm_latency_s": move_info["latency_s"],  # positive = PM moves after LSports
            }
            results.append(record)

    return results


def main():
    fm = pd.read_parquet(DATA_ROOT / "polymarket/fixture_market_matches.parquet")
    fi = pd.read_parquet(DATA_ROOT / "fixture_index.parquet").set_index("fixture_id")

    lsports_ids = {
        int(Path(f).parent.name)
        for f in glob.glob(str(DATA_ROOT / "hyper/football/*/*/messages.parquet"))
    }
    fixture_cids = fm.groupby("fixture_id")["condition_id"].apply(list).to_dict()
    overlap = sorted(lsports_ids & set(fixture_cids.keys()))

    all_results = []
    for fid in overlap:
        row = fi.loc[fid] if fid in fi.index else {}
        meta = {
            "home": row.get("home", ""),
            "away": row.get("away", ""),
            "league_name": row.get("league_name", ""),
        }
        results = analyze_fixture(fid, fixture_cids.get(fid, []), meta)
        if results:
            all_results.extend(results)
            n_goals = len(set(r["score_after"] for r in results))
            print(f"  {fid}: {meta['home']} vs {meta['away']} — {len(results)} goal-market pairs")

    if not all_results:
        print("No results!")
        return

    df = pd.DataFrame(all_results)
    df.to_parquet(OUT_DIR / "latency_analysis.parquet", index=False)

    # Summary stats
    valid = df.dropna(subset=["pm_latency_s"])
    print(f"\n=== LATENCY ANALYSIS (n={len(valid)} goal-market observations) ===")
    print(f"PM latency vs LSports first report (positive = PM slower):")
    print(f"  Median: {valid['pm_latency_s'].median():.1f}s")
    print(f"  Mean:   {valid['pm_latency_s'].mean():.1f}s")
    print(f"  P25:    {valid['pm_latency_s'].quantile(0.25):.1f}s")
    print(f"  P75:    {valid['pm_latency_s'].quantile(0.75):.1f}s")
    print(f"  PM faster than LSports: {(valid['pm_latency_s'] < 0).mean():.1%}")
    print(f"  PM within 5s of LSports: {(valid['pm_latency_s'].abs() < 5).mean():.1%}")
    print(f"  PM within 30s of LSports: {(valid['pm_latency_s'].abs() < 30).mean():.1%}")
    print(f"\nLSports confirmation lag (first report → confidence=1.0):")
    conf_valid = df.dropna(subset=["lsports_confirm_lag_s"])
    print(f"  Median: {conf_valid['lsports_confirm_lag_s'].median():.1f}s")
    print(f"  Mean:   {conf_valid['lsports_confirm_lag_s'].mean():.1f}s")
    print(f"\nPrice impact (max delta in 60s):")
    delta_valid = df.dropna(subset=["pm_max_delta_60s"])
    print(f"  Median |ΔP|: {delta_valid['pm_max_delta_60s'].abs().median():.3f}")
    print(f"  Mean   |ΔP|: {delta_valid['pm_max_delta_60s'].abs().mean():.3f}")

    summary = {
        "n_fixture_goal_market_pairs": len(valid),
        "n_fixtures": int(df["fixture_id"].nunique()),
        "pm_latency_median_s": float(valid["pm_latency_s"].median()),
        "pm_latency_mean_s": float(valid["pm_latency_s"].mean()),
        "pm_latency_p25_s": float(valid["pm_latency_s"].quantile(0.25)),
        "pm_latency_p75_s": float(valid["pm_latency_s"].quantile(0.75)),
        "pm_faster_pct": float((valid["pm_latency_s"] < 0).mean()),
        "pm_within_5s_pct": float((valid["pm_latency_s"].abs() < 5).mean()),
        "pm_within_30s_pct": float((valid["pm_latency_s"].abs() < 30).mean()),
        "lsports_confirm_lag_median_s": float(conf_valid["lsports_confirm_lag_s"].median()),
        "mean_abs_price_impact": float(delta_valid["pm_max_delta_60s"].abs().mean()),
    }
    with open(OUT_DIR / "latency_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: outputs/latency_analysis.parquet, outputs/latency_summary.json")


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)
    main()
