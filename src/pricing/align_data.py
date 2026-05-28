"""
Phase 1: Data Alignment
Pairs LSports event stream with Polymarket price paths for the 54 overlapping fixtures.
Uses vectorized searchsorted for fast timestamp matching.
Outputs: data/aligned/master.parquet
"""

from __future__ import annotations
from typing import Optional, List

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path

DATA_ROOT = Path("data")
ALIGNED_OUT = DATA_ROOT / "aligned"
ALIGNED_OUT.mkdir(parents=True, exist_ok=True)

KEY_INCIDENTS = {
    "Score", "Period", "Attacks", "DangerousAttacks",
    "Total Shots", "Corners", "YellowCard", "RedCard",
    "FreeKicks", "Fouls", "Substitutions", "Penalties",
    "ShotsOnTarget", "ShotsOffTarget", "BlockedShots",
}


def load_lsports(fixture_id: int) -> Optional[pd.DataFrame]:
    files = glob.glob(str(DATA_ROOT / "hyper/football/*" / str(fixture_id) / "messages.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    return df.sort_values("timestamp_utc").reset_index(drop=True)


def load_pm_prices(condition_ids: List[str], fixture_id: int) -> Optional[pd.DataFrame]:
    dfs = []
    price_dir = DATA_ROOT / "polymarket/prices"

    # Direct fixture file
    fid_file = price_dir / f"{fixture_id}.parquet"
    if fid_file.exists():
        dfs.append(pd.read_parquet(fid_file))

    # Condition-id based files (md5 hash filenames)
    all_price_files = {Path(f).stem: f for f in glob.glob(str(price_dir / "*.parquet"))}
    for cid in condition_ids:
        short = cid.replace("0x", "")[:32]
        for stem, fpath in all_price_files.items():
            if stem == short or short.startswith(stem):
                dfs.append(pd.read_parquet(fpath))

    if not dfs:
        return None

    df = pd.concat(dfs, ignore_index=True).drop_duplicates()
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("UTC")
    df["mid"] = (df["best_bid"] + df["best_ask"]) / 2
    return df.sort_values("timestamp").reset_index(drop=True)


def build_game_state(msgs: pd.DataFrame) -> pd.DataFrame:
    """Extract key events with forward-filled score state."""
    df = msgs[msgs["incident_name"].isin(KEY_INCIDENTS)].copy()

    score_mask = df["incident_name"] == "Score"
    df["home_score"] = np.where(score_mask, pd.to_numeric(df["home_value"], errors="coerce"), np.nan)
    df["away_score"] = np.where(score_mask, pd.to_numeric(df["away_value"], errors="coerce"), np.nan)
    df["home_score"] = df["home_score"].ffill().fillna(0).astype(int)
    df["away_score"] = df["away_score"].ffill().fillna(0).astype(int)
    df["score_diff"] = df["home_score"] - df["away_score"]
    df["total_goals"] = df["home_score"] + df["away_score"]
    df["elapsed_sec"] = pd.to_numeric(df["seconds"], errors="coerce").fillna(0)
    df["time_remaining_min"] = (90 * 60 - df["elapsed_sec"]).clip(lower=0) / 60

    return df.reset_index(drop=True)


def align_vectorized(events: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized alignment: for each event, find price changes at multiple horizons.
    Uses searchsorted — O(n log n) instead of O(n*m).
    """
    if prices is None or len(prices) == 0:
        return pd.DataFrame()

    # Convert to int64 nanoseconds for fast comparison
    # Normalize both to ns to avoid us/ns mismatch
    ev_ts = events["timestamp_utc"].values.astype("datetime64[ns]").astype("int64")
    pr_ts = prices["timestamp"].values.astype("datetime64[ns]").astype("int64")
    pr_mid = prices["mid"].values

    horizons = [10, 30, 60, 120, 300]
    horizon_ns = [int(h * 1e9) for h in horizons]

    rows = []
    for i, ev_t in enumerate(ev_ts):
        # Last price at or before event
        idx_before = np.searchsorted(pr_ts, ev_t, side="right") - 1
        if idx_before < 0:
            continue
        mid_before = pr_mid[idx_before]

        row = {
            "fixture_id": events.at[i, "fixture_id"] if "fixture_id" in events.columns else None,
            "timestamp_utc": events.at[i, "timestamp_utc"],
            "incident_name": events.at[i, "incident_name"],
            "confidence_grade": events.at[i, "confidence_grade"],
            "home_score": events.at[i, "home_score"],
            "away_score": events.at[i, "away_score"],
            "score_diff": events.at[i, "score_diff"],
            "total_goals": events.at[i, "total_goals"],
            "elapsed_sec": events.at[i, "elapsed_sec"],
            "time_remaining_min": events.at[i, "time_remaining_min"],
            "period_name": events.at[i, "period_name"] if "period_name" in events.columns else None,
            "mid_before": mid_before,
        }

        for h, h_ns in zip(horizons, horizon_ns):
            idx_start = np.searchsorted(pr_ts, ev_t, side="right")
            idx_end = np.searchsorted(pr_ts, ev_t + h_ns, side="right")
            if idx_start < idx_end:
                mid_after = pr_mid[idx_start:idx_end].mean()
                row[f"delta_p_{h}s"] = mid_after - mid_before
                row[f"abs_delta_{h}s"] = abs(mid_after - mid_before)
            else:
                row[f"delta_p_{h}s"] = np.nan
                row[f"abs_delta_{h}s"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def process_fixture(fixture_id: int, meta: dict) -> Optional[pd.DataFrame]:
    msgs = load_lsports(fixture_id)
    if msgs is None:
        return None

    prices = load_pm_prices(meta["condition_ids"], fixture_id)
    if prices is None:
        return None

    events = build_game_state(msgs)
    events["fixture_id"] = fixture_id
    aligned = align_vectorized(events, prices)
    if len(aligned) == 0:
        return None

    for k in ("home", "away", "league_name", "event_date"):
        aligned[k] = meta.get(k, "")

    return aligned


def main():
    fm = pd.read_parquet(DATA_ROOT / "polymarket/fixture_market_matches.parquet")
    fi = pd.read_parquet(DATA_ROOT / "fixture_index.parquet").set_index("fixture_id")

    lsports_ids = {
        int(Path(f).parent.name)
        for f in glob.glob(str(DATA_ROOT / "hyper/football/*/*/messages.parquet"))
    }
    fixture_cids = fm.groupby("fixture_id")["condition_id"].apply(list).to_dict()
    overlap = sorted(lsports_ids & set(fixture_cids.keys()))
    print(f"Aligned fixtures to process: {len(overlap)}")

    all_dfs = []
    ok = 0
    for fid in overlap:
        row = fi.loc[fid] if fid in fi.index else {}
        meta = {
            "condition_ids": fixture_cids.get(fid, []),
            "home": row.get("home", ""),
            "away": row.get("away", ""),
            "league_name": row.get("league_name", ""),
            "event_date": row.get("event_date", ""),
        }
        result = process_fixture(fid, meta)
        if result is not None:
            all_dfs.append(result)
            ok += 1
            print(f"  OK [{ok:02d}] {fid}: {meta['home']} vs {meta['away']} — {len(result)} events aligned")
        else:
            print(f"  SKIP {fid}: no overlap")

    if not all_dfs:
        print("No aligned fixtures found!")
        return

    master = pd.concat(all_dfs, ignore_index=True)
    out = ALIGNED_OUT / "master.parquet"
    master.to_parquet(out, index=False)
    print(f"\nMaster dataset: {len(master):,} rows, {master['fixture_id'].nunique()} fixtures → {out}")

    # Quick stats
    print("\nIncident counts:")
    print(master["incident_name"].value_counts().to_string())
    print("\nAvg |ΔP| at 30s by incident (top 12):")
    stats = master.groupby("incident_name")["abs_delta_30s"].agg(["mean", "count"]).sort_values("mean", ascending=False)
    print(stats.head(12).to_string())


if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent.parent)
    main()
