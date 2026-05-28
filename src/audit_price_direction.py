from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import requests

from project_paths import DATA_DIR, OUTPUTS_DIR


CLOB_API = "https://clob.polymarket.com"


def target_outcome(market_type: str, question: str) -> str:
    text = f"{market_type} {question}".lower()
    if "o/u" in text or "over" in text or "under" in text:
        return "Over"
    return "Yes"


def get_market(condition_id: str) -> dict:
    r = requests.get(f"{CLOB_API}/markets/{condition_id}", timeout=20)
    r.raise_for_status()
    return r.json()


def price_history(token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    r = requests.get(
        f"{CLOB_API}/prices-history",
        params={"market": token_id, "resolution": 60, "startTs": start_ts, "endTs": end_ts},
        timeout=20,
    )
    r.raise_for_status()
    hist = r.json().get("history", [])
    if not hist:
        return pd.DataFrame()
    df = pd.DataFrame(hist)
    if {"t", "p"}.issubset(df.columns):
        out = pd.DataFrame({"timestamp": pd.to_datetime(df["t"].astype("int64"), unit="s", utc=True), "mid": df["p"].astype(float)})
    else:
        out = pd.DataFrame(hist, columns=["t", "p"])
        out["timestamp"] = pd.to_datetime(out["t"].astype("int64"), unit="s", utc=True)
        out["mid"] = out["p"].astype(float)
    return out[["timestamp", "mid"]].sort_values("timestamp")


def main() -> None:
    fmm = pd.read_parquet(DATA_DIR / "polymarket/fixture_market_matches.parquet")
    pred_path = OUTPUTS_DIR / "real_dataset_v4.parquet"
    if not pred_path.exists():
        pred_path = OUTPUTS_DIR / "test_predictions.parquet"
    aligned = pd.read_parquet(pred_path, columns=["fixture_id", "timestamp", "mid_price"])
    aligned["fixture_id"] = aligned["fixture_id"].astype(str)
    aligned["timestamp"] = pd.to_datetime(aligned["timestamp"], utc=True, errors="coerce")

    rows = []
    # Keep audit bounded; one representative target market per fixture.
    for fid, grp in fmm.groupby(fmm["fixture_id"].astype(str)):
        sub = aligned[aligned["fixture_id"] == fid].sort_values("timestamp")
        if sub.empty:
            continue
        market_row = grp.sort_values("market_type").iloc[0]
        cid = str(market_row["condition_id"])
        wanted = target_outcome(str(market_row.get("market_type", "")), str(market_row.get("question", "")))
        try:
            market = get_market(cid)
            token = next((t for t in market.get("tokens", []) if str(t.get("outcome", "")).lower() == wanted.lower()), None)
            if token is None:
                continue
            start_ts = int((sub["timestamp"].min() - pd.Timedelta(hours=1)).timestamp())
            end_ts = int((sub["timestamp"].max() + pd.Timedelta(hours=1)).timestamp())
            ph = price_history(str(token["token_id"]), start_ts, end_ts)
            if ph.empty:
                continue
            merged = pd.merge_asof(
                sub[["timestamp", "mid_price"]].sort_values("timestamp"),
                ph.rename(columns={"mid": "target_mid"}).sort_values("timestamp"),
                on="timestamp",
                direction="nearest",
                tolerance=pd.Timedelta(seconds=90),
            ).dropna()
            if merged.empty:
                continue
            corr = float(merged["mid_price"].corr(merged["target_mid"]))
            corr_inv = float(merged["mid_price"].corr(1 - merged["target_mid"]))
            mae = float((merged["mid_price"] - merged["target_mid"]).abs().mean())
            mae_inv = float((merged["mid_price"] - (1 - merged["target_mid"])).abs().mean())
            rows.append({
                "fixture_id": fid,
                "condition_id": cid,
                "market_type": str(market_row.get("market_type", "")),
                "question": str(market_row.get("question", "")),
                "target_outcome": wanted,
                "rows": int(len(merged)),
                "corr_target": round(corr, 4),
                "corr_inverse": round(corr_inv, 4),
                "mae_target": round(mae, 4),
                "mae_inverse": round(mae_inv, 4),
                "likely_inverted_or_wrong_market": bool((mae_inv + 0.03 < mae) or (corr_inv > corr + 0.2)),
            })
        except Exception as exc:
            rows.append({
                "fixture_id": fid,
                "condition_id": cid,
                "error": str(exc),
            })

    out = pd.DataFrame(rows).sort_values(["likely_inverted_or_wrong_market", "mae_target"], ascending=[False, False])
    OUTPUTS_DIR.mkdir(exist_ok=True)
    out_path = OUTPUTS_DIR / "price_direction_audit.csv"
    out.to_csv(out_path, index=False)
    print(json.dumps({
        "audit_rows": int(len(out)),
        "flagged": int(out.get("likely_inverted_or_wrong_market", pd.Series(dtype=bool)).fillna(False).sum()),
        "output": str(out_path),
    }, indent=2))
    if not out.empty:
        print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
