"""
large_scale_bbo_microstructure_eval.py - PMXT BBO-only alert validation.

This expands beyond the few LSports-aligned games by evaluating BBO
microstructure alerts on Polymarket soccer moneyline markets directly.

Target:
    future 3c BBO mid repricing within the next N seconds.

This does not claim the move was caused by a goal. It validates whether
microstructure rules can anticipate BBO repricing at larger coverage.
"""
import _bootstrap  # noqa: F401
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

from src import load_config, resolve_path
from src import signal_evaluation as SE
from src.polymarket_loader import PolymarketPriceLoader


def _soccer_moneyline_events(cfg: dict, start: str, end: str,
                             limit: int = 500) -> pd.DataFrame:
    base = cfg["polymarket"]["gamma_base_url"]
    params = {
        "limit": limit,
        "offset": 0,
        "tag_id": 100350,
        "closed": "true",
        "start_date_min": f"{start}T00:00:00Z",
        "start_date_max": f"{end}T23:59:59Z",
    }
    r = requests.get(f"{base}/events", params=params,
                     timeout=cfg["polymarket"].get("request_timeout_sec", 15))
    r.raise_for_status()
    rows = []
    for e in r.json():
        slug = e.get("slug", "")
        if any(x in slug for x in [
            "player-props", "total-corners", "halftime-result",
            "exact-score", "more-markets",
        ]):
            continue
        markets = e.get("markets") or []
        money = [m for m in markets if m.get("sportsMarketType") == "moneyline"]
        if not money:
            continue
        # Use the highest-volume binary moneyline market for BBO microstructure.
        m = sorted(money, key=lambda x: float(x.get("volumeNum") or x.get("volume") or 0),
                   reverse=True)[0]
        token_ids = json.loads(m.get("clobTokenIds", "[]"))
        rows.append({
            "event_slug": slug,
            "title": e.get("title"),
            "event_date": e.get("eventDate"),
            "start_time": e.get("startTime") or m.get("gameStartTime"),
            "market_id": str(m.get("id")),
            "condition_id": m.get("conditionId"),
            "token_id": str(token_ids[0]) if token_ids else "",
            "market_slug": m.get("slug"),
            "market_title": m.get("question"),
            "volume": float(m.get("volumeNum") or m.get("volume") or 0),
        })
    return pd.DataFrame(rows)


def _fetch_pmxt_window(loader: PolymarketPriceLoader, market_id: str,
                       start_ts: int, end_ts: int,
                       chunk_sec: int = 900) -> pd.DataFrame:
    frames = []
    t = int(start_ts)
    while t < int(end_ts):
        u = min(t + chunk_sec, int(end_ts))
        frames.append(loader.load_pmxt_historical_orderbooks(
            market_id, "yes", t * 1000, u * 1000, limit=1000))
        t = u
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, format="mixed")
    return (out.dropna(subset=["timestamp", "price"])
            .drop_duplicates(subset=["timestamp"])
            .sort_values("timestamp")
            .reset_index(drop=True))


def _micro_features(bbo: pd.DataFrame) -> pd.DataFrame:
    df = bbo.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")
    df["abs_spread"] = (pd.to_numeric(df["best_ask"], errors="coerce") -
                        pd.to_numeric(df["best_bid"], errors="coerce")).abs()
    df["depth"] = (pd.to_numeric(df["bid_size"], errors="coerce").fillna(0) +
                   pd.to_numeric(df["ask_size"], errors="coerce").fillna(0))
    denom = df["depth"].replace(0, np.nan)
    df["imbalance"] = ((pd.to_numeric(df["bid_size"], errors="coerce").fillna(0) -
                        pd.to_numeric(df["ask_size"], errors="coerce").fillna(0)) /
                       denom).fillna(0)
    sec = (df.set_index("timestamp")
             .sort_index()
             .resample("1s")
             .agg(price=("price", "last"),
                  abs_spread=("abs_spread", "median"),
                  depth=("depth", "last"),
                  imbalance=("imbalance", "last"),
                  update_count=("price", "size")))
    sec["price"] = sec["price"].ffill()
    sec["abs_spread"] = sec["abs_spread"].ffill()
    sec["depth"] = sec["depth"].ffill()
    sec["imbalance"] = sec["imbalance"].ffill()
    sec["update_count"] = sec["update_count"].fillna(0)
    sec["abs_mid_change_5s"] = (sec["price"] - sec["price"].shift(5)).abs()
    sec["abs_mid_change_10s"] = (sec["price"] - sec["price"].shift(10)).abs()
    sec["update_rate_5s"] = sec["update_count"].rolling(5, min_periods=1).sum()
    sec["update_rate_10s"] = sec["update_count"].rolling(10, min_periods=1).sum()
    sec["spread_median_10s"] = sec["abs_spread"].rolling(10, min_periods=1).median()
    sec["spread_shock"] = sec["abs_spread"] - sec["spread_median_10s"]
    sec["depth_median_10s"] = sec["depth"].rolling(10, min_periods=1).median()
    sec["depth_drop_frac"] = 1 - (sec["depth"] / sec["depth_median_10s"].replace(0, np.nan))
    return sec.reset_index().rename(columns={"timestamp": "ts"})


def _detect_moves(feat: pd.DataFrame, jump: float, refractory_sec: int) -> pd.DataFrame:
    moved = feat[(feat["price"] - feat["price"].shift(10)).abs() >= jump].copy()
    rows, last = [], pd.NaT
    for _, r in moved.iterrows():
        t = r["ts"]
        if pd.isna(last) or (t - last).total_seconds() >= refractory_sec:
            rows.append({"move_ts": t, "move_price": r["price"]})
            last = t
    return pd.DataFrame(rows)


def _episode_times(feat: pd.DataFrame, mask: pd.Series,
                   cooldown_sec: int = 10) -> pd.DataFrame:
    rows, last, was_on = [], pd.NaT, False
    for (_, row), on in zip(feat.iterrows(), mask.fillna(False).to_numpy()):
        t = row["ts"]
        if on and not was_on:
            if pd.isna(last) or (t - last).total_seconds() >= cooldown_sec:
                rows.append({"alert_ts": t})
                last = t
        was_on = bool(on)
    return pd.DataFrame(rows)


def _score(eps: pd.DataFrame, moves: pd.DataFrame, horizon_sec: int) -> dict:
    if eps.empty:
        return {"n_alerts": 0, "true_alerts": 0, "false_alerts": 0,
                "precision": np.nan, "covered_moves": 0, "move_recall": 0.0,
                "median_lead_sec": np.nan}
    true_alerts, covered, leads = 0, set(), []
    for _, a in eps.iterrows():
        fut = moves[(moves["move_ts"] > a["alert_ts"]) &
                    (moves["move_ts"] <= a["alert_ts"] + pd.Timedelta(seconds=horizon_sec))]
        if not fut.empty:
            true_alerts += 1
            mt = fut["move_ts"].iloc[0]
            covered.add(str(mt))
            leads.append((mt - a["alert_ts"]).total_seconds())
    return {
        "n_alerts": len(eps),
        "true_alerts": true_alerts,
        "false_alerts": len(eps) - true_alerts,
        "precision": true_alerts / len(eps),
        "covered_moves": len(covered),
        "move_recall": len(covered) / len(moves) if len(moves) else np.nan,
        "median_lead_sec": float(np.median(leads)) if leads else np.nan,
    }


def _score_rules(feat: pd.DataFrame, moves: pd.DataFrame,
                 horizon_sec: int) -> pd.DataFrame:
    rules = {
        "mid1c": feat["abs_mid_change_5s"] >= 0.01,
        "mid1c_and_spread2c": (feat["abs_mid_change_5s"] >= 0.01) &
            (feat["spread_shock"] >= 0.02),
        "mid1c_and_update10": (feat["abs_mid_change_5s"] >= 0.01) &
            (feat["update_rate_5s"] >= 10),
        "spread2c_or_depth50_update10": ((feat["spread_shock"] >= 0.02) |
                                         (feat["depth_drop_frac"] >= 0.50)) &
            (feat["update_rate_5s"] >= 10),
    }
    rows = []
    for name, mask in rules.items():
        score = _score(_episode_times(feat, mask), moves, horizon_sec)
        score["strategy"] = name
        rows.append(score)
    return pd.DataFrame(rows)


def _plot_large_scale(summary: pd.DataFrame, out_path: str) -> None:
    if summary.empty:
        return
    agg = summary.groupby("strategy").agg(
        n_alerts=("n_alerts", "sum"),
        true_alerts=("true_alerts", "sum"),
        covered_moves=("covered_moves", "sum"),
        total_moves=("n_moves", "sum"),
    ).reset_index()
    agg["precision"] = agg["true_alerts"] / agg["n_alerts"].replace(0, np.nan)
    agg["move_recall"] = agg["covered_moves"] / agg["total_moves"].replace(0, np.nan)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(agg))
    ax.bar(x - 0.18, agg["precision"], width=0.36, label="precision", color="#2f9e44")
    ax.bar(x + 0.18, agg["move_recall"], width=0.36, label="move recall", color="#1c7ed6")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["strategy"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Large-scale BBO-only microstructure alert validation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default="2026-05-06")
    ap.add_argument("--end-date", default="2026-05-29")
    ap.add_argument("--max-events", type=int, default=20)
    ap.add_argument("--hours", type=float, default=3.0)
    ap.add_argument("--chunk-sec", type=int, default=900)
    ap.add_argument("--jump-threshold", type=float, default=0.03)
    ap.add_argument("--horizon-sec", type=int, default=10)
    ap.add_argument("--refractory-sec", type=int, default=30)
    args = ap.parse_args()

    loader = PolymarketPriceLoader(cfg)
    events = _soccer_moneyline_events(cfg, args.start_date, args.end_date)
    events = events.sort_values("volume", ascending=False).head(args.max_events)

    coverage, scores = [], []
    for _, e in events.iterrows():
        start = pd.to_datetime(e["start_time"], utc=True, format="mixed",
                               errors="coerce")
        if pd.isna(start):
            coverage.append({**e.to_dict(), "status": "no_start_time"})
            continue
        print(f"[large-bbo] {e['event_slug']} market={e['market_id']}")
        bbo = _fetch_pmxt_window(
            loader, str(e["market_id"]), int(start.timestamp()),
            int((start + pd.Timedelta(hours=args.hours)).timestamp()),
            chunk_sec=args.chunk_sec)
        if bbo.empty:
            coverage.append({**e.to_dict(), "status": "no_bbo",
                             "n_snapshots": 0, "n_moves": 0})
            continue
        feat = _micro_features(bbo)
        moves = _detect_moves(feat, args.jump_threshold, args.refractory_sec)
        coverage.append({**e.to_dict(), "status": "ok",
                         "n_snapshots": len(bbo), "n_seconds": len(feat),
                         "n_moves": len(moves)})
        if moves.empty:
            continue
        rs = _score_rules(feat, moves, args.horizon_sec)
        rs.insert(0, "event_slug", e["event_slug"])
        rs.insert(1, "market_id", e["market_id"])
        rs["n_moves"] = len(moves)
        scores.append(rs)

    coverage = pd.DataFrame(coverage)
    scores = pd.concat(scores, ignore_index=True) if scores else pd.DataFrame()
    tdir = resolve_path(cfg["paths"]["table_dir"])
    fdir = resolve_path(cfg["paths"]["fig_dir"])
    rdir = resolve_path(cfg["paths"]["report_dir"])
    coverage.to_csv(f"{tdir}/large_scale_bbo_micro_coverage.csv",
                    index=False, encoding="utf-8-sig")
    scores.to_csv(f"{tdir}/large_scale_bbo_micro_strategy_scores.csv",
                  index=False, encoding="utf-8-sig")
    _plot_large_scale(scores, f"{fdir}/large_scale_bbo_micro_strategy.png")

    if not scores.empty:
        agg = scores.groupby("strategy").agg(
            n_events=("event_slug", "nunique"),
            n_alerts=("n_alerts", "sum"),
            true_alerts=("true_alerts", "sum"),
            false_alerts=("false_alerts", "sum"),
            covered_moves=("covered_moves", "sum"),
            total_moves=("n_moves", "sum"),
            median_lead_sec=("median_lead_sec", "median"),
        ).reset_index()
        agg["precision"] = agg["true_alerts"] / agg["n_alerts"].replace(0, np.nan)
        agg["move_recall"] = agg["covered_moves"] / agg["total_moves"].replace(0, np.nan)
    else:
        agg = pd.DataFrame()

    lines = [
        "# Large-Scale BBO-Only Microstructure Validation",
        "",
        "## Scope",
        "",
        "This evaluates BBO microstructure alerts on Polymarket soccer moneyline "
        "markets directly. The target is future BBO repricing, not confirmed goals.",
        "",
        f"- Date range: `{args.start_date}` to `{args.end_date}`",
        f"- Candidate events evaluated: {len(events)}",
        f"- Events with BBO snapshots: {int((coverage['status'] == 'ok').sum()) if not coverage.empty else 0}",
        "",
        "## Coverage",
        "",
        SE.markdown_table(coverage) if not coverage.empty else "_No coverage._",
        "",
        "## Aggregated Strategy Scores",
        "",
        SE.markdown_table(agg) if not agg.empty else "_No scored strategies._",
        "",
        "## Interpretation",
        "",
        "This is the correct larger-scale test for BBO microstructure: predict BBO "
        "repricing itself across many markets. To connect back to LSports goals, "
        "we still need more LSports fixture mappings for these Polymarket markets.",
        "",
        "Figure:",
        "",
        "- `outputs/figures/large_scale_bbo_micro_strategy.png`",
    ]
    with open(f"{rdir}/large_scale_bbo_microstructure_report.md", "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report: {rdir}/large_scale_bbo_microstructure_report.md")


if __name__ == "__main__":
    main()
