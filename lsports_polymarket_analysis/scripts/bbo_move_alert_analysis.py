"""
bbo_move_alert_analysis.py - Batch BBO move vs LSports goal analysis.

This script focuses only on fixtures where we can retrieve enough historical
BBO snapshots around LSports goals. It builds compact microstructure features
and tests simple pre-BBO-move alert heuristics:

- early_mid_drift: BBO mid moves by at least 1c before the 3c reaction label
- update_burst: order-book update count in the last 5s is elevated
- spread_shock: absolute top-of-book spread proxy widens sharply
- depth_drop: top-of-book displayed depth drops materially

These are exploratory diagnostics, not a production model. They answer whether
BBO itself gives an earlier warning before the larger repricing event.
"""
import _bootstrap  # noqa: F401
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL
from src import latency_analysis as LA
from src import lsports_parser as P
from src import signal_evaluation as SE
from src.polymarket_loader import PolymarketPriceLoader, load_mapping_table


def _load_or_fetch_bbo(loader: PolymarketPriceLoader, token_id: str,
                       market_id: str, fixture_id: str, goals: pd.DataFrame,
                       lookback_sec: int, lookahead_sec: int) -> pd.DataFrame:
    local = f"{resolve_path('outputs/tables')}/single_match_historical_bbo_{fixture_id}.csv"
    try:
        cached = pd.read_csv(local)
        if not cached.empty:
            cached["timestamp"] = pd.to_datetime(cached["timestamp"], utc=True, format="mixed")
            return cached
    except Exception:  # noqa: BLE001
        pass

    frames = []
    for _, g in goals.iterrows():
        goal_ts = float(g["ts"].timestamp())
        windows = [
            (int(goal_ts) - lookback_sec, int(goal_ts) - 1),
            (int(goal_ts) - 1, int(goal_ts) + lookahead_sec),
        ]
        for start_ts, end_ts in windows:
            frames.append(loader.load_historical_bbo(token_id, market_id,
                                                     start_ts, end_ts))
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, format="mixed")
    return (out.dropna(subset=["timestamp", "price"])
            .drop_duplicates(subset=["timestamp", "source"])
            .sort_values("timestamp")
            .reset_index(drop=True))


def _micro_features(bbo: pd.DataFrame) -> pd.DataFrame:
    if bbo.empty:
        return pd.DataFrame()
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
    sec["mid_change_5s"] = sec["price"] - sec["price"].shift(5)
    sec["abs_mid_change_5s"] = sec["mid_change_5s"].abs()
    sec["update_rate_5s"] = sec["update_count"].rolling(5, min_periods=1).sum()
    sec["spread_median_10s"] = sec["abs_spread"].rolling(10, min_periods=1).median()
    sec["spread_shock"] = sec["abs_spread"] - sec["spread_median_10s"]
    sec["depth_median_10s"] = sec["depth"].rolling(10, min_periods=1).median()
    sec["depth_drop_frac"] = 1 - (sec["depth"] / sec["depth_median_10s"].replace(0, np.nan))
    return sec.reset_index().rename(columns={"timestamp": "ts"})


def _first_signal(feat: pd.DataFrame, start_ts, end_ts, reaction_ts,
                  goal_ts) -> list[dict]:
    win = feat[(feat["ts"] >= start_ts) & (feat["ts"] < reaction_ts)].copy()
    if win.empty:
        return []
    update_thr = max(5, float(win["update_rate_5s"].quantile(0.80)))
    spread_thr = max(0.02, float(win["spread_shock"].quantile(0.80)))
    conditions = {
        "early_mid_drift_1c": win["abs_mid_change_5s"] >= 0.01,
        "update_burst_5s": win["update_rate_5s"] >= update_thr,
        "spread_shock": win["spread_shock"] >= spread_thr,
        "depth_drop_50pct": win["depth_drop_frac"] >= 0.50,
    }
    rows = []
    for name, mask in conditions.items():
        hit = win[mask.fillna(False)]
        if hit.empty:
            rows.append({"signal": name, "alert_ts": pd.NaT,
                         "lead_to_bbo_sec": np.nan, "lead_to_goal_sec": np.nan})
        else:
            t = hit["ts"].iloc[0]
            rows.append({
                "signal": name,
                "alert_ts": t,
                "lead_to_bbo_sec": (reaction_ts - t).total_seconds(),
                "lead_to_goal_sec": (goal_ts - t).total_seconds(),
            })
    return rows


def _plot_latency(reactions: pd.DataFrame, out_path: str) -> None:
    df = reactions.dropna(subset=["reaction_latency_sec"]).copy()
    if df.empty:
        return
    labels = df["fixture_id"].astype(str) + "-" + df["goal_idx"].astype(str)
    colors = np.where(df["reaction_latency_sec"] < 0, "#7048e8", "#f03e3e")
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(labels, df["reaction_latency_sec"], color=colors)
    ax.axhline(0, color="#222", lw=1)
    ax.set_ylabel("BBO move - LSports goal (seconds)")
    ax.set_title("Batch BBO move timing around LSports goals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_signal_leads(alerts: pd.DataFrame, out_path: str) -> None:
    df = alerts.dropna(subset=["lead_to_bbo_sec"]).copy()
    if df.empty:
        return
    order = ["early_mid_drift_1c", "update_burst_5s", "spread_shock", "depth_drop_50pct"]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    data = [df[df["signal"] == s]["lead_to_bbo_sec"].to_numpy() for s in order]
    ax.boxplot(data, tick_labels=order, showfliers=False)
    ax.set_ylabel("Lead to 3c BBO move (seconds)")
    ax.set_title("Exploratory BBO microstructure alert lead times")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_windows(features: pd.DataFrame, reactions: pd.DataFrame,
                  alerts: pd.DataFrame, out_path: str) -> None:
    valid = reactions.dropna(subset=["reaction_ts"]).copy()
    if valid.empty:
        return
    n = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(10, max(3.0, 2.4 * n)), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (_, r) in zip(axes, valid.iterrows()):
        fid = str(r["fixture_id"])
        gidx = int(r["goal_idx"])
        goal_ts = pd.to_datetime(r["goal_ts"], utc=True, format="mixed")
        reaction_ts = pd.to_datetime(r["reaction_ts"], utc=True, format="mixed")
        start = goal_ts - pd.Timedelta(seconds=60)
        end = goal_ts + pd.Timedelta(seconds=30)
        fr = features[(features["fixture_id"].astype(str) == fid) &
                      (features["ts"] >= start) & (features["ts"] <= end)]
        ax.plot(fr["ts"], fr["price"], color="#1c7ed6", lw=1.5, label="BBO mid")
        ax.axvline(reaction_ts, color="#7048e8", ls="--", lw=1.2, label="3c BBO move")
        ax.axvline(goal_ts, color="#f03e3e", ls="-", lw=1.2, label="LSports goal")
        al = alerts[(alerts["fixture_id"].astype(str) == fid) &
                    (alerts["goal_idx"] == gidx)].dropna(subset=["alert_ts"])
        for _, a in al.iterrows():
            ax.axvline(pd.to_datetime(a["alert_ts"], utc=True, format="mixed"),
                       color="#2f9e44", alpha=0.25, lw=1)
        ax.set_title(f"fixture {fid} goal {gidx}")
        ax.set_ylabel("mid")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*", default=None)
    ap.add_argument("--lookback-sec", type=int, default=60)
    ap.add_argument("--lookahead-sec", type=int, default=300)
    ap.add_argument("--jump-threshold", type=float, default=0.03)
    args = ap.parse_args()

    mapping = load_mapping_table(resolve_path(cfg["polymarket"]["mapping_file"]))
    if args.fixtures:
        keep = {str(x) for x in args.fixtures}
        mapping = mapping[mapping["fixture_id"].astype(str).isin(keep)].copy()
    loader = PolymarketPriceLoader(cfg)

    all_reactions, all_alerts, all_features, all_bbo, fixture_rows = [], [], [], [], []
    for _, row in mapping.iterrows():
        fid = str(row["fixture_id"])
        print(f"[bbo-alert] fixture={fid}")
        res = DL.download_fixture(cfg, str(row["event_date"]), fid)
        events = P.parse_messages(DL.load_messages(res["messages"]))
        goals = P.extract_goals(events)
        if goals.empty:
            continue
        bbo = _load_or_fetch_bbo(
            loader, str(row["polymarket_market_id"]), str(row["primary_market_id"]),
            fid, goals, args.lookback_sec, args.lookahead_sec)
        feat = _micro_features(bbo)
        if feat.empty:
            fixture_rows.append({"fixture_id": fid, "status": "no_bbo"})
            continue
        bbo_out = bbo.copy()
        bbo_out.insert(0, "fixture_id", fid)
        all_bbo.append(bbo_out)
        feat.insert(0, "fixture_id", fid)
        all_features.append(feat)
        reaction = LA.event_price_reaction_signed(
            goals, bbo[["timestamp", "price"]].copy(),
            jump_threshold=args.jump_threshold,
            baseline_sec=10,
            lookahead_sec=args.lookahead_sec,
        )
        reaction.insert(0, "fixture_id", fid)
        reaction.insert(1, "event_date", str(row["event_date"]))
        reaction.insert(2, "market_id", str(row["primary_market_id"]))
        reaction.insert(3, "goal_idx", range(1, len(reaction) + 1))
        all_reactions.append(reaction)

        valid = reaction.dropna(subset=["reaction_ts"]).copy()
        for _, r in valid.iterrows():
            goal_ts = pd.to_datetime(r["goal_ts"], utc=True, format="mixed")
            reaction_ts = pd.to_datetime(r["reaction_ts"], utc=True, format="mixed")
            start_ts = pd.to_datetime(r["base_ts"], utc=True, format="mixed")
            rows = _first_signal(feat, start_ts, goal_ts, reaction_ts, goal_ts)
            for a in rows:
                a.update({
                    "fixture_id": fid,
                    "goal_idx": int(r["goal_idx"]),
                    "goal_ts": goal_ts,
                    "reaction_ts": reaction_ts,
                    "bbo_lag_sec": float(r["reaction_latency_sec"]),
                })
                all_alerts.append(a)
        lat = pd.to_numeric(reaction["reaction_latency_sec"], errors="coerce")
        fixture_rows.append({
            "fixture_id": fid,
            "home": row.get("team_home", ""),
            "away": row.get("team_away", ""),
            "n_goals": len(goals),
            "n_bbo_snapshots": len(bbo),
            "n_bbo_moves": int(lat.notna().sum()),
            "bbo_first_count": int((lat < 0).sum()),
            "lsports_first_count": int((lat > 0).sum()),
            "median_bbo_lag_sec": float(lat.dropna().median()) if lat.notna().any() else np.nan,
            "status": "ok",
        })

    reactions = pd.concat(all_reactions, ignore_index=True) if all_reactions else pd.DataFrame()
    features = pd.concat(all_features, ignore_index=True) if all_features else pd.DataFrame()
    bbo_snapshots = pd.concat(all_bbo, ignore_index=True) if all_bbo else pd.DataFrame()
    alerts = pd.DataFrame(all_alerts)
    summary = pd.DataFrame(fixture_rows)

    tdir = resolve_path(cfg["paths"]["table_dir"])
    fdir = resolve_path(cfg["paths"]["fig_dir"])
    rdir = resolve_path(cfg["paths"]["report_dir"])
    reactions.to_csv(f"{tdir}/batch_bbo_move_reactions.csv",
                     index=False, encoding="utf-8-sig")
    alerts.to_csv(f"{tdir}/batch_bbo_micro_alert_candidates.csv",
                  index=False, encoding="utf-8-sig")
    summary.to_csv(f"{tdir}/batch_bbo_fixture_summary.csv",
                   index=False, encoding="utf-8-sig")
    bbo_snapshots.to_csv(f"{tdir}/batch_bbo_compact_snapshots.csv",
                         index=False, encoding="utf-8-sig")

    _plot_latency(reactions, f"{fdir}/batch_bbo_move_latency.png")
    _plot_signal_leads(alerts, f"{fdir}/batch_bbo_micro_alert_leads.png")
    _plot_windows(features, reactions, alerts,
                  f"{fdir}/batch_bbo_microstructure_windows.png")

    valid = reactions.dropna(subset=["reaction_latency_sec"])
    lead_stats = alerts.dropna(subset=["lead_to_bbo_sec"]).groupby("signal").agg(
        n=("lead_to_bbo_sec", "size"),
        median_lead_to_bbo_sec=("lead_to_bbo_sec", "median"),
        median_lead_to_goal_sec=("lead_to_goal_sec", "median"),
    ).reset_index() if not alerts.empty else pd.DataFrame()

    lines = [
        "# Batch BBO Move And Microstructure Alert Analysis",
        "",
        "## Scope",
        "",
        "This report uses mapped fixtures with retrievable historical BBO snapshots "
        "around LSports goal timestamps. It does not save raw orderbook archives.",
        "",
        "## Fixture Summary",
        "",
        SE.markdown_table(summary) if not summary.empty else "_No fixtures._",
        "",
        "## BBO Move Timing",
        "",
        SE.markdown_table(valid[["fixture_id", "goal_idx", "goal_ts", "reaction_ts",
                                 "reaction_latency_sec", "price_jump_magnitude"]])
        if not valid.empty else "_No detected BBO moves._",
        "",
        "Negative latency means BBO moved before the LSports goal timestamp.",
        "",
        "## Exploratory Pre-Move Signals",
        "",
        SE.markdown_table(lead_stats) if not lead_stats.empty else "_No signal hits._",
        "",
        "Interpretation: BBO-only early alerts are often just early repricing, "
        "quote churn, or liquidity withdrawal. They should be combined with "
        "LSports event intensity to distinguish true goal risk from noisy market "
        "microstructure.",
        "",
        "Figures:",
        "",
        "- `outputs/figures/batch_bbo_move_latency.png`",
        "- `outputs/figures/batch_bbo_micro_alert_leads.png`",
        "- `outputs/figures/batch_bbo_microstructure_windows.png`",
    ]
    with open(f"{rdir}/batch_bbo_move_alert_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report: {rdir}/batch_bbo_move_alert_report.md")


if __name__ == "__main__":
    main()
