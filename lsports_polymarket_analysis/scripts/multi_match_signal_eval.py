"""
multi_match_signal_eval.py - Test the LSports alert formula on mapped soccer games.

For each fixture in config/polymarket_fixture_map.csv:
1. Load LSports event stream and goals.
2. Pull PMXT/Dome historical BBO only around each goal window.
3. Detect first signed BBO mid move around each goal.
4. Fit the same prototype LSports hazard score and evaluate alert thresholds.

The script writes compact reaction/metric summaries, not raw orderbook archives.
"""
import _bootstrap  # noqa: F401
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL
from src import feature_engineering as FE
from src import latency_analysis as LA
from src import lsports_parser as P
from src import modeling as M
from src import signal_evaluation as SE
from src.polymarket_loader import PolymarketPriceLoader, load_mapping_table


def _historical_bbo_for_goals(loader: PolymarketPriceLoader, token_id: str,
                              market_id: str, goals: pd.DataFrame,
                              lookback_sec: int,
                              lookahead_sec: int) -> pd.DataFrame:
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
    return (pd.concat(frames, ignore_index=True)
            .dropna(subset=["timestamp", "price"])
            .drop_duplicates(subset=["timestamp", "source"])
            .sort_values("timestamp")
            .reset_index(drop=True))


def _hazard_frame(events: pd.DataFrame, fixture_id: str, cfg: dict,
                  step_sec: int, horizon_sec: int) -> tuple[pd.DataFrame, np.ndarray]:
    frame = FE.build_feature_frame(
        events,
        step_sec=step_sec,
        windows_sec=tuple(cfg["analysis"]["intensity_windows_sec"]),
        horizon_sec=horizon_sec,
        fixture_id=fixture_id,
    )
    if frame.empty:
        return frame, np.array([])
    fit = M.fit_logistic_hazard(frame, test_frac=0.0)
    hazard = fit["pred"] if fit["pred"] is not None else np.zeros(len(frame))
    hazard = pd.Series(hazard).rolling(3, min_periods=1).mean().to_numpy()
    return frame, hazard


def _pick_operating_point(metrics: pd.DataFrame, fallback: float) -> float:
    if metrics.empty:
        return fallback
    candidates = metrics[(metrics["precision"] >= 0.5) &
                         (metrics["n_alerts"] <= 8)]
    if candidates.empty:
        return fallback
    return float(candidates.sort_values(
        ["goal_recall", "precision"], ascending=False)["threshold"].iloc[0])


def _plot_latency_distribution(reactions: pd.DataFrame, out_path: str) -> None:
    if reactions.empty:
        return
    df = reactions.dropna(subset=["reaction_latency_sec"]).copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = np.where(df["reaction_latency_sec"] < 0, "#7048e8", "#f03e3e")
    labels = df["fixture_id"].astype(str) + "\n" + df["goal_idx"].astype(str)
    ax.bar(range(len(df)), df["reaction_latency_sec"], color=colors)
    ax.axhline(0, color="#222", lw=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("BBO move - LSports goal (seconds)")
    ax.set_title("Historical BBO reaction timing around LSports goals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_threshold_precision(metrics: pd.DataFrame, out_path: str) -> None:
    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for fixture_id, g in metrics.groupby("fixture_id"):
        ax.plot(g["threshold"], g["precision"], marker="o", lw=1.5,
                label=str(fixture_id))
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("alert threshold")
    ax.set_ylabel("precision")
    ax.set_title("False-alert tradeoff by fixture")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, title="fixture")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures", nargs="*", default=None,
                    help="Optional fixture ids. Defaults to every row in mapping CSV.")
    ap.add_argument("--step-sec", type=int, default=10)
    ap.add_argument("--horizon-sec", type=int, default=180)
    ap.add_argument("--lookback-sec", type=int, default=10)
    ap.add_argument("--lookahead-sec", type=int, default=300)
    ap.add_argument("--jump-threshold", type=float, default=0.03)
    args = ap.parse_args()

    mapping = load_mapping_table(resolve_path(cfg["polymarket"]["mapping_file"]))
    if args.fixtures:
        keep = {str(x) for x in args.fixtures}
        mapping = mapping[mapping["fixture_id"].astype(str).isin(keep)].copy()
    if mapping.empty:
        raise SystemExit("No mapped fixtures to evaluate.")

    loader = PolymarketPriceLoader(cfg)
    all_reactions, all_metrics, summaries = [], [], []
    for _, row in mapping.iterrows():
        fixture_id = str(row["fixture_id"])
        event_date = str(row["event_date"])
        print(f"[multi] fixture={fixture_id} date={event_date}")
        res = DL.download_fixture(cfg, event_date, fixture_id)
        events = P.parse_messages(DL.load_messages(res["messages"]))
        goals = P.extract_goals(events)
        if events.empty or goals.empty:
            summaries.append({"fixture_id": fixture_id, "status": "no_goals"})
            continue

        market_id = str(row["primary_market_id"])
        token_id = str(row["polymarket_market_id"])
        bbo = _historical_bbo_for_goals(loader, token_id, market_id, goals,
                                        args.lookback_sec, args.lookahead_sec)
        reaction = LA.event_price_reaction_signed(
            goals, bbo[["timestamp", "price"]].copy() if not bbo.empty else bbo,
            jump_threshold=args.jump_threshold,
            baseline_sec=args.lookback_sec,
            lookahead_sec=args.lookahead_sec,
        )
        reaction.insert(0, "fixture_id", fixture_id)
        reaction.insert(1, "event_date", event_date)
        reaction.insert(2, "market_id", market_id)
        reaction.insert(3, "goal_idx", range(1, len(reaction) + 1))
        all_reactions.append(reaction)

        frame, hazard = _hazard_frame(events, fixture_id, cfg,
                                      args.step_sec, args.horizon_sec)
        if frame.empty or hazard.size == 0:
            summaries.append({"fixture_id": fixture_id, "status": "no_features"})
            continue
        thresholds = np.unique(np.round(
            np.nanpercentile(hazard, [70, 75, 80, 85, 90, 95]), 3))
        metrics = pd.DataFrame([
            SE.score_threshold(frame, hazard, goals, reaction, float(t),
                               warning_sec=args.horizon_sec)
            for t in thresholds
        ])
        metrics.insert(0, "fixture_id", fixture_id)
        metrics.insert(1, "event_date", event_date)
        metrics.insert(2, "market_id", market_id)
        all_metrics.append(metrics)
        operating_threshold = _pick_operating_point(
            metrics, float(np.nanpercentile(hazard, 85)))
        chosen_idx = (metrics["threshold"] - operating_threshold).abs().idxmin()
        chosen = metrics.loc[chosen_idx].to_dict()
        operating_threshold = float(chosen["threshold"])
        lat = pd.to_numeric(reaction["reaction_latency_sec"], errors="coerce")
        summaries.append({
            "fixture_id": fixture_id,
            "event_date": event_date,
            "home": row.get("team_home", ""),
            "away": row.get("team_away", ""),
            "market_id": market_id,
            "n_goals": len(goals),
            "n_bbo_snapshots": len(bbo),
            "n_bbo_moves": int(lat.notna().sum()),
            "bbo_first_count": int((lat < 0).sum()),
            "lsports_first_count": int((lat > 0).sum()),
            "median_bbo_lag_sec": float(lat.dropna().median()) if lat.notna().any() else np.nan,
            "operating_threshold": operating_threshold,
            "alerts": int(chosen["n_alerts"]),
            "true_alerts": int(chosen["true_alerts"]),
            "false_alerts": int(chosen["false_alerts"]),
            "precision": float(chosen["precision"]) if pd.notna(chosen["precision"]) else np.nan,
            "goal_recall": float(chosen["goal_recall"]) if pd.notna(chosen["goal_recall"]) else np.nan,
            "unique_bbo_moves_preempted": int(chosen["unique_bbo_moves_preempted"]),
            "mean_goal_lead_sec": float(chosen["mean_goal_lead_sec"])
            if pd.notna(chosen["mean_goal_lead_sec"]) else np.nan,
            "status": "ok",
        })

    tdir = resolve_path(cfg["paths"]["table_dir"])
    fdir = resolve_path(cfg["paths"]["fig_dir"])
    rdir = resolve_path(cfg["paths"]["report_dir"])
    reactions = pd.concat(all_reactions, ignore_index=True) if all_reactions else pd.DataFrame()
    metrics = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    summary = pd.DataFrame(summaries)

    reactions.to_csv(f"{tdir}/multi_match_bbo_reactions.csv",
                     index=False, encoding="utf-8-sig")
    metrics.to_csv(f"{tdir}/multi_match_signal_threshold_metrics.csv",
                   index=False, encoding="utf-8-sig")
    summary.to_csv(f"{tdir}/multi_match_signal_eval_summary.csv",
                   index=False, encoding="utf-8-sig")
    _plot_latency_distribution(reactions,
                               f"{fdir}/multi_match_bbo_latency_distribution.png")
    _plot_threshold_precision(metrics,
                              f"{fdir}/multi_match_signal_precision.png")

    lines = [
        "# Multi-Match LSports Signal Evaluation",
        "",
        "## What `alert` Means",
        "",
        "`alert` is not an official abnormal-match flag from LSports. It is a "
        "prototype model threshold crossing computed from LSports event-flow features.",
        "",
        "At every 10-second step `t`, the feature vector includes score state, "
        "match minute, cards, period, and rolling 60/180/300/600 second counts "
        "or xT-weighted sums of attacks, dangerous attacks, shots, corners, and "
        "all live events. The label is:",
        "",
        "`y_t = 1{a goal occurs in (t, t + 180 seconds]}`",
        "",
        "The fitted score is:",
        "",
        "`hazard_t = sigmoid(beta_0 + sum_j beta_j * zscore(x_{t,j}))`",
        "",
        "The plotted line is a 3-step rolling average of `hazard_t`. An alert "
        "episode starts when the smoothed score exceeds a threshold; repeated "
        "crossings inside 90 seconds are collapsed into one episode.",
        "",
        "An alert is a true alert if a goal happens within the next 180 seconds. "
        "Otherwise it is counted as a false alert. A BBO move is preempted only "
        "when the alert timestamp is before the first BBO mid-price move linked "
        "to that goal.",
        "",
        "## Fixture Summary",
        "",
        SE.markdown_table(summary),
        "",
        "## Threshold Metrics",
        "",
        SE.markdown_table(metrics),
        "",
        "## Interpretation",
        "",
        "This is still a prototype, because each fixture is fitted on its own "
        "event stream. The results are useful for checking whether the formula "
        "has intuitive behavior and whether thresholding produces too many false "
        "alerts. Production use should train on many prior matches and validate "
        "by held-out fixture/date/league.",
        "",
        "Figures:",
        "",
        "- `outputs/figures/multi_match_bbo_latency_distribution.png`",
        "- `outputs/figures/multi_match_signal_precision.png`",
    ]
    with open(f"{rdir}/multi_match_signal_eval_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"multi summary: {tdir}/multi_match_signal_eval_summary.csv")
    print(f"multi report: {rdir}/multi_match_signal_eval_report.md")


if __name__ == "__main__":
    main()
