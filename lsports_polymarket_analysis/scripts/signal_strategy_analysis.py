"""
signal_strategy_analysis.py - Prototype LSports early-warning signal design.

Goal: explore whether LSports event-flow features can warn before Polymarket
BBO jumps caused by goals, while limiting false positives.
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL
from src import lsports_parser as P
from src import feature_engineering as FE
from src import modeling as M
from src import visualization as V
from src import signal_evaluation as SE


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=cfg["single_match"]["fixture_id"])
    ap.add_argument("--date", default=cfg["single_match"]["event_date"])
    ap.add_argument("--step-sec", type=int, default=10)
    ap.add_argument("--horizon-sec", type=int, default=180)
    args = ap.parse_args()

    res = DL.download_fixture(cfg, args.date, args.fixture)
    events = P.parse_messages(DL.load_messages(res["messages"]))
    goals = P.extract_goals(events)
    bbo_path = resolve_path(cfg["paths"]["table_dir"]) + \
        f"/single_match_bbo_reaction_{args.fixture}.csv"
    bbo_reaction = pd.read_csv(bbo_path) if pd.io.common.file_exists(bbo_path) \
        else pd.DataFrame()

    frame = FE.build_feature_frame(
        events, step_sec=args.step_sec,
        windows_sec=tuple(cfg["analysis"]["intensity_windows_sec"]),
        horizon_sec=args.horizon_sec, fixture_id=args.fixture)
    fit = M.fit_logistic_hazard(frame, test_frac=0.0)
    hazard = fit["pred"] if fit["pred"] is not None else np.zeros(len(frame))
    # Smooth noisy single-match fitted probabilities into a usable alert score.
    hazard = pd.Series(hazard).rolling(3, min_periods=1).mean().to_numpy()

    thresholds = np.unique(np.round(np.nanpercentile(hazard, [70, 75, 80, 85, 90, 95]), 3))
    metrics = pd.DataFrame([
        SE.score_threshold(frame, hazard, goals, bbo_reaction, float(t),
                           warning_sec=args.horizon_sec)
        for t in thresholds
    ])
    tdir = resolve_path(cfg["paths"]["table_dir"])
    metrics.to_csv(f"{tdir}/single_match_signal_threshold_metrics.csv",
                   index=False, encoding="utf-8-sig")

    # Pick a high-precision operating point if available; otherwise 85th percentile.
    candidates = metrics[(metrics["precision"] >= 0.5) &
                         (metrics["n_alerts"] <= 8)]
    threshold = float(candidates.sort_values(
        ["goal_recall", "precision"], ascending=False)["threshold"].iloc[0]) \
        if not candidates.empty else float(np.nanpercentile(hazard, 85))
    V.plot_signal_timeline(
        frame, hazard, goals, bbo_reaction, threshold,
        "LSports pre-goal signal vs BBO reaction times",
        resolve_path(cfg["paths"]["fig_dir"]) +
        "/single_match_signal_vs_bbo.png")

    lines = [
        "# LSports Signal Strategy Prototype",
        "",
        "## Objective",
        "",
        "Use LSports event-flow features to forecast goal-driven BBO repricing "
        "before the BBO itself jumps, while keeping false alerts low.",
        "",
        "## Prototype",
        "",
        f"- Fixture: `{args.fixture}`",
        f"- Feature step: {args.step_sec}s",
        f"- Prediction horizon: {args.horizon_sec}s",
        "- Score: logistic goal-hazard fitted on rolling LSports event features, "
        "smoothed over 3 steps.",
        "- Alert scoring: collapse repeated threshold crossings into episodes "
        "with 90s cooldown.",
        "",
        "## Threshold Sweep",
        "",
        SE.markdown_table(metrics),
        "",
        "## Recommended Modeling Path",
        "",
        "1. Use historical BBO mid moves as the target: future `abs(mid - baseline) >= 3c` "
        "within 10-180s, not just future goals.",
        "2. Train cross-match models with grouped validation by fixture/date/league.",
        "3. Prefer calibrated gradient boosting or discrete-time survival models over "
        "single-match logistic regression.",
        "4. Add false-positive controls: minimum liquidity/spread filters, cooldown, "
        "probability calibration, and precision-at-k alert thresholds.",
        "5. Deploy as two-stage system: LSports goal-hazard alert first; trade/avoid only "
        "when BBO has not already moved or spread/liquidity allow execution.",
        "",
        "## Interpretation",
        "",
        "For this match, PMXT BBO moved a few seconds before the LSports goal timestamp. "
        "Therefore the direct edge is not simply 'trade after LSports goal'. The research "
        "opportunity is earlier: use LSports event intensity before the goal to anticipate "
        "the BBO jump, and treat BBO movement as the benchmark to beat.",
    ]
    out = resolve_path(cfg["paths"]["report_dir"]) + "/signal_strategy_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in lines))
    print(f"Signal strategy report: {out}")


if __name__ == "__main__":
    main()
