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


def _alert_episodes(frame: pd.DataFrame, hazard: np.ndarray, threshold: float,
                    cooldown_sec: int = 90) -> pd.DataFrame:
    ts = pd.to_datetime(frame["ts"], utc=True, format="mixed").reset_index(drop=True)
    h = pd.Series(hazard).reset_index(drop=True)
    alerts = ts[h >= threshold]
    rows, last = [], pd.NaT
    for t in alerts:
        if pd.isna(last) or (t - last).total_seconds() >= cooldown_sec:
            rows.append({"alert_ts": t})
            last = t
    return pd.DataFrame(rows)


def _score_threshold(frame: pd.DataFrame, hazard: np.ndarray,
                     goals: pd.DataFrame, bbo_reaction: pd.DataFrame,
                     threshold: float, warning_sec: int = 180) -> dict:
    alerts = _alert_episodes(frame, hazard, threshold)
    goals_ts = pd.to_datetime(goals["ts"], utc=True, format="mixed") \
        if goals is not None and not goals.empty else pd.Series(dtype="datetime64[ns, UTC]")
    bbo_ts = pd.to_datetime(bbo_reaction["reaction_ts"], utc=True, format="mixed").dropna() \
        if bbo_reaction is not None and not bbo_reaction.empty else pd.Series(dtype="datetime64[ns, UTC]")
    true_alerts = 0
    bbo_preempt = 0
    covered_goals = set()
    covered_bbo = set()
    lead_secs = []
    for _, a in alerts.iterrows():
        t = a["alert_ts"]
        future_goals = goals_ts[(goals_ts >= t) & (goals_ts <= t + pd.Timedelta(seconds=warning_sec))]
        if not future_goals.empty:
            true_alerts += 1
            g = future_goals.iloc[0]
            covered_goals.add(str(g))
            lead_secs.append((g - t).total_seconds())
            future_bbo = bbo_ts[(bbo_ts >= t) & (bbo_ts <= g + pd.Timedelta(seconds=10))]
            if not future_bbo.empty and t < future_bbo.iloc[0]:
                bbo_preempt += 1
                covered_bbo.add(str(future_bbo.iloc[0]))
    n_alerts = len(alerts)
    precision = true_alerts / n_alerts if n_alerts else np.nan
    recall = len(covered_goals) / len(goals_ts) if len(goals_ts) else np.nan
    return {
        "threshold": threshold,
        "n_alerts": n_alerts,
        "true_alerts": true_alerts,
        "false_alerts": n_alerts - true_alerts,
        "precision": precision,
        "goal_recall": recall,
        "bbo_preempt_alerts": bbo_preempt,
        "unique_bbo_moves_preempted": len(covered_bbo),
        "mean_goal_lead_sec": np.mean(lead_secs) if lead_secs else np.nan,
    }


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |",
             "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("" if np.isnan(v) else f"{v:.3f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


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
        _score_threshold(frame, hazard, goals, bbo_reaction, float(t),
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
        _markdown_table(metrics),
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
