"""
bbo_micro_alert_strategy.py - Rule/model tests for pre-BBO-move alerts.

Question:
    Can BBO microstructure itself provide a high-precision early alert before a
    larger goal-driven repricing?

Target used here:
    alert is true if a detected 3c BBO move occurs within the next N seconds.

Important limitation:
    The available complete historical BBO sample is currently tiny (4 usable
    goal-driven BBO moves across 2 fixtures). Results are exploratory only.
"""
import _bootstrap  # noqa: F401
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import load_config, resolve_path
from src import signal_evaluation as SE


FEATURE_COLS = [
    "abs_mid_change_5s",
    "abs_mid_change_10s",
    "update_rate_5s",
    "update_rate_10s",
    "spread_shock",
    "depth_drop_frac",
    "abs_spread",
    "depth",
    "imbalance",
]


def _load_cached_bbo_features() -> pd.DataFrame:
    """Use compact BBO files already produced by prior scripts."""
    batch = resolve_path("outputs/tables/batch_bbo_compact_snapshots.csv")
    try:
        df = pd.read_csv(batch)
        if not df.empty and "fixture_id" in df.columns:
            return df
    except Exception:  # noqa: BLE001
        pass

    paths = [
        "outputs/tables/single_match_historical_bbo_18746260.csv",
    ]
    frames = []
    for rel in paths:
        p = resolve_path(rel)
        try:
            df = pd.read_csv(p)
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        fixture_id = rel.rsplit("_", 1)[-1].replace(".csv", "")
        df.insert(0, "fixture_id", fixture_id)
        frames.append(df)

    # For non-cached fixtures, reconstruct compact features from the generated
    # reaction-window plot inputs is not possible. The strategy test therefore
    # evaluates cached raw BBO plus any future cached files with the same naming
    # convention.
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _micro_features(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fixture_id, bbo in raw.groupby("fixture_id"):
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
        sec = sec.reset_index().rename(columns={"timestamp": "ts"})
        sec.insert(0, "fixture_id", str(fixture_id))
        rows.append(sec)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _reaction_events() -> pd.DataFrame:
    p = resolve_path("outputs/tables/batch_bbo_move_reactions.csv")
    r = pd.read_csv(p)
    r = r.dropna(subset=["reaction_ts"]).copy()
    r["fixture_id"] = r["fixture_id"].astype(str)
    r["reaction_ts"] = pd.to_datetime(r["reaction_ts"], utc=True, format="mixed")
    r["goal_ts"] = pd.to_datetime(r["goal_ts"], utc=True, format="mixed")
    return r


def _label_future_moves(feat: pd.DataFrame, reactions: pd.DataFrame,
                        horizon_sec: int) -> pd.DataFrame:
    feat = feat.copy()
    feat["label_future_bbo_move"] = 0
    feat["next_reaction_ts"] = ""
    for fixture_id, rr in reactions.groupby("fixture_id"):
        idx = feat["fixture_id"].astype(str) == str(fixture_id)
        ts = pd.to_datetime(feat.loc[idx, "ts"], utc=True, format="mixed")
        labels = []
        nexts = []
        for t in ts:
            fut = rr[(rr["reaction_ts"] > t) &
                     (rr["reaction_ts"] <= t + pd.Timedelta(seconds=horizon_sec))]
            labels.append(int(not fut.empty))
            nexts.append(str(fut["reaction_ts"].iloc[0]) if not fut.empty else "")
        feat.loc[idx, "label_future_bbo_move"] = labels
        feat.loc[idx, "next_reaction_ts"] = nexts
    return feat


def _episode_times(feat: pd.DataFrame, mask: pd.Series,
                   cooldown_sec: int = 10) -> pd.DataFrame:
    rows, last = [], pd.NaT
    was_on = False
    for (_, row), on in zip(feat.iterrows(), mask.fillna(False).to_numpy()):
        t = row["ts"]
        if on and not was_on:
            if pd.isna(last) or (t - last).total_seconds() >= cooldown_sec:
                rows.append({"fixture_id": row["fixture_id"], "alert_ts": t})
                last = t
        was_on = bool(on)
    return pd.DataFrame(rows)


def _score_episodes(episodes: pd.DataFrame, reactions: pd.DataFrame,
                    horizon_sec: int) -> dict:
    if episodes.empty:
        return {
            "n_alerts": 0, "true_alerts": 0, "false_alerts": 0,
            "precision": np.nan, "covered_moves": 0, "move_recall": 0.0,
            "median_lead_sec": np.nan,
        }
    true_alerts, covered, leads = 0, set(), []
    for _, a in episodes.iterrows():
        rr = reactions[reactions["fixture_id"].astype(str) == str(a["fixture_id"])]
        fut = rr[(rr["reaction_ts"] > a["alert_ts"]) &
                 (rr["reaction_ts"] <= a["alert_ts"] + pd.Timedelta(seconds=horizon_sec))]
        if not fut.empty:
            true_alerts += 1
            rts = fut["reaction_ts"].iloc[0]
            covered.add((str(a["fixture_id"]), str(rts)))
            leads.append((rts - a["alert_ts"]).total_seconds())
    n = len(episodes)
    return {
        "n_alerts": n,
        "true_alerts": true_alerts,
        "false_alerts": n - true_alerts,
        "precision": true_alerts / n if n else np.nan,
        "covered_moves": len(covered),
        "move_recall": len(covered) / len(reactions) if len(reactions) else np.nan,
        "median_lead_sec": float(np.median(leads)) if leads else np.nan,
    }


def _rule_grid(feat: pd.DataFrame, reactions: pd.DataFrame,
               horizon_sec: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rule_defs = {
        "mid1c": feat["abs_mid_change_5s"] >= 0.01,
        "mid2c": feat["abs_mid_change_5s"] >= 0.02,
        "spread2c": feat["spread_shock"] >= 0.02,
        "depth50": feat["depth_drop_frac"] >= 0.50,
        "update10": feat["update_rate_5s"] >= 10,
    }
    combos = {
        "mid1c": rule_defs["mid1c"],
        "mid1c_and_spread2c": rule_defs["mid1c"] & rule_defs["spread2c"],
        "mid1c_and_depth50": rule_defs["mid1c"] & rule_defs["depth50"],
        "mid1c_and_update10": rule_defs["mid1c"] & rule_defs["update10"],
        "mid1c_and_any_liquidity_stress": rule_defs["mid1c"] & (
            rule_defs["spread2c"] | rule_defs["depth50"] | rule_defs["update10"]),
        "spread2c_or_depth50_update10": (rule_defs["spread2c"] | rule_defs["depth50"]) &
            rule_defs["update10"],
    }
    rows, all_eps = [], []
    for name, mask in combos.items():
        eps = _episode_times(feat, mask)
        score = _score_episodes(eps, reactions, horizon_sec)
        score["strategy"] = name
        rows.append(score)
        if not eps.empty:
            tmp = eps.copy()
            tmp["strategy"] = name
            all_eps.append(tmp)
    return pd.DataFrame(rows), pd.concat(all_eps, ignore_index=True) if all_eps else pd.DataFrame()


def _cross_fixture_model(feat: pd.DataFrame, reactions: pd.DataFrame,
                         horizon_sec: int) -> pd.DataFrame:
    rows = []
    fixtures = sorted(feat["fixture_id"].astype(str).unique())
    for test_fixture in fixtures:
        train = feat[feat["fixture_id"].astype(str) != test_fixture].copy()
        test = feat[feat["fixture_id"].astype(str) == test_fixture].copy()
        if train["label_future_bbo_move"].nunique() < 2 or test.empty:
            continue
        Xtr = train[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        ytr = train["label_future_bbo_move"].astype(int)
        Xte = test[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler = StandardScaler()
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(scaler.fit_transform(Xtr), ytr)
        test = test.copy()
        test["model_score"] = model.predict_proba(scaler.transform(Xte))[:, 1]
        for threshold in [0.5, 0.7, 0.9]:
            eps = _episode_times(test, test["model_score"] >= threshold)
            score = _score_episodes(eps, reactions[reactions["fixture_id"].astype(str) == test_fixture],
                                    horizon_sec)
            score.update({"test_fixture": test_fixture, "threshold": threshold})
            rows.append(score)
    return pd.DataFrame(rows)


def _plot_rule_results(rules: pd.DataFrame, out_path: str) -> None:
    if rules.empty:
        return
    df = rules.sort_values(["precision", "move_recall"], ascending=[False, False])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df))
    ax.bar(x - 0.18, df["precision"], width=0.36, label="precision", color="#2f9e44")
    ax.bar(x + 0.18, df["move_recall"], width=0.36, label="move recall", color="#1c7ed6")
    ax.set_xticks(x)
    ax.set_xticklabels(df["strategy"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("Rule-based pre-BBO-move alert tests")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_mid_vs_microstructure(feat: pd.DataFrame, reactions: pd.DataFrame,
                                out_path: str) -> None:
    valid = reactions.dropna(subset=["reaction_ts"]).copy()
    if valid.empty or feat.empty:
        return
    n = len(valid)
    fig, axes = plt.subplots(n, 1, figsize=(12, max(3.2, 2.7 * n)),
                             sharex=False)
    if n == 1:
        axes = [axes]
    micro_cols = [
        ("abs_mid_change_5s", "#f08c00"),
        ("spread_shock", "#7048e8"),
        ("depth_drop_frac", "#2f9e44"),
        ("update_rate_5s", "#e03131"),
    ]
    for ax, (_, r) in zip(axes, valid.iterrows()):
        fid = str(r["fixture_id"])
        goal_ts = pd.to_datetime(r["goal_ts"], utc=True, format="mixed")
        reaction_ts = pd.to_datetime(r["reaction_ts"], utc=True, format="mixed")
        start = reaction_ts - pd.Timedelta(seconds=20)
        end = goal_ts + pd.Timedelta(seconds=15)
        w = feat[(feat["fixture_id"].astype(str) == fid) &
                 (feat["ts"] >= start) & (feat["ts"] <= end)].copy()
        if w.empty:
            continue
        ax.plot(w["ts"], w["price"], color="#1c7ed6", lw=1.8,
                label="BBO mid")
        ax.set_ylabel("BBO mid")
        ax2 = ax.twinx()
        for col, color in micro_cols:
            vals = pd.to_numeric(w[col], errors="coerce").fillna(0)
            denom = vals.abs().max()
            scaled = vals / denom if denom and denom > 0 else vals
            ax2.plot(w["ts"], scaled, color=color, lw=1.1, alpha=0.75,
                     label=f"{col} normalized")
        ax.axvline(reaction_ts, color="#7048e8", ls="--", lw=1.3,
                   label="3c BBO move")
        ax.axvline(goal_ts, color="#f03e3e", ls="-", lw=1.3,
                   label="LSports goal")
        ax2.set_ylabel("normalized microstructure")
        ax.set_title(f"fixture {fid} goal {int(r['goal_idx'])}: mid vs microstructure")
        ax.grid(alpha=0.22)
    legend_items = [
        plt.Line2D([0], [0], color="#1c7ed6", lw=1.8, label="BBO mid"),
        plt.Line2D([0], [0], color="#7048e8", ls="--", lw=1.3, label="3c BBO move"),
        plt.Line2D([0], [0], color="#f03e3e", lw=1.3, label="LSports goal"),
        plt.Line2D([0], [0], color="#f08c00", lw=1.1, label="mid drift"),
        plt.Line2D([0], [0], color="#7048e8", lw=1.1, label="spread shock"),
        plt.Line2D([0], [0], color="#2f9e44", lw=1.1, label="depth drop"),
        plt.Line2D([0], [0], color="#e03131", lw=1.1, label="update burst"),
    ]
    fig.legend(handles=legend_items, loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon-sec", type=int, default=10)
    args = ap.parse_args()

    raw = _load_cached_bbo_features()
    reactions = _reaction_events()
    if raw.empty or reactions.empty:
        raise SystemExit("Need cached BBO and batch_bbo_move_reactions first.")
    feat = _micro_features(raw)
    cached_fixtures = set(feat["fixture_id"].astype(str).unique())
    reactions = reactions[reactions["fixture_id"].astype(str).isin(cached_fixtures)].copy()
    if reactions.empty:
        raise SystemExit("No reaction events match cached BBO fixtures.")
    feat = _label_future_moves(feat, reactions, args.horizon_sec)

    # Keep rows with enough lagged history and within cached reaction windows.
    feat = feat.dropna(subset=["price"]).copy()
    rule_summary, rule_episodes = _rule_grid(feat, reactions, args.horizon_sec)
    model_summary = _cross_fixture_model(feat, reactions, args.horizon_sec)

    tdir = resolve_path(cfg["paths"]["table_dir"])
    fdir = resolve_path(cfg["paths"]["fig_dir"])
    rdir = resolve_path(cfg["paths"]["report_dir"])
    feat[["fixture_id", "ts", "price", "label_future_bbo_move"] + FEATURE_COLS].to_csv(
        f"{tdir}/bbo_micro_model_frame.csv", index=False, encoding="utf-8-sig")
    rule_summary.to_csv(f"{tdir}/bbo_micro_rule_strategy_summary.csv",
                        index=False, encoding="utf-8-sig")
    rule_episodes.to_csv(f"{tdir}/bbo_micro_rule_alert_episodes.csv",
                         index=False, encoding="utf-8-sig")
    model_summary.to_csv(f"{tdir}/bbo_micro_model_leave_fixture_out.csv",
                         index=False, encoding="utf-8-sig")
    _plot_rule_results(rule_summary,
                       f"{fdir}/bbo_micro_rule_strategy_precision.png")
    _plot_mid_vs_microstructure(feat, reactions,
                                f"{fdir}/bbo_micro_mid_vs_microstructure.png")

    best = rule_summary.sort_values(["precision", "move_recall", "n_alerts"],
                                    ascending=[False, False, True]).head(3)
    lines = [
        "# BBO Microstructure Alert Strategy Test",
        "",
        "## Objective",
        "",
        "Test whether BBO microstructure can trigger a higher-precision alert before "
        "a larger 3c BBO repricing move.",
        "",
        "## Target",
        "",
        f"`true alert = a 3c BBO move occurs within the next {args.horizon_sec} seconds`",
        "",
        "## Important Limitation",
        "",
        "The current complete cached raw BBO sample is tiny: this strategy test "
        f"uses {len(cached_fixtures)} cached fixture(s) and {len(reactions)} "
        "detected 3c BBO move(s). Treat these results as strategy diagnostics, "
        "not out-of-sample proof.",
        "",
        "## Rule Strategy Summary",
        "",
        SE.markdown_table(rule_summary),
        "",
        "## Best Rule Candidates",
        "",
        SE.markdown_table(best),
        "",
        "## Leave-Fixture-Out Logistic Sanity Check",
        "",
        SE.markdown_table(model_summary) if not model_summary.empty else
        "_Not enough cached fixtures with both positive and negative labels._",
        "",
        "## Interpretation",
        "",
        "The most promising logic is not a single BBO signal. It is a confirmation "
        "rule: require early mid drift plus at least one liquidity-stress symptom "
        "(spread shock, depth drop, or update burst). This reduces noisy quote churn "
        "compared with raw update bursts, but the current sample is too small to "
        "claim stable high precision. The next robust test needs longer BBO windows "
        "and non-goal control periods.",
        "",
        "Figure:",
        "",
        "- `outputs/figures/bbo_micro_rule_strategy_precision.png`",
        "- `outputs/figures/bbo_micro_mid_vs_microstructure.png`",
    ]
    with open(f"{rdir}/bbo_micro_alert_strategy_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"report: {rdir}/bbo_micro_alert_strategy_report.md")


if __name__ == "__main__":
    main()
