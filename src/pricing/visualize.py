"""
Phase 4: Visualization
Generates key plots for the pricing research:
1. Price path + event timeline for example fixture (Manchester City vs Brentford, O/U 2.5)
2. Pre-goal event lift ratio heatmap
3. Price move distribution by event type
4. Fair value deviation over time

All plots saved to outputs/plots/
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

DATA_ROOT = Path("data")
OUT_DIR = Path("outputs")
PLOT_DIR = OUT_DIR / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Color scheme
COLORS = {
    "price": "#2196F3",
    "fair": "#FF9800",
    "goal": "#F44336",
    "danger": "#9C27B0",
    "corner": "#4CAF50",
    "yellow": "#FFEB3B",
    "red_card": "#B71C1C",
    "period": "#607D8B",
    "background": "#FAFAFA",
    "grid": "#E0E0E0",
}


def load_fixture_data(fixture_id: int, condition_id: str):
    """Load LSports messages + Polymarket prices for one fixture."""
    files = glob.glob(str(DATA_ROOT / f"hyper/football/*/{fixture_id}/messages.parquet"))
    if not files:
        return None, None
    msgs = pd.read_parquet(files[0])
    msgs["timestamp_utc"] = pd.to_datetime(msgs["timestamp_utc"], utc=True)
    msgs = msgs.sort_values("timestamp_utc").reset_index(drop=True)

    # Load prices
    short = condition_id.replace("0x", "")[:32]
    prices = None
    for f in glob.glob(str(DATA_ROOT / "polymarket/prices/*.parquet")):
        stem = Path(f).stem
        if stem == short or short.startswith(stem):
            prices = pd.read_parquet(f)
            prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.tz_convert("UTC")
            prices["mid"] = (prices["best_bid"] + prices["best_ask"]) / 2
            if "asset_id" in prices.columns:
                # Pick YES token by higher or lower mean based on market type
                asset_means = prices.groupby("asset_id")["mid"].mean()
                # For O/U 2.5, YES (OVER) should be ~48% pre-game; pick closest to 0.48
                yes_asset = (asset_means - 0.48).abs().idxmin()
                prices = prices[prices["asset_id"] == yes_asset].reset_index(drop=True)
            prices = prices.sort_values("timestamp").reset_index(drop=True)
            break

    return msgs, prices


def plot_price_event_timeline(fixture_id: int, msgs: pd.DataFrame, prices: pd.DataFrame,
                               title: str, market_type: str = "O/U 2.5"):
    """
    Main visualization: price path overlaid with key events.
    Shows the real-time pricing story of one match.
    """
    if prices is None or len(prices) == 0:
        print(f"  No prices for {fixture_id}")
        return

    fig, axes = plt.subplots(3, 1, figsize=(16, 12),
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.patch.set_facecolor(COLORS["background"])

    # === Panel 1: Price path ===
    ax1 = axes[0]
    ax1.set_facecolor(COLORS["background"])

    # Resample prices to 30s bars for clarity
    prices_plot = prices.copy().set_index("timestamp").sort_index()
    price_30s = prices_plot["mid"].resample("30s").last().dropna()

    if len(price_30s) == 0:
        plt.close()
        return

    game_start = price_30s.index.min()
    price_minutes = [(t - game_start).total_seconds() / 60 for t in price_30s.index]

    ax1.plot(price_minutes, price_30s.values, color=COLORS["price"], linewidth=1.8,
             label=f"{market_type} YES price", alpha=0.9, zorder=3)
    ax1.fill_between(price_minutes, price_30s.values, alpha=0.1, color=COLORS["price"])

    # Add price 50% reference line
    ax1.axhline(0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)

    # === Overlay key events ===
    # Score events (goals)
    score_events = msgs[
        (msgs["incident_name"] == "Score") &
        (msgs["confidence_grade"] == 1.0)
    ].copy()
    score_events["home_s"] = pd.to_numeric(score_events["home_value"], errors="coerce").fillna(0).astype(int)
    score_events["away_s"] = pd.to_numeric(score_events["away_value"], errors="coerce").fillna(0).astype(int)
    score_events = score_events.sort_values("timestamp_utc")
    score_events["total"] = score_events["home_s"] + score_events["away_s"]
    score_events["total_diff"] = score_events["total"].diff().fillna(0)
    goals = score_events[score_events["total_diff"] > 0].copy()

    goal_y_vals = []
    for _, goal in goals.iterrows():
        t_min = (goal["timestamp_utc"].tz_convert("UTC") - game_start).total_seconds() / 60
        # Get price at this time
        idx = np.searchsorted([t - game_start for t in price_30s.index if hasattr(t, "total_seconds")],
                               pd.Timedelta(minutes=t_min))
        price_here = price_30s.iloc[min(idx, len(price_30s) - 1)] if len(price_30s) > 0 else 0.5

        ax1.axvline(t_min, color=COLORS["goal"], alpha=0.8, linewidth=2, zorder=4)
        ax1.annotate(f"⚽ {int(goal['home_s'])}-{int(goal['away_s'])}",
                     xy=(t_min, price_here),
                     xytext=(t_min + 0.5, price_here + 0.04),
                     fontsize=8, color=COLORS["goal"], fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=COLORS["goal"], lw=1.2),
                     zorder=5)
        goal_y_vals.append(price_here)

    # Period markers
    period_events = msgs[msgs["incident_name"] == "Period"].sort_values("timestamp_utc")
    for _, pev in period_events.iterrows():
        t_min = (pev["timestamp_utc"].tz_convert("UTC") - game_start).total_seconds() / 60
        ax1.axvline(t_min, color=COLORS["period"], alpha=0.4, linewidth=1, linestyle=":")

    # Yellow / Red cards
    for card_type, color, marker in [("YellowCard", COLORS["yellow"], "^"),
                                       ("RedCard", COLORS["red_card"], "v")]:
        cards = msgs[msgs["incident_name"] == card_type].copy()
        for _, card in cards.iterrows():
            t_min = (card["timestamp_utc"].tz_convert("UTC") - game_start).total_seconds() / 60
            ax1.scatter(t_min, ax1.get_ylim()[0] if ax1.get_ylim()[0] != 0 else 0.05,
                       marker=marker, color=color, s=60, zorder=5, alpha=0.8)

    ax1.set_ylabel("Market Mid Price (YES)", fontsize=11)
    ax1.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3, color=COLORS["grid"])
    ax1.set_ylim(0, 1)

    # === Panel 2: DangerousAttacks rate ===
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["background"])

    da = msgs[msgs["incident_name"] == "DangerousAttacks"].copy()
    if len(da) > 0:
        da["t_min"] = (da["timestamp_utc"] - game_start).dt.total_seconds() / 60
        # Count per 2-minute window
        bins = np.arange(0, price_minutes[-1] + 2, 2)
        da_counts, bin_edges = np.histogram(da["t_min"].values, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        ax2.bar(bin_centers, da_counts, width=1.8, color=COLORS["danger"], alpha=0.7,
                label="DangerousAttacks per 2min")

        # Overlay goal lines
        for _, goal in goals.iterrows():
            t_min = (goal["timestamp_utc"].tz_convert("UTC") - game_start).total_seconds() / 60
            ax2.axvline(t_min, color=COLORS["goal"], alpha=0.8, linewidth=2)

    ax2.set_ylabel("DA count / 2min", fontsize=10)
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, alpha=0.3)

    # === Panel 3: Corners rate ===
    ax3 = axes[2]
    ax3.set_facecolor(COLORS["background"])

    corners = msgs[msgs["incident_name"] == "Corners"].copy()
    if len(corners) > 0:
        corners["t_min"] = (corners["timestamp_utc"] - game_start).dt.total_seconds() / 60
        c_counts, _ = np.histogram(corners["t_min"].values, bins=bins)
        ax3.bar(bin_centers, c_counts, width=1.8, color=COLORS["corner"], alpha=0.7,
                label="Corners per 2min")
        for _, goal in goals.iterrows():
            t_min = (goal["timestamp_utc"].tz_convert("UTC") - game_start).total_seconds() / 60
            ax3.axvline(t_min, color=COLORS["goal"], alpha=0.8, linewidth=2)

    ax3.set_ylabel("Corners / 2min", fontsize=10)
    ax3.set_xlabel("Minutes from game start", fontsize=11)
    ax3.legend(loc="upper right", fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Sync x-axis
    x_max = max(price_minutes) + 2
    for ax in axes:
        ax.set_xlim(0, x_max)

    plt.tight_layout()
    out_path = PLOT_DIR / f"timeline_{fixture_id}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["background"])
    plt.close()
    print(f"  Saved {out_path}")


def plot_lift_heatmap(lead_stats_path: str = None):
    """Heatmap of pre-goal event lift ratios."""
    if lead_stats_path is None:
        lead_stats_path = str(OUT_DIR / "lead_signal_stats.json")

    try:
        with open(lead_stats_path) as f:
            stats = json.load(f)
    except FileNotFoundError:
        print("  lead_signal_stats.json not found, skipping heatmap")
        return

    events = list(stats.keys())
    windows = ["10s", "30s", "60s", "120s"]

    data = np.zeros((len(events), len(windows)))
    for i, ev in enumerate(events):
        for j, ws in enumerate(windows):
            key = f"{ws.replace('s', '')}s_lift_vs_no_goal"
            data[i, j] = stats[ev].get(key, 1.0)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    im = ax.imshow(data, cmap="RdYlGn", vmin=0.5, vmax=3.0, aspect="auto")
    plt.colorbar(im, ax=ax, label="Lift ratio (event rate / background rate)")

    ax.set_xticks(range(len(windows)))
    ax.set_xticklabels([f"{w} before goal" for w in windows])
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(events)

    for i in range(len(events)):
        for j in range(len(windows)):
            val = data[i, j]
            ax.text(j, i, f"{val:.2f}x", ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if (val > 2.0 or val < 0.8) else "black")

    ax.set_title("Pre-Goal Event Lift Ratios\n(How much more frequent vs background rate)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Time window before goal")
    ax.set_ylabel("Event type")

    plt.tight_layout()
    out_path = PLOT_DIR / "lift_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_latency_distribution():
    """Distribution of Polymarket price reaction latency after LSports first report."""
    try:
        la = pd.read_parquet(OUT_DIR / "latency_analysis.parquet")
    except FileNotFoundError:
        print("  latency_analysis.parquet not found, skipping")
        return

    valid = la.dropna(subset=["pm_latency_s"])
    if len(valid) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(COLORS["background"])

    # Panel 1: Latency distribution
    ax1 = axes[0]
    latencies = valid["pm_latency_s"].clip(0, 60)
    ax1.hist(latencies, bins=30, color=COLORS["price"], alpha=0.7, edgecolor="white")
    ax1.axvline(latencies.median(), color="red", linestyle="--", linewidth=2,
                label=f"Median: {latencies.median():.1f}s")
    ax1.axvline(5, color="orange", linestyle=":", linewidth=1.5,
                label="5s threshold")
    ax1.set_xlabel("PM price move latency after LSports first report (seconds)", fontsize=11)
    ax1.set_ylabel("Count", fontsize=11)
    ax1.set_title("Polymarket Reaction Time After LSports Goal Report", fontsize=12, fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Price impact by total goals at time of goal
    ax2 = axes[1]
    valid_impact = la.dropna(subset=["pm_max_delta_60s"])
    if len(valid_impact) > 5:
        impact = valid_impact["pm_max_delta_60s"].abs()
        ax2.hist(impact.clip(0, 0.3), bins=25, color=COLORS["goal"], alpha=0.7, edgecolor="white")
        ax2.axvline(impact.median(), color="red", linestyle="--", linewidth=2,
                    label=f"Median: {impact.median():.3f}")
        ax2.axvline(0.05, color="orange", linestyle=":", linewidth=1.5, label="5¢ threshold")
        ax2.set_xlabel("|ΔPrice| in 60s after goal (market units)", fontsize=11)
        ax2.set_ylabel("Count", fontsize=11)
        ax2.set_title("Price Impact of Goals (60-second window)", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = PLOT_DIR / "latency_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_fair_value_vs_market(fv_path: str = None):
    """Compare Poisson fair value to Polymarket price over a game."""
    try:
        fv = pd.read_parquet(OUT_DIR / "fair_value_paths.parquet")
    except FileNotFoundError:
        print("  fair_value_paths.parquet not found, skipping")
        return

    # Pick Manchester City vs Brentford O/U 2.5
    sample = fv[
        (fv["fixture_id"] == 16066981) &
        (fv["market_type"] == "O/U 2.5")
    ].sort_values("elapsed_min").reset_index(drop=True)

    if len(sample) == 0:
        print("  No fair value data for Man City vs Brentford O/U 2.5")
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    fig.patch.set_facecolor(COLORS["background"])

    ax1 = axes[0]
    ax1.set_facecolor(COLORS["background"])

    ax1.plot(sample["elapsed_min"], sample["market_mid"], color=COLORS["price"],
             linewidth=2, label="Polymarket mid price", alpha=0.9)
    ax1.plot(sample["elapsed_min"], sample["fair_price"], color=COLORS["fair"],
             linewidth=1.5, linestyle="--", label="Poisson fair price", alpha=0.9)

    # Shade deviation
    ax1.fill_between(sample["elapsed_min"], sample["fair_price"], sample["market_mid"],
                     where=sample["market_mid"] > sample["fair_price"],
                     alpha=0.15, color="red", label="Market > Fair (expensive)")
    ax1.fill_between(sample["elapsed_min"], sample["fair_price"], sample["market_mid"],
                     where=sample["market_mid"] < sample["fair_price"],
                     alpha=0.15, color="green", label="Market < Fair (cheap)")

    # Mark goals
    goals = sample[sample["incident_name"] == "Score"].copy()
    goal_changes = goals[goals["total_goals"].diff().fillna(0) > 0]
    for _, row in goal_changes.iterrows():
        ax1.axvline(row["elapsed_min"], color=COLORS["goal"], alpha=0.8, linewidth=2)
        ax1.text(row["elapsed_min"] + 0.5, 0.85, f"⚽ {row['home_score']}-{row['away_score']}",
                 fontsize=8, color=COLORS["goal"])

    ax1.set_ylabel("Price (O/U 2.5 YES)", fontsize=11)
    ax1.set_title("Manchester City 3-0 Brentford: Polymarket vs Poisson Fair Value (O/U 2.5)",
                  fontsize=12, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    ax1.axhline(0.5, color="gray", linestyle=":", alpha=0.4)
    ax1.axvline(45, color=COLORS["period"], alpha=0.4, linewidth=1, linestyle=":")
    ax1.text(45.5, 0.05, "HT", fontsize=8, color=COLORS["period"])

    # Deviation panel
    ax2 = axes[1]
    ax2.set_facecolor(COLORS["background"])
    ax2.plot(sample["elapsed_min"], sample["deviation"], color="purple",
             linewidth=1.5, alpha=0.8)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.axhline(0.05, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="+5¢")
    ax2.axhline(-0.05, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="-5¢")
    ax2.fill_between(sample["elapsed_min"], sample["deviation"], 0,
                     where=sample["deviation"] > 0.05, alpha=0.3, color="red")
    ax2.fill_between(sample["elapsed_min"], sample["deviation"], 0,
                     where=sample["deviation"] < -0.05, alpha=0.3, color="green")
    for _, row in goal_changes.iterrows():
        ax2.axvline(row["elapsed_min"], color=COLORS["goal"], alpha=0.8, linewidth=2)

    ax2.set_ylabel("Fair - Market (deviation)", fontsize=10)
    ax2.set_xlabel("Elapsed minutes", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)

    plt.tight_layout()
    out_path = PLOT_DIR / "fair_value_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def plot_confidence_trajectory():
    """Show how LSports confidence escalates after a goal, vs PM price."""
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(COLORS["background"])
    ax.set_facecolor(COLORS["background"])

    # Data from FSV Mainz vs Union Berlin analysis (from earlier exploration)
    # First goal at 18:07:09
    conf_trajectory = [
        (0, 0.141, "LSports: conf=0.14 (first report)"),
        (2, 0.290, "conf=0.29"),
        (6, 0.483, "conf=0.48"),
        (9, 0.610, "conf=0.61"),
        (12, 1.000, "LSports: conf=1.0 (confirmed)"),
    ]
    # Polymarket price trajectory (O/U 3.5 YES token)
    pm_trajectory = [
        (-30, 0.770, "PM price before goal"),
        (0, 0.550, "PM price drops at goal time"),
        (21, 0.560, "PM stabilizing"),
        (60, 0.580, "PM post-adjustment"),
    ]

    times_conf = [t for t, _, _ in conf_trajectory]
    vals_conf = [v for _, v, _ in conf_trajectory]
    times_pm = [t for t, _, _ in pm_trajectory]
    vals_pm = [v for _, v, _ in pm_trajectory]

    ax2 = ax.twinx()

    l1, = ax.plot(times_conf, vals_conf, "o-", color=COLORS["danger"], linewidth=2.5,
                  markersize=8, label="LSports confidence_grade")
    ax.axvline(0, color=COLORS["goal"], linestyle="--", linewidth=2, alpha=0.7, label="LSports first report")
    ax.axvline(12, color="green", linestyle="--", linewidth=2, alpha=0.7, label="LSports confirmed (conf=1.0)")

    l2, = ax2.step(times_pm, vals_pm, "s--", color=COLORS["price"], linewidth=2,
                   markersize=8, where="post", label="Polymarket price (O/U 3.5)")

    ax.set_xlabel("Seconds from LSports first goal report", fontsize=11)
    ax.set_ylabel("LSports confidence_grade", fontsize=11, color=COLORS["danger"])
    ax2.set_ylabel("Polymarket mid price", fontsize=11, color=COLORS["price"])
    ax.set_title("Goal Detection: LSports Confidence Escalation vs Polymarket Price\n"
                 "FSV Mainz vs Union Berlin (first goal at min 47, O/U 3.5 market)",
                 fontsize=12, fontweight="bold")

    ax.set_xlim(-35, 70)
    ax.set_ylim(0, 1.1)
    ax2.set_ylim(0.4, 0.9)

    lines = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="center left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Annotation
    ax.annotate("← 12-second\nconfidence window\n(pricing opportunity)",
                xy=(6, 0.5), xytext=(20, 0.25),
                fontsize=9, color="green",
                arrowprops=dict(arrowstyle="->", color="green"))

    plt.tight_layout()
    out_path = PLOT_DIR / "confidence_trajectory.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_path}")


def main():
    os.chdir(Path(__file__).parent.parent.parent)
    print("=== Generating visualizations ===\n")

    # 1. Confidence trajectory diagram
    print("1. Confidence trajectory...")
    plot_confidence_trajectory()

    # 2. Latency distribution
    print("2. Latency analysis plots...")
    plot_latency_distribution()

    # 3. Lift heatmap
    print("3. Pre-goal lift heatmap...")
    plot_lift_heatmap()

    # 4. Fair value comparison (Man City vs Brentford)
    print("4. Fair value vs market price...")
    plot_fair_value_vs_market()

    # 5. Price + event timeline for 2 example fixtures
    print("5. Price + event timelines...")
    fm = pd.read_parquet(DATA_ROOT / "polymarket/fixture_market_matches.parquet")
    fi = pd.read_parquet(DATA_ROOT / "fixture_index.parquet").set_index("fixture_id")

    example_fixtures = [
        (16066981, "Manchester City vs Brentford", "O/U 2.5"),
        (16045142, "FSV Mainz 3-2 Union Berlin", "O/U 3.5"),
    ]
    ou25_cids = fm[fm["market_type"] == "O/U 2.5"].groupby("fixture_id")["condition_id"].first().to_dict()
    ou35_cids = fm[fm["market_type"] == "O/U 3.5"].groupby("fixture_id")["condition_id"].first().to_dict()

    for fid, name, mtype in example_fixtures:
        cid_map = ou25_cids if mtype == "O/U 2.5" else ou35_cids
        cid = cid_map.get(fid)
        if cid is None:
            continue
        msgs, prices = load_fixture_data(fid, cid)
        if msgs is not None:
            plot_price_event_timeline(fid, msgs, prices, f"{name} — {mtype}", mtype)

    print(f"\nAll plots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    main()
