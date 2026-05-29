"""
visualization.py - 所有图表 (matplotlib, 保存到 outputs/figures)。

单赛事图: timeline / score / event intensity / goal hazard / 价格对齐 (若有)。
批量图: 每日比赛数 / 每日进球数 / 每场进球分布 / 归档延迟 / stale window 分布等。
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3})

_KIND_STYLE = {
    "goal": ("G", "#d62728"), "red_card": ("R", "#b30000"),
    "yellow_card": ("Y", "#e6b800"), "penalty": ("P", "#9467bd"),
}


def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_match_timeline(timeline: dict, title: str, out_path: str):
    """单赛事完整事件 timeline: x=真实时间, 标注进球/红黄牌, 叠加 (可选) 价格曲线。"""
    events = timeline["events"]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    live = timeline["live_events"]
    if not live.empty:
        cats = live["category"].astype("category")
        ax.scatter(live["ts"], cats.cat.codes, s=8, c="#1f77b4", alpha=0.35,
                   label="LSports events")
        ax.set_yticks(range(len(cats.cat.categories)))
        ax.set_yticklabels(cats.cat.categories, fontsize=7)
    ke = timeline["key_events"]
    for _, e in ke.iterrows():
        sym, color = _KIND_STYLE.get(e["kind"], ("*", "black"))
        ax.axvline(e["ts"], color=color, alpha=0.5, lw=1)
        ax.annotate(sym, (e["ts"], ax.get_ylim()[1]), color=color,
                    fontsize=11, ha="center", va="bottom")
    if timeline.get("has_prices"):
        ax2 = ax.twinx()
        pr = timeline["prices"]
        ax2.plot(pr["timestamp"], pr["price"], color="#ff7f0e", lw=1.4,
                 label="Polymarket price")
        ax2.set_ylabel("Polymarket price", color="#ff7f0e")
        ax2.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time"); ax.set_ylabel("event category")
    ax.set_title(title)
    return _save(fig, out_path)


def plot_score_timeline(timeline: dict, title: str, out_path: str):
    """比分随真实时间变化的阶梯图。"""
    score = timeline["score"]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    if not score.empty:
        ax.step(score["ts"], score["home_score"], where="post",
                label="home", color="#1f77b4", lw=2)
        ax.step(score["ts"], score["away_score"], where="post",
                label="away", color="#d62728", lw=2)
        ax.set_yticks(range(0, int(score["total_goals"].max()) + 2))
    for _, g in timeline["goals"].iterrows():
        ax.axvline(g["ts"], color="gray", ls="--", alpha=0.4)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time"); ax.set_ylabel("goals")
    ax.set_title(title); ax.legend()
    return _save(fig, out_path)


def plot_event_intensity(intensity_df: pd.DataFrame, goals: pd.DataFrame,
                         title: str, out_path: str):
    """滚动事件强度 (xT 加权) 时间序列, 叠加进球竖线。"""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    if intensity_df is not None and not intensity_df.empty:
        ax.fill_between(intensity_df["ts"], intensity_df["intensity"],
                        color="#2ca02c", alpha=0.35)
        ax.plot(intensity_df["ts"], intensity_df["intensity"],
                color="#2ca02c", lw=1.2, label="rolling xT intensity")
    if goals is not None and not goals.empty:
        for _, g in goals.iterrows():
            ax.axvline(g["ts"], color="#d62728", ls="--", alpha=0.7)
        ax.axvline(goals["ts"].iloc[0], color="#d62728", ls="--",
                   alpha=0.7, label="goal")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time"); ax.set_ylabel("xT intensity")
    ax.set_title(title); ax.legend()
    return _save(fig, out_path)


def plot_goal_hazard(frame: pd.DataFrame, pred: np.ndarray, goals: pd.DataFrame,
                     title: str, out_path: str, threshold: float = 0.5):
    """模型输出的 next-goal hazard 概率曲线, 叠加进球时刻与阈值线。"""
    fig, ax = plt.subplots(figsize=(11, 3.5))
    if frame is not None and not frame.empty and pred is not None:
        ax.plot(frame["ts"], pred, color="#1f77b4", lw=1.4,
                label="predicted goal hazard")
        ax.axhline(threshold, color="gray", ls=":", alpha=0.7,
                   label=f"threshold={threshold}")
    if goals is not None and not goals.empty:
        for _, g in goals.iterrows():
            ax.axvline(g["ts"], color="#d62728", ls="--", alpha=0.7)
        ax.axvline(goals["ts"].iloc[0], color="#d62728", ls="--",
                   alpha=0.7, label="goal")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time"); ax.set_ylabel("P(goal in next window)")
    ax.set_title(title); ax.legend(loc="upper left")
    return _save(fig, out_path)


def plot_price_reaction(prices: pd.DataFrame, goals: pd.DataFrame,
                        title: str, out_path: str):
    """Polymarket price history with LSports goal timestamps."""
    fig, ax = plt.subplots(figsize=(11, 3.8))
    if prices is not None and not prices.empty:
        pr = prices.sort_values("timestamp")
        ax.plot(pr["timestamp"], pr["price"], color="#ff7f0e", lw=1.5,
                label="Polymarket price")
        ax.scatter(pr["timestamp"], pr["price"], color="#ff7f0e", s=10, alpha=0.55)
    if goals is not None and not goals.empty:
        for _, g in goals.iterrows():
            label = f"{g.get('scoring_side', '')} goal"
            ax.axvline(g["ts"], color="#d62728", ls="--", lw=1.0, alpha=0.75)
            ax.annotate(label, (g["ts"], 0.98), xycoords=("data", "axes fraction"),
                        rotation=90, va="top", ha="right", fontsize=8,
                        color="#d62728")
        ax.axvline(goals["ts"].iloc[0], color="#d62728", ls="--",
                   lw=1.0, alpha=0.75, label="LSports goal")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time"); ax.set_ylabel("price / implied probability")
    ax.set_title(title); ax.legend(loc="best")
    return _save(fig, out_path)


def plot_bbo_goal_windows(bbo: pd.DataFrame, reaction: pd.DataFrame,
                          title: str, out_path: str):
    """Small multiples: BBO mid around each LSports goal and reaction timestamp."""
    if bbo is None or bbo.empty or reaction is None or reaction.empty:
        return None
    b = bbo.copy()
    b["timestamp"] = pd.to_datetime(b["timestamp"], utc=True, format="mixed")
    r = reaction.copy()
    r["goal_ts"] = pd.to_datetime(r["goal_ts"], utc=True, format="mixed")
    r["reaction_ts"] = pd.to_datetime(r["reaction_ts"], utc=True, format="mixed")
    n = len(r)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.8 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, r.iterrows()):
        gts = row["goal_ts"]
        win = b[(b["timestamp"] >= gts - pd.Timedelta(seconds=20)) &
                (b["timestamp"] <= gts + pd.Timedelta(seconds=60))]
        if not win.empty:
            ax.plot(win["timestamp"], win["price"], color="#0b7285", lw=1.5,
                    label="BBO mid")
            ax.scatter(win["timestamp"], win["price"], color="#0b7285",
                       s=8, alpha=0.45)
        ax.axvline(gts, color="#d62728", lw=1.4, label="LSports goal")
        if pd.notna(row["reaction_ts"]):
            ax.axvline(row["reaction_ts"], color="#6f42c1", ls="--", lw=1.4,
                       label="BBO move")
            lag = row["reaction_latency_sec"]
            ax.set_title(f"Goal {gts.strftime('%H:%M:%S')} | BBO lag {lag:+.1f}s")
        else:
            ax.set_title(f"Goal {gts.strftime('%H:%M:%S')} | no BBO jump detected")
        ax.set_ylim(0, 1)
        ax.set_ylabel("BBO mid")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.legend(loc="best", fontsize=8)
    fig.suptitle(title, y=1.01, fontsize=11)
    return _save(fig, out_path)


def plot_latency_bars(reaction: pd.DataFrame, title: str, out_path: str):
    """Bar chart where negative latency means BBO moved before LSports."""
    if reaction is None or reaction.empty:
        return None
    r = reaction.copy()
    r["goal_ts"] = pd.to_datetime(r["goal_ts"], utc=True, format="mixed")
    labels = r["goal_ts"].dt.strftime("%H:%M:%S")
    vals = pd.to_numeric(r["reaction_latency_sec"], errors="coerce")
    colors = ["#6f42c1" if v < 0 else "#d62728" for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(labels, vals, color=colors, alpha=0.85)
    ax.axhline(0, color="black", lw=1)
    for i, v in enumerate(vals):
        if pd.notna(v):
            ax.text(i, v + (0.4 if v >= 0 else -0.4), f"{v:+.1f}s",
                    ha="center", va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set_ylabel("Reaction latency (seconds)")
    ax.set_xlabel("LSports goal time (UTC)")
    ax.set_title(title)
    ax.text(0.01, 0.02, "negative = BBO moved before LSports goal timestamp",
            transform=ax.transAxes, fontsize=8, color="#555")
    return _save(fig, out_path)


def plot_signal_timeline(frame: pd.DataFrame, hazard: pd.Series,
                         goals: pd.DataFrame, bbo_reaction: pd.DataFrame,
                         threshold: float, title: str, out_path: str):
    """LSports hazard signal with goals and BBO reaction markers."""
    fig, ax = plt.subplots(figsize=(11, 4))
    if frame is not None and not frame.empty:
        ts = pd.to_datetime(frame["ts"], utc=True, format="mixed")
        ax.plot(ts, hazard, color="#2f9e44", lw=1.5, label="LSports hazard")
        ax.axhline(threshold, color="#495057", ls=":", lw=1.2,
                   label=f"alert threshold={threshold:.2f}")
    if bbo_reaction is not None and not bbo_reaction.empty:
        rr = bbo_reaction.copy()
        rr["reaction_ts"] = pd.to_datetime(rr["reaction_ts"], utc=True, format="mixed")
        for _, row in rr.dropna(subset=["reaction_ts"]).iterrows():
            ax.axvline(row["reaction_ts"], color="#6f42c1", ls="--",
                       alpha=0.95, lw=1.8, zorder=4)
            lag = pd.to_numeric(row.get("reaction_latency_sec"), errors="coerce")
            label = f"BBO {lag:+.1f}s" if pd.notna(lag) else "BBO"
            ax.scatter([row["reaction_ts"]], [0.985], color="#6f42c1",
                       marker="v", s=65, zorder=6, clip_on=False)
            ax.annotate(label, (row["reaction_ts"], 0.96),
                        xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        rotation=90, va="top", ha="left",
                        fontsize=8, color="#6f42c1", zorder=7)
        if rr["reaction_ts"].notna().any():
            ax.axvline(rr["reaction_ts"].dropna().iloc[0], color="#6f42c1",
                       ls="--", alpha=0.95, lw=1.8, label="BBO move")
    if goals is not None and not goals.empty:
        for _, g in goals.iterrows():
            ax.axvline(g["ts"], color="#d62728", alpha=0.85, lw=1.2, zorder=3)
        ax.axvline(goals["ts"].iloc[0], color="#d62728", alpha=0.75,
                   label="LSports goal")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("UTC time"); ax.set_ylabel("hazard")
    ax.set_title(title); ax.legend(loc="best")
    return _save(fig, out_path)


# ----------------------------- 批量统计图 -----------------------------

def plot_bar(series: pd.Series, title: str, xlabel: str, ylabel: str,
             out_path: str, rotate: int = 45):
    fig, ax = plt.subplots(figsize=(9, 4))
    series.plot(kind="bar", ax=ax, color="#1f77b4", alpha=0.85)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.setp(ax.get_xticklabels(), rotation=rotate, ha="right")
    return _save(fig, out_path)


def plot_hist(values, title: str, xlabel: str, out_path: str, bins: int = 30,
              color: str = "#2ca02c"):
    fig, ax = plt.subplots(figsize=(8, 4))
    v = pd.Series(values).dropna()
    if not v.empty:
        ax.hist(v, bins=bins, color=color, alpha=0.8)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("count")
    return _save(fig, out_path)
