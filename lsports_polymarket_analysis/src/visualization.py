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
    "goal": ("⚽", "#d62728"), "red_card": ("R", "#b30000"),
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
