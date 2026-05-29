"""
feature_engineering.py - 基于 LSports 事件流构建时间步特征帧。

在统一真实时间轴上按固定步长采样, 每个时间步计算:
- 比赛进行分钟数, 当前比分 / 分差 / 总进球
- 多窗口滚动事件计数与 xT 加权强度 (60/180/300/600s)
- 各类危险事件计数 (危险进攻 / 射门 / 角球 / 黄牌 / 红牌)
- 当前 period
标签: 未来 horizon 秒内是否发生进球 (goal hazard 二分类标签)。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import lsports_parser as P


def build_feature_frame(events: pd.DataFrame,
                        step_sec: int = 30,
                        windows_sec: tuple[int, ...] = (60, 180, 300, 600),
                        horizon_sec: int = 120,
                        fixture_id: str | None = None) -> pd.DataFrame:
    """构建单场比赛的时间步特征帧 (含 goal hazard 标签)。

    Returns:
        DataFrame, 每行一个时间步, 含特征列 + 'label_goal_next' + 'ts'。
        事件不足时返回空表。
    """
    if events is None or events.empty:
        return pd.DataFrame()
    live = events[events["is_live"]].copy() if "is_live" in events.columns else events.copy()
    if live.empty or live["match_seconds"].max() <= 0:
        return pd.DataFrame()

    t0 = live[live["match_seconds"] > 0]["ts"].min()
    t_end = live["ts"].max()
    if pd.isna(t0) or pd.isna(t_end) or t_end <= t0:
        return pd.DataFrame()

    grid = pd.date_range(t0, t_end, freq=f"{step_sec}s")
    score = P.reconstruct_score(events)
    goals = P.extract_goals(events)
    goal_ts = goals["ts"].tolist() if not goals.empty else []

    ev = live.sort_values("ts")
    rows = []
    for T in grid:
        feat = {"ts": T, "minute": (T - t0).total_seconds() / 60.0}
        # 当前比分 (T 之前最后一次比分)
        if not score.empty:
            prior = score[score["ts"] <= T]
            if not prior.empty:
                hs, as_ = int(prior["home_score"].iloc[-1]), int(prior["away_score"].iloc[-1])
            else:
                hs, as_ = 0, 0
        else:
            hs, as_ = 0, 0
        feat["home_score"], feat["away_score"] = hs, as_
        feat["score_diff"], feat["total_goals"] = hs - as_, hs + as_

        # 多窗口滚动统计
        for w in windows_sec:
            win = ev[(ev["ts"] > T - pd.Timedelta(seconds=w)) & (ev["ts"] <= T)]
            feat[f"n_events_{w}s"] = int(len(win))
            feat[f"xt_sum_{w}s"] = float(win["xt_weight"].sum())
            feat[f"n_danger_{w}s"] = int((win["category"] == "danger").sum())
            feat[f"n_shot_{w}s"] = int((win["category"] == "shot").sum())
            feat[f"n_attack_{w}s"] = int((win["category"] == "attack").sum())
            feat[f"n_corner_{w}s"] = int((win["category"] == "corner").sum())

        # 累计红黄牌 (T 之前)
        past = ev[ev["ts"] <= T]
        feat["cum_yellow"] = int((past["category"] == "card_yellow").sum())
        feat["cum_red"] = int((past["category"] == "card_red").sum())
        # 当前 period
        if "period_id" in past.columns and not past.empty:
            pid = past["period_id"].dropna()
            feat["period_id"] = float(pid.iloc[-1]) if not pid.empty else 0.0
        else:
            feat["period_id"] = 0.0

        # 标签: 未来 horizon 秒内是否进球
        feat["label_goal_next"] = int(any(T < g <= T + pd.Timedelta(seconds=horizon_sec)
                                          for g in goal_ts))
        rows.append(feat)

    fr = pd.DataFrame(rows)
    if fixture_id is not None:
        fr.insert(0, "fixture_id", str(fixture_id))
    return fr


FEATURE_COLS = [
    "minute", "home_score", "away_score", "score_diff", "total_goals",
    "n_events_60s", "xt_sum_60s", "n_danger_60s", "n_shot_60s",
    "n_attack_60s", "n_corner_60s",
    "n_events_180s", "xt_sum_180s", "n_danger_180s", "n_shot_180s",
    "n_events_300s", "xt_sum_300s", "n_danger_300s", "n_shot_300s",
    "n_events_600s", "xt_sum_600s",
    "cum_yellow", "cum_red", "period_id",
]


def feature_matrix(frame: pd.DataFrame):
    """从特征帧抽取 (X, y, used_cols), 自动忽略缺失列。"""
    cols = [c for c in FEATURE_COLS if c in frame.columns]
    X = frame[cols].fillna(0.0).to_numpy(dtype=float)
    y = frame["label_goal_next"].to_numpy(dtype=int) if "label_goal_next" in frame else None
    return X, y, cols
