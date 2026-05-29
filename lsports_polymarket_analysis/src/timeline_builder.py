"""
timeline_builder.py - 构建单场比赛的统一时间线。

把以下信息对齐到统一的真实时间轴 (wall clock, UTC):
- LSports 事件流 (按真实发布时间 ts)
- 重建的比分变化点 / 关键事件 (进球 / 红黄牌)
- (可选) Polymarket 历史价格曲线

当 Polymarket 价格缺失时, 仍返回完整 LSports 时间线, price 部分为空。
"""
from __future__ import annotations

import pandas as pd

from . import lsports_parser as P


def build_timeline(events: pd.DataFrame,
                   prices: pd.DataFrame | None = None) -> dict:
    """构建统一时间线。

    Args:
        events: parse_messages() 输出的干净事件表。
        prices: PolymarketPriceLoader 输出的价格表 (可空)。

    Returns:
        dict: {
          'events', 'live_events', 'score', 'goals', 'key_events',
          'prices', 't0' (kickoff 估计), 'has_prices'
        }
    """
    if events is None or events.empty:
        return {"events": pd.DataFrame(), "live_events": pd.DataFrame(),
                "score": pd.DataFrame(), "goals": pd.DataFrame(),
                "key_events": pd.DataFrame(),
                "prices": prices if prices is not None else pd.DataFrame(),
                "t0": None, "has_prices": False}

    score = P.reconstruct_score(events)
    goals = P.extract_goals(events)
    key_events = P.extract_key_events(events)
    live = events[events["is_live"]].copy() if "is_live" in events.columns else events

    # kickoff 估计: 第一条 match_seconds>0 的事件时间
    t0 = None
    live_pos = events[events["match_seconds"] > 0]
    if not live_pos.empty:
        t0 = live_pos["ts"].iloc[0]

    has_prices = prices is not None and not prices.empty
    return {"events": events, "live_events": live, "score": score,
            "goals": goals, "key_events": key_events,
            "prices": prices if prices is not None else pd.DataFrame(),
            "t0": t0, "has_prices": has_prices}


def event_intensity_series(events: pd.DataFrame, window_sec: int = 60,
                           step_sec: int = 30,
                           weighted: bool = True) -> pd.DataFrame:
    """计算滚动事件强度时间序列 (用于强度图)。

    Args:
        events: 干净事件表。
        window_sec: rolling 窗口长度 (秒)。
        step_sec: 采样步长 (秒)。
        weighted: True 用 xt_weight 加权, False 用事件计数。

    Returns:
        DataFrame[ts, intensity]。
    """
    if events is None or events.empty:
        return pd.DataFrame(columns=["ts", "intensity"])
    live = events[events["is_live"]] if "is_live" in events.columns else events
    if live.empty:
        return pd.DataFrame(columns=["ts", "intensity"])
    s = live.set_index("ts").sort_index()
    val = s["xt_weight"].fillna(0.0) if weighted else pd.Series(1.0, index=s.index)
    # 按时间 rolling 窗口累计, 再按步长重采样
    roll = val.rolling(f"{window_sec}s").sum()
    res = roll.resample(f"{step_sec}s").max().ffill().fillna(0.0)
    out = res.reset_index()
    out.columns = ["ts", "intensity"]
    return out
