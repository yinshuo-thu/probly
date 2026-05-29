"""
latency_analysis.py - LSports 事件发布 与 Polymarket 价格反应的时间关系分析。

诚实边界 (务必阅读 docs_data_understanding.md):
- 本 HF 数据集是 **按小时批量归档**: `ingested_at_utc` 聚集在下载批次时刻,
  因此 (ingested - ts) 衡量的是"归档批处理延迟", 不是 LSports 实时推送延迟。
  我们用 `archive_lag_stats()` 报告该量, 但明确标注其语义。
- 真正的 "LSports 是否领先 Polymarket" 需要 Polymarket 价格历史。本模块提供
  `goal_price_reaction()` / `detect_stale_window()`, 只要传入真实价格即可计算;
  无价格时返回 NaN 占位, 绝不伪造。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def archive_lag_stats(events: pd.DataFrame) -> dict:
    """统计 (ingested_at_utc - timestamp_utc) 的分布。

    注意: 这是批量归档延迟, 不能解释为实时推送延迟。仅用于刻画数据集性质。
    """
    if events is None or events.empty or "ingested" not in events.columns:
        return {}
    lag = (events["ingested"] - events["ts"]).dt.total_seconds().dropna()
    if lag.empty:
        return {}
    return {
        "n": int(lag.shape[0]),
        "lag_min_sec": float(lag.min()),
        "lag_p50_sec": float(lag.median()),
        "lag_p95_sec": float(lag.quantile(0.95)),
        "lag_max_sec": float(lag.max()),
        "note": "batch archive lag, NOT real-time push latency",
    }


def event_cadence_stats(events: pd.DataFrame) -> dict:
    """LSports 实时事件之间的时间间隔统计 (刻画事件流时间分辨率)。"""
    if events is None or events.empty:
        return {}
    live = events[events["is_live"]] if "is_live" in events.columns else events
    ts = live["ts"].sort_values()
    if len(ts) < 2:
        return {}
    gaps = ts.diff().dt.total_seconds().dropna()
    gaps = gaps[gaps >= 0]
    return {
        "n_live_events": int(len(ts)),
        "median_gap_sec": float(gaps.median()),
        "p90_gap_sec": float(gaps.quantile(0.90)),
        "max_gap_sec": float(gaps.max()),
    }


def goal_price_reaction(goals: pd.DataFrame, prices: pd.DataFrame,
                        jump_threshold: float = 0.03,
                        lookback_sec: int = 120,
                        lookahead_sec: int = 600) -> pd.DataFrame:
    """对每个进球, 计算 Polymarket 价格的首次显著反应时间与领先/滞后。

    Args:
        goals: extract_goals() 输出 (含 ts)。
        prices: PolymarketPriceLoader 输出 (含 timestamp, price)。
        jump_threshold: 认定"价格显著变化"的绝对价格阈值。
        lookback_sec: 进球前用于确定基准价的窗口。
        lookahead_sec: 进球后搜索价格反应的窗口。

    Returns:
        DataFrame[goal_ts, base_price, reaction_ts, reaction_latency_sec,
                  price_jump_magnitude, lsports_leads]; 无价格 -> 空表。
    """
    cols = ["goal_ts", "base_price", "reaction_ts", "reaction_latency_sec",
            "price_jump_magnitude", "lsports_leads"]
    if goals is None or goals.empty or prices is None or prices.empty:
        return pd.DataFrame(columns=cols)
    pr = prices.dropna(subset=["price"]).sort_values("timestamp")
    rows = []
    for _, g in goals.iterrows():
        gts = g["ts"]
        base_window = pr[(pr["timestamp"] >= gts - pd.Timedelta(seconds=lookback_sec)) &
                         (pr["timestamp"] <= gts)]
        base_price = base_window["price"].iloc[-1] if not base_window.empty else np.nan
        fwd = pr[(pr["timestamp"] > gts) &
                 (pr["timestamp"] <= gts + pd.Timedelta(seconds=lookahead_sec))]
        reaction_ts, lat, jump = pd.NaT, np.nan, np.nan
        if not fwd.empty and not np.isnan(base_price):
            moved = fwd[(fwd["price"] - base_price).abs() >= jump_threshold]
            if not moved.empty:
                reaction_ts = moved["timestamp"].iloc[0]
                lat = (reaction_ts - gts).total_seconds()
                jump = float(moved["price"].iloc[0] - base_price)
        rows.append({
            "goal_ts": gts, "base_price": base_price, "reaction_ts": reaction_ts,
            "reaction_latency_sec": lat, "price_jump_magnitude": jump,
            "lsports_leads": (lat > 0) if not np.isnan(lat) else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


def detect_stale_window(goals: pd.DataFrame, prices: pd.DataFrame,
                        jump_threshold: float = 0.03) -> pd.DataFrame:
    """检测进球后 Polymarket 价格未更新的 "stale window" (滞后窗口)。

    stale window = 进球时刻到价格首次显著变化之间的时长, 这段时间内市场价格
    相对真实赛况是"陈旧"的, 可能存在 adverse selection / 可交易窗口。
    无真实价格时返回空表。
    """
    reac = goal_price_reaction(goals, prices, jump_threshold=jump_threshold)
    if reac.empty:
        return reac
    reac = reac.copy()
    reac["stale_window_sec"] = reac["reaction_latency_sec"]
    return reac[["goal_ts", "stale_window_sec", "price_jump_magnitude",
                 "lsports_leads"]]
