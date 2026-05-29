"""
lsports_parser.py - 解析 LSports Hyper messages.parquet 事件流。

核心职责:
1. 规范化时间戳 (timestamp_utc = LSports 事件发布时间)。
2. 把 incident_name 分类为有比赛语义的"实时事件" vs 赛前/累计型球员统计快照。
3. 从 `Score` incident 稳健地重建比分时间线与进球时刻。
4. 抽取关键事件 (进球 / 红黄牌)。

数据质量提示 (详见 docs_data_understanding.md):
- 该数据集为按小时批量归档, `ingested_at_utc` 聚集在下载批次时间, 不能用作实时延迟。
- 大量 `Player <stat>` 行是累计统计, 多数 seconds=0, 不是真实比赛事件。
- `Score` 的 home_value/away_value 存在跨半场重置为 0 的噪声, 因此进球用
  "每一侧比分的累计最大值 (cummax) 的递增点" 来识别, 对重置噪声鲁棒。
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# 关键事件 incident_name (实时比赛语义)
CARD_YELLOW = {"YellowCard", "Player Yellow Card"}
CARD_RED = {"RedCard", "Player Red Card"}

# incident -> xT 风格威胁权重 (用于事件强度/风险建模, 数值为经验设定)
XT_WEIGHTS = {
    "goal": 5.0,
    "penalty": 4.0,
    "redcard": 3.5,
    "dangerousattacks": 1.2,
    "shotontarget": 1.5,
    "shotofftarget": 0.6,
    "blockedshots": 0.7,
    "corners": 0.8,
    "freekicks": 0.4,
    "yellowcard": 0.5,
    "attacks": 0.3,
    "fouls": 0.2,
    "throwins": 0.1,
    "offsides": 0.2,
}


def _to_num(s: pd.Series) -> pd.Series:
    """把字符串列稳健转换为数值 (无法解析的置 NaN)。"""
    return pd.to_numeric(s, errors="coerce")


def classify_incident(name: str) -> str:
    """把 incident_name 归类为粗粒度 category。

    Returns: 'goal_stat'|'card_yellow'|'card_red'|'shot'|'danger'|'attack'|
             'corner'|'foul'|'freekick'|'throwin'|'offside'|'sub'|'penalty'|
             'save'|'timer'|'score'|'player_stat'|'other'
    """
    if not isinstance(name, str):
        return "other"
    n = name.lower()
    if name in CARD_RED:
        return "card_red"
    if name in CARD_YELLOW:
        return "card_yellow"
    if name == "Score":
        return "score"
    if "penalt" in n and "area" not in n and "outside" not in n:
        return "penalty"
    if "red card" in n or "redcard" in n:
        return "card_red"
    if "yellow card" in n or "yellowcard" in n:
        return "card_yellow"
    if "shot" in n and "target" in n:
        return "shot"
    if "blocked shot" in n or "total shots" in n or n == "player shots":
        return "shot"
    if "dangerousattack" in n or "dangerous attack" in n:
        return "danger"
    if n == "attacks":
        return "attack"
    if "corner" in n:
        return "corner"
    if "free kick" in n or "freekick" in n:
        return "freekick"
    if "throwin" in n or "throw in" in n:
        return "throwin"
    if "offside" in n:
        return "offside"
    if "substitut" in n:
        return "sub"
    if "save" in n:
        return "save"
    if n == "timer":
        return "timer"
    if "goal" in n:           # Player Goals / Header Goals 等多为累计统计
        return "goal_stat"
    if name.startswith("Player ") or "(raw)" in n or "percentage" in n:
        return "player_stat"
    if "foul" in n:
        return "foul"
    return "other"


def parse_messages(df: pd.DataFrame) -> pd.DataFrame:
    """把原始 messages.parquet 规范化为干净的事件表。

    Args:
        df: 原始 messages DataFrame。

    Returns:
        干净事件 DataFrame, 关键列:
        ts (UTC), ingested (UTC), match_seconds, period_name, incident_name,
        category, xt_weight, home_value, away_value, value, player_name,
        confidence_grade, is_live。按 ts 升序。空输入返回空表。
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    # 时间戳: LSports 事件发布时间 与 归档批次时间
    out["ts"] = pd.to_datetime(out.get("timestamp_utc"), format="ISO8601",
                               utc=True, errors="coerce")
    out["ingested"] = pd.to_datetime(out.get("ingested_at_utc"), format="ISO8601",
                                     utc=True, errors="coerce")
    out = out.dropna(subset=["ts"]).copy()
    if out.empty:
        return out

    out["match_seconds"] = _to_num(out.get("seconds")).fillna(0).astype("int64")
    for c in ["home_value", "away_value", "value"]:
        if c in out.columns:
            out[c + "_num"] = _to_num(out[c])
    out["confidence_grade"] = _to_num(out.get("confidence_grade"))
    out["category"] = out["incident_name"].map(classify_incident)
    out["xt_weight"] = out["category"].map(
        lambda c: XT_WEIGHTS.get(c, XT_WEIGHTS.get(c.replace("_", ""), 0.0))
    ).fillna(0.0)
    # is_live: 在比赛时钟内 (match_seconds>0) 且不是纯累计统计
    out["is_live"] = (out["match_seconds"] > 0) & (~out["category"].isin(
        ["player_stat", "goal_stat"]))

    out = out.sort_values("ts").reset_index(drop=True)
    keep = ["ts", "ingested", "match_seconds", "period_id", "period_name",
            "incident_name", "category", "xt_weight", "home_value_num",
            "away_value_num", "value_num", "player_name", "confidence_grade",
            "is_live"]
    keep = [c for c in keep if c in out.columns]
    return out[keep].rename(columns={"home_value_num": "home_value",
                                     "away_value_num": "away_value",
                                     "value_num": "value"})


def reconstruct_score(events: pd.DataFrame) -> pd.DataFrame:
    """从 `Score` incident 重建比分时间线 (对跨半场重置噪声鲁棒)。

    方法: 取 category=='score' 行, 分别对 home/away 取累计最大值 (cummax),
    比分状态在任一侧 cummax 递增时更新。返回比分变化点。

    Returns:
        DataFrame[ts, match_seconds, period_name, home_score, away_score,
                  total_goals]; 空则返回空表。
    """
    if events is None or events.empty or "category" not in events.columns:
        return pd.DataFrame()
    sc = events[events["category"] == "score"].copy()
    sc = sc.dropna(subset=["home_value", "away_value"])
    if sc.empty:
        return pd.DataFrame()
    sc = sc.sort_values("ts")
    sc["home_score"] = sc["home_value"].clip(lower=0).cummax().astype(int)
    sc["away_score"] = sc["away_value"].clip(lower=0).cummax().astype(int)
    sc["total_goals"] = sc["home_score"] + sc["away_score"]
    # 仅保留比分发生变化的点
    chg = sc[(sc["home_score"] != sc["home_score"].shift()) |
             (sc["away_score"] != sc["away_score"].shift())]
    cols = ["ts", "match_seconds", "period_name", "home_score", "away_score",
            "total_goals"]
    cols = [c for c in cols if c in chg.columns]
    return chg[cols].reset_index(drop=True)


def extract_goals(events: pd.DataFrame) -> pd.DataFrame:
    """从重建的比分时间线抽取进球时刻 (total_goals 递增点)。

    Returns:
        DataFrame[ts, match_seconds, period_name, scoring_side, home_score,
                  away_score]; 第一行 (0-0 初始) 不计为进球。
    """
    score = reconstruct_score(events)
    if score.empty:
        return pd.DataFrame()
    score = score.copy()
    score["d_home"] = score["home_score"].diff().fillna(score["home_score"])
    score["d_away"] = score["away_score"].diff().fillna(score["away_score"])
    goals = []
    for _, r in score.iterrows():
        if r["total_goals"] == 0:
            continue
        if r["d_home"] > 0:
            goals.append({**r, "scoring_side": "home"})
        if r["d_away"] > 0:
            goals.append({**r, "scoring_side": "away"})
    if not goals:
        return pd.DataFrame()
    g = pd.DataFrame(goals)
    cols = ["ts", "match_seconds", "period_name", "scoring_side",
            "home_score", "away_score"]
    return g[[c for c in cols if c in g.columns]].reset_index(drop=True)


def extract_key_events(events: pd.DataFrame) -> pd.DataFrame:
    """抽取关键离散事件 (进球/红黄牌/点球) 用于时间线标注。"""
    if events is None or events.empty:
        return pd.DataFrame()
    rows = []
    goals = extract_goals(events)
    for _, g in goals.iterrows():
        rows.append({"ts": g["ts"], "match_seconds": g.get("match_seconds"),
                     "kind": "goal", "detail": f"{g['scoring_side']} "
                     f"{int(g['home_score'])}-{int(g['away_score'])}"})
    live = events[events["is_live"]] if "is_live" in events.columns else events
    for cat, kind in [("card_red", "red_card"), ("card_yellow", "yellow_card"),
                      ("penalty", "penalty")]:
        sub = live[live["category"] == cat]
        # 去重: 同一秒同类只保留一条
        sub = sub.drop_duplicates(subset=["match_seconds"])
        for _, r in sub.iterrows():
            rows.append({"ts": r["ts"], "match_seconds": r.get("match_seconds"),
                         "kind": kind,
                         "detail": str(r.get("player_name") or "")})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)


def match_summary(events: pd.DataFrame, meta: dict | None = None) -> dict:
    """生成单场比赛的摘要统计 (用于批量分析)。"""
    if events is None or events.empty:
        return {"n_events": 0, "n_live_events": 0, "total_goals": 0}
    score = reconstruct_score(events)
    goals = extract_goals(events)
    live = events[events["is_live"]] if "is_live" in events.columns else events
    summary = {
        "n_events": int(len(events)),
        "n_live_events": int(len(live)),
        "n_goals": int(len(goals)),
        "final_home": int(score["home_score"].iloc[-1]) if not score.empty else 0,
        "final_away": int(score["away_score"].iloc[-1]) if not score.empty else 0,
        "n_yellow": int((live["category"] == "card_yellow").sum()),
        "n_red": int((live["category"] == "card_red").sum()),
        "ts_start": events["ts"].min(),
        "ts_end": events["ts"].max(),
        "match_seconds_max": int(events["match_seconds"].max()),
        "mean_confidence": float(events["confidence_grade"].mean()),
    }
    if meta:
        summary.update({
            "fixture_id": meta.get("fixture_id"),
            "home": meta.get("home"),
            "away": meta.get("away"),
            "league_name": meta.get("league_name"),
            "status": meta.get("status"),
            "start_date_utc": meta.get("start_date_utc"),
            "event_date": meta.get("event_date"),
        })
    return summary
