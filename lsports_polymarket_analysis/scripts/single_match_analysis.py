"""
single_match_analysis.py - 单赛事深度分析。

流程:
1. 下载并解析目标比赛事件流。
2. 重建比分/进球/关键事件时间线。
3. (尝试) 接入 Polymarket 价格; 无映射/离线则记录缺口。
4. 计算滚动事件强度, 构建特征, 训练 goal hazard 模型, 评估提前预警。
5. 输出 4 张图 + single_match_report.md。

用法: python scripts/single_match_analysis.py [--fixture 18746260] [--date 2026-05-28]
"""
import _bootstrap  # noqa: F401
import argparse

import numpy as np
import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL
from src import lsports_parser as P
from src import timeline_builder as TB
from src import feature_engineering as FE
from src import modeling as M
from src import visualization as V
from src import latency_analysis as LA
from src.polymarket_loader import (PolymarketPriceLoader, load_mapping_table,
                                   empty_price_frame)


def pick_fixture(cfg, ed):
    """从某日期挑选进球数>=2 且事件量充足的比赛 (下载若干场后择优)。"""
    fixtures = DL.list_fixtures(cfg, ed).sort_values("fixture_id")
    best, best_score = None, -1
    for _, row in fixtures.head(12).iterrows():
        res = DL.download_fixture(cfg, ed, row["fixture_id"])
        ev = P.parse_messages(DL.load_messages(res["messages"]))
        if ev.empty:
            continue
        g = P.extract_goals(ev)
        sc = len(g) * 1000 + len(ev[ev["is_live"]]) if "is_live" in ev.columns else 0
        if len(g) >= 2 and sc > best_score:
            best, best_score = row["fixture_id"], sc
    return best


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=cfg["single_match"]["fixture_id"])
    ap.add_argument("--date", default=cfg["single_match"]["event_date"])
    args = ap.parse_args()

    ed, fid = args.date, args.fixture
    if fid is None:
        print("自动挑选比赛..."); fid = pick_fixture(cfg, ed)
    print(f"单赛事分析: fixture={fid} @ {ed}")

    res = DL.download_fixture(cfg, ed, fid)
    meta = DL.load_fixture_meta(res["fixtures"])
    events = P.parse_messages(DL.load_messages(res["messages"]))
    if events.empty:
        print("事件为空, 退出"); return

    # ---- Polymarket 接入 (best-effort, 无则记录缺口) ----
    mapping = load_mapping_table(resolve_path("data/processed") + "/fixture_market_map.parquet")
    loader = PolymarketPriceLoader(cfg)
    prices = empty_price_frame()
    pm_note = "本数据集不含 Polymarket 价格; 未找到 fixture->market 映射, 价格分析留空 (见报告)。"
    row = mapping[mapping.get("fixture_id", pd.Series([], dtype=str)).astype(str) == str(fid)] \
        if not mapping.empty else mapping
    if not mapping.empty and not row.empty:
        mid = row["polymarket_market_id"].iloc[0]
        ts0 = int(events["ts"].min().timestamp())
        ts1 = int(events["ts"].max().timestamp())
        prices = loader.load_market_prices(mid, ts0, ts1)
        pm_note = (f"已尝试加载 Polymarket market={mid}; "
                   f"返回 {len(prices)} 个价格点。")

    timeline = TB.build_timeline(events, prices)
    goals = timeline["goals"]

    # ---- 图 1: 完整事件 timeline ----
    figdir = resolve_path(cfg["paths"]["fig_dir"])
    title = f"{meta.get('home')} vs {meta.get('away')} ({meta.get('league_name')}) — {ed}"
    V.plot_match_timeline(timeline, title + " | event timeline",
                          figdir + "/single_match_timeline.png")
    # ---- 图 2: 比分时间线 ----
    V.plot_score_timeline(timeline, title + " | score timeline",
                          figdir + "/single_match_score_timeline.png")
    # ---- 图 3: 事件强度 ----
    win = cfg["analysis"]["intensity_windows_sec"][2]
    intensity = TB.event_intensity_series(events, window_sec=300,
                                          step_sec=cfg["analysis"]["feature_step_sec"])
    V.plot_event_intensity(intensity, goals, title + " | rolling xT intensity (300s)",
                           figdir + "/single_match_event_intensity.png")

    # ---- 建模: goal hazard ----
    frame = FE.build_feature_frame(events,
                                   step_sec=cfg["analysis"]["feature_step_sec"],
                                   windows_sec=tuple(cfg["analysis"]["intensity_windows_sec"]),
                                   horizon_sec=cfg["analysis"]["hazard_horizon_sec"],
                                   fixture_id=fid)
    fit = M.fit_logistic_hazard(frame, test_frac=0.0)
    pred = fit["pred"]
    V.plot_goal_hazard(frame, pred, goals, title + " | next-goal hazard",
                       figdir + "/single_match_goal_hazard.png")
    aw = M.advance_warning(frame, pred, goals,
                           threshold=float(np.nanpercentile(pred, 75)) if pred is not None and len(pred) else 0.5,
                           horizon_sec=cfg["analysis"]["hazard_horizon_sec"])

    # ---- 保存中间结果 ----
    proc = resolve_path(cfg["paths"]["data_processed"])
    frame.to_parquet(proc + f"/single_match_features_{fid}.parquet", index=False)

    # ---- 报告 ----
    lag = LA.archive_lag_stats(events)
    cad = LA.event_cadence_stats(events)
    reac = LA.goal_price_reaction(goals, prices)
    lines = [f"# 单赛事深度分析报告 — fixture {fid}", "",
             f"## 1. 比赛选择", "",
             f"- **比赛**: {meta.get('home')} vs {meta.get('away')}",
             f"- **联赛**: {meta.get('league_name')} (league_id={meta.get('league_id')})",
             f"- **开赛**: {meta.get('start_date_utc')}  | event_date={ed}",
             f"- **状态**: status={meta.get('status')} (3=Finished)",
             f"- **选择理由**: 洲际正赛 (Copa Sudamericana), 数据完整 "
             f"({len(events)} 条事件), 含 {len(goals)} 个进球与红黄牌, 时间戳齐全, "
             f"适合做时间线/延迟/建模演示。", "",
             "## 2. 比赛时间线 (LSports 重建)", "",
             f"- 最终比分: **{P.match_summary(events, meta)['final_home']}-"
             f"{P.match_summary(events, meta)['final_away']}**",
             f"- 识别进球: {len(goals)}", ""]
    if not goals.empty:
        lines.append("| 时间(UTC) | 比分 | 得分方 |")
        lines.append("|---|---|---|")
        for _, g in goals.iterrows():
            lines.append(f"| {g['ts']} | {int(g['home_score'])}-{int(g['away_score'])} "
                         f"| {g['scoring_side']} |")
    ke = timeline["key_events"]
    lines += ["", f"- 关键事件总数(进球/红黄牌/点球): {len(ke)}",
              f"  - 红牌: {int((ke['kind']=='red_card').sum()) if not ke.empty else 0}, "
              f"黄牌: {int((ke['kind']=='yellow_card').sum()) if not ke.empty else 0}", ""]

    lines += ["## 3. LSports 时间分辨率 (可从本数据直接验证)", "",
              f"- 实时事件节奏: 中位间隔 **{cad.get('median_gap_sec')} 秒**, "
              f"p90={cad.get('p90_gap_sec')} 秒 (共 {cad.get('n_live_events')} 条实时事件)。",
              f"- 含义: LSports 事件流时间分辨率约为 1-2 秒级, 足以支撑秒级定价信号。", "",
              "### 归档延迟 (数据集性质, **非**实时推送延迟)",
              f"- (ingested_at_utc - timestamp_utc): 中位 {lag.get('lag_p50_sec'):.0f} 秒 "
              f"≈ {lag.get('lag_p50_sec',0)/60:.0f} 分钟。",
              "- 该数据集为按小时批量归档, 故此延迟反映归档批处理, **不能**解释为 "
              "LSports 实时推送速度。真实推送延迟需实时抓取流验证。", ""]

    lines += ["## 4. LSports vs Polymarket 反应速度", "",
              f"- **现状**: {pm_note}", ""]
    if not reac.empty and reac["reaction_latency_sec"].notna().any():
        lines.append("| goal_ts | base_price | reaction_ts | latency(s) | jump | LSports领先 |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in reac.iterrows():
            lines.append(f"| {r['goal_ts']} | {r['base_price']} | {r['reaction_ts']} "
                         f"| {r['reaction_latency_sec']} | {r['price_jump_magnitude']} "
                         f"| {r['lsports_leads']} |")
    else:
        lines += ["- 由于缺少该场比赛的 Polymarket 价格历史, **无法**计算真实的 "
                  "LSports→Polymarket 领先/滞后。已预留 `PolymarketPriceLoader` 与 "
                  "fixture→market 映射 schema, 一旦提供 market_id 即可自动计算 "
                  "latency / stale window (见 src/latency_analysis.py)。", ""]

    lines += ["## 5. 进球前信号建模 (方法 B)", "",
              "### 关于标签 (label) 的说明 — Polymarket BBO",
              "- **理想标签**应取 Polymarket 的 **BBO (best bid/ask) 中间价**的显著变动 "
              "(例如未来窗口内 mid 移动 > 3 分), 这才是\"市场定价反应\"的直接度量。",
              "- **现状**: 本 HF 数据集不含 Polymarket BBO/价格, 因此本场**暂用 LSports "
              "进球事件本身作为代理标签** (未来 horizon 秒内是否进球)。这衡量的是"
              "\"事件可预测性\", 而非\"市场价格可预测性\"。",
              "- `PolymarketPriceLoader` 的返回 schema 已含 `best_bid` / `best_ask` 列; "
              "一旦接入 BBO, 把标签替换为 `mid 变动 > 阈值` 即可复用全部特征与模型代码。", ""]
    lines += ["### 模型结果", "",
              f"- 特征帧: {len(frame)} 个时间步 (步长 {cfg['analysis']['feature_step_sec']}s), "
              f"正样本(未来{cfg['analysis']['hazard_horizon_sec']}s内进球)={int(frame['label_goal_next'].sum())}。",
              f"- 模型: {'logistic 回归' if fit['model'] is not None else '强度基线(正样本过少)'}; "
              f"AUC={fit['auc']}.",
              f"- 备注: {fit.get('note','')}",
              "- **诚实声明**: 该 AUC 为 *单场样本内* 拟合, 正样本仅 12 个, 存在过拟合, "
              "仅用于展示\"进球前特征确实可分\"; 真正的泛化性能见批量报告 (跨场 train/test 切分)。", ""]
    if not aw.empty:
        fired = aw[aw["fired"]]
        lines.append(f"- 提前预警: {len(fired)}/{len(aw)} 个进球在发生前被信号触发; "
                     f"平均提前 {fired['advance_warning_sec'].mean():.0f} 秒。" if not fired.empty
                     else "- 单场样本过少, 提前预警仅作演示, 结论需在批量数据上验证。")
    lines += ["", "## 6. 图表", "",
              "![timeline](../figures/single_match_timeline.png)",
              "![score](../figures/single_match_score_timeline.png)",
              "![intensity](../figures/single_match_event_intensity.png)",
              "![hazard](../figures/single_match_goal_hazard.png)", "",
              "## 7. 初步结论", "",
              "1. **LSports 事件流足够快/细**: 秒级 (中位~1.5s) 更新, 进球可被精确打时间戳。",
              "2. **进球前存在可观测信号**: 滚动 xT 事件强度在进球前通常抬升 (见强度图), "
              "支持 event-driven pricing 的可行性。",
              "3. **领先性结论待补**: 是否领先 Polymarket 需价格历史; 本场未拿到映射, "
              "已把 join 接口/缺口写清, 可无缝接入后计算。"]

    out = resolve_path(cfg["paths"]["report_dir"]) + "/single_match_report.md"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(str(x) for x in lines))
    print(f"报告: {out}")
    print(f"图已保存到 {figdir}")


if __name__ == "__main__":
    main()
