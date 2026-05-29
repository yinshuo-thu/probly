"""
inspect_dataset.py - 自动识别 LSports messages.parquet 的 schema 与事件分类。

下载 1 场样本比赛, 输出:
- 列结构与 dtype
- incident_name 频次 + 分类映射
- 归档延迟 / 事件节奏统计
结果写入 outputs/tables/schema_inspection.md 与 incident_taxonomy.csv,
并把一份小样本 (前 500 行精简列) 存到 data/sample/ 以便入库展示。
"""
import _bootstrap  # noqa: F401
import os

import pandas as pd

from src import load_config, resolve_path
from src import data_loader as DL
from src import lsports_parser as P
from src import latency_analysis as LA


def main():
    cfg = load_config()
    ed = cfg["single_match"]["event_date"]
    fid = cfg["single_match"]["fixture_id"]
    if fid is None:
        fixtures = DL.list_fixtures(cfg, ed).sort_values("fixture_id")
        fid = fixtures["fixture_id"].iloc[0]
    print(f"inspect fixture {fid} @ {ed}")

    res = DL.download_fixture(cfg, ed, fid)
    raw = DL.load_messages(res["messages"])
    meta = DL.load_fixture_meta(res["fixtures"])
    if raw.empty:
        print("messages 为空, 退出"); return

    lines = ["# LSports messages.parquet Schema 自动识别报告", ""]
    lines.append(f"- 样本: fixture_id={fid}, event_date={ed}")
    lines.append(f"- home={meta.get('home')} away={meta.get('away')} "
                 f"league={meta.get('league_name')}")
    lines.append(f"- 原始行数: {len(raw)}, 列数: {raw.shape[1]}")
    lines.append("\n## 列与 dtype\n")
    lines.append("| column | dtype | 非空率 | 示例 |")
    lines.append("|---|---|---|---|")
    for c in raw.columns:
        nonnull = raw[c].notna().mean()
        sample = str(raw[c].dropna().iloc[0])[:40] if raw[c].notna().any() else ""
        lines.append(f"| {c} | {raw[c].dtype} | {nonnull:.0%} | {sample} |")

    events = P.parse_messages(raw)
    vc = raw["incident_name"].value_counts()
    tax = pd.DataFrame({"incident_name": vc.index, "count": vc.values})
    tax["category"] = tax["incident_name"].map(P.classify_incident)
    tax_path = resolve_path(cfg["paths"]["table_dir"]) + "/incident_taxonomy.csv"
    tax.to_csv(tax_path, index=False, encoding="utf-8-sig")

    lines.append(f"\n## incident_name 分类 (共 {len(tax)} 种, 详见 incident_taxonomy.csv)\n")
    lines.append("| category | 种类数 | 总行数 |")
    lines.append("|---|---|---|")
    agg = tax.groupby("category").agg(kinds=("incident_name", "count"),
                                      rows=("count", "sum")).sort_values("rows",
                                                                         ascending=False)
    for cat, r in agg.iterrows():
        lines.append(f"| {cat} | {int(r['kinds'])} | {int(r['rows'])} |")

    lines.append("\n## 归档延迟 与 事件节奏\n")
    lines.append(f"- 归档延迟 (ingested - timestamp): `{LA.archive_lag_stats(events)}`")
    lines.append(f"- 事件节奏: `{LA.event_cadence_stats(events)}`")
    score = P.reconstruct_score(events)
    goals = P.extract_goals(events)
    lines.append(f"\n## 比分重建\n- 最终比分: "
                 f"{int(score['home_score'].iloc[-1]) if not score.empty else 0}-"
                 f"{int(score['away_score'].iloc[-1]) if not score.empty else 0}")
    lines.append(f"- 识别进球数: {len(goals)}")
    if not goals.empty:
        lines.append("\n进球时刻:\n")
        for _, g in goals.iterrows():
            lines.append(f"  - {g['ts']} | {g['scoring_side']} | "
                         f"{int(g['home_score'])}-{int(g['away_score'])}")

    out_md = resolve_path(cfg["paths"]["table_dir"]) + "/schema_inspection.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"schema 报告: {out_md}")
    print(f"分类表: {tax_path}")

    # 存一份小样本入库 (精简列, 前 800 行)
    sample_cols = [c for c in ["ts", "match_seconds", "period_name",
                               "incident_name", "category", "home_value",
                               "away_value", "player_name", "confidence_grade"]
                   if c in events.columns]
    sample = events[sample_cols].head(800)
    sp = resolve_path(cfg["paths"]["data_sample"]) + f"/sample_messages_{fid}.csv"
    sample.to_csv(sp, index=False, encoding="utf-8-sig")
    print(f"小样本: {sp}")


if __name__ == "__main__":
    main()
