# LSports × Polymarket 价值分析

研究 **LSports 体育事件流** 对 **Polymarket 体育预测市场定价** 的价值: LSports 的进球/红黄牌/
比赛状态等事件, 是否比 Polymarket 价格反应更快? 能否用于 faster pricing / early signal /
stale price detection / adverse selection avoidance / event-driven pricing?

> 本目录是仓库内独立子项目, **不修改** 仓库已有结构。所有产出提交到 `test` 分支。

---

## 1. 项目目标

回答以下问题, 并明确区分"已可验证"与"需补数据":
1. LSports 事件数据是否比 Polymarket 价格反应更快?
2. 能否提前判断 Polymarket 价格滞后?
3. 能否用事件流对进球概率/价格方向做提前预测?
4. 如何把发现拓展为批量统计结论与可复用方法?

## 2. 数据来源

- **LSports Hyper**: Hugging Face `probly/lsports-hyper-dataset`, football 分区
  `sport_id=6046__sport=Football/event_date=YYYY-MM-DD/fixture_id=<id>/messages.parquet`。
- 聚焦日期: **2026-05-24 .. 2026-05-28**。
- 结构与字段详见 [`docs_data_understanding.md`](docs_data_understanding.md)。
- **重要**: 该数据集**不含 Polymarket 价格**, 且为**按小时批量归档** (非实时流) ——
  这两点直接决定了哪些结论现在可下、哪些需补数据 (见 §8 与最终报告)。

## 3. Git 分支说明

- 工作分支: **`test`** (从 `main` 切出)。请勿直接改 `main`。
- 本子项目位于 `lsports_polymarket_analysis/`。

## 4. Hugging Face 数据下载

```bash
# token 优先环境变量 (推荐); 也可放在未入库的 config/.hf_token
export HF_TOKEN="<your_hf_token>"
pip install -r lsports_polymarket_analysis/requirements.txt
```

下载只针对所需分区/采样, 不会拉全量 (~25GB)。
若没有 HF token, 脚本会自动回退到本地已缓存 fixture, 但无法扩展批量样本。

## 5. 复现实验 (运行顺序)

所有脚本从本子项目根目录运行:

```bash
cd lsports_polymarket_analysis

# (1) 识别 schema / 事件分类 / 比分重建 (下载 1 场样本)
python scripts/inspect_dataset.py

# (2) 下载采样数据 (默认每日 40 场, 可调 --max / --dates)
python scripts/download_hf_data.py --docs

# (3) 单赛事深度分析 -> outputs/figures + outputs/reports/single_match_report.md
python scripts/single_match_analysis.py            # 默认 fixture 18746260

# (4) 批量统计 -> outputs/tables + figures + reports/batch_statistics_report.md
python scripts/batch_event_analysis.py --max 40

# (可选) 为指定比赛导出特征帧
python scripts/build_features.py --fixture 18746260 --date 2026-05-28
```

## 6. 主要分析流程

```
messages.parquet
   └─ lsports_parser   : 清洗事件 / 分类 incident / 重建比分+进球
        └─ timeline_builder : 统一真实时间轴 + (可选)Polymarket价格
        └─ latency_analysis : 事件节奏 / 归档延迟 / (有价格时)领先-滞后+stale window
        └─ feature_engineering : 多窗口滚动事件强度(xT) + 比分/牌面 + 标签
             └─ modeling   : goal hazard (强度基线 + logistic, 跨场 train/test)
                  └─ visualization : 时间线/比分/强度/hazard/批量分布图
```

## 7. 代码结构

```
lsports_polymarket_analysis/
├── config/config.yaml          # 路径/日期/采样/建模参数 (不硬编码)
├── scripts/                    # 可执行入口 (download/inspect/single/batch/features)
├── src/                        # 库模块 (见 src/__init__.py 注释)
├── notebooks/                  # 3 个交互式 notebook
├── data/{raw,processed,sample} # raw/processed 不入库; sample 小样本入库
└── outputs/{figures,tables,reports}
```

## 8. 当前已完成 vs 已知问题

**已完成 (基于真实数据)**
- 完整下载/解析/清洗 pipeline; incident 自动分类; 稳健比分+进球重建。
- 单赛事: 时间线 / 比分 / 事件强度 / goal hazard 四图 + 报告。
- 批量: 跨 5 日采样的比赛/进球/时间分辨率统计 + 跨场 held-out hazard 模型。
- LSports 时间分辨率结论: 实时事件**中位间隔 ~1.5 秒** (秒级)。

**已知问题 / 下一步**
- **缺 Polymarket 价格历史** → 无法直接给"LSports 领先 Polymarket"的交易级结论;
  已实现 `PolymarketPriceLoader` (真实 CLOB 客户端) + fixture→market 映射 schema, 接入即可算。
- 数据是**小时归档**, `ingested_at_utc` 非实时延迟; 端到端推送延迟需实时流验证。
- 比分/比赛时钟存在噪声, 已用 cummax 等启发式处理, 但极端场次可能漏判, 需扩样验证。

更多结论见 `outputs/reports/{single_match_report,batch_statistics_report,final_report}.md`。
