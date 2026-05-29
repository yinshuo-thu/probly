# LSports Hyper 数据集结构理解

> 数据来源: Hugging Face `probly/lsports-hyper-dataset`
> football 分区: `sport_id=6046__sport=Football`
> 本文档结论均基于对真实数据 (event_date=2026-05-28, fixture_id=18746260 等) 的实测。

---

## 1. 数据集整体结构

```
probly/lsports-hyper-dataset/
├── docs/                                  # 数据字典与建模文档
│   ├── data_dictionary/
│   │   ├── README.md
│   │   ├── lsports_hyper_事件类型汇总.csv   # incident 类型表
│   │   ├── lsports_hyper_状态转移矩阵.csv
│   │   └── lsports_hyper_完整事件清单.csv
│   ├── MODEL_SPEC_markov_pricing.md
│   ├── REQUIREMENTS_sports_pricing.md
│   ├── kalshi-api-knowledge.md
│   └── pmxt-*.md                           # Polymarket archive 索引/查询说明
├── _manifests/
└── sport_id=<id>__sport=<name>/            # 共 22 个运动, Football=6046
    └── event_date=YYYY-MM-DD/              # 按日期分区
        └── fixture_id=<id>/                # 单场比赛
            ├── fixtures.parquet            # 比赛元数据 (1 行)
            ├── manifest.json               # 下载批次信息
            └── messages.parquet            # 事件流 (核心)
```

- football 当前可用日期: **2026-05-06 .. 2026-05-28** (本任务聚焦 05-24..05-28)。
- 单日比赛量级很大: 例如 2026-05-28 有 **1418 个 fixture**。全量约 25GB+, 必须采样。

## 2. 每个文件的含义

| 文件 | 含义 | 规模 |
|---|---|---|
| `fixtures.parquet` | 比赛元数据: home/away/league/start_date/status 等, 仅 1 行 | ~10KB |
| `manifest.json` | 该次归档的批次信息: `downloaded_at_utc`, `run_id`(如 `hourly_2026...`), `message_rows_written` | <1KB |
| `messages.parquet` | **事件流**: 一场比赛全部 incident 消息, 数千~数万行 | 1~4MB |

## 3. messages.parquet 核心字段 (实测 23 列)

| 字段 | 类型 | 含义 |
|---|---|---|
| `fixture_id` | int | 比赛 ID (= event_id / match_id) |
| `sport_id` | int | 运动 ID (Football=6046) |
| `event_date` | str | 比赛日期分区 |
| `message_id` | str | 消息唯一 ID |
| **`timestamp_utc`** | str | **LSports 事件发布时间** (亚秒精度, 如 `...T00:46:54.96Z`) |
| `ingested_at_utc` | str | 本数据集**归档写入时间** (见 §5 重要警告) |
| `incident_id` / `incident_name` | int / str | 事件类型 (72 种, 见 §4) |
| `confidence_grade` | float | 事件置信度 [0,1] |
| `seconds` | int | **比赛时钟秒数** (注意: 大量行=0, 有重置噪声) |
| `period_id` / `period_name` | float / str | 半场 (1st Half / 2nd Half 等), 非空率约 26% |
| `status` | str | 比赛状态文本 (Finished / AboutToStart 等), 非空率约 13% |
| `value` / `home_value` / `away_value` | str | 通用值 / 主队值 / 客队值 (比分等, 需转数值) |
| `player_name` | str | 关联球员 |
| `message_json` / `raw_json` | str | 原始 JSON (含 header/payload, 可深挖) |

字段映射对照任务要求:
- `event_id / fixture_id / match_id` → `fixture_id`
- `update_time / timestamp / received_at` → `timestamp_utc` (发布) 与 `ingested_at_utc` (归档)
- `score / home_score / away_score` → 由 `Score` incident 的 `home_value/away_value` 重建 (§6)
- `event type` (goal/card/...) → `incident_name` → 经 `classify_incident()` 归类

## 4. incident_name 分类 (实测 72 种)

`src/lsports_parser.py: classify_incident()` 把 72 种 incident 归为粗类。实测某场分布:

| category | 含义 | 典型行数 |
|---|---|---|
| `player_stat` | **累计型球员统计** (Player Passes/Minutes...), 多数 seconds=0, **非实时事件** | 最多 |
| `timer` | 计时器心跳 | 2043 |
| `shot` | 射门类 (on/off target, blocked, total) | ~1000 |
| `attack` / `danger` | 进攻 / 危险进攻 | 388 / 296 |
| `score` | **比分快照** (含 home_value/away_value) | 62 |
| `card_yellow` / `card_red` | 黄牌 / 红牌 | 145 / 48 |
| `corner` `foul` `freekick` `throwin` `offside` `sub` `save` | 角球/犯规/任意球/界外球/越位/换人/扑救 | 各数十~数百 |
| `goal_stat` | "Header Goals / Player Goals" 等**累计统计**, 非真实进球 | 121 |

> **关键陷阱**: `incident_name` 含 "Goal" 的多为累计统计 (射门方式分布等), 真实进球必须从
> `Score` incident 的比分变化推断 (§6), 不能直接数 "Goal" 行。

## 5. ⚠ 数据质量与关键警告

1. **这是按小时批量归档, 不是实时流捕获。**
   实测同一场比赛所有 `ingested_at_utc` 聚集在某个下载批次时刻 (53 秒窗口内, run_id=`hourly_...`),
   与事件真实时间相差 40~180 分钟。因此 `ingested_at_utc - timestamp_utc` 衡量的是
   **归档批处理延迟 (中位 ~89 分钟)**, **不能**解释为 LSports 实时推送延迟。
2. **`seconds` (比赛时钟) 不可靠**: 大量行=0; 跨半场可能重置。时间线应以 `timestamp_utc` (真实时钟) 为准。
3. **比分有重置噪声**: `Score` 的 home/away 在 2nd Half 起始可能短暂回到 0; 故进球用
   "每侧比分累计最大值 (cummax) 的递增点" 识别, 对噪声鲁棒 (实测可正确还原 2-1)。
4. **累计统计混入事件流**: `player_stat` / `goal_stat` 占比很高, 必须用 `is_live`
   (`seconds>0` 且非累计类) 过滤后再做时间线与建模。

## 6. 进球与比分重建逻辑

`reconstruct_score()`: 取 `category=='score'` 行 → 对 home/away 分别 `cummax` → 比分变化点。
`extract_goals()`: 在重建比分上取 `total_goals` 递增点, 标注得分方。
实测 fixture 18746260 (Bragantino vs Carabobo) 还原:
`00:46:54 客队 0-1` → `02:07:21 主队 1-1` → `02:24:12 主队 2-1`, 与 manifest 状态一致。

## 7. 是否包含 Polymarket 数据?

- **不包含。** football 分区与 messages.parquet 中**没有任何 Polymarket 价格 / 订单簿 / market 字段**。
- docs 下有 `pmxt-archive-indexing.md` / `pmxt-duckdb-https.md` 等说明文档, 提示 Polymarket
  数据存在于**独立的 archive / API**, 而非本事件数据集内。
- 因此 "LSports vs Polymarket 反应速度" 无法仅凭本数据集计算, 必须额外接入价格历史。

## 8. 后续需要补充的数据 (用于交易级结论)

1. **Polymarket 价格历史**: CLOB `/prices-history` (分钟级) 或 Gamma API; 需 fixture→market 映射。
2. **fixture → polymarket market 映射表**: schema 见 `src/polymarket_loader.py: MAPPING_SCHEMA`
   (`event_id, fixture_id, team_home, team_away, event_start_time, polymarket_market_id,
   polymarket_slug, outcome_mapping`)。
3. **真实实时流时间戳** (而非小时归档): 才能测量 LSports 端到端推送延迟。
4. (可选) `probly/lsports-trade-dataset` 的 bookmaker odds, 作为第二个对照源。

> 同仓库已有 `src/collect_polymarket.py` (Polymarket 采集脚本) 可作为接入起点。
