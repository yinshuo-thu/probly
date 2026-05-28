# Data Catalog

提取时间：2026-05-26。

## 总览

- 数据文件数：15,966 个。
- 数据目录校验：源目录与目标目录的非 `._*` 文件数量一致，抽样 SHA256 一致。
- 目标目录数据大小：约 749MB 实际占用。
- 源目录在外置盘上的 `du` 读数更大，但逐文件大小校验一致，差异来自文件系统占用统计方式。

## 顶层数据

| 文件 | 行数 | 说明 |
| --- | ---: | --- |
| `data/fixture_index.parquet` | 15,606 | LSports football fixture 索引。 |
| `data/matched_fixtures.parquet` | 184 | 已匹配到 Polymarket 的 fixture 摘要。 |
| `data/hyper/synthetic_matches.parquet` | 280,000 | 合成样本，可用于方法 smoke test，不应作为真实结论。 |

## Polymarket 数据

| 路径 | 行数/数量 | 说明 |
| --- | ---: | --- |
| `data/polymarket/fixture_market_matches.parquet` | 287 | fixture 到 Polymarket condition/market 的匹配表。 |
| `data/polymarket/markets.parquet` | 100 | Polymarket market 元数据。 |
| `data/polymarket/football_markets.json` | - | football market 原始 JSON。 |
| `data/polymarket/football_markets_clean.json` | - | 清洗后的 football market JSON。 |
| `data/polymarket/prices/*.parquet` | 约 100MB | PMXT/CLOB 价格路径。字段包括 `timestamp`, `event_type`, `price`, `size`, `side`, `best_bid`, `best_ask`, `asset_id`。 |

## LSports Hyper Football

| 路径 | 说明 |
| --- | --- |
| `data/hyper/football/<event_date>/<fixture_id>/messages.parquet` | 真实 football fixture 消息流。样例字段包括 `run_id`, `ingested_at_utc`, `sport_id`, `sport_name`, `fixture_id`, `home`, `away`, `event_date`, `message_id`, `timestamp_utc`, `incident_id`, `incident_name`, `confidence_grade`, `seconds`, `period_id`, `period_name`, `status`, `value`, `home_value`, `away_value`, `player_name`, `message_json`, `raw_json`。 |
| `data/hyper/football_sample/` | football 样本子集。 |

## 关键 Schema 样例

`data/fixture_index.parquet`:

```text
fixture_id, event_date, home, away, league_id, league_name, league_type, location_name, start_date_utc, status
```

`data/polymarket/fixture_market_matches.parquet`:

```text
fixture_id, condition_id, question, market_type, home, away, market_team_a, market_team_b, event_date, start_date_utc, league_name, match_score, market_date, market_hour
```

`data/hyper/football/.../messages.parquet`:

```text
run_id, ingested_at_utc, sport_id, sport_name, fixture_id, home, away, event_date, message_id, timestamp_utc, incident_id, incident_name, confidence_grade, seconds, period_id, period_name, status, value, home_value, away_value, player_name, message_json, raw_json
```

`data/polymarket/prices/*.parquet`:

```text
timestamp, event_type, price, size, side, best_bid, best_ask, asset_id
```

## 建模注意事项

- `fixture_market_matches.parquet` 是连接 LSports 与 Polymarket 的关键入口。
- `prices/*.parquet` 是价格标签和未来窗口计算的主要输入。
- `messages.parquet` 中的 `confidence_grade` 是任务要求中的核心信号。
- 新研究应重新生成 aligned modeling table，而不是从当前仓库 `outputs/real_dataset_v4.parquet` 继续。
