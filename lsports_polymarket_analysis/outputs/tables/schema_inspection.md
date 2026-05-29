# LSports messages.parquet Schema 自动识别报告

- 样本: fixture_id=18746260, event_date=2026-05-28
- home=Red Bull Bragantino away=Carabobo FC league=Copa Sudamericana
- 原始行数: 16053, 列数: 23

## 列与 dtype

| column | dtype | 非空率 | 示例 |
|---|---|---|---|
| run_id | str | 100% | hourly_20260528T030603Z |
| ingested_at_utc | str | 100% | 2026-05-28T03:15:04.299965+00:00 |
| sport_id | int64 | 100% | 6046 |
| sport_name | str | 100% | Football |
| fixture_id | int64 | 100% | 18746260 |
| home | str | 100% | Red Bull Bragantino |
| away | str | 100% | Carabobo FC |
| event_date | str | 100% | 2026-05-28 |
| message_id | str | 100% | 2f43a291-e64b-4fef-988d-3a26af4c811e |
| timestamp_utc | str | 100% | 2026-05-28T00:15:34.6077776Z |
| incident_id | int64 | 100% | 27 |
| incident_name | str | 100% | Score |
| confidence_grade | float64 | 100% | 1.0 |
| seconds | int64 | 100% | 0 |
| period_id | float64 | 26% | 10.0 |
| period_name | str | 26% | 1st Half |
| status | str | 13% | Finished |
| value | str | 74% | AboutToStart |
| home_value | str | 14% | 0 |
| away_value | str | 14% | 0 |
| player_name | str | 74% | Gustavo Marques |
| message_json | str | 100% | {"header":{"messageId":"2f43a291-e64b-4f |
| raw_json | str | 100% | {"confidenceGrade": 1.0, "incidentId": 2 |

## incident_name 分类 (共 72 种, 详见 incident_taxonomy.csv)

| category | 种类数 | 总行数 |
|---|---|---|
| player_stat | 11 | 8956 |
| timer | 1 | 2043 |
| other | 19 | 1704 |
| shot | 10 | 1050 |
| attack | 1 | 388 |
| foul | 2 | 340 |
| danger | 1 | 296 |
| offside | 2 | 268 |
| save | 2 | 221 |
| throwin | 1 | 162 |
| card_yellow | 2 | 145 |
| goal_stat | 8 | 126 |
| freekick | 4 | 109 |
| corner | 1 | 68 |
| score | 1 | 62 |
| sub | 1 | 56 |
| card_red | 2 | 48 |
| penalty | 3 | 11 |

## 归档延迟 与 事件节奏

- 归档延迟 (ingested - timestamp): `{'n': 16053, 'lag_min_sec': 2388.385305, 'lag_p50_sec': 5327.5624744, 'lag_p95_sec': 9729.512570739998, 'lag_max_sec': 10769.6923234, 'note': 'batch archive lag, NOT real-time push latency'}`
- 事件节奏: `{'n_live_events': 2755, 'median_gap_sec': 1.47462705, 'p90_gap_sec': 5.324640090000002, 'max_gap_sec': 414.1720006}`

## 比分重建
- 最终比分: 2-1
- 识别进球数: 3

进球时刻:

  - 2026-05-28 00:46:54.967075500+00:00 | away | 0-1
  - 2026-05-28 02:07:21.622370400+00:00 | home | 1-1
  - 2026-05-28 02:24:12.788107900+00:00 | home | 2-1