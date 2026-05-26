# pmxt Archive 索引与定位

更新日期: `2026-05-18`

## 目标

实现一套可复用能力，用来解决这几个问题：

1. 列出 `v1 / v2` 各自在哪些小时文件上有数据。
2. 给定 `event slug` 或 `event id`，快速定位它在指定时间范围内落在哪些 pmxt 文件里。
3. 如果事件跨天、跨小时，支持一次性找到多个文件并下载过滤后的子集。
4. 让后续其他管理工具既可以直接调 CLI，也可以调用 HTTP API。

## 一个重要验证

pmxt archive 是**按小时切分**的。

文件名规则：

- `v1`: `https://r2.pmxt.dev/polymarket_orderbook_YYYY-MM-DDTHH.parquet`
- `v2`: `https://r2v2.pmxt.dev/polymarket_orderbook_YYYY-MM-DDTHH.parquet`

因此：

- **同一个 UTC 小时只对应一个 pmxt 文件**
- **一个事件如果跨多个小时/跨天，就会分布在多个文件里**
- 不存在“同一个小时又拆成多个不同 pmxt 小时文件”的情况

这也是为什么定位逻辑必须先按时间范围找到候选小时文件，再按事件的 `conditionId` 做过滤。

## 已实现接口

### 1. CLI

脚本：

- [scripts/pmxt_archive_locator.py](/Users/wilson/Documents/New%20project/polymarket-live-lab/scripts/pmxt_archive_locator.py)

#### 查询 V1/V2 覆盖时间

```bash
cd "/Users/wilson/Documents/New project/polymarket-live-lab"

python3 scripts/pmxt_archive_locator.py coverage \
  --version v2 \
  --start-hour 2026-04-13T00 \
  --end-hour 2026-05-18T23
```

输出里会包含：

- `existing_urls`
- `url_probe`
- `coverage_ranges`

其中 `coverage_ranges` 是压缩后的连续小时区间。

#### 按事件定位文件

```bash
python3 scripts/pmxt_archive_locator.py locate \
  --version v2 \
  --event-slug 2026-fifa-world-cup-winner-595 \
  --start-hour 2026-05-17T22 \
  --end-hour 2026-05-18T04
```

输出里会包含：

- `event`
- `filter.condition_ids`
- `match.matched_files`
- `match.per_file_rows`
- `match.spans_multiple_files`

#### 按事件定位并下载

```bash
python3 scripts/pmxt_archive_locator.py locate \
  --version v2 \
  --event-slug 2026-fifa-world-cup-winner-595 \
  --start-hour 2026-05-17T22 \
  --end-hour 2026-05-18T04 \
  --download \
  --output-name fifa_worldcup_v2_crossday_subset
```

这会把多个匹配小时文件里的目标事件行过滤后，合并导出到一个本地 parquet。

### 2. HTTP API

#### 查询覆盖范围

```bash
curl -sS 'http://127.0.0.1:8010/api/pmxt/coverage?version=v2&start_hour=2026-04-13T00&end_hour=2026-05-18T23'
```

#### 按事件定位/下载

```bash
curl -sS -X POST 'http://127.0.0.1:8010/api/pmxt/locate' \
  -H 'Content-Type: application/json' \
  --data '{
    "version":"v2",
    "event_slug":"2026-fifa-world-cup-winner-595",
    "start_hour":"2026-05-17T22",
    "end_hour":"2026-05-18T04",
    "download":true,
    "output_name":"fifa_worldcup_v2_crossday_subset"
  }'
```

请求字段：

- `version`: `v1` 或 `v2`
- `start_hour`: `YYYY-MM-DDTHH`
- `end_hour`: `YYYY-MM-DDTHH`
- `event_slug`: 可选
- `event_id`: 可选
- `condition_ids`: 可选，逗过 Gamma 直接指定
- `event_types`: 可选，缺省时 `v1=book_snapshot`, `v2=book`
- `download`: 是否直接导出本地 parquet
- `output_dir`: 可选
- `output_name`: 可选

## 定位逻辑

### 如果给的是事件

1. 先用 Gamma 解析 `event slug / event id`
2. 拿到事件下所有 `conditionId`
3. 在指定时间范围内枚举 pmxt 小时文件 URL
4. 先 `HEAD` 探测哪些文件存在
5. 用 DuckDB 远程读 parquet，并按：
   - `v1: market_id`
   - `v2: market`
   过滤到该事件的 `conditionId`
6. 返回每个文件里的行数、时间范围、匹配文件列表

### 如果只给了 `condition_ids`

就跳过 Gamma，直接按条件过滤。

## 为什么这套方式适合后续工具调用

因为它把问题拆成了两个稳定步骤：

1. **时间索引**
   - 某个小时有没有文件
   - 连续小时范围有哪些

2. **事件定位**
   - 事件对应哪些 `conditionId`
   - 这些 `conditionId` 在哪些小时文件里出现

这意味着：

- 后台管理工具可以先查 `coverage`
- 再查 `locate`
- 需要数据时再 `download`

而不是每次都盲扫所有小时文件。

## 当前实现文件

- [backend/app/pmxt_archive.py](/Users/wilson/Documents/New%20project/polymarket-live-lab/backend/app/pmxt_archive.py)
- [backend/app/main.py](/Users/wilson/Documents/New%20project/polymarket-live-lab/backend/app/main.py)
- [backend/app/models.py](/Users/wilson/Documents/New%20project/polymarket-live-lab/backend/app/models.py)
- [scripts/pmxt_archive_locator.py](/Users/wilson/Documents/New%20project/polymarket-live-lab/scripts/pmxt_archive_locator.py)

## 建议的后续用法

如果你后面要做“按足球事件快速落历史 orderbook 子集”，推荐标准流程：

1. 先调 `coverage`，确认 `v1 / v2` 的实际可用小时区间。
2. 再调 `locate`，确认某个事件在目标时间窗里命中了哪些小时文件。
3. 如果 `spans_multiple_files=true`，就直接用 `download=true` 把多个文件里的目标行合并导出。
4. 再基于导出的 parquet 做回测或因子计算。
