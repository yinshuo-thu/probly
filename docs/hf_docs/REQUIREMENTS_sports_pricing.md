# 体育事件实时定价研究：LSports × Polymarket 跨市场价格发现

## 1. 项目概述

本项目旨在利用 LSports Hyper 实时体育事件数据（状态转移 + 置信度评分），结合 Polymarket 预测市场的订单簿与成交数据，构建一个**更快速、更精准的体育事件定价模型**。核心目标是验证：基于体育数据源的状态信号，能否在 Polymarket 市场价格调整之前完成定价，从而实现信息优势。

---

## 2. 数据源说明

### 2.1 LSports Hyper（体育事件数据源）

**来源**: https://platform.lsports.eu/engage/hyper/livescore

LSports Hyper 提供赛事的**实时状态流**（LiveScore Messages），每个赛事产生一系列按时间排列的事件消息。

#### 数据结构

| 字段 | 说明 |
|------|------|
| `fixture_id` | 赛事唯一 ID |
| `sport_id` / `sport_name` | 运动类型（Football=6046, Basketball=48242 等） |
| `home` / `away` | 对阵双方 |
| `start_date_utc` | 开赛时间 |
| `incident_name` | 事件类型（Score, Period, Possession, YellowCard, Corners 等） |
| `confidence_grade` | **置信度评分（0~1）**— 核心字段 |
| `timestamp_utc` | 事件发生时间戳（毫秒级） |
| `home_value` / `away_value` | 事件数值（如比分 "2"-"1"） |
| `period_name` | 当前阶段（1st Half, 2nd Half, FT 等） |

#### 关键概念：Confidence Grade（置信度）

- **confidence_grade = 1.0**: 完全确认的事实（如最终比分、换人）
- **confidence_grade ∈ (0, 1)**: 带有不确定性的推断/预测
  - 例如：`0.75` 表示该事件有 75% 的置信度
  - 常见于进球争议判定、VAR 审查期间、数据源未完全确认时
- **应用价值**: 置信度直接映射为市场定价的概率调整因子

#### 事件类型（Incident Types）

完整的足球赛事包含以下事件流：
- `Score` — 比分变动（最核心）
- `Period` — 阶段转换（上半场→下半场→加时→点球）
- `Possession` — 控球率
- `Attacks` / `DangerousAttacks` — 进攻次数
- `ShotsOnTarget` / `ShotsOffTarget` / `BlockedShots` — 射门统计
- `Corners` — 角球
- `YellowCard` / `RedCard` — 牌照
- `Fouls` / `FreeKicks` — 犯规
- `Substitutions` — 换人
- `Penalties` / `MissedPenalty` — 点球
- `Timer` — 比赛计时器
- `FixtureStatus` — 赛事状态（未开始→进行中→结束）

#### 当前数据规模

| 运动 | 赛事数量 |
|------|----------|
| Football（足球） | 46,637 |
| Table Tennis（乒乓球） | 31,341 |
| Basketball（篮球） | 13,409 |
| Tennis（网球） | 5,967 |
| Ice Hockey（冰球） | 5,662 |
| Baseball（棒球） | 1,634 |
| 其他 15 项运动 | ~3,500 |
| **合计** | **~108,000+** |

数据存储格式: **Parquet**，按 `sport_id/event_date/fixture_id/` 分区存储。

---

### 2.2 Polymarket（预测市场数据源）

**来源**: https://polymarket.com

Polymarket 是基于 Polygon 链的预测市场，体育赛事（特别是 NFL、NBA、足球等）的二元期权合约（YES/NO Token）在此交易。

#### 关键数据获取方式

##### 方式 A: PMXT（开源工具）

- **仓库**: https://github.com/kevinh/pmxt（或类似开源 Polymarket 数据工具）
- **功能**: 获取 Polymarket 的 **Orderbook 快照**（Best Bid/Offer）
- **用途**: 对齐 LSports 事件时间戳与 Polymarket 价格变动时间戳
- **核心对比维度**:
  - 事件发生时刻的 BBO（Best Bid/Offer）价格
  - 事件发生后 BBO 的变动延迟（ms 级别）
  - 价格调整幅度 vs 事件 confidence 预期

##### 方式 B: Goalsky Token 成交数据

- **来源**: Goalsky API / 链上数据
- **功能**: 获取 Polymarket 合约的**逐笔成交数据（Trades/Fills）**
- **核心字段**: 成交时间、价格、数量、方向（buy/sell）、maker/taker 标识
- **用途**: 区分价格变动的驱动来源

---

## 3. 核心研究问题

### 3.1 事件-市场匹配（Event Matching）

**目标**: 将 LSports 体育赛事与 Polymarket 对应市场合约进行精确匹配。

匹配维度：
- 赛事名称（队伍名）→ Polymarket 市场标题
- 赛事时间 → 合约到期时间
- 运动类型 → 市场分类

**挑战**:
- 命名差异（如 "Man United" vs "Manchester United"）
- Polymarket 不覆盖所有赛事（主要覆盖大型联赛）
- 一个赛事可能对应多个市场（胜负盘、大小球、角球数等）

### 3.2 定价速度评估（Pricing Latency）

**核心问题**: LSports 状态信号能否比 Polymarket 市场价格调整更快？

评估方法：
1. 记录 LSports 事件 `timestamp_utc`（如进球事件 T₀）
2. 记录 Polymarket 价格开始变动的时刻 T₁
3. 计算 **信息延迟** = T₁ - T₀
4. 若 T₁ - T₀ > 0 → 存在定价优势窗口

### 3.3 价格驱动来源分析（Price Discovery Attribution）

**核心问题**: Polymarket 的价格调整是由用户主动交易驱动，还是做市商主动调价？

两种机制：
- **用户驱动（Taker-driven）**: 有信息优势的交易者提交市价单 → 吃掉做市商挂单 → 价格被动调整
- **做市商驱动（Maker-driven）**: 做市商收到外部信号 → 主动撤单/调整报价 → 价格主动调整

**区分方法**:
1. 从 Goalsky 获取逐笔成交数据
2. 分析事件发生后的成交流：
   - 如果先有大量 Taker 成交（吃单）→ 再有价格跳变 → **用户驱动**
   - 如果直接 BBO 跳变、无明显成交量 → **做市商驱动**
3. 统计两种模式的比例和响应时间

### 3.4 置信度 → 定价映射（Confidence-to-Price）

**目标**: 建立 `confidence_grade` 到市场合理价格的映射关系。

模型思路：
- 对于二元市场（如"A 队获胜"）：`fair_price ≈ f(current_score, time_remaining, confidence)`
- confidence < 1.0 时，市场可能尚未完全反应 → 存在交易机会
- confidence = 1.0 且价格未调整 → 确定性套利窗口

---

## 4. 项目目标

### Phase 1: 数据对齐与匹配
- [ ] 建立 LSports 足球/篮球赛事 ↔ Polymarket 市场的映射表
- [ ] 对齐时间轴：LSports 事件时间戳 vs Polymarket orderbook/trade 时间戳
- [ ] 数据质量验证：确保时间精度（ms 级）和事件完整性

### Phase 2: 信息延迟量化
- [ ] 计算每个 Score 事件后 Polymarket 价格首次变动的延迟（中位数、P95、P99）
- [ ] 按联赛/赛事热度分组分析延迟差异
- [ ] 确定可利用的定价优势窗口

### Phase 3: 价格驱动归因
- [ ] 对每个价格调整事件进行 Taker/Maker 归因
- [ ] 统计做市商主动调价 vs 被动被吃的比例
- [ ] 识别"信息交易者"的行为模式

### Phase 4: 定价模型构建
- [ ] 基于 LSports 状态转移 + confidence 构建实时定价模型
- [ ] 回测：模拟在信息窗口内下单的 PnL
- [ ] 优化：最优下单时机、仓位管理、滑点评估

---

## 5. 技术架构

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐
│  LSports Hyper  │     │   Polymarket     │     │    Goalsky     │
│  (Event Stream) │     │  (Orderbook/BBO) │     │ (Trade/Fills)  │
└────────┬────────┘     └────────┬─────────┘     └───────┬────────┘
         │                       │                        │
         ▼                       ▼                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Event Matching Engine                         │
│  (LSports fixture ↔ Polymarket market_id mapping)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Time-Aligned Dataset                           │
│  (event_ts, confidence, bbo_before, bbo_after, trades_in_window)│
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌───────────────────┐   ┌───────────────────┐   ┌────────────────┐
│ Latency Analysis  │   │ Attribution Model │   │ Pricing Model  │
│ (T1 - T0 stats)  │   │ (Maker vs Taker)  │   │ (Conf → Price) │
└───────────────────┘   └───────────────────┘   └────────────────┘
```

---

## 6. 数据仓库

所有数据已上传至 HuggingFace 私有仓库：

| 数据集 | 仓库 | 说明 |
|--------|------|------|
| LSports Hyper | [probly/lsports-hyper-dataset](https://huggingface.co/datasets/probly/lsports-hyper-dataset) | 108K+ 赛事历史数据（Parquet） |
| LSports Trade | [probly/lsports-trade-dataset](https://huggingface.co/datasets/probly/lsports-trade-dataset) | 赔率快照数据（Parquet） |

---

## 7. 协作说明

### 数据使用方式

```python
# 加载单场赛事数据
import pandas as pd
from huggingface_hub import hf_hub_download

# 或直接从本地路径加载
fixtures = pd.read_parquet("data/lsports_hyper/sport_id=6046__sport=Football/event_date=2026-05-18/fixture_id=18934236/fixtures.parquet")
messages = pd.read_parquet("data/lsports_hyper/sport_id=6046__sport=Football/event_date=2026-05-18/fixture_id=18934236/messages.parquet")

# 按时间排序事件流
timeline = messages.sort_values("timestamp_utc")

# 筛选高置信度得分事件
scores = timeline[(timeline["incident_name"] == "Score") & (timeline["confidence_grade"] >= 0.9)]
```

### 关键交付物
1. **匹配表**: LSports fixture_id ↔ Polymarket condition_id 映射
2. **延迟报告**: 每个事件类型的价格反应延迟统计
3. **归因报告**: Maker/Taker 驱动比例分析
4. **定价模型**: 可部署的实时信号生成器

---

## 8. 参考资料

- LSports Hyper API 文档: https://docs.lsports.eu/u/hyper/hyper
- Polymarket API: https://docs.polymarket.com
- PMXT (Polymarket Exchange Tool): 开源 orderbook 数据采集工具
- Goalsky: Token 级成交数据获取
- HuggingFace 数据集: https://huggingface.co/wilsonwangwang
