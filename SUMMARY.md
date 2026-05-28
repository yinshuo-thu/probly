# Probly Sports Pricing — 研究进度 Summary

> 目标：用 LSports 实时体育事件流，在 Polymarket 价格调整之前实现合理定价，提前预知高波动窗口。

---

## 当前状态（2026-05-28）

### ✅ 完成：Phase 1 — 数据对齐

**数据集概览**：
- 26场足球比赛同时有 LSports 事件流 + Polymarket 价格数据（共54个候选，28个因时间段不重叠排除）
- 对齐后：**28,847条事件记录**，分布在26场比赛
- 核心输出：`data/aligned/master.parquet`

**核心数据来源**：
| 数据 | 规模 | 字段 |
|------|------|------|
| LSports Hyper Football | 195个fixtures的事件流 | incident_name, confidence_grade, timestamp_utc, seconds, period_name |
| Polymarket prices | 58个市场价格文件（微秒级） | best_bid, best_ask, asset_id, timestamp |
| 对齐匹配表 | 287条 fixture↔condition_id 映射 | fixture_id, condition_id, market_type |

**事件分布**：
```
Attacks            6,926   ← 主要噪声
DangerousAttacks   5,763   ← 关键信号
Total Shots        2,782
FreeKicks          2,032
Fouls              1,804
Corners            1,754
Score              1,444   ← 核心标签事件
Substitutions      1,359
ShotsOnTarget      1,283
ShotsOffTarget     1,265
BlockedShots       1,149
YellowCard           883
RedCard              155
Penalties            130
Period               118
```

---

### 🔍 关键发现（初步分析）

**案例：FSV Mainz vs Union Berlin，O/U 3.5 市场，第47分钟进球**

```
时间         LSports事件                          Polymarket价格(O/U 3.5 YES)
18:06:30    无进球事件                              0.765
18:07:00    —                                      0.550  ← 价格跳变 (-0.215)
18:07:09    Score(0-1), confidence=0.14            ← LSports首报
18:07:11    Score(0-1), confidence=0.29
18:07:15    Score(0-1), confidence=0.48
18:07:18    Score(0-1), confidence=0.61
18:07:21    Score(0-1), confidence=1.00            ← 确认
18:07:30    —                                      0.570
```

**发现**：
1. Polymarket 价格在 18:07:00 bar 内跳变（精度30s），LSports 首报在 18:07:09
   → 两者几乎同时（待毫秒级精确测量）
2. **LSports confidence trajectory 是关键信号**：
   - 0.14 → 0.29 → 0.48 → 0.61 → 1.00 （共12秒从首报到确认）
   - 价格在这12秒内仍在探寻正确位置
3. 价格从 0.765 → 0.55 = **-21.5¢** 的跳变（O/U 3.5市场）

**方法论确认**：
- LSports 比分事件总是先以低置信度（~0.1-0.15）报告，然后快速升至1.0
- **这个置信度上升窗口（约12秒）就是定价优势窗口**

---

### 📐 对齐方法（已实现）

```python
# src/pricing/align_data.py
# 对每个LSports事件，用searchsorted快速找到：
# 1. 事件前的最近价格 mid_before
# 2. 事件后10/30/60/120/300秒的平均价格
# 3. 计算各窗口的价格变化 delta_p_Xs

# 关键修复：timestamp精度对齐
# LSports: datetime64[ns, UTC]  → int64 nanoseconds
# Polymarket: datetime64[us, UTC] → int64 nanoseconds (需normalize)
ev_ts = events["timestamp_utc"].values.astype("datetime64[ns]").astype("int64")
pr_ts = prices["timestamp"].values.astype("datetime64[ns]").astype("int64")
```

---

## 🔄 进行中：Phase 2 — 精细分析

### 下一步计划

**2.1 毫秒级延迟测量**
- 对每个goal事件，找 Polymarket 价格首次移动的精确时间戳
- 计算 T_lsports - T_polymarket（正数=LSports领先，负数=市场领先）

**2.2 Confidence Trajectory 定价模型**
- 当 confidence 从低值上升时，mid-price 应该如何调整
- `fair_price(t) = price_before + delta_p_expected × confidence(t)`

**2.3 市场类型分层分析**
- O/U 2.5 vs O/U 3.5 vs BTTS 对同一进球的价格反应差异
- Win market 对进球的方向性预测

**2.4 预进球信号（Lead-time）**
- DangerousAttacks + ShotsOnTarget 序列 → 进球概率升高
- 目标：进球前30-120秒给出警告信号

---

## 📁 文件结构

```
probly/
├── data/
│   ├── aligned/
│   │   └── master.parquet          ← 对齐数据集（28,847行）
│   ├── hyper/football/             ← LSports事件（195个fixture）
│   ├── polymarket/prices/          ← 58个市场价格文件
│   └── fixture_index.parquet       ← fixture元数据
├── src/pricing/
│   └── align_data.py               ← Phase 1 对齐脚本
├── docs/hf_docs/
│   ├── REQUIREMENTS_sports_pricing.md
│   └── MODEL_SPEC_markov_pricing.md
└── SUMMARY.md                      ← 本文件
```

---

## 📊 模型方向

### 定价公式（草稿）

对于 O/U N.5 市场，基于当前比赛状态的合理价格：

```
fair_price(t) = P(total_goals > N.5 | score_so_far, time_remaining, game_pace)
```

使用泊松过程估计：
- `λ(t)` = 当前比赛进球率（基于历史 + 当前比赛节奏）
- `P(k more goals in t_remaining | current_total)` = Poisson CDF

关键调整因子：
- `confidence_grade` → 未确认事件的概率折扣
- `time_remaining` → 接近终场时事件影响指数级放大
- `score_diff` → 领先队防守策略导致进球率下降

### 信号格式（目标输出）

```json
{
  "fixture_id": 16045142,
  "timestamp": "2026-05-10T18:07:09Z",
  "signal_type": "GOAL_LIKELY",
  "confidence": 0.14,
  "market_type": "O/U 3.5",
  "current_price": 0.765,
  "fair_price_estimate": 0.55,
  "price_edge": -0.215,
  "lead_time_s": 12
}
```

---

## 迭代记录

| 版本 | 日期 | 内容 |
|------|------|------|
| v0.1 | 2026-05-28 | Phase 1 数据对齐完成，28,847事件，26场比赛 |
| v0.2 | 进行中 | 毫秒延迟分析 + 进球信号定价 |
