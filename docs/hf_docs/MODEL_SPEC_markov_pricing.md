# 状态转移定价模型：LSports 事件序列 → Polymarket 价格路径

## 1. 模型目标

构建一个轻量级、可解释的状态转移模型，将 LSports Hyper 实时事件流（带 confidence_grade）映射到 Polymarket 合约的 mid-price 变动路径，实现：

1. **定价路径置信率** — 给定当前赛事状态序列，预测 Polymarket 价格的期望变动及其置信度
2. **关键事件识别** — 哪些事件类型对价格影响最大
3. **关键路径发现** — 哪些事件组合（路径）导致最大价格跳跃
4. **价格跳跃归因** — 不同幅度的价格跳跃对应何种物理事件

---

## 2. 理论基础：Markov Chain × Expected Threat (xT) 框架

### 2.1 核心思想

借鉴足球分析领域的 **Expected Threat (xT)** 和 **Possession Value (PV)** 模型框架，将赛事状态空间离散化，用转移矩阵描述事件序列的演进，并计算每个状态到"结算事件"（如进球 → 市场价格跳变）的路径概率。

**关键创新**: 我们不是预测进球本身，而是预测 **Polymarket mid-price 的期望变动幅度**。

### 2.2 学术支撑

| 论文 | 年份 | 核心贡献 | 与本项目关联 |
|------|------|----------|-------------|
| Van Arem et al., "The trade-off between model flexibility and accuracy of the Expected Threat model" | 2025 | xT 状态划分对 goal probability 估计的影响；误差量化与灵活性权衡 | 状态空间设计、confidence 校准 |
| Jesse Davis et al., "Three key design decisions for possession state value models" | 2024 | PV 框架设计选择（状态定义、转移、价值迭代）的实验对比 | 模型设计决策参考 |
| Shelopugin et al., "Expected Possession Value of Control and Duel Actions" | 2024 | EPV 扩展：融入 decay effect（近期事件权重更高）+ confidence 加权 | 直接借鉴 confidence 加权机制 |
| Van Roy & Davis, "A Markov Framework for Learning and Reasoning About Strategies in Professional Soccer" | 2023, JAIR | MDP 学习进攻策略 + probabilistic model checking | MDP 路径推理方法论 |
| "Understanding soccer possessions via path signatures" | 2025 | Path Signatures + Markov-style xT/EPV 处理变长序列 | 变长事件路径建模 |
| Van Roy et al., "Valuing On-the-Ball Actions: xT vs VAEP" | 2020 (仍被广泛引用) | xT（纯 Markov）vs VAEP（ML 增强）对比 | 验证轻量 Markov 方案有效性 |

**趋势共识 (2024-2026)**: Markov / xT / PV 仍是解释性最强、最轻量的主流基线。标准数据集上 goal probability 路径预测 AUC 常达 0.75+，尤其适合事件序列 + confidence 输入。

---

## 3. 状态空间定义

### 3.1 赛事状态向量 S(t)

每个时刻 t 的赛事状态定义为：

```
S(t) = (score_diff, time_remaining, period, momentum, last_event_type, confidence)
```

| 维度 | 说明 | 离散化方式 |
|------|------|-----------|
| `score_diff` | 主队 - 客队得分差 | 整数：{-4, -3, -2, -1, 0, +1, +2, +3, +4} |
| `time_remaining` | 剩余比赛时间（分钟） | 分桶：{0-15, 15-30, 30-45, 45-60, 60-75, 75-90, 90+} |
| `period` | 当前阶段 | {1H, 2H, ET1, ET2, PEN} |
| `momentum` | 近 N 分钟事件密度（进攻压力） | {low, medium, high} |
| `last_event_type` | 最近发生的关键事件 | {Score, Corner, FreeKick, YellowCard, RedCard, Penalty, ShotOnTarget, ...} |
| `confidence` | LSports confidence_grade | 连续 [0, 1] 或分桶 {low<0.5, mid 0.5-0.8, high>0.8, confirmed=1.0} |

### 3.2 状态空间规模估算

```
|S| ≈ 9 × 7 × 5 × 3 × 10 × 4 ≈ 37,800 states
```

完全可在 CPU 上用 NumPy 矩阵运算处理（内存 < 100MB）。

---

## 4. 核心模型：Event × Confidence → Price Path

### 4.1 转移矩阵 T

从历史数据学习状态转移概率：

```
T[s_i → s_j] = P(S(t+1) = s_j | S(t) = s_i)
```

**confidence 加权转移**:

```
T_weighted[s_i → s_j] = Σ_k [ conf_k × I(transition_k = s_i→s_j) ] / Σ_k [ conf_k × I(state_k = s_i) ]
```

其中 `conf_k` 是第 k 次观测的 confidence_grade。低置信度事件对转移矩阵贡献更小。

### 4.2 价格影响函数 ΔP(event, state)

核心目标：学习每种事件在不同状态下对 Polymarket mid-price 的影响：

```
ΔP(event_type, S(t)) = E[ P_poly(t + Δt) - P_poly(t) | event, S(t) ]
```

其中 Δt 为价格反应窗口（建议 1s / 5s / 30s 多尺度）。

### 4.3 路径价值函数 V(s)

类似 xT 的价值迭代，计算从状态 s 出发到比赛结束时的**期望累计价格变动**：

```
V(s) = Σ_j T[s→s_j] × [ ΔP(transition_event, s) + γ × V(s_j) ]
```

γ 为时间衰减因子（剩余时间越少，事件对价格影响越大）。

### 4.4 路径置信率 PathConf(s₁→s₂→...→sₙ)

给定一条事件路径，计算其发生概率与价格影响的联合置信度：

```
PathConf = Π_i T[s_i→s_{i+1}] × Π_i conf_i × |ΔP_cumulative|
```

高 PathConf 的路径 = **高置信度 + 高概率 + 高价格影响** → 最佳交易信号。

---

## 5. 价格跳跃分析框架

### 5.1 价格跳跃分级

| 跳跃等级 | ΔP 幅度 | 典型物理事件 | 剩余时间影响 |
|----------|---------|-------------|-------------|
| **L5 (极端)** | >30¢ | 进球（比分领先/扳平，最后15分钟） | 剩余时间↓ → 影响↑↑↑ |
| **L4 (大幅)** | 15-30¢ | 进球（上半场）、红牌、点球判定 | 剩余时间↓ → 影响↑↑ |
| **L3 (中等)** | 5-15¢ | 进球（大比分差追加）、点球未进、VAR取消进球 | 视比分差而定 |
| **L2 (小幅)** | 2-5¢ | 角球连续、危险进攻密集、射正目标 | momentum 累积效应 |
| **L1 (微调)** | <2¢ | 任意球、控球率变化、黄牌 | 仅在关键时刻有效 |

### 5.2 剩余时间 × 事件类型的交互效应

**核心洞察**: 相同事件在不同剩余时间下对价格的影响完全不同。

```
ΔP_effective(event, t_remaining) = ΔP_base(event) × TimeDecay(t_remaining) × ScoreContext(score_diff)
```

**TimeDecay 函数**（非线性放大效应）：

```python
def time_decay(t_remaining_minutes, total_minutes=90):
    """剩余时间越少，事件影响越大（指数放大）"""
    ratio = 1 - (t_remaining_minutes / total_minutes)
    return 1 + 2 * (ratio ** 2)  # 最后时刻影响 ×3
```

**ScoreContext 函数**：

```python
def score_context(score_diff, event_changes_score=True):
    """比分越接近，进球影响越大"""
    if event_changes_score:
        if score_diff == 0:  # 打破僵局
            return 2.0
        elif abs(score_diff) == 1:  # 扳平/反超
            return 2.5
        else:  # 扩大优势
            return 0.8
    return 1.0
```

### 5.3 事件组合路径（关键路径模式）

通过频繁子序列挖掘，识别导致大价格跳跃的**事件组合模式**：

| 路径模式 | 价格影响 | 解释 |
|----------|---------|------|
| `DangerousAttack → Corner → ShotOnTarget → Score` | L5 | 经典进攻得分路径 |
| `Foul → FreeKick(dangerous) → Score` | L4 | 任意球直接得分 |
| `Corner → Corner → Corner → ShotOnTarget` | L2→L3 | 角球围攻，momentum 累积 |
| `RedCard → [5min window] → DangerousAttack` | L3 | 红牌后进攻压力 |
| `Score(equalizer) → [last 10min]` | L5 | 终场扳平 |
| `Penalty → MissedPenalty` | L4 (reverse) | 点球未进，价格反转 |

---

## 6. 实现方案

### 6.1 Phase 1: 数据对齐 + 基线统计

```python
import numpy as np
import pandas as pd

class EventPriceAligner:
    """将 LSports 事件与 Polymarket 价格对齐"""
    
    def align(self, events_df, prices_df, window_ms=5000):
        """
        对每个 LSports 事件，找到:
        1. 事件前的 mid-price (P_before)
        2. 事件后 window_ms 内的 mid-price (P_after)
        3. 计算 ΔP = P_after - P_before
        """
        results = []
        for _, event in events_df.iterrows():
            t = event['timestamp_utc']
            p_before = prices_df[prices_df['ts'] <= t].iloc[-1]['mid']
            p_after = prices_df[
                (prices_df['ts'] > t) & 
                (prices_df['ts'] <= t + pd.Timedelta(ms=window_ms))
            ]
            if len(p_after) > 0:
                results.append({
                    'event': event['incident_name'],
                    'confidence': event['confidence_grade'],
                    'score_diff': event['home_value'] - event['away_value'],
                    't_remaining': event['time_remaining_min'],
                    'delta_p': p_after.iloc[-1]['mid'] - p_before,
                    'delta_p_max': p_after['mid'].max() - p_before,
                })
        return pd.DataFrame(results)
```

### 6.2 Phase 2: Markov 转移矩阵学习

```python
class MarkovPriceModel:
    """基于 xT 框架的价格预测模型"""
    
    def __init__(self, state_dims):
        self.n_states = np.prod(state_dims)
        self.T = np.zeros((self.n_states, self.n_states))  # 转移矩阵
        self.R = np.zeros(self.n_states)  # 状态价值（期望价格变动）
        
    def fit(self, state_sequences, price_changes, confidences):
        """从历史数据学习转移概率和价格影响"""
        for seq, prices, confs in zip(state_sequences, price_changes, confidences):
            for i in range(len(seq) - 1):
                s_from, s_to = seq[i], seq[i+1]
                weight = confs[i]  # confidence 加权
                self.T[s_from, s_to] += weight
                self.R[s_to] += weight * prices[i]
        
        # 归一化
        row_sums = self.T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        self.T /= row_sums
        
        counts = self.T.sum(axis=0)
        counts[counts == 0] = 1
        self.R /= counts
        
    def value_iteration(self, gamma=0.95, n_iter=100):
        """计算每个状态的期望累计价格变动"""
        V = np.zeros(self.n_states)
        for _ in range(n_iter):
            V_new = self.R + gamma * self.T @ V
            if np.max(np.abs(V_new - V)) < 1e-6:
                break
            V = V_new
        return V
    
    def path_confidence(self, path, confidences):
        """计算路径的联合置信率"""
        transition_prob = np.prod([
            self.T[path[i], path[i+1]] 
            for i in range(len(path)-1)
        ])
        conf_prod = np.prod(confidences)
        value_gain = abs(self.R[path[-1]] - self.R[path[0]])
        return transition_prob * conf_prod * value_gain
```

### 6.3 Phase 3: 价格跳跃检测与归因

```python
class PriceJumpDetector:
    """检测并归因价格跳跃"""
    
    THRESHOLDS = {
        'L1': 0.02, 'L2': 0.05, 'L3': 0.15, 'L4': 0.30, 'L5': 0.50
    }
    
    def detect_jumps(self, prices_df, min_level='L2'):
        """检测所有超过阈值的价格跳跃"""
        threshold = self.THRESHOLDS[min_level]
        jumps = []
        for i in range(1, len(prices_df)):
            delta = abs(prices_df.iloc[i]['mid'] - prices_df.iloc[i-1]['mid'])
            if delta >= threshold:
                jumps.append({
                    'timestamp': prices_df.iloc[i]['ts'],
                    'delta': delta,
                    'level': self._classify_level(delta),
                    'direction': 'up' if delta > 0 else 'down'
                })
        return pd.DataFrame(jumps)
    
    def attribute_to_events(self, jumps_df, events_df, lookback_s=10):
        """将价格跳跃归因到最近的体育事件"""
        attributions = []
        for _, jump in jumps_df.iterrows():
            t = jump['timestamp']
            recent_events = events_df[
                (events_df['timestamp_utc'] >= t - pd.Timedelta(seconds=lookback_s)) &
                (events_df['timestamp_utc'] <= t)
            ]
            attributions.append({
                **jump,
                'attributed_events': recent_events['incident_name'].tolist(),
                'max_confidence': recent_events['confidence_grade'].max(),
            })
        return pd.DataFrame(attributions)
```

---

## 7. 预期输出与交付物

### 7.1 关键事件影响热力图

```
事件类型 × 剩余时间 → 平均 ΔP

              0-15min  15-30min  30-45min  45-60min  60-75min  75-90min
Score(lead)    +32¢     +25¢      +18¢      +15¢      +12¢      +8¢
Score(equal)   +28¢     +22¢      +15¢      +12¢      +10¢      +6¢
RedCard        +15¢     +12¢      +8¢       +6¢       +5¢       +4¢
Penalty(award) +20¢     +18¢      +14¢      +12¢      +10¢      +8¢
Corner         +1.5¢    +1.2¢     +1.0¢     +0.8¢     +0.5¢     +0.3¢
ShotOnTarget   +2.0¢    +1.5¢     +1.2¢     +0.8¢     +0.5¢     +0.3¢
```

### 7.2 关键路径 Top-K

```
Rank | Path                                          | PathConf | Avg ΔP
  1  | DangerousAttack → ShotOnTarget → Score        | 0.82     | +28¢
  2  | Foul → DangerousFreeKick → Score              | 0.76     | +25¢
  3  | Corner → Corner → ShotOnTarget → Score        | 0.71     | +22¢
  4  | RedCard → [decay 5min] → DangerousAttack      | 0.65     | +12¢
  5  | Penalty(awarded) → Penalty(scored)            | 0.88     | +18¢
```

### 7.3 定价信号输出格式

```json
{
  "signal_id": "sig_20260520_match12345_ev47",
  "fixture_id": 18934236,
  "timestamp_utc": "2026-05-18T19:32:15.123Z",
  "current_state": {
    "score_diff": 0,
    "time_remaining_min": 12,
    "period": "2H",
    "momentum": "high",
    "last_event": "DangerousAttack"
  },
  "lsports_confidence": 0.92,
  "predicted_price_move": {
    "direction": "up",
    "magnitude_cents": 18.5,
    "confidence": 0.78,
    "time_horizon_ms": 5000
  },
  "path_context": ["Corner", "ShotOnTarget", "DangerousAttack"],
  "recommended_action": "BUY_YES",
  "edge_estimate": 0.12
}
```

---

## 8. 发散思考：扩展维度

### 8.1 多市场联动

单场赛事在 Polymarket 上可能有多个合约：
- "Team A 获胜" — 直接受比分影响
- "总进球数 Over 2.5" — 受所有进球事件影响
- "两队都进球 (BTTS)" — 受特定一方进球影响
- "角球数 Over 9.5" — 受角球事件直接影响

→ 同一事件序列可驱动多个市场的定价信号。

### 8.2 Confidence 动态衰减

LSports 的 confidence_grade 不是静态的：
- 初始报告 confidence = 0.7（可能进球）
- VAR 审查中 confidence = 0.5（存疑）
- VAR 确认 confidence = 1.0（确认进球）

→ 监控 confidence 的**变化轨迹**本身就是交易信号：
- confidence 上升 → 提前 BUY
- confidence 突然下降 → 立即 EXIT/SELL

### 8.3 跨运动泛化

模型核心（状态转移 × confidence → 价格）可跨运动迁移：
- 篮球：每次得分频率更高，价格跳跃更小但更频繁
- 网球：Set/Game/Match point 是关键状态节点
- 拳击：Round 结果 + 判定概率

### 8.4 做市策略推导

基于模型输出，构建做市策略：
1. **Spread 调整**: 高 momentum 时 → 扩大 spread（保护自己）
2. **Quote 偏移**: 路径置信率高 → 向预测方向偏移报价
3. **库存管理**: 连续同向信号 → 逐步建仓，PathConf 下降 → 平仓

---

## 9. 技术栈与性能要求

| 组件 | 选型 | 理由 |
|------|------|------|
| 核心计算 | NumPy / SciPy | xT 转移矩阵 + value iteration，CPU 友好 |
| 数据存储 | Parquet + ClickHouse | 历史回测 + 实时查询 |
| 路径挖掘 | PrefixSpan / 自定义 | 频繁子序列发现 |
| 实时推理 | Python / Rust (PyO3) | 低延迟信号生成 |
| 可视化 | Plotly / Matplotlib | 热力图 + 路径可视化 |

**性能目标**:
- 模型训练: < 5 分钟（单 CPU，全部历史数据）
- 单事件推理: < 1ms
- 价值迭代收敛: < 100 轮

---

## 10. 实施路线图

```
Week 1-2: 数据对齐
├── LSports Football 事件 ↔ Polymarket 市场匹配
├── 时间轴对齐（ms 精度）
└── 基线统计：每种事件的 avg/median/p95 ΔP

Week 3-4: Markov 模型 v1
├── 状态空间定义 + 离散化
├── 转移矩阵学习（confidence 加权）
├── Value iteration → 状态价值表
└── 验证：预测 ΔP vs 实际 ΔP 的相关性

Week 5-6: 价格跳跃分析
├── 跳跃检测 + 分级
├── 事件归因
├── 关键路径挖掘
└── 剩余时间交互效应建模

Week 7-8: 信号生成 + 回测
├── 实时信号生成器
├── 历史回测 PnL
├── 做市策略参数优化
└── 报告 + 部署方案
```

---

## 11. 数据仓库

| 资源 | 链接 |
|------|------|
| LSports Hyper 数据 | https://huggingface.co/datasets/probly/lsports-hyper-dataset |
| LSports Trade 赔率 | https://huggingface.co/datasets/probly/lsports-trade-dataset |
| 项目需求文档 | 本文档 |
| 代码仓库 | https://gitlab.agesail.ai/wangwangwilson/probly_lab |
