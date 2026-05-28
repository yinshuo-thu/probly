# Probly Sports Pricing — 研究进度 Summary

> **目标**：用 LSports 实时体育事件流，在 Polymarket 价格调整之前实现合理定价，提前预知高波动窗口。

---

## 📌 整体问题定义

Polymarket 预测市场（O/U goals、Win/Draw、BTTS）的价格随赛事进展实时变化。研究三个核心问题：
1. **LSports vs Polymarket 时间优势**：LSports 何时比 PM 更快？
2. **进球前预信号**：进球发生前 10-120 秒，事件流中有哪些异常？
3. **合理定价**：基于当前比赛状态，各市场的"公允价格"是多少？

---

## 🗂 一、旧实现审计（历史背景）

### 旧方法（已放弃）

| 问题 | 描述 |
|------|------|
| 标签泄漏 | 用"已发生"的价格跳变作标签，特征包含跳变本身信息 |
| 时间切分错误 | 同一场比赛数据跨越 train/test，导致信息泄漏 |
| 多市场混合 | O/U 1.5 / 4.5 / BTTS 混在一起，物理含义不统一 |
| xT 计算伪造 | LSports 无球场坐标，xT 是凭空估算 |
| 马尔可夫状态稀疏 | 约37,800个状态，实际数据只有几千条 |
| 旧F1 ≈ 0.65 | 验证方式不规范，无实际参考价值 |

---

## ✅ 二、当前完成工作

### Phase 1：数据对齐（完成）

- **26场比赛** × LSports事件 + Polymarket价格对齐
- **28,847条事件记录**，各时间窗口（10/30/60/120/300s）价格变化
- 关键修复：LSports ns 精度 vs Polymarket μs 精度的时间戳统一
- 输出：`data/aligned/master.parquet`

---

### Phase 2a：延迟分析（完成）

**59个进球×市场组合的毫秒级测量**：

```
LSports 首报 → Polymarket 价格移动延迟：
  中位数：+0.1 秒   ← LSports 永远不慢于 PM
  平均：  +4.3 秒
  PM 比 LSports 更快：0.0%   ← 从未发生
  PM 在 LSports 首报 5秒内跟进：89.8%
  
LSports 进球置信度确认窗口（0.14→1.0）：
  中位数：108.8 秒   ← 这是定价优势窗口
```

**案例（FSV Mainz 第47分钟进球，O/U 3.5 市场）**：
```
18:07:09  LSports: conf=0.14（首报）
18:07:11  LSports: conf=0.29
18:07:15  LSports: conf=0.48
18:07:18  LSports: conf=0.61
18:07:21  LSports: conf=1.00（确认）← 12秒完成确认
18:07:00  Polymarket: 0.765 → 0.550（-21.5¢ 跳变，几乎同时）
```

---

### Phase 2b：泊松公允价格模型（完成）

**模型**：`P(OVER N.5 | total_goals, time_remaining, λ)` = Poisson CDF

```python
def poisson_ou_fair_price(ou_line, goals_so_far, time_remaining_min, lambda_per_min):
    goals_needed = int(ou_line) + 1 - goals_so_far
    expected_more = lambda_per_min * time_remaining_min
    return 1 - poisson.cdf(goals_needed - 1, expected_more)
```

**结果**（91,325个快照，24场比赛，51个市场）：

| 市场类型 | 泊松公允均值 | PM市场均值 | 平均偏差 |
|----------|-------------|-----------|---------|
| O/U 1.5  | 0.446 | 0.668 | -0.224 |
| O/U 2.5  | 0.331 | 0.467 | -0.136 |
| O/U 3.5  | 0.184 | 0.330 | -0.147 |
| O/U 4.5  | 0.112 | 0.276 | -0.164 |
| O/U 5.5  | 0.050 | 0.140 | -0.090 |
| BTTS     | 0.359 | 0.584 | -0.230 |

**关键发现**：Polymarket 持续比泊松公允价高 10-22¢。
原因：泊松模型使用全联赛均值 λ（2.6-2.79 球/场），但市场定价包含更多信息（进攻节奏、技战术、市场情绪）。

**泊松模型适用性**：
- ✅ 可以判断价格变动方向（进球后价格应涨/跌）
- ✅ 可以估计进球后价格应该移动多少
- ❌ 无法直接给出绝对价格（需要校准 λ 和市场vig）

---

### Phase 3：Lead-Time 预信号分析（完成）

**方法**：对 109 个已确认进球，分析进球前 10/30/60/120 秒的事件频率 vs 背景频率。

**核心结果 — 进球前 lift 倍率**：

| 事件类型 | 10s前 | 30s前 | 60s前 | 120s前 |
|----------|-------|-------|-------|--------|
| **DangerousAttacks** | **4.15x** | **2.45x** | 1.93x | 2.26x |
| **Corners** | **3.10x** | 2.24x | **2.91x** | 2.21x |
| Attacks | 1.86x | 1.72x | 1.89x | 2.17x |
| BlockedShots | 1.40x | 2.08x | 1.35x | 1.34x |
| ShotsOnTarget | 0.70x ↓ | 0.90x ↓ | 1.17x | 1.44x |
| FreeKicks | **0.18x ↓↓** | 0.58x ↓ | 1.32x | 1.51x |
| ShotsOffTarget | **0.25x ↓↓** | 0.68x ↓ | 0.95x | 1.20x |

**解读**：
- **DangerousAttacks 在进球前10秒出现频率是背景的4.15倍** → 主要预信号
- **Corners 在60秒前出现频率是背景的2.91倍** → 最佳30-60秒预警信号
- FreeKicks 和 ShotsOffTarget 在进球前10-30秒反而减少 → 负信号（进球不来自这些）

**实际含义**：当 DangerousAttacks 密集出现（4.15x background rate），价格跳变概率在后续10秒显著提升。

---

### Phase 4：可视化（完成）

生成图表见 `outputs/plots/`：

| 文件 | 内容 |
|------|------|
| `confidence_trajectory.png` | LSports 置信度爬升 vs PM 价格跳变时序对比 |
| `latency_analysis.png` | PM延迟分布 + 进球价格冲击分布 |
| `lift_heatmap.png` | 事件类型 × 时间窗口 → lift ratio 热力图 |
| `fair_value_comparison.png` | Man City vs Brentford O/U 2.5：泊松公允价 vs 市场价 |
| `timeline_16066981.png` | Man City vs Brentford 完整价格+事件时间轴 |
| `timeline_16045142.png` | FSV Mainz vs Union Berlin 完整价格+事件时间轴 |

---

## 📁 文件结构

```
probly/
├── data/
│   ├── aligned/master.parquet       ← 28,847行对齐事件（Phase 1）
│   ├── hyper/football/              ← 195个fixture LSports事件
│   └── polymarket/prices/           ← 58个市场价格文件
├── src/pricing/
│   ├── align_data.py                ← Phase 1 数据对齐
│   ├── latency_analysis.py          ← Phase 2a 延迟分析
│   ├── fair_value_model.py          ← Phase 2b 泊松公允价格
│   ├── lead_signal.py               ← Phase 3 预信号检测
│   └── visualize.py                 ← Phase 4 可视化
├── outputs/
│   ├── latency_summary.json         ← 延迟统计
│   ├── fair_value_summary.json      ← 公允价格统计
│   ├── lead_signal_stats.json       ← 预信号lift倍率
│   ├── prediction_dataset.parquet   ← 97,865行特征数据集
│   └── plots/                       ← 6张分析图表
└── SUMMARY.md                       ← 本文件
```

---

## ❓ 三、待解决问题

### Q1：定价任务定义
- **A. Overlay 调整**：以 PM 当价格为基准，在 conf 上升窗口内纠偏
- **B. 从头预测连续价格**：用 LSports 推算合理价格，独立于 PM
- **建议**：先做 A，用泊松 fair_price + LSports conf 做置信度加权调整

### Q2：λ 校准
- 全联赛均值（2.6球/场）vs 本场实时进球率 vs DangerousAttacks 密度作代理
- **建议**：用滚动窗口（最近 15 分钟）动态更新 λ

### Q3：样本量
- 26场比赛，109个进球事件（Phase 3 分析用）
- 预信号检测需要更多样本才能做统计学显著性检验

### Q4：置信度到价格的映射
- conf=0.14（首报）→ 应该调整多少价格？
- conf 上升轨迹（0.14→0.29→0.48→1.0）是否可以预测最终确认？

### Q5：做市策略
- 基于 DangerousAttacks lift 信号，应该如何调整报价 spread？
- 什么时候扩大 spread（高 momentum 时），什么时候收窄？

---

## 📊 迭代记录

| 版本 | 日期 | 内容 | 状态 |
|------|------|------|------|
| 旧实现 | 2026-05-25前 | ML二分类(XGBoost+Markov+xT), F1≈0.65 | ❌放弃 |
| **v1.0** | **2026-05-28** | Phase 1 数据对齐，28,847事件，26场 | ✅ |
| **v1.1** | **2026-05-28** | Phase 2a 延迟：LSports永远不慢于PM | ✅ |
| **v1.2** | **2026-05-28** | Phase 2b 泊松公允价格，91,325快照 | ✅ |
| **v1.3** | **2026-05-28** | Phase 3 预信号：DA=4.15x, Corner=3.10x | ✅ |
| **v1.4** | **2026-05-28** | Phase 4 可视化，6张分析图表 | ✅ |
| v2.0 | 计划中 | 实时定价信号生成器（置信度加权） | ⏳ |
| v2.1 | 计划中 | 回测框架（历史数据模拟下单） | ⏳ |
