# Probly Sports Pricing — 研究进度 Summary

> **目标**：用 LSports 实时体育事件流，在 Polymarket 价格调整之前实现合理定价，提前预知高波动窗口。

---

## 📌 整体问题定义

### 核心问题

Polymarket 预测市场（O/U goals、Win/Draw、BTTS 等）的价格随赛事进展实时变化。问题是：

1. **定价合理性**：当前 Polymarket 价格是否等于"合理概率"？什么时候会出现偏差？
2. **提前预测**：能否在 Polymarket 价格大幅调整之前，用 LSports 事件数据识别出即将到来的价格跳变？
3. **标签定义**：我们的学习目标是什么——预测价格方向？幅度？还是连续的合理价格？

---

## 🗂 一、之前的机器学习尝试（历史背景）

> 参考 `reference/` 和 `docs/worklogs/`。以下是对旧实现的梳理。

### 1.1 旧方法

旧代码（在 `/Volumes/T7/probly/src/` 中，本研究包未复制）做的是：
- **二分类任务**：预测未来 2 分钟内 Polymarket 价格是否变动 >3%（高波动 flag）
- **特征**：Markov 状态 ID + xT 值 + LSports 事件密度 + 信心度
- **模型**：Logistic Regression → Random Forest → XGBoost → LSTM
- **产物**：`outputs/real_dataset_v4.parquet`、多个 dashboard 截图

### 1.2 已知问题

| 问题类型 | 描述 |
|----------|------|
| **标签泄漏** | 用"过去已发生"的价格跳变作为标签，但特征可能包含跳变本身的信息 |
| **时间切分错误** | 没有严格按 fixture 做 train/test 切分，同一场比赛的事件可能同时出现在训练和测试集 |
| **标签定义模糊** | "高波动"定义为3%，但不同市场（O/U 1.5 vs O/U 4.5）基础波动率差异很大 |
| **多市场混合** | 把 Win/O/U/BTTS 不同性质的市场混在一起训练，物理含义不统一 |
| **同步与预测混淆** | 模型实际上是在解释"已发生的价格跳变"，而不是"预测未来的跳变" |
| **xT 计算无依据** | 缺少位置数据（LSports 数据没有球场坐标），xT 是伪造的 |
| **马尔可夫状态爆炸** | 状态空间太大（~37,800 个），但实际观测数据只有几千条，转移矩阵极度稀疏 |
| **旧流程混乱** | `build_dataset_v4.py` 有多个版本，逻辑复杂，中间变量命名模糊 |

### 1.3 旧代码指标

在旧 dashboard 里可见 F1 ≈ 0.65-0.72，但置信度不足，因为验证方式不规范。

---

## ✅ 二、当前已完成工作（从头重建）

### 2.1 数据整理与对齐（Phase 1）

**已完成**：

- **26场比赛**同时具有 LSports 事件流 + Polymarket 价格数据（从54个候选筛选）
- **28,847条事件记录**，对齐后包含各时间窗口（10s/30s/60s/120s/300s）的价格变化
- 关键修复：LSports 时间戳精度为 ns，Polymarket 为 μs，必须统一转换后才能正确对齐
- 输出：`data/aligned/master.parquet`

**26场比赛涵盖联赛**：德甲、西甲、英超、法甲、土超、MLS、欧洲杯友谊赛

**事件分布**（28,847行）：
```
Attacks             6,926
DangerousAttacks    5,763
Total Shots         2,782
FreeKicks           2,032
Fouls               1,804
Corners             1,754
Score               1,444  ← 核心目标事件
Substitutions       1,359
ShotsOnTarget       1,283
ShotsOffTarget      1,265
YellowCard            883
RedCard               155
Penalties             130
```

---

### 2.2 关键实证发现（Phase 2a：延迟分析）

对所有进球事件做了毫秒级时间戳对比分析（n=59个进球×市场组合）：

#### 🔑 核心发现 #1：LSports 永远不慢于 Polymarket

```
Polymarket 价格移动 vs LSports 首报时间（正=PM慢）：
  中位数延迟：+0.1 秒   （PM 比 LSports 晚 0.1 秒）
  平均延迟：  +4.3 秒
  P25:        0.0 秒
  P75:        0.2 秒
  
  PM 比 LSports 更快的比例：0.0%     ← 从未出现
  PM 在 LSports 首报后 5秒 内移动：89.8%
  PM 在 LSports 首报后 30秒 内移动：93.2%
```

**结论：LSports 总是比 Polymarket 市场早或同时报告进球**，不存在 PM 领先的情况。

#### 🔑 核心发现 #2：LSports 置信度窗口是定价优势

```
LSports 进球确认过程（从首报 conf≈0.14 到 conf=1.0）：
  中位数确认时间：108.8 秒
  平均确认时间：  260.3 秒
  
示例（FSV Mainz vs Union Berlin，第47分钟进球）：
  18:07:09  conf=0.14  ← 首报（市场价已开始跳）
  18:07:11  conf=0.29
  18:07:15  conf=0.48
  18:07:18  conf=0.61
  18:07:21  conf=1.00  ← 确认（约12秒后）
```

**意义**：在 conf=0.14 到 conf=1.0 的这段时间内，市场价格仍在调整中，存在定价偏差窗口。

#### 🔑 核心发现 #3：进球的价格冲击

```
各市场进球后60秒内价格变动：
  中位数 |ΔP|：0.011（1.1分）
  平均   |ΔP|：0.102（10.2分）    ← 高方差，部分市场影响极大
```

O/U 3.5 市场示例：
```
18:06:30  mid = 0.765  （进球前）
18:07:00  mid = 0.550  （进球后，跌 21.5 分）
→ 原因：比赛进入第47分钟，单球之后仍需3球才能 OVER 3.5，概率下调
```

---

### 2.3 文件结构（当前状态）

```
probly/
├── data/
│   ├── aligned/
│   │   └── master.parquet          ← 28,847行对齐事件（新建）
│   ├── hyper/football/             ← LSports事件（195个fixture）
│   ├── polymarket/prices/          ← 58个市场价格文件
│   └── fixture_index.parquet
├── src/pricing/
│   ├── align_data.py               ← Phase 1：数据对齐脚本（新建）
│   └── latency_analysis.py         ← Phase 2a：延迟分析（新建）
├── outputs/
│   ├── latency_analysis.parquet    ← 进球×市场延迟数据（新建）
│   └── latency_summary.json        ← 汇总统计（新建）
└── SUMMARY.md                      ← 本文件
```

---

## 🔄 三、进行中 / 计划工作

### 3.1 Phase 2b：公允价格模型（Fair Value）

**思路**：对于 O/U N.5 市场，合理价格可以用泊松过程计算：

```python
# 当前比分 h:a，剩余时间 T 分钟，当前进球率 λ
# 需要再进 k = ceil(N+1-h-a) 球才算 OVER
from scipy.stats import poisson
def fair_price_ou(h, a, total_line, lambda_per_min, time_remaining_min):
    goals_needed = max(0, int(total_line) + 1 - h - a)
    expected_more = lambda_per_min * time_remaining_min
    return 1 - poisson.cdf(goals_needed - 1, expected_more)
```

**待解决**：`λ`（进球率）如何实时更新？
- 选项A：用全联赛历史均值（约0.028球/分钟）
- 选项B：用本场前N分钟进球密度动态调整
- 选项C：用 DangerousAttacks 密度作为即时进球率代理

### 3.2 Phase 3：预信号检测（Lead-time Signals）

目标：在进球发生 30-120 秒前，识别出高风险状态。

**候选特征**：
- 过去 60 秒 DangerousAttacks 密度
- 过去 30 秒 ShotsOnTarget 数量
- 过去 5 分钟 Corners 累计数
- 当前比分差 × 剩余时间（决定市场价格敏感度）

### 3.3 Phase 4：可视化

- 价格路径 + 事件标注时间轴图（每场比赛）
- 事件类型 × 剩余时间 → 价格影响热力图
- 置信度轨迹可视化

---

## ❓ 四、需要专业判断的核心问题

### Q1：定价任务定义

> **目前我们做的是"overlay 调整"还是"从头预测"？**

两种方向本质不同：
- **A. Overlay 调整**：以 Polymarket 当前价格为基准，识别"偏离合理值"的时机，仅在置信度上升窗口内下注纠偏
- **B. 从头预测连续价格**：不依赖 Polymarket 现价，直接用 LSports 事件流推算合理价格，作为独立定价引擎

目前数据支持 A，因为我们能测量 PM 与 LSports 的延迟。B 需要更多历史数据验证泊松假设。

### Q2：标签设计

> **什么是"正确"的 label？**

- **选项1**：`high_vol_flag`（未来60s内价格变动>X%）— 二分类，直接用，但阈值X如何定？
- **选项2**：`delta_p_60s`（未来60s价格实际变化量）— 回归，连续标签，更丰富
- **选项3**：`fair_price_deviation`（当前PM价格 - 泊松公允价格）— 需要验证泊松模型准确性
- **推荐**：先做选项2（连续回归），再根据分布确定阈值做分类

### Q3：市场选择

> **应该分市场类型建模，还是统一模型+市场特征？**

现有数据：O/U 1.5 / 2.5 / 3.5 / 4.5 / 5.5 / BTTS / Win
- 不同市场对同一进球的反应方向甚至相反（O/U 3.5 vs BTTS）
- **建议**：先做 O/U 2.5 单独分析（最流动、最通用），再泛化

### Q4：LSports 置信度的含义

> **confidence_grade 是"进球已确认概率"还是"事件分类置信度"？**

从数据看，每个进球首报时 conf≈0.10-0.15，然后快速升到1.0（通常12-30秒），偶尔出现 conf 下降又上升（VAR 审查）。需要确认：
- conf < 0.5 时，Polymarket 市场应该调整多少？
- 有没有进球首报后 conf 最终没到 1.0（即进球被取消）的案例？

### Q5：样本量

> **26场比赛够用吗？**

目前有约 50-100 个进球事件。对于统计分析基本够，但如果用机器学习预测 lead-time 信号可能不足。
- 需要更多历史数据：是否能从 HuggingFace 获取更多 fixture_id 的 Polymarket 价格？
- 或者：能否直接从 Polymarket API 拉取历史比赛的价格数据？

---

## 📊 迭代记录

| 版本 | 日期 | 内容 | 状态 |
|------|------|------|------|
| v0.1（旧实现） | 2026-05-25之前 | ML 二分类（XGBoost+Markov+xT），F1≈0.65 | ❌已放弃，问题多 |
| **v1.0** | **2026-05-28** | **Phase 1 数据对齐，28,847事件，26场** | ✅已完成 |
| **v1.1** | **2026-05-28** | **Phase 2a 延迟分析：LSports永远不慢于PM** | ✅已完成 |
| v1.2 | 计划中 | Phase 2b 泊松公允价格模型 | 🔄进行中 |
| v2.0 | 计划中 | Phase 3 预信号检测（lead-time） | ⏳待 |
| v2.1 | 计划中 | Phase 4 可视化 dashboard | ⏳待 |
