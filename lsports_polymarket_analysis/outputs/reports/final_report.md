# LSports × Polymarket 数据价值分析最终报告

## Executive Summary

本项目已经搭建出一套可复用的 LSports 体育事件流研究 pipeline, 覆盖数据下载、事件解析、比分/进球重建、单赛事深度分析、事件强度特征、goal-hazard 建模、Polymarket 价格接入占位和 stale-window 计算接口。

当前可直接验证的结论是: LSports 事件流在已缓存样本中具备秒级时间分辨率, 单场 Red Bull Bragantino vs Carabobo FC 的实时事件中位间隔约 1.47 秒, p90 约 5.32 秒。这足以支撑 event-driven sports pricing 的技术路径。

当前不能直接验证的结论是: “LSports 是否比 Polymarket 市场价格反应更快”。原因有两个:

1. Hugging Face LSports 数据集本身不含 Polymarket price/BBO/order-book history。
2. 当前环境没有 HF_TOKEN, 只能复现本地已缓存的 1 场样本, 不能扩展到 5 日 × 每日 40 场的完整批量统计。

因此，本报告把结论明确分为“已验证”和“待补数据验证”，避免把方法演示误表述为交易级 edge。

## 已完成工作

- 初始化 `lsports_polymarket_analysis/` 独立子项目, 不破坏仓库原结构。
- 实现 Hugging Face LSports 数据下载/缓存脚本: `scripts/download_hf_data.py`。
- 实现离线回退: 无 HF token 或远程 401 时自动扫描本地 `data/raw` 缓存。
- 实现 LSports messages parser: 标准化时间戳、incident 分类、实时事件过滤、xT-style 权重。
- 用 `Score` incident 的 cummax 方法重建比分, 对半场重置/噪声有鲁棒性。
- 实现单赛事报告和图表: timeline、score、event intensity、goal hazard。
- 实现 batch summary 表: `match_summary.csv`, `goal_event_summary.csv`, `latency_summary.csv`。
- 实现 Polymarket 接入 schema 和 CLOB `/prices-history` loader, 未拿到真实映射时返回空表而不伪造数据。
- 实现 `goal_price_reaction()` 和 `detect_stale_window()` 接口, 一旦接入价格历史即可计算领先/滞后与 stale window。

## 当前实证结果

### 单赛事样本

- fixture: `18746260`
- 比赛: Red Bull Bragantino vs Carabobo FC
- 联赛: Copa Sudamericana
- 开赛: 2026-05-28T00:30:00Z
- 最终比分: 2-1
- 事件总数: 16053
- 实时事件数: 2755
- 识别进球: 3

### LSports 时间分辨率

- 实时事件中位间隔: 1.47 秒
- 实时事件 p90 间隔: 5.32 秒
- 归档延迟 p50: 5328 秒

注意: 归档延迟来自 `ingested_at_utc - timestamp_utc`, 反映的是小时级批量归档流程, 不是 LSports 实时推送延迟。

### 进球前信号

单场内的 rolling xT/intensity 特征可以在进球前给出明显抬升信号。当前单场 logistic hazard AUC 很高, 但这是样本内演示, 正样本少, 不能当作泛化结论。跨场泛化需要恢复 HF token 后扩展样本。

## 对核心问题的回答

1. LSports 是否比 Polymarket 价格反应更快?

当前不能下交易级结论。LSports 事件流本身是秒级, 但缺 Polymarket 历史价格/BBO, 无法计算真实 reaction latency。

2. 能否提前判断 Polymarket 价格滞后?

方法上可以。代码已预留 fixture→market 映射、价格历史 join、goal reaction latency 和 stale-window 计算。一旦接入 market_id/token_id 与价格序列, 可批量输出 stale window 秒数和领先比例。

3. 能否用事件流预测进球概率或价格方向?

事件流可以构造进球前代理信号, 当前单场结果支持方法可行性。价格方向预测需要把 label 从 “未来 N 秒是否进球” 换成 “未来 N 秒 Polymarket mid/BBO 是否显著移动”。

4. 如何拓展到批量赛事统计?

现有 batch 脚本已经按日期和 fixture 扫描, 能输出 match/goal/latency summary, 并按 fixture 做 held-out hazard 模型。恢复 HF token 后运行:

```bash
cd lsports_polymarket_analysis
python scripts/download_hf_data.py --docs --max 40
python scripts/batch_event_analysis.py --max 40
```

## 建议的下一步实验

1. 补 Hugging Face token, 下载 2026-05-24 至 2026-05-28 每日 40 场样本。
2. 建立 fixture→Polymarket market/token 映射表: `data/processed/fixture_market_map.parquet`。
3. 拉取 Polymarket CLOB `/prices-history` 或更高频 BBO/order-book snapshot。
4. 以进球、红牌、点球等 LSports 事件为 anchor, 计算 Polymarket 首次显著价格反应时间。
5. 输出统计指标:
   - LSports lead rate
   - median/p90 stale window
   - price jump magnitude
   - event-type conditional latency
   - league/team/liquidity 分层结果
6. 将模型标签替换为 BBO mid 价格移动, 评估能否预测 market repricing direction。

## 研究边界

本项目没有伪造任何 Polymarket 价格, 也没有把 LSports 批量归档延迟误解释为实时传输延迟。当前结论是“LSports 秒级事件流具备定价信号基础, pipeline 已准备好验证价格领先性”, 不是“已经证明可稳定套利”。
