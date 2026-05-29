# 单重要赛事先行结论

## Match

- Fixture: `18746260`
- Match: Red Bull Bragantino vs Carabobo FC
- League: Copa Sudamericana
- Kickoff: 2026-05-28 00:30:00 UTC
- Polymarket event slug: `sud-bra-car-2026-05-27`
- Polymarket market analyzed: Red Bull Bragantino win, Yes token

## Key Answer

用更真实的 **PMXT historical BBO/orderbook** 口径看, 本场进球附近是 **Polymarket BBO 先动, LSports goal timestamp 后到**。

| LSports goal time | BBO first significant move | BBO lead |
|---|---:|---:|
| 00:46:54.967 UTC | 00:46:45.178 UTC | 9.8s |
| 02:07:21.622 UTC | 02:07:16.022 UTC | 5.6s |
| 02:24:12.788 UTC | 02:24:10.969 UTC | 1.8s |

口径: 进球前 10 秒 BBO mid 作为基准, 在进球前 1 秒到进球后 300 秒内寻找首次 `>=3c` 的 BBO mid 跳变。负延迟表示 BBO 早于 LSports 进球时间戳。

## Why This Differs From `/prices-history`

官方 CLOB `/prices-history` 的分钟级价格序列显示 LSports 领先可观测价格点约 9/44/51 秒。但该序列不是 BBO, 分辨率更粗。PMXT historical BBO 更接近可交易盘口, 因此对于“谁先反应”应优先采用 BBO 结论。

## Implication

这场比赛里, 不能简单依赖“LSports 报进球后交易”来领先盘口。真正值得研究的是更早的阶段: 用 LSports 的危险进攻、射门、角球、事件强度等赛中流特征, 在进球前几十秒预测 BBO repricing 风险。

## Visuals

- `outputs/figures/single_match_bbo_goal_windows.png`
- `outputs/figures/single_match_bbo_latency_bars.png`
- `outputs/figures/single_match_signal_vs_bbo.png`
- `outputs/figures/single_match_polymarket_price_reaction.png`

## Signal Prototype

单场 LSports hazard 原型显示, 通过 rolling event intensity 可以提前覆盖 3/3 个 BBO repricing moments, 但存在误报。较实用的下一步不是单场调阈值, 而是跨场训练:

- target: future BBO mid jump, not just goal
- model: calibrated gradient boosting / survival model / temporal sequence model
- validation: grouped by fixture/date/league
- execution filter: only alert when BBO has not moved, spread/liquidity still tradable
