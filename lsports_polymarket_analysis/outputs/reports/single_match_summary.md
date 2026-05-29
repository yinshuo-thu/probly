# 单重要赛事先行结论

## Match

- Fixture: `18746260`
- Match: Red Bull Bragantino vs Carabobo FC
- League: Copa Sudamericana
- Kickoff: 2026-05-28 00:30:00 UTC
- Polymarket event slug: `sud-bra-car-2026-05-27`
- Polymarket market analyzed: Red Bull Bragantino win, Yes token

## LSports vs Polymarket

本场已经完成 LSports 进球事件和 Polymarket 官方 CLOB `/prices-history` 的时间对齐。结论是: **LSports 三次进球均早于 Polymarket 可观测价格显著变动**。

| LSports goal time (UTC) | Scoring side | Base price | Polymarket reaction time | Lag |
|---|---:|---:|---|---:|
| 2026-05-28 00:46:54.967 | away | 0.705 | 2026-05-28 00:47:04 | 9.0s |
| 2026-05-28 02:07:21.622 | home | 0.405 | 2026-05-28 02:08:06 | 44.4s |
| 2026-05-28 02:24:12.788 | home | 0.965 | 2026-05-28 02:25:04 | 51.2s |

中位价格反应滞后约 **44 秒**。第一球是 away goal, Bragantino-win Yes 价格从约 0.705 下跳; 后两球是 home goal, 价格上跳。

## BBO Boundary

上述结果来自官方 CLOB `/prices-history`, 不是 tick-level BBO。对已关闭市场, 官方 `/book` 当前 orderbook 返回空, 因此无法赛后从官方 CLOB 重建历史 bid/ask。历史 BBO 需要:

- PMXT Archive 的 historical prices/orderbook feed, 或
- 赛中运行 WebSocket/orderbook recorder 自己采样 best bid/ask。

## Visuals

- `outputs/figures/single_match_polymarket_price_reaction.png`
- `outputs/figures/single_match_timeline.png`
- `outputs/figures/single_match_event_intensity.png`
- `outputs/figures/single_match_goal_hazard.png`

## Takeaway

单场证据支持 LSports 对价格反应具有 faster-pricing 价值: 进球事件先到, Polymarket 价格随后调整。但交易级结论还需要扩大到多场、并用历史 BBO 或实时 BBO 采样验证 stale-window 是否可成交。
