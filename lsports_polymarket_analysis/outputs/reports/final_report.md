# LSports × Polymarket 数据价值分析最终报告

## Executive Summary

本项目已完成 LSports 事件流到 Polymarket 价格/BBO 研究的可复用 pipeline。当前最重要的实证结果分两层:

1. **官方 CLOB `/prices-history` 价格层面**: 单重要赛事 `18746260` 中, LSports 三次进球均早于 Polymarket 可观测价格显著变动, 领先约 **9 / 44 / 51 秒**, 中位约 **44 秒**。
2. **历史 BBO 层面**: 不能用官方 CLOB 赛后还原。官方 `/book` 是当前 orderbook, 已关闭市场无当前 book; 历史 BBO 需要 PMXT Archive 或 DomeAPI 这类归档服务。代码已实现 `PMXT_API_KEY` / `DOME_API_KEY` 接入, 当前环境未配置 key, 因此未产出 BBO 结论。

批量 LSports 侧已扩展到 **5 日 × 每日 40 场 = 200 场**, 共识别 **596 个进球**。事件更新时间分辨率为秒级: 每场实时事件间隔中位数的中位约 **3.06 秒**, p90 的中位约 **9.05 秒**。

## 单重要赛事结论

- Fixture: `18746260`
- Match: Red Bull Bragantino vs Carabobo FC
- Polymarket slug: `sud-bra-car-2026-05-27`
- Market: Red Bull Bragantino win, Yes token

| LSports goal time (UTC) | Scoring side | Base price | Polymarket reaction time | Lag |
|---|---:|---:|---|---:|
| 2026-05-28 00:46:54.967 | away | 0.705 | 2026-05-28 00:47:04 | 9.0s |
| 2026-05-28 02:07:21.622 | home | 0.405 | 2026-05-28 02:08:06 | 44.4s |
| 2026-05-28 02:24:12.788 | home | 0.965 | 2026-05-28 02:25:04 | 51.2s |

Interpretation: 在官方历史价格序列上, LSports 明显先于市场价格调整。第一球是客队进球, Bragantino-win Yes 价格下跳; 后两球是主队进球, 价格上跳。

## BBO 状态

已实现:

- `scripts/single_match_bbo_analysis.py`
- `PolymarketPriceLoader.load_historical_bbo()`
- PMXT Archive path: `PMXT_API_KEY`
- DomeAPI path: `DOME_API_KEY`

当前运行结果:

- `PMXT_API_KEY present: False`
- `DOME_API_KEY present: False`
- 未检索到历史 BBO snapshots

因此, 对“LSports 进球 vs Polymarket BBO 变动谁更快”的严格答案是: **当前还不能下 BBO 结论**。对“LSports 进球 vs Polymarket 官方历史价格变动谁更快”的答案是: **单场证据显示 LSports 更快**。

## 批量 LSports 结果

- 覆盖日期: 2026-05-24 至 2026-05-28
- 每日样本: 40 场
- 总比赛: 200 场
- 总进球: 596
- 平均每场进球: 2.98
- 有进球比赛占比: 92.5%
- 实时事件间隔中位数: 3.06 秒
- 实时事件间隔 p90 中位: 9.05 秒
- 归档延迟 p50 中位: 5982 秒, 这是批量归档延迟, 不是实时推送延迟

跨场 goal-hazard 模型 held-out AUC 约 0.56, 说明事件强度有弱信号, 可作为风险过滤器雏形, 但不足以单独交易。

## 可复现命令

```bash
cd lsports_polymarket_analysis

# LSports 数据
python scripts/download_hf_data.py --docs --max 40
python scripts/inspect_dataset.py
python scripts/single_match_analysis.py
python scripts/batch_event_analysis.py --max 40

# 历史 BBO, 需至少一个历史 orderbook provider key
export PMXT_API_KEY="<your_pmxt_key>"   # 或 DOME_API_KEY
python scripts/single_match_bbo_analysis.py
```

## 研究边界

本项目不伪造 Polymarket BBO。官方 Polymarket CLOB read endpoints 可读当前 orderbook、prices、midpoints、spreads 和 price history; 但历史 BBO/orderbook 对已关闭市场需要独立 archive。当前仓库已经把这条路径工程化, 缺的是 provider credential 或可下载的对应小时 archive。
