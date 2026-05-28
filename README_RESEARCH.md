# Probly Independent Modeling Research Package

本目录是从 `/Volumes/T7/probly` 提取出的独立研究包，用于重新开展建模研究，避免被当前实现代码和已有训练产物影响。

## 目录结构

- `data/`: 可用数据。包含 LSports Hyper 足球事件、Polymarket 市场匹配、Polymarket 价格路径、fixture 索引和 synthetic baseline 数据。
- `docs/`: 当前任务、需求、模型规格和研究记录。
- `environment/requirements.txt`: 建模环境依赖。
- `reference/`: 原仓库 README 与 requirements 的只读参考副本。
- `NOT_COPIED_FROM_CURRENT_IMPLEMENTATION.md`: 被刻意排除的当前实现内容。

## 当前任务

研究 LSports 实时体育事件流能否在 Polymarket 市场价格完全调整之前提供可用的提前定价信号。

核心问题：

1. 将 LSports football fixture/event stream 与 Polymarket 市场及价格路径对齐。
2. 在事件时间 `t` 仅使用当时及过去信息，预测未来价格波动窗口。
3. 尽量实现至少 30 秒提前量，而不是解释同步或已发生的价格反应。
4. 建立可解释的状态转移 / Markov / xT 风格模型，理解哪些状态、事件、路径带来价格跳跃。
5. 后续研究应优先验证连续价格路径、方向性、滑点和成交可执行性，而不只做二分类高波动检测。

## 主要参考文档

- `docs/TASK_SUMMARY.md`: 原始任务摘要。
- `docs/hf_docs/REQUIREMENTS_sports_pricing.md`: 体育事件实时定价研究需求。
- `docs/hf_docs/MODEL_SPEC_markov_pricing.md`: Markov/xT 状态转移定价模型规格。
- `docs/hf_docs/pmxt-archive-indexing.md`: PMXT archive 索引和定位说明。
- `docs/worklogs/2026-05-25_markov_xt_rewrite.md`: 历史实现记录，仅作为背景，不建议作为新研究起点。

## 建议研究原则

- 从 `data/` 重新定义数据集、标签和切分。
- 不直接复用当前仓库 `src/` 代码、`outputs/models` 模型或预测 parquet。
- 明确区分“提前预测”与“同步解释”：目标窗口必须从 `t + gap` 开始，例如 `[t+30s, t+150s]`。
- 按 fixture 做时间/赛事级切分，避免同一赛事事件泄漏到 train/test 两侧。
- 单独评估市场类型、流动性、联赛和价格区间，不把所有市场混成一个结论。

## 快速开始

```bash
cd /Users/yinshuo/Documents/code-iCloud/probly
python -m venv .venv
source .venv/bin/activate
pip install -r environment/requirements.txt
```

然后从 `DATA_CATALOG.md` 选择输入数据，重新设计 notebook 或训练脚本。
