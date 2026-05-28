# Not Copied From Current Implementation

为避免新建模研究被当前实现影响，以下内容没有复制到本研究包：

- `src/`: 当前数据构建、训练、优化和 dashboard 后端实现。
- `viz/`: 当前 Flask/Plotly dashboard 实现。
- `outputs/models/*.pkl`: 当前已训练模型。
- `outputs/*predictions.parquet`: 当前预测结果。
- `outputs/*features.parquet`: 当前特征表。
- `outputs/real_dataset*.parquet`, `outputs/training_dataset.parquet`, `outputs/aligned_dataset.parquet`: 当前实现生成的建模表。
- dashboard 截图和当前部署产物。

可以参考 `reference/README.md` 理解旧项目背景，但新研究建议只把它作为问题描述，不作为实现蓝图。

如果后续确实需要和旧结果对照，应单独从 `/Volumes/T7/probly/outputs` 读取，并在实验记录中标记为 current-implementation baseline。
