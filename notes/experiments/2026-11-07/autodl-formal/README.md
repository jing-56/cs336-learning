# AutoDL 正式主训练结果

每个子目录包含 202 个原始训练日志点、正式配置、摘要、200 步测速、全量验证和生成结果。源文件来自 AutoDL 下载包 `a1-formal-results.tar.gz`，SHA-256 为 `9c7dff85bdcb8c8958822f3ffa91b1e69791adc1dba88799681c8fa4df7aa8bc`。归档配置改用相对数据路径，摘要中的存档路径只保留文件名，CSV 保留原始数值。

| 实验 | 训练日志 | 全量验证 | 生成 |
| --- | --- | --- | --- |
| TinyStories | [metrics.csv](tiny/metrics.csv) | [full-validation.json](tiny/full-validation.json) | [generation.json](tiny/generation.json) |
| OpenWebText | [metrics.csv](owt/metrics.csv) | [full-validation.json](owt/full-validation.json) | [generation.json](owt/generation.json) |

归档核对覆盖步数、token 预算、固定批次 loss 与末行日志的一致性、验证数组指纹、全量预测位置数、交叉熵与困惑度、生成长度和停止原因。存档指纹保留在全量评估 JSON 中。

[正式学习曲线](learning-curves.svg) · [运行入口与数据清单](../../../../implementations/a1-basics/experiments/autodl/) · [总报告](../../../A1-TRAINING-REPORT.md)
