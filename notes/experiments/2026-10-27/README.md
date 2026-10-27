# 2026-10-27 A1 样本实验数据

本目录保存实际运行产生的配置、指标、曲线 CSV 与来源清单。模型存档、原始故事和 token 数组不上传。

| 实验 | 配置与结果 | 曲线与汇总 |
| --- | --- | --- |
| RoPE，种子 42 | [配置](rope/seed42/summary.json) | [有 RoPE](rope/seed42/with_rope_loss.csv)、[无 RoPE](rope/seed42/without_rope_loss.csv) |
| RoPE，种子 43 | [配置](rope/seed43/summary.json) | [有 RoPE](rope/seed43/with_rope_loss.csv)、[无 RoPE](rope/seed43/without_rope_loss.csv) |
| RoPE，种子 44 | [配置](rope/seed44/summary.json) | [有 RoPE](rope/seed44/with_rope_loss.csv)、[无 RoPE](rope/seed44/without_rope_loss.csv) |
| RoPE，三种子汇总 | [JSON](rope/multi_seed_summary.json) | [CSV](rope/multi_seed_results.csv) |
| 学习率扫描 | [完整配置与结果](lr/lr_sweep_summary.json) | [汇总 CSV](lr/learning_rate_results.csv)，各组逐步数据在同目录 |

共享模型：V=10000，d=64，2 层，4 头，d_ff=128，T=64；CPU 4 线程、B=4、1000 步、50 步预热、梯度裁剪 1.0、AdamW weight_decay=0.01。训练集 1,143,921 token，验证集 127,340 token。

所有实验使用 validation_seed=12345 的 10 个固定验证批次，覆盖 2560 个目标位置，片段可能重叠。RoPE 每个种子内部严格配对，三种子只描述随机初始化和抽样顺序的波动。学习率扫描使用种子 42，min_lr=0.1×max_lr，0.003 的完整基线复用自 RoPE 实验；其他六组均为新训练。

CSV 保留原始字节，JSON 仅将机器绝对路径改成相对引用或 checkpoint_filename，并标记存档未上传。哈希清单见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。汇总 CSV 中的 source_summary 是相对于该 CSV 所在目录的引用。

现有曲线与结论属于 5 MB 样本实验。七个学习率都完成训练，没有观察到非有限数值；1.0 出现大幅波动，不能写成已经定位数值发散边界。

详细讲解与生成样例见[当天晚间记录](../../2026-10-27-A1-TRAINING-GENERATION-AND-EXPERIMENTS.md)。图：[RoPE](../../assets/2026-10-27/rope-validation-curves.png)、[学习率](../../assets/2026-10-27/learning-rate-curves.png)；同目录 SVG 可放大。
