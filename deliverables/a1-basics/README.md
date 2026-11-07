# CS336 A1 个人自学交付

| 文件 | 内容 |
| --- | --- |
| [writeup.pdf](writeup.pdf) | 本地小模型与 AutoDL 正式尺寸训练、全量验证、曲线和生成全文，以及分词、调参与消融和资源核算 |
| [code.zip](code.zip) | 个人实现、测试接入函数、配置、AutoDL 入口、绘图与分析脚本及所需的少量指标文件，内含 `README.md` 和 `FILES.sha256` |

上一版代码包解压复验 **46 项通过、2 项跳过**，耗时 17.11 秒。本版核心源码和接入函数与受测文件逐字节一致。两项跳过的用例依赖 Linux 的 `resource` 内存限制。代码包只收录个人实现与复现材料，按包内说明另行准备课程环境、上游测试、数据、词表和存档。

TinyStories 与 OpenWebText 都已完成 20,000 步的正式尺寸训练，各处理 327,680,000 个目标位置。全量验证 loss 分别为 **1.3517** 和 **3.9645**。更宽的 batch 扫描、精确学习率稳定边界、原尺寸消融和排行榜仍在扩展范围。

[可编辑报告](../../notes/A1-TRAINING-REPORT.md) · [AutoDL 正式结果归档](../../notes/experiments/2026-11-07/autodl-formal/) · [云端复现入口](../../implementations/a1-basics/experiments/autodl/)
