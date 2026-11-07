# CS336 A1 个人自学交付

| 文件 | 内容 |
| --- | --- |
| [writeup.pdf](writeup.pdf) | Unicode、分词、全量 BPE 性能分析、语言模型训练与验证、生成全文、调参与消融、Transformer 和 AdamW 核算 |
| [code.zip](code.zip) | 个人实现、测试接入函数、配置、复现脚本及所需的少量指标文件，内含 `README.md` 和 `FILES.sha256` |

解压后用包内源码及接入函数运行现有 A1 测试集，Windows 实测 **46 项通过、2 项跳过**，耗时 17.11 秒。两项跳过的用例依赖 Linux 的 `resource` 内存限制。代码包只收录个人实现与复现材料，按包内说明另行准备课程环境、上游测试、数据、词表和存档。

完整数据上的训练与验证采用本地小模型。课程原尺寸实验、更宽的 batch 扫描、精确学习率稳定边界和排行榜仍待覆盖。

[可编辑报告与原始数据链接](../../notes/A1-TRAINING-REPORT.md) · [个人源码目录](../../implementations/a1-basics/README.md)
