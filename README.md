# CS336 从零实现语言模型

斯坦福 [CS336: Language Modeling from Scratch](https://cs336.stanford.edu/) 的个人学习记录。用 PyTorch 从 BPE 分词开始，逐步实现 Transformer、训练循环和文本生成，并用实验检验每个环节。

**当前进度：A1 Basics 自学主线完成。**

[实验报告 PDF](deliverables/a1-basics/writeup.pdf) · [代码包 ZIP](deliverables/a1-basics/code.zip) · [浏览源码](implementations/a1-basics/README.md) · [学习笔记](notes/INDEX.md)

## A1 做了什么

- **分词**：训练字节级 BPE，完成 TinyStories 和 OpenWebText 的词表训练、编码和还原检查。
- **模型**：实现带 RoPE、RMSNorm 和 SwiGLU 的自回归 Transformer。
- **训练与生成**：实现 AdamW、学习率调度、存档续训和文本采样，并完成两套模型的训练与评估。

两套模型各训练 **20,000 步**，每套共预测 **3.28 亿个训练目标 token**。下表指标来自各自的完整验证集。

| 数据集 | 验证 loss | 困惑度 | 生成样例 |
| --- | ---: | ---: | --- |
| TinyStories | **1.3517** | 3.86 | [儿童故事](notes/experiments/2026-11-07/autodl-formal/tiny/generation.json) |
| OpenWebText | **3.9645** | 52.69 | [新闻风格续写](notes/experiments/2026-11-07/autodl-formal/owt/generation.json) |

[查看训练曲线](notes/experiments/2026-11-07/autodl-formal/learning-curves.svg) · [查看完整日志和评估](notes/experiments/2026-11-07/autodl-formal/) · [阅读分词、调参与消融实验](notes/A1-TRAINING-REPORT.md)

## 从哪里开始

| 想做什么 | 入口 |
| --- | --- |
| 看学习路线和后续作业 | [学习路线](plan/00-ROADMAP.md) · [A1–A5 作业导航](plan/03-ASSIGNMENTS.md) |
| 阅读 A1 源码或恢复到官方作业 | [模块导航与恢复步骤](implementations/a1-basics/README.md) · [环境准备](plan/01-SETUP.md) |
| 复现正式训练 | [运行说明与数据清单](implementations/a1-basics/experiments/autodl/) |
| 查看交付与测试范围 | [A1 交付说明](deliverables/a1-basics/README.md) |

仓库收录个人实现、学习笔记和实验结果。复现时需要另行准备官方作业环境、训练数据和所需的模型存档。
