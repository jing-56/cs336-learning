# 笔记索引

看一讲 → 复制 `_TEMPLATE.md` 成对应文件 → 填完打勾。
状态：`[ ]` 未看 · `[~]` 看了没记 · `[x]` 记完 · `[!]` 有没搞懂的问题（写进文件里）

| # | 主题 | 本地文件 | 对应作业 | 状态 |
| --- | --- | --- | --- | --- |
| 1 | Overview, tokenization | `lecture_01.py` | A1 | [ ] |
| 2 | PyTorch (einops), 资源核算 | `lecture_02.py` | A1 / A2 | [ ] |
| 3 | Architectures, hyperparameters | `lecture_03.pdf` | A1 | [ ] |
| 4 | Attention 变体, MoE | `lecture_04.pdf` | A1 | [ ] |
| 5 | GPUs, TPUs | `lecture_05.pdf` | A2 | [ ] |
| 6 | Kernels, Triton | `lecture_06.py` | A2 | [ ] |
| 7 | Parallelism（一） | `lecture_07.py` | A2 | [ ] |
| 8 | Parallelism（二） | `lecture_08.pdf` | A2 | [ ] |
| 9 | Scaling laws（一） | `lecture_09.pdf` | A3 | [ ] |
| 10 | Inference | `lecture_10.py` | A5 | [ ] |
| 11 | Scaling laws（二） | `lecture_11.pdf` | A3 | [ ] |
| 12 | Evaluation | `lecture_12.py` | A5 / writeup | [ ] |
| 13 | Data（来源、数据集） | `lecture_13.py` | A4 | [ ] |
| 14 | Data（过滤、去重、混合、合成） | `lecture_14.py` | A4 | [ ] |
| 15 | 中期/后训练（SFT/RLHF） | `lecture_15.pdf` | A5 | [ ] |
| 16 | 后训练 · RLVR | `lecture_16.pdf` | A5 | [ ] |
| 17 | 对齐 · 多模态 | `lecture_17.py` | A5 | [ ] |
| 18 | 客座：Daniel Selsam | — | — | [ ] |
| 19 | 客座：Dan Fu | — | — | [ ] |

> 课件本体在 `E:\CS336\lectures\cs336-lectures\`。`.py` 是 live-coding 脚本，可以当 notebook 逐块跑（文件里用 `# %%` 分块）。

## 精选补充阅读（按需，不要通读）

| 主题 | 材料 |
| --- | --- |
| Transformer 原始论文 | Vaswani et al. 2017 |
| BPE / 分词 | Sennrich 2016；GPT-2 论文 §2.2 |
| Scaling laws | Kaplan 2020；Hoffmann 2022（Chinchilla） |
| FlashAttention | Dao 2022 / Dao 2023（FA2） |
| 分布式训练 | ZeRO（Rajbhandari 2020）；Megatron-LM（Shoeybi 2019） |
| 数据 | Gopher（Rae 2021）§2 过滤规则；Deduplicating Training Data（Lee 2021） |
| 对齐 | InstructGPT（Ouyang 2022）；DPO（Rafailov 2023）；DeepSeek-R1（2025，RLVR） |

## A1 每日实现记录

以下为依据对话、文件与实验记录整理的工作说明，不代表已看完对应课程讲次，也不代表 A1 已全部完成。

| 日期 | 具体说明 | 本日重点 |
| --- | --- | --- |
| 2026-10-23 | [BPE 基础与单轮合并](2026-10-23-A1-BPE-FOUNDATIONS.md) | UTF-8、预分词、加权相邻对、非重叠合并、词表和规则 |
| 2026-10-24 | [BPE 训练与优化](2026-10-24-A1-BPE-TRAINING.md) | 循环、特殊标记、函数封装、倒排索引、局部更新、3 项训练测试 |
| 2026-10-25 | [分词器、实验、数据与模型组件](2026-10-25-A1-TOKENIZER-AND-MODEL.md) | 编解码、流式和文件接口、样本实验、数据采样、基础层、注意力与拆头演示 |

另见：[BPE 5 MB 样本实验摘要](A1-BPE-SAMPLE-EXPERIMENT.md)。

截至 2026-10-25：

- 已完成并有分项通过回执：BPE 训练，分词器编解码与文件逐行流式测试，batch 采样，Embedding、Linear、RMSNorm、SiLU/SwiGLU、softmax、缩放点积注意力。
- 多头注意力当前仅有拆头演示；下次从正式多头模块继续，再学习 RoPE 和组合完整模型。
- Linux 流式内存约束、任意切块的编码边界、完整语料实验和模型训练尚待验证或完成。
- 仓库忽略 `assignments/`；本次同步说明文档，不包含其中的本地源码、模型词表、数据数组或性能记录文件。
