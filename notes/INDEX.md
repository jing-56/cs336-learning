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
| 2026-10-26 | [完整 Transformer 与交叉熵](2026-10-26-A1-TRANSFORMER-AND-LOSS.md) | 正式多头、RoPE、Block、LM、稳定交叉熵和梯度演示 |
| 2026-10-27 | [训练组件与首次拟合](2026-10-27-A1-TRAINING-AND-CHECKPOINT.md) | AdamW、调度、裁剪、checkpoint、CPU 单批次 100 步实验 |
| 2026-10-27（续） | [训练、生成与对照实验](2026-10-27-A1-TRAINING-GENERATION-AND-EXPERIMENTS.md) | 随机批次与验证、5 MB 小模型、top-p、RoPE 三种子、学习率扫描 |

另见：[BPE 5 MB 样本实验摘要](A1-BPE-SAMPLE-EXPERIMENT.md)。

截至 2026-10-27：

- 分项通过范围保持原记录；已有随机批次训练、固定验证、调度、存档恢复及温度/top-p 生成链路。
- 5 MB 样本完成 1000 步小模型训练；RoPE 三种子配对方向一致，学习率七候选中 0.003 在当前设置下最好。
- [38 个活动源码模块与哈希](../implementations/a1-basics/README.md)、[实验结果与来源清单](experiments/2026-10-27/README.md)已保存；数据、模型存档和上游测试不上传。
- 下一步：batch size、其余三个结构实验、资源核算、完整语料实验和 writeup。尚未定位学习率数值发散边界，也没有完成正式低资源训练目标。
- 任意切块编码边界与 Linux 内存约束仍未完成验证；上述进度不据此修改课程讲次勾选。
