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
| 2026-10-29 | [批次大小](2026-10-29-A1-BATCH-SIZE.md)、[移除 RMSNorm](2026-10-29-A1-RMSNORM-ABLATION.md)、[pre/post-norm](2026-10-29-A1-PRE-POST-NORM.md)、[SwiGLU/SiLU](2026-10-29-A1-SWIGLU-SILU.md) | 固定预算对照与四类结构实验的后续记录 |
| 2026-10-29（续） | [参数量](2026-10-29-A1-PARAMETER-COUNT.md)、[前向 FLOPs](2026-10-29-A1-FORWARD-FLOPS.md)、[训练内存](2026-10-29-A1-TRAINING-MEMORY.md)、[正式配置核算](2026-10-29-A1-TRANSFORMER-ACCOUNTING.md) | 从当前小模型到 XL 配置的资源计算 |
| 2026-10-30 | [AdamW 激活与 batch](2026-10-30-A1-ADAMW-MEMORY-ACCOUNTING.md)、[更新计算量与时间](2026-10-30-A1-ADAMW-COMPUTE-AND-TIME.md) | 带明确假设的内存/运算与 MFU 估算 |
| 2026-10-30（续） | [A1 完成度核对](2026-10-30-A1-COMPLETION-AUDIT.md) | 46 通过、2 跳过；逐项缺口和执行顺序 |
| 2026-10-30（再续） | [统一训练/生成工具与 PyCharm 使用](2026-10-30-A1-TRAINING-TOOLS.md) | 可配置入口、累计耗时、可复现续训、256-token 样例 |
| 2026-11-01 | [Unicode 与 SGD 学习率](2026-11-01-A1-UNICODE-AND-SGD.md) | 逐题回答、四组十步实验、闭式核验与适用范围 |
| 2026-11-01（续） | [完整 TinyStories 数据与 BPE 分块读取](2026-11-01-A1-FULL-TINYSTORIES-PREPARATION.md) | 完整数据校验、验证集 10k 调试、读取等价检查、3 项 BPE 测试 |
| 2026-11-01（再续） | [完整训练集 10k BPE](2026-11-01-A1-TINYSTORIES-FULL-BPE.md) | 单次正常训练、实测时间/内存、词表/merges 与验证文档检查 |
| 2026-11-01（编码） | [完整训练/验证 token 数组](2026-11-01-A1-TINYSTORIES-FULL-ENCODING.md) | 固定正式词表、片段缓存、uint16 数组、全量还原和写盘验证 |
| 2026-11-01（训练） | [新数据训练起步与 PyCharm 续训](2026-11-01-A1-FULL-DATA-TRAINING-START.md) | CPU 速度基准、40M 计划、实际完成 100 步，验证 loss 9.2332→6.0035 |
| 2026-11-01（完成训练） | [20,000 步训练与故事生成](2026-11-01-A1-FULL-TRAINING-AND-GENERATION.md) | 4,096 万目标 token、验证 loss=2.0210、200 新 token 后 EOS、56 个模块归档 |
| 2026-11-01（GPU） | [GPU 环境、设备迁移与续训](2026-11-01-A1-GPU-TRAINING.md) | RTX 3050 实测、CPU100→GPU200、约 4.3 倍提速、全套 46 passed/2 skipped |
| 2026-11-04 | [完整验证、采样和 OpenWebText BPE](2026-11-04-A1-EVALUATION-AND-OWT-BPE.md) | 全量验证 loss 1.9715、高学习率失稳、跨域对照、堆与紧凑索引 |
| 2026-11-07 | [OpenWebText 全量分词、编码和训练](2026-11-07-A1-OPENWEBTEXT-FULL.md) | 32K 词表、完整编码、20,000 步训练和生成、全量验证 loss 4.9895、最终词表跨域对照 |

另见：[BPE 5 MB 样本实验摘要](A1-BPE-SAMPLE-EXPERIMENT.md)。

[A1 实验与核算总报告](A1-TRAINING-REPORT.md)汇总全量分词、训练与生成、学习率、批量大小、四组结构消融和资源核算。

[A1 自学版 PDF 和代码包](../deliverables/a1-basics/README.md)包含交付文件及解压复验结果。

截至 2026-11-08，TinyStories 与 OpenWebText 的两次正式主训练已完成归档，全量验证 loss 分别为 1.3517 和 3.9645。[正式曲线与日志](experiments/2026-11-07/autodl-formal/)

- 2026-11-01 更换 GPU 版 PyTorch 后完整收集 48 项，46 项通过、两项 Linux 内存测试跳过，17.06 秒；见[本次 GPU 切换记录](2026-11-01-A1-GPU-TRAINING.md)。
- 四类结构消融和 batch 对比已有小配置曲线，资源核算已有带假设的说明。已有教学实验不能替代全语料和正式训练预算。
- Unicode/SGD 逐题回答、十步实测和完整 TinyStories 训练集 profile 已补齐。更宽的 batch 扫描、原尺寸消融和精确稳定边界仍待补。[TinyStories 全量 profile](experiments/2026-11-07/tinystories-bpe-profile/README.md)
- 训练/生成工具已完成短运行验证，完整 TinyStories 数据已校验，分块读取已验证；完整训练集 10k BPE 已完成；完整 TinyStories 编码已完成，新模型已从 CPU100 步迁移至 GPU，[完成 20,000 步、4,096 万目标 token 及故事生成](2026-11-01-A1-FULL-TRAINING-AND-GENERATION.md)，最终固定批次验证 loss=2.0210，下一步补剩余实验，再按[交付清单](2026-10-30-A1-COMPLETION-AUDIT.md)补齐。未完成 Linux 内存验证，不额外把任意切块等价升级为讲义未明确要求的必交项。
- [源码快照](../implementations/a1-basics/README.md)现保存 56 个活动模块、最小包标识、接入函数和三个训练配置，含新增 18 个模块及 bpe.py 分块读取改动；数据、模型存档和上游测试不上传。课程讲次状态不依据代码进度自动勾选。
