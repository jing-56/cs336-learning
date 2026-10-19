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
