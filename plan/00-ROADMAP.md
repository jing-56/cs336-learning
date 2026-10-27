# 13 周路线图

目标：**完整跟完 19 讲 + 做完 A1–A5**，每周 12–16 小时。
这套排期按"作业驱动"设计：先看支撑本周作业的讲，其余讲当作背景音，别为了"看完"而看。

> 每周的"过关"只写**可验证的硬指标**（测试绿、曲线出来、数字达标），不写"理解了 XX"。理解是副产品，测试才是证据。

---

## 当前实际落点（2026-10-27）

已跑通 A1 数据→模型训练/验证→存档恢复→文本生成闭环，完成 5 MB 样本上的 CPU 小模型实验。RoPE 三种子配对及学习率七候选扫描有配置、曲线和记录，开始进入第 5 周的实验分析内容。

当前样本的模型、语料和训练 token 预算远小于正式要求，生成也仍有连贯性问题；不据此把第 4—6 周全部标为通过。下一步做 batch size 对比、其余结构消融、资源核算，补齐分词器/完整语料实验和 writeup。学习率数值发散边界、RNG 恢复及部分平台约束仍未完成。

详见[实际作业进度](03-ASSIGNMENTS.md)与[当天晚间记录](../notes/2026-10-27-A1-TRAINING-GENERATION-AND-EXPERIMENTS.md)。课程讲次进度与实现进度分别记录。

## 总表

| 周 | 讲次 | 本周主要产出 | 过关标准（硬） | 工时 |
| --- | --- | --- | --- | --- |
| 1 | L1 tokenization · L2 PyTorch/资源核算 | A1 环境 + `train_bpe` | `tests/test_train_bpe.py` 全绿 | 12 |
| 2 | L3 架构 · L4 注意力变体/MoE | A1 `Tokenizer` 类 + 分词实验 | `test_tokenizer.py` 全绿，TinyStories 压到 4 倍以内 | 14 |
| 3 | L5 GPU/TPU · L6 kernel/Triton | A1 模型层全部 | `test_model.py` + `test_nn_utils.py` 全绿 | 16 |
| 4 | L7 并行（一） | A1 训练循环 + 小规模训练跑通 | 本地跑出 loss 下降曲线，能生成通顺小故事 | 16 |
| 5 | L8 并行（二） | A1 实验：lr / batch / 4 个消融 | 6 组学习曲线 + 一句话结论 | 16 |
| 6 | L9 缩放定律（一） | A1 OWT 主实验 + writeup | val loss 达标或曲线可解释 + writeup 成稿 | 16 |
| 7 | L10 推理 | A2 profiling 三件套 | benchmark 脚本 + nsys 截图 + 显存曲线 | 14 |
| 8 | L11 缩放定律（二） | A2 FlashAttention-2 前向 | `test_flash_forward_pass_triton` 绿 | 16 |
| 9 | L12 评测 · L13 数据源 | A2 反向 + benchmark + DDP | 自己的 kernel 快过 PyTorch 基线，2 卡 DDP 跑通 | 16 |
| 10 | L14 数据清洗/去重/混合 | A2 分片 + FSDP + writeup | A2 全部测试绿 + 通信量计算题做完 | 16 |
| 11 | L15 后训练 SFT/RLHF | A3 缩放定律（自学者改版） | IsoFLOP 曲线 + 最优 N/D 预测 | 14 |
| 12 | L16 RLVR | A4 数据流水线 + 训练评测 | 过滤/去重脚本跑通，tokens 数报出 | 16 |
| 13 | L17 多模态 · L18/19 客座 | A5 SFT/GRPO | GRPO 训练脚本跑通 + GSM8K 提升曲线 | 16 |

合计 ≈ **194 小时**。如果你每周只有 10 小时，把这份表拉成 18–20 周，别压缩内容——A1 和 A2 压缩了等于没做。

---

## 每周细节

### 第 1 周 · L1 + L2：tokenization 与"算力账本"

- 看：L1（BPE 全流程、Unicode/UTF-8、词表大小的取舍）、L2（einops、FLOPs/显存/算术强度的估算）
- 读：A1 handout §2（BPE），只读到你动手需要的那几页
- 写：`cs336_basics/` 里的 `train_bpe`；把 `tests/adapters.py` 的对应函数接上
- 坑：纯 Python 的 BPE 在 TinyStories 上如果 >30 分钟，说明你在每轮重扫全表（见 `04-PITFALLS.md` 第 1 条）
- 过关：`uv run pytest tests/test_train_bpe.py -x -q`
- 加分：把 vocab_size 32000 跑到 ≤ 30 分钟、峰值内存 ≤ 30 GB（handout 给的上限）

### 第 2 周 · L3 + L4：Tokenizer 与架构直觉

- 看：L3（Transformer 各超参怎么选、d_ff 为什么是 8/3 倍）、L4（MHA 变体、GQA/MQA、MoE、线性注意力）
- 写：`Tokenizer.encode/decode`（含特殊 token 不被切碎）、`tokenizer_experiments`
- 实验：TinyStories vs OWT 的压缩率（bytes/token）；同一段文本的 token 数对比
- 过关：`test_tokenizer.py` 全绿；能回答"为什么 OWT 上词表大一点更划算"
- 提醒：这一周先别碰模型，tokenizer 的正确性会污染后面所有实验

### 第 3 周 · L5 + L6：从 GPU 到 kernel

- 看：L5（SM、warp、HBM 带宽、roofline、TPU 的异同）、L6（Triton 编程模型、融合为什么快）
- 写：A1 模型层 —— Linear / Embedding / RMSNorm / SwiGLU / RoPE / softmax / SDPA / MHA / TransformerBlock / TransformerLM
- 过关：`test_model.py`、`test_nn_utils.py` 全绿
- 坑：RoPE 的"交错"与"半半分"两种写法只差一个 reshape，测试会抓（见 `04-PITFALLS.md`）
- 提示：`test_model.py` 用 numpy snapshot 对比，误差阈值很紧；用 fp32 调试，别用 bf16

### 第 4 周 · L7：把训练跑起来

- 看：L7（数据并行、张量并行、流水线并行的取舍；通信量估算）
- 写：cross-entropy、AdamW、梯度裁剪、cosine 学习率、checkpoint 存取、数据采样、训练脚本
- 跑：本地 RTX 3050（4 GB）上 TinyStories，`d_model=256/384, ctx=256, batch=32`，几千步
- 过关：loss 从 ~9 降到 ~2 以下，`generate` 出通顺的儿童故事；`test_optimizer.py`、`test_serialization.py`、`test_data.py` 全绿
- 若 OOM：先减 batch，再减 ctx，最后减 d_model —— 顺序别反，ctx 太短会让 attention 形状错误暴露不出来

### 第 5 周 · L8 + A1 实验周

- 看：L8（FSDP/ZeRO 的分片粒度、all-gather/reduce-scatter 的通信量、2D 并行）
- 做实验（handout 给了算力配额，按配额规划，别乱烧）：
  - 学习率扫描（2 B200·h）：至少 5 个 lr，找出边界与发散点
  - batch size 扫描（1 B200·h）：从 1 到显存上限
  - 4 个消融各 0.5 B200·h：去 RMSNorm、post-norm、NoPE、SwiGLU→SiLU
- 过关：6 张学习曲线 + 每张一句话结论（"为什么"比"是什么"值钱）
- 省钱：这些实验在 3050 上把模型缩到 1/8 也能看出趋势，正式数字再上云复现

### 第 6 周 · L9 + A1 收尾

- 看：L9（Chinchilla 的 IsoFLOP 方法、怎么拟合 L(N,D)、数据受限怎么办）
- 做：OWT 主实验（2 B200·h）+ leaderboard（10 B200·h，可选）
- 写：`writeup.pdf` —— 每个实验给图 + 结论 + 你踩的坑
- 过关：A1 全部测试绿 + writeup 成稿 + 实验日志（`experiment_log` 那题要的是可复现的记录，不是流水账）
- 里程碑：**A1 完成**。这是全程最重的一份，做完你已经超过大多数"调过 LLM API"的人

### 第 7 周 · L10 + A2 起步

- 看：L10（KV cache、paged attention、投机解码、batching 策略）
- 写：benchmark harness（warmup + `torch.cuda.synchronize()` + 多次取中位数）、nsys profile、显存 profile
- 过关：能说出"forward 时间是花在哪个 kernel 上"，`benchmarking_script` + `nsys_profile` + `memory_profiling` 三题的图都有
- 前提：A2 需要 NVIDIA GPU + Linux 环境跑 Triton/nsys。Windows 上装不了官方 Triton，走 WSL2 或云（见 `02-COMPUTE.md`）

### 第 8 周 · L11 + FlashAttention（最硬的一周）

- 看：L11（缩放定律续：数据质量、重复数据、推理最优）
- 写：先写 PyTorch 版 FlashAttention-2（online softmax + 分块），再搬进 Triton
- 过关：`test_flash_forward_pass_triton` 绿（15 分那题）
- 策略：先用 `TRITON_INTERPRET=1` 在 CPU 上验证逻辑，再上 GPU 调 tile size；别一上来就调 block
- 心理准备：这题是全程最容易卡三天的地方。卡住时对照算法 1 逐行检查：`m_i`、`l_i`、`acc` 三个量的更新顺序

### 第 9 周 · L12 + L13 + A2 中段

- 看：L12（评测的陷阱：污染、prompt 敏感、pass@k）、L13（数据来源、许可、配比）
- 写：FlashAttention 反向（5 分）+ benchmark 对比（5 分）+ naive DDP
- 过关：你的 kernel 在长序列上快过 PyTorch SDPA；2 卡 DDP 的吞吐线性度能画出来
- 提醒：benchmark 一定要报"从空 cache 开始 10 分钟内完成"，否则 leaderboard 判违规

### 第 10 周 · L14 + A2 收尾

- 看：L14（Gopher 质量过滤、MinHash 去重、数据混合、合成数据）
- 写：优化器状态分片（15 分）+ FSDP（15 分）+ 各种通信量计算题
- 过关：A2 全部测试绿 + writeup（含 FSDP 显存/通信的代数推导）
- 里程碑：**A2 完成**。到这里你已经能答"训练一个 7B 模型需要多少卡、瓶颈在哪"

### 第 11 周 · L15 + A3 缩放定律

- 看：L15（SFT、RLHF、DPO 的目标函数与数据构造）
- 做：A3 的 IsoFLOP 拟合 + 预算内选最优 N/D
- ⚠️ **自学者注意**：A3 的训练 API 需要 **Stanford 学号 + 校园网**，非在校生用不了。改版方案见 `02-COMPUTE.md` 的"A3 替代方案"——用自己跑的 8–12 个小模型拟合，结论一样成立
- 过关：IsoFLOP 曲线 + 对 10²³/10²⁴ FLOPs 的最优 N、D 预测 + 一段误差分析

### 第 12 周 · L16 + A4 数据

- 看：L16（RLVR、可验证奖励、GRPO 与 PPO 的差别）
- 做：CC → 正文抽取 → 语言识别 → PII 脱敏 → 质量过滤 → 精确/MinHash 去重 → 分词
- 过关：流水线跑通、报出 token 数、抽 20 条样本人工看质量；`train_model`（8 分）按 handout 的 Modal 脚本提交
- 现实：8×B200 跑 2 小时 ≈ 16 B200·h ≈ 100 美元。预算紧就砍到 1 卡 + 1/8 token，写清"缩水版结论的适用边界"
- 提醒：磁盘至少留 10–20 GB；CC 的 WARC 文件是 gzip 套 tar，别整个读进内存

### 第 13 周 · L17 + A5 对齐

- 看：L17（多模态对齐）、L18/19（客座：Daniel Selsam、Dan Fu）
- 做：prompting baseline → 打 token/logprob 地基 → GRPO 训练 → 变体（RFT、长度归一化、off-policy）
- 过关：GRPO 让 OLMo-2-1B 在 GSM8K 上比 baseline 有可见提升，曲线能解释
- 现实：vLLM 占据一张卡、训练占一张卡，1B 模型要 2×A100/L40S 级别。省钱做法：只在最后几次实验上用云，前面用 `prompting_baselines` 的结果做对照
- 里程碑：**A5 完成**，全课通关

---

## 里程碑与缓冲

| 里程碑 | 时间点 | 判断标准 |
| --- | --- | --- |
| M1 能写 tokenizer | 第 2 周末 | 测试绿 + 压缩率数字 |
| M2 端到端训出模型 | 第 4 周末 | 本地 3050 跑出会讲故事的模型 |
| M3 A1 交付 | 第 6 周末 | writeup + 全部测试 |
| M4 kernel 能写 | 第 8 周末 | Triton 前向绿 |
| M5 A2 交付 | 第 10 周末 | 分布式跑通 |
| M6 A3 交付 | 第 11 周末 | 拟合曲线 |
| M7 A4 交付 | 第 12 周末 | 数据流水线 |
| M8 A5 交付 | 第 13 周末 | GRPO 曲线 |

**落后了怎么砍（按优先级从高到低砍）**：

1. 砍客座讲座（L18/L19）和 L17 —— 不影响作业
2. A3 用"改版"（自己跑小模型）代替 API 版，省 2 天
3. A5 只做 on-policy GRPO，跳过变体与 DPO 附加题（值 10+ 分，但最耗时）
4. A4 的 `train_model` 用缩水配置，写清边界
5. **不要砍**：A1 的实验部分、A2 的 FlashAttention。这两块是这门课的全部含金量

**提前跑完了（有余力）**：

- A1 leaderboard 排名（6 分，纯优化，好玩）
- A2 leaderboard：最快训练步（10 分，需要调 Triton block size + `torch.compile`）
- A5 的"自己设计一个策略梯度估计器"（10 分，最能出彩的一题）

---

## 每周节奏建议（怎么分配这 12–16 小时）

| 环节 | 时间 | 说明 |
| --- | --- | --- |
| 看讲 | 2–3 h | 1.5 倍速看一遍，第二遍只看卡住的部分 |
| 读 handout 对应章节 | 2 h | 只看本周要写的那几节，不要通读 47 页 |
| 写代码 | 6–8 h | 分成 2–3 次，每次带着一个"这次要让哪个测试变绿"的目标 |
| 跑实验 + 记笔记 | 2–3 h | 每次跑之前先写假设，跑完记录结果，否则一周后全忘 |
| 复盘 | 0.5 h | 填 `notes/` 和进度表 |

一条硬规则：**每周至少让一个测试从红变绿**。连续两周没有绿灯，说明你在"看"而不是在"写"，停下来改方式。
