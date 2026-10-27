# A1–A5 拆解

每份作业的结构都一样：**`tests/` 是规格书 → `tests/adapters.py` 是你唯一要碰的胶水层 → 你的实现放在自己的包里 → 测试绿了才算对**。
下面每份给出：要写什么、分值在哪、推荐顺序、自检命令、writeup 该有什么图。

> 分值来自 2026 春季 handout，用于判断"哪里值得多花时间"，个别题目分值未逐条核对，以 PDF 为准。

---

## A1 · Basics（最重要，占全程 40% 时间）

**目标**：手写 BPE tokenizer → Transformer → AdamW → 训练循环 → 跑实验。
**交付**：`code.zip` + `writeup.pdf`（含实验图）。
**算力**：CPU 能完成全部代码与测试；正式实验需 B200（配额约 17 B200·h）。

### 实际进度（2026-10-27）

| 部分 | 状态与证据 |
| --- | --- |
| BPE / Tokenizer / 数据采样 | 核心功能已有分项通过；任意切块边界、Linux 内存限制和完整语料实验未完成 |
| 模型组件到完整 LM | 对应前向测试通过，支持 RoPE 与较短输入 |
| 交叉熵、AdamW、调度、裁剪、checkpoint | 对应单元测试通过；checkpoint 曾因加载适配器未接入失败，修复后通过 |
| 单批次训练 | CPU 100 步，loss 9.2533 → 0.0124，固定批次准确率 100% |
| 训练与生成闭环 | 已有随机批次、固定验证、调度、分段存档恢复和温度/top-p 生成；5 MB 样本训练 1000 步，用户运行验证 loss 3.6202；CLI、完整 RNG 恢复与正式训练目标待补 |
| 分析与正式实验 | 样本规模 RoPE 三种子对照和七候选学习率扫描完成；0.003 当前最好。batch、其余三种结构对照、数值发散边界、资源核算、全语料和 writeup 待完成 |

详细变更与测试范围见 [10 月 26 日](../notes/2026-10-26-A1-TRANSFORMER-AND-LOSS.md)、[10 月 27 日早期训练组件](../notes/2026-10-27-A1-TRAINING-AND-CHECKPOINT.md)及[晚间训练与实验](../notes/2026-10-27-A1-TRAINING-GENERATION-AND-EXPERIMENTS.md)。[源码归档](../implementations/a1-basics/README.md)保存累计实现；上游测试、数据和 checkpoint 未上传。

当前下一步：batch size 对比先明确固定处理 token 预算或固定更新步数；后做移除 RMSNorm、post-norm、SwiGLU→SiLU。正式生成题至少 256 token（或至 EOS）、完整数据与推导题仍需补齐。本地讲义 CPU/MPS 建议约处理 4000 万 token，本次每组仅 25.6 万；样本数值不与正式目标直接比较。

### 要写的东西（按依赖顺序）

| # | 模块 | 对应测试 | 分值 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | Unicode / UTF-8 理解题 | — | 1 + 3 | 纯文字题，10 分钟，先做掉 |
| 2 | `train_bpe` | `test_train_bpe.py` | **15** | 全程第一个硬骨头；性能是评分点 |
| 3 | BPE 在 TinyStories/OWT 上的实验 | — | 2 + 2 | 报压缩率、耗时、内存 |
| 4 | `Tokenizer.encode/decode` | `test_tokenizer.py` | **15** | 特殊 token 必须整体切分 |
| 5 | 分词实验 | — | 4 | 两种语料的 token 数对比 |
| 6 | Linear / Embedding / RMSNorm / SwiGLU / SiLU | `test_model.py` | 1+1+1+1 | 都要自己算，不许调 `nn.Linear` |
| 7 | RoPE | `test_model.py` | 2 | 交错 vs 半半分，见 pitfalls |
| 8 | softmax / SDPA / MHA | `test_model.py`、`test_nn_utils.py` | 1 + 5 + 5 | SDPA 支持布尔 mask 和任意前导维度，因果 mask 在多头模块生成 |
| 9 | TransformerBlock / TransformerLM | `test_model.py` | 3 + 3 | pre-norm + RoPE；当前 Embedding 与输出层权重独立 |
| 10 | 资源核算题（FLOPs/显存） | — | 5 | 纯推导，但对后面理解 systems 极有帮助 |
| 11 | cross-entropy | `test_nn_utils.py` | 1 | 用 log-sum-exp 稳定写法 |
| 12 | AdamW | `test_optimizer.py` | 2 | 自己实现，别 wrap `torch.optim.AdamW` |
| 13 | AdamW 显存核算 | — | 2 | 同上，推导题 |
| 14 | 梯度裁剪 / cosine 调度 / checkpoint | `test_nn_utils.py`、`test_optimizer.py`、`test_serialization.py` | 1+1+1 | 简单但容易错边界 |
| 15 | 数据采样 `get_batch` | `test_data.py` | 2 | 随机 offset，注意别越界 |
| 16 | 训练脚本（拼起来） | — | 4 | 支持配置、断点续训、独立验证与周期存档；日志可先输出到终端 |
| 17 | 解码（top-p / temperature） | — | 3 | 采样要可复现 |
| 18 | 实验日志 | — | 3 | 每次 run 记超参 + 曲线 |
| 19 | 学习率扫描 | — | 3 | **2 B200·h**，找出会发散的上界 |
| 20 | batch size 扫描 | — | 1 | 1 B200·h |
| 21 | 生成文本样例 | — | 1 | 贴 256 token |
| 22 | 4 个消融（去 RMSNorm / post-norm / NoPE / SiLU） | — | 1×4 | 各 0.5 B200·h，**最划算的分数** |
| 23 | OWT 主实验 | — | 2 | 2 B200·h，同一模型换数据集 |
| 24 | leaderboard | — | 6 | 10 B200·h，可选，投入产出比最低 |

### 自检

```powershell
cd E:\CS336\assignments\a1-basics
uv run pytest tests/test_train_bpe.py -q -p no:cacheprovider
uv run pytest tests/test_model.py tests/test_nn_utils.py -q -p no:cacheprovider
uv run pytest tests/test_optimizer.py tests/test_serialization.py tests/test_data.py -q -p no:cacheprovider
# Windows 上当前已为 resource 做条件导入，普通测试可运行；Linux 专属内存限额测试仍需 Linux 验证
```

### writeup 该有的东西

- 一张"lr vs 最终 val loss"的曲线（含至少一条发散曲线，这是得分点）
- 一张 batch size 对比曲线 + 一句"为什么大 batch 不一定更好"
- 4 个消融各一条曲线，每条一句话结论（"去掉 RMSNorm 后 loss 上升 X，说明……"）
- TinyStories 与 OWT 的 loss/生成样例对比
- 实验日志链接（wandb 或表格）

### 常见失败

- BPE 训练 3 小时还没跑完 → 每轮重扫全表，见 `04-PITFALLS.md` #1
- `test_model.py` 全红但形状对 → 多半是 RoPE 的排布方式或 RMSNorm 的 `eps` 位置
- 训练不收敛但测试全绿 → 学习率太大 / 没 warmup / loss 没做 log-sum-exp

---

## A2 · Systems（含金量最高，A1 之后第二重要）

**目标**：把 A1 的模型跑快、跑在多卡上。**交付**：`writeup.pdf` + `code.zip`。
**算力**：需要 NVIDIA GPU + Linux（Triton/nsys）；2 卡用于 DDP/FSDP 实验。

### 要写的东西

| 部分 | 内容 | 分值 | 难度 |
| --- | --- | --- | --- |
| 2.1 | benchmark 脚本（warmup + sync + 中位数） | 4 | 低 |
| 2.1 | Nsight Systems profile（找出最耗时的 kernel） | 5 | 低，但要会读图 |
| 2.2 | 混合精度 + 梯度累积的数值问题 | 1 + 2 | 中（bf16 累加误差） |
| 2.3 | 显存 profile（`torch.cuda.memory._record_memory_history`） | 4 | 中 |
| 3 | 激活重算（gradient checkpointing） | 4 | 中 |
| 4.1 | PyTorch SDPA 基准 + `torch.compile` | 2 + 2 | 低 |
| 4.2 | **FlashAttention-2 前向（Triton）** | **15** | **高，全程最硬** |
| 4.2 | FlashAttention-2 反向 | 5 | 高（可选加分） |
| 4.2 | kernel benchmark vs PyTorch | 5 | 中 |
| 5 | naive DDP + 基准 | 5 + 3 | 中 |
| 6 | 优化器状态分片 | **15** | 高 |
| 7 | FSDP | **15** | 高 |
| 8 | 通信量/显存计算题（all-reduce、TP、2D 并行） | 1+3+3+4+6 | 纯推导，性价比高 |
| 9 | leaderboard：最快训练步 | 10 | 可选 |

### 推荐顺序（重要）

1. **先做所有计算题和 profiling**（约 25 分，半天到一天）。它们不依赖你的 kernel，还能让你后面调优时知道该看什么。
2. 激活重算 → DDP（先 naive，再分片）→ FSDP。这条线是"照论文写代码"，思路清晰。
3. **FlashAttention 放最后集中攻**。前向先写 PyTorch 版（纯 `for` 循环 + online softmax），跑通数值再搬 Triton。
4. leaderboard 只在前面全绿后再碰。

### 自检

```bash
uv run pytest tests/test_attention.py -q     # Triton kernel
uv run pytest tests/test_ddp.py -q           # 需要 2 卡
uv run pytest tests/test_sharded_optimizer.py -q
uv run pytest tests/test_fsdp.py -q
```

### writeup 该有的东西

- nsys 截图：标出耗时 top-3 的 kernel
- 显存曲线：标明峰值出现在 forward 还是 backward 的哪一层
- FlashAttention 加速比 vs 序列长度（128 → 8192），并解释为什么短序列更慢
- DDP/FSDP 的吞吐与通信占比随卡数变化
- 所有计算题的代数推导（这部分白给分，别丢）

---

## A3 · Scaling（最省算力，但自学者要改版）

**目标**：用有限实验预算拟合缩放定律，选出一个 48 B200·h 预算下的最优模型配置。
**交付**：`writeup.pdf`（重分析与图，代码量很小）。
**算力**：在校生用课程 API（12 B200·h 额度，需 Stanford 学号 + 校园网）；**自学者必须改版**，见 `02-COMPUTE.md` 第 5 节。

### 要写的东西

| 部分 | 内容 | 分值 |
| --- | --- | --- |
| 1 | IsoFLOP 曲线分析（给定数据，拟合幂律） | 5 |
| 2 | 用 API 跑实验、拟合 scaling law | 50（主体） |
| 2 | 预测 10²³/10²⁴ FLOPs 下的最优 N 与 D | 含在上面 |
| 2 | 花 48 B200·h 预算选配置 + 报预测 loss + 解释 | 含在上面 |

### 关键方法（写 writeup 的骨架）

1. 固定预算 C，扫 N（从而 D = C/(6N)），得到每个 N 的最终 loss
2. 对每个 C 取最低点 → 得到 (C, N_opt, D_opt) 三元组
3. 拟合 `N_opt ∝ C^a`、`D_opt ∝ C^b`（log-log 最小二乘）
4. 拟合 `L(N,D) = E + A/N^α + B/D^β`（在 log 空间拟合，用 `scipy.optimize.curve_fit` + 合理初值）
5. 外推 → 报数 → 讨论：你的指数和 Chinchilla 的 0.5/0.5 差多少，为什么（数据重复？tokenizer？学习率没调够？）

### 自学者改版

用 A1 的代码跑 10–14 个小模型（6 个 N × 2 个 D 网格），全部在单卡 4090 上，成本 < $10。writeup 里明确写"数据与 tokenizer 与官方 API 不同，绝对数值不可比，方法与结论可比"。

---

## A4 · Data（最"脏"，但最像真实工作）

**目标**：把 Common Crawl 的原始 WARC 变成能训模型的语料。**交付**：`writeup.pdf` + `code.zip`。
**算力**：数据处理 CPU 密集（磁盘 10–20 GB）；最终评测 8×B200 × 2 h（用 `modal run`）。

### 要写的东西

| 部分 | 内容 | 分值 | 实现要点 |
| --- | --- | --- | --- |
| 1 | 观察 CC 数据 | 4 | WARC/WET 各是什么，抽几条看 |
| 2 | HTML → 文本 | 3 | `resiliparse` 或 `trafilatura`，别自己写正则 |
| 3 | 语言识别 | 6 | fastText lid.176，阈值取舍要说明 |
| 4 | PII 脱敏（邮箱/电话/IP） | 3 | 正则 + 计数返回 |
| 5 | 有害内容分类 | 6 | fastText 分类器，阈值与误杀 trade-off |
| 6 | Gopher 质量过滤 | 3 | 规则过滤，注意中文/代码的误伤 |
| 7 | 质量分类器 | **15** | 训练一个分类器（正负样本从哪来是关键） |
| 8 | 精确行去重 | 3 | 哈希 + 计数 |
| 9 | **MinHash + LSH 文档去重** | 8 | 分桶、band 数、Jaccard 阈值 |
| 10 | 串成流水线 | 6 | 顺序很重要：先去重还是先过滤 |
| 11 | 检查过滤后数据 | 4 | 抽 20 条人工看，贴出来 |
| 12 | 分词并序列化 | 2 | `np.uint16.tofile`，注意词表 ≤ 65535 |
| 13 | 训练 + 评测 | 8 | 用官方 `scripts/train.py`，不许改训练 |

### 自检

```bash
uv run pytest tests/test_extract.py tests/test_langid.py tests/test_pii.py -q
uv run pytest tests/test_toxicity.py tests/test_quality.py tests/test_deduplication.py -q
```

### writeup 该有的东西

- 每一步过滤掉了多少比例（瀑布图），这是最有说服力的一张图
- MinHash 的参数（num_bands × rows）与检出的近重复对数
- 最终 token 数、训练 loss 曲线、C4 100 domains 上的评测
- 至少 3 个"过滤错了"的例子（过度过滤比过滤不足更难发现）

---

## A5 · Alignment & Reasoning RL（最新、最贴近当下）

**目标**：用 SFT + GRPO 让 OLMo-2-0425-1B 在 GSM8K 上变强。**交付**：`writeup.pdf` + `code.zip`。
**算力**：vLLM 占一张卡、训练占一张卡；1B 模型需 2×24 GB 级别（Kaggle 2×T4 可勉强跑通小 batch）。

### 要写的东西

| 部分 | 内容 | 分值 |
| --- | --- | --- |
| 1 | prompting baseline（zero/few-shot、CoT） | 5 |
| 2 | prompt/output 分词（mask prompt 的 loss） | 1 |
| 2 | 取 log-probs 与 entropy | 1 |
| 2 | rollout 奖励（GSM8K 答案抽取 + 比对） | 1 |
| 2 | group 归一化奖励（GRPO 的 advantage） | 1 |
| 2 | on-policy 策略梯度损失 | 1 |
| 2 | 完整 GRPO 训练步（含梯度累积） | 5 |
| 3 | 学习率调参 | 3（4 B200·h） |
| 3 | prompt 消融 | 3（4 B200·h） |
| 3 | 长度归一化的思考题 | 1 |
| 3 | RFT（拒绝采样微调）对比 | 2 |
| 4 | GRPO 变体（Dr.GRPO 等） | 2.5 |
| 4 | off-policy GRPO | 2.5 |
| 5 | 自己设计一个估计器 | **10**（最能出彩） |
| 附 | 安全对齐：SFT / DPO / red-teaming | 另一份 handout |

### 自检

```bash
uv run pytest tests/test_data.py tests/test_metrics.py -q
uv run pytest tests/test_grpo.py -q      # 需要 GPU + vLLM 的部分会 skip/失败
uv run pytest tests/test_dpo.py -q
```

### writeup 该有的东西

- GSM8K 准确率：baseline → +SFT → +GRPO，三个数并列
- 训练曲线：reward、response 长度、entropy（这三条一起看才能判断有没有 reward hacking）
- 长度归一化的取舍：不加约束时长度会不会爆炸
- 至少一个失败案例的分析（模型学会了什么"假技巧"）

---

## 五份作业的共性经验

1. **先读测试，再读 handout**。测试里写死了张量形状、边界条件、容差，比讲义精确。
2. **adapter 只是胶水**。真正的东西放你自己的包里（`cs336_basics/`、`cs336_systems/`），A2 会 import A1 的实现，一开始就分好目录。
3. **每通过一个测试就 `git commit`**。作业仓库都是 git repo，出问题能回滚。
4. **writeup 用 LaTeX 或 Markdown→PDF**。图必须能看清轴标签；评审对"图漂亮 + 一句话结论"极其宽容。
5. **A1 的 checkpoint 与数据加载会被 A2–A5 复用**，写的时候就当公共库写（类型标注、docstring、单元测试）。
