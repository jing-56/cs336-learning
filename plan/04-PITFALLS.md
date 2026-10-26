# 踩坑手册

按"症状 → 原因 → 怎么查"组织。这些都是这门课的经典卡点，遇到时先在这里对一遍，能省掉半天。

---

## A1 · Tokenizer

### 1. BPE 训练跑了 3 小时还没完

**症状**：`train_bpe` 在 TinyStories 上远超 handout 给的 30 分钟。
**原因**：每轮合并后**重扫整个语料**重算所有 pair 频次，复杂度变成 O(merge 次数 × 语料长度)。
**怎么查**：数一下每个 merge 迭代里你遍历了多少个 token。如果 ≈ 语料总长度，就是这个病。
**正解方向**：维护一张相邻 pair 的频次表，每轮只从表里取最大值；合并后**只更新受影响的相邻位置**（被合并的 pair 周围两三个位置），其余不动。再用堆/有序结构取 max，避免每轮 O(V) 扫全表。
**加速前提**：先用 GPT-2 的正则把文本切成预 token（`tiktoken` 的 pattern），再在预 token 内部做 BPE——这也是 handout 要求的 "pretokenization"。

### 2. 特殊 token 被切碎了

**症状**：`<|endoftext|>` 被编码成 5 个 token。
**原因**：编码时没有先按特殊 token 切分文本，或者切分后仍对特殊 token 走 BPE。
**怎么查**：`tok.encode("<|endoftext|>")` 应该返回长度 1。
**正解方向**：编码前先按特殊 token 切段 → 特殊段直接映射为对应 id，普通段才做 BPE。注意训练时特殊 token 也占词表位置，且出现在语料里时不应被合并。

### 3. 分词器的压缩率"太好了"

**症状**：TinyStories 上 bytes/token 超过 5（正常约 4.0–4.5）。
**原因**：词表里塞了语料中不存在的合并，或训练集与测试集混了。
**怎么查**：拿 1 MB 未参与训练的文本算 `len(text.encode('utf-8')) / len(tokens)`。

---

## A1 · 模型

### 4. 所有模型测试都差一点点（1e-5 级误差）

**症状**：`test_model.py` 报 `AssertionError`，数值很接近但不过。
**原因**：几乎总是 **RoPE 的排布方式**。存在两种等价但不同的实现：
- 交错式（interleaved）：把最后一维按 `(x0,x1), (x2,x3)` 两两配对旋转
- 半半分（half-split）：把维度切成前后两半，`x[:d/2]` 与 `x[d/2:]` 配对

**怎么查**：用 `d_k=4`、位置 `[0,1]`、`theta=10000` 手算一个向量，和测试的 snapshot 对。两种写法只在第一组旋转角的正负/顺序上不同。
**正解方向**：先确认测试用的哪种（看 `tests/fixtures` 或 snapshot 形状），再统一。RoPE 还要注意：`token_positions` 可能是任意位置（不一定是 0..T-1），要按它去索引 cos/sin 表。

### 5. Attention 输出有 nan

**症状**：loss 变 nan，或者全 mask 的行输出 nan。
**原因**：mask 用 `-1e9` 在 bf16 下不够（bf16 最大约 3e38，但 softmax 的 exp 后会溢出/下溢），或用了 `float('-inf')` 之后整行都是 `-inf` → `softmax` 得到 nan。
**怎么查**：打印 mask 之后、softmax 之前的 score 的 min/max/是否有 nan。
**正解方向**：mask 用 `torch.finfo(dtype).min`，并且**先减行最大值**再 exp；如果存在全被 mask 的行（padding 场景），要显式把这些行置零。

### 6. SDPA 在前导维度上崩了

**症状**：`test_scaled_dot_product_attention` 里 4 维输入通过、3 维失败。
**原因**：实现里写死了 `B, H, T, D = q.shape`。
**怎么查**：测试用的是 `... queries d_k` 的任意前导维度。
**正解方向**：只在最后两维上做 `matmul`（`q @ k.transpose(-2,-1)`），batch 维靠广播；mask 同理。

### 7. RMSNorm 对不上

**原因**：三个高频错法——(a) `eps` 加在开方**外** vs **内**；(b) 忘了乘 weight；(c) 用了 mean 而不是 mean of squares。
**怎么查**：`x = [3, 4]`，`eps=0`，正确结果应是 `[3,4]/sqrt(12.5)*w`。

### 8. SwiGLU 形状错

**原因**：`d_ff` 取成 `4*d_model`（那是普通 FFN 的比例），或者 `w2` 的转置方向错。
**正解方向**：`d_ff = round(8/3 * d_model / 64) * 64`，`w1,w3: (d_ff, d_model)`，`w2: (d_model, d_ff)`，`SwiGLU(x) = w2(silu(w1 x) * w3 x)`。

---

## A1 · 优化与训练

### 9. AdamW 测试过不了

**症状**：`test_adamw` 在几十步后出现偏差。
**原因**：三个细节——(a) 忘了 bias correction `1-β^t`；(b) weight decay 用的是 L2 正则（加进梯度）而不是 decoupled；(c) `eps` 加在 `sqrt(v)` 之外还是之内。
**怎么查**：跑 1 步、2 步、3 步，和手算对照；`t` 从 0 还是 1 开始也要对。

### 10. 学习率调度在边界上跳变

**原因**：三段（warmup / cosine / 之后）的边界条件写成了 `<=` 与 `<` 混用。
**正解方向**：`it < warmup` 线性升到 max；`warmup <= it <= cosine_cycle_iters` 余弦降到 min；之后保持 min。

### 11. 数据加载报 "expected scalar type Long"

**原因**：`np.uint16` 数组直接 `torch.from_numpy` 得到 `uint16` tensor，embedding 只接受 `int64`。
**正解方向**：`.astype(np.int64)` 再转 tensor；随机起点范围是 `[0, len - context_length - 1]`，且 `y = x` 右移一位。

### 12. 训练 loss 不降

**排查顺序**（别跳步）：
1. **能不能过拟合一个 batch**：拿固定一个 batch 训 200 步，loss 应该趋近 0。不行 → 模型/损失/反传有 bug。
2. **学习率**：太大→前几步就飞；太小→loss 几乎不动。先试 `1e-3`（小模型）。
3. **warmup**：没有 warmup + 大 lr 是新手最常见的发散原因。
4. **mask**：把 causal mask 去掉训一遍，如果 loss 反而更低，说明 mask 方向错了。
5. **数据**：打印一批 `x` 和 `y`，确认 `y` 真的是 `x` 右移一位（不是同一条）。

### 13. bf16 / 混合精度下 loss 抖

**原因**：loss 在 bf16 里累加，或者梯度累积时没做平均。
**正解方向**：loss 用 fp32 算、梯度累积时除以累积步数；bf16 不需要 GradScaler，fp16 才需要。

---

## A2 · Kernel 与分布式

### 14. Triton kernel 结果对不上

**查的顺序**：
1. `TRITON_INTERPRET=1` 在 CPU 上跑一遍（能打断点、能 print），先确认算法对。
2. 把 tile 设成"能盖住整行"的大小（比如 `BLOCK=128` 且序列 ≤128），消除边界因素。
3. 检查 `tl.load(..., mask=..., other=0.0)` 里的 mask 是不是少了 `other`——不加会读到脏数据。
4. 检查 `tl.store` 的 mask 是否覆盖了所有应该写的元素。
5. 最后才调 `num_warps` / `BLOCK_SIZE`。

### 15. FlashAttention 前向"大概对但误差大"

**原因**：online softmax 的三个量更新顺序错。每处理一个新块，必须严格按：
`m_new = max(m_old, rowmax(scores))` → `alpha = exp(m_old - m_new)` → `l = l*alpha + rowsum(exp(scores - m_new))` → `acc = acc*alpha + exp(scores - m_new) @ v`。
**怎么查**：先令块大小 = 整个序列（只有一块），此时结果应该和普通 attention **完全一致**；再缩小块，看误差是否只来自 fp 累加。

### 16. 基准测出来比 PyTorch 慢很多

**原因**：没 warmup、没 `torch.cuda.synchronize()`、测的是第一次编译（Triton JIT）。
**正解方向**：warmup 10 次以上，用 `triton.testing.do_bench`，报中位数；`torch.compile` 的首次编译时间要排除。

### 17. 2 卡 DDP 不加速甚至更慢

**原因**：模型太小 → 通信占比过高；或者每个 GPU 的 batch 太小。
**怎么查**：同时看"单步时间"和"通信时间"。用 nsys 或 `torch.distributed` 的 profiling 看 all-reduce 占比。
**正解方向**：先确认单卡吞吐随 batch 上升是饱和的，再上多卡；梯度累积 + `no_sync()` 能减少通信次数。

### 18. FSDP 报 "sharded param accessed outside all-gather"

**原因**：在 forward 之外持有了参数引用（比如在 `__init__` 里存了 `self.w = param` 的别名）。
**正解方向**：只通过 `module.parameters()` 访问；自定义 forward 用 `torch.nn.Module` 的标准写法，让 hooks 能插进去。

---

## A4 · 数据

### 19. WARC 解析直接把内存吃爆

**原因**：整个 `.warc.gz` 读进内存再解压。单个 WARC 可能几 GB。
**正解方向**：流式读 `gzip` → 逐 record 解析（`warcio` 或自己按 header 切）；过滤阶段也全用生成器，不要 `list(...)`。

### 20. MinHash 去重后剩不下东西

**原因**：Jaccard 阈值太低（比如 0.5）或 band 数太少，导致误杀；也可能把"模板化页面"当成了重复（该杀）连带杀掉了正常页面。
**怎么查**：抽 20 对判为重复的文档，人工看它们真的重复吗；再抽 20 对没被判重的近似文档。
**正解方向**：MinHash 的 band/row 配置决定"能检出多相似的文档"——先用理论公式算一遍检出的概率曲线，再定参数。

### 21. 分词后 token 数少得可疑

**原因**：用了 A1 训的 tokenizer 但词表与 `np.uint16` 上限（65535）冲突，或者过滤后数据几乎空了。
**怎么查**：`wc -c` 看原始字节，`len(ids)` 看 token 数，比值应在 4 左右。

---

## A5 · 对齐

### 22. GRPO 的 reward 一直不涨

**排查顺序**：
1. 先确认 **prompting baseline** 能拿到分（说明答案抽取、评分逻辑是对的）。
2. 打印一个 batch 的 reward 分布：全 0 或全 1 → 没有学习信号（group 内方差为 0）。
3. 检查 advantage 归一化：除以的是 `std + eps` 还是 `std`，有没有处理 std=0。
4. 检查 loss 是不是只对 response token 算（prompt 部分要 mask 掉）。

### 23. 训练几轮后 response 长度爆炸

**原因**：长度归一化把"长答案"的惩罚摊薄了，模型学会"写长一点赌对"。
**怎么查**：把 reward、长度、entropy 三条曲线画在一起。长度单调上升而 reward 平 → 就是这个。
**正解方向**：改用 token 级平均而不是序列级归一化（Dr.GRPO 的思路），或者加长度惩罚。

### 24. vLLM 和训练进程抢显存

**原因**：vLLM 默认按 `gpu_memory_utilization=0.9` 预分配。
**正解方向**：训练和推理分卡（官方建议 gpu 0 训练、gpu 1 推理）；单卡时把 utilization 降到 0.3–0.5，并在训练前调用 vLLM 的 sleep/offload。

---

## 通用

### 25. 测试通过但作业分低

**原因**：测试只覆盖正确性，不覆盖性能、实验质量和 writeup。
**正解方向**：A1 有 40% 的分在"实验 + 分析"，A2 有 25 分在纯计算题。**先做性价比高的题**（见 `03-ASSIGNMENTS.md` 的分值表）。

### 26. 一周后回来看不懂自己的实验

**原因**：没记超参。
**正解方向**：每次 run 前在 `notes/` 写一行假设（"lr 3e-4 应该比 1e-3 稳"），run 完贴图和结论。A1 的 `experiment_log` 那题就是逼你养成这个习惯。


### 27. Checkpoint 加载报 NotImplementedError（2026-10-27 实际排错）

**症状**：保存后，`test_checkpointing` 调用 `run_load_checkpoint`，报错停在 `tests/adapters.py` 的 `raise NotImplementedError`。

**原因**：本地加载适配器还没接上，尚未执行 `checkpoint.py` 中的读取逻辑。

**处理**：保留官方函数参数与说明，将函数体接到自己的 `load_checkpoint(src, model, optimizer)` 并返回已完成步数。保存文件后重新运行该测试，得到 `1 passed in 2.15s`；不需要修改测试断言。

### 28. 单批次 100% 准确率应怎样解读（2026-10-27）

固定 B=2、T=16 的批次训练 100 步，loss 从 9.2533 降到 0.0124，32 个位置最高分预测均正确。这验证训练链路能拟合该批次；新文本表现要另外划分验证集评估。准确率 100% 不意味着正确 token 的概率为 1。

日志中的 step=100 loss=0.0126 是该步更新前的值，最后重新前向得到 0.0124 是更新后的值，二者不矛盾。
