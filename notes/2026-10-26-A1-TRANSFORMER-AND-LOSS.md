# 2026-10-26｜A1：多头注意力、RoPE、完整模型与交叉熵

## 范围与证据

这篇补齐上次 10 月 25 日记录之后的工作。日期依据对话顺序与本地文件创建、修改时间整理；当前源码是累计版本，不代表每个文件在当天结束时的独立 Git 快照。`nn_utils.py` 后来又加入了 27 日的梯度裁剪。

下表的单项测试时间来自当时的运行回执。10 月 27 日同步时另做的组合复查见[当天记录](2026-10-27-A1-TRAINING-AND-CHECKPOINT.md)。

源码现已单独归档至 [implementations/a1-basics](../implementations/a1-basics/README.md)，不上传整个上游作业仓库、测试和数据。

## 逐文件变更

以下文件名均相对于本地 `assignments/a1-basics/cs336_basics/`；归档中的同名文件位于 `implementations/a1-basics/cs336_basics/`。

| 文件 | 具体改了什么 | 为什么这样做 |
| --- | --- | --- |
| `multihead_demo.py` | 在拆头演示后加入各头注意力计算、换轴、拼接 | 看清 `[B,T,d] → [B,H,T,d/H] → [B,T,d]` 的全过程 |
| `multihead_attention.py` | 新增 `MultiHeadSelfAttention`；四个独立投影层、拆头、因果遮罩、注意力、拼接与输出投影；后来增加可选 `rope` 和 `token_positions` | 将已通过测试的缩放点积注意力组成正式多头模块 |
| `rope_demo.py` | 从二维向量旋转，逐步增加位置×频率角度表和整段序列的两两旋转 | 将旋转公式与张量切片、配对、广播联系起来 |
| `rope.py` | 新增 `RotaryPositionalEmbedding`；预计算 sin/cos buffer、按位置查表、相邻分量旋转、恢复输入 dtype | 为 Q/K 注入相对位置信息，固定表不参与训练 |
| `transformer_block.py` | 新增两个独立 RMSNorm、带 RoPE 的注意力和 SwiGLU；两次残差相加 | 实现 pre-norm Block，输入输出维度相同 |
| `transformer_lm.py` | 新增 Embedding、独立 Block 的 ModuleList、最终 RMSNorm、LM Head；检查最大上下文长度，支持较短输入 | 连通 token ID 到每个位置词表 logits 的完整前向计算 |
| `nn_utils.py` | 在已有 softmax 后加入稳定交叉熵 | 直接计算 `log(sum(exp(shifted))) - shifted_target`，平均所有预测位置的损失 |
| `gradient_demo.py` | 用单参数 `(w-3)^2` 演示 backward 与一次手动 SGD 更新 | 区分计算梯度与更新参数；观察 `w: 2 → 2.2`、loss `1 → 0.64` |
| 本地 `tests/adapters.py` | 接通无/有 RoPE 的 MHA、RoPE、Block、LM 和交叉熵接口 | 将测试提供的固定权重加载到自己的模块；只归档已实现的接入函数，不上传上游测试文件 |

## 关键理解

### 多头注意力

每个头覆盖整个可见序列，使用不同的投影特征独立形成注意力权重。头的维度与位置维度交换后，批量矩阵乘法会分别处理各个头；不需要 Python 循环逐头调用。

演示中令 Q、K、V 相同且数值较大，输出看起来与输入相同，是因为各位置几乎只关注自身，并不表示注意力一般都不改变输入。

### RoPE

每头维度为 `d_k`，相邻分量两两配对，第 i 对频率为 `theta ** (-2*i/d_k)`，位置 m 的角度为 `m * frequency[i]`。只旋转 Q、K；V 保持原样。`theta=10000` 是生成频率的基数，不是直接旋转的角度。

sin/cos 使用 `register_buffer(..., persistent=False)`：跟随模块移动设备，不是可训练参数，也不写进 state_dict；恢复模型时按配置重新生成。

### 残差与完整模型

Block 的两步是：

```text
y = x + Attention(RMSNorm_1(x))
z = y + SwiGLU(RMSNorm_2(y))
```

加回的是该子层的原始输入。注意力混合位置信息，SwiGLU 对各位置分别加工特征。完整 LM 在多个独立 Block 之后，再经过 RMSNorm 和无偏置 Linear，返回原始 logits。

### 交叉熵与梯度

输入 `[B,T,V]`、目标 `[B,T]`，最后得到一个平均损失。目标是下一个 token，已经在数据构造时错开一位，不再对 logits 或目标重复移动。

`loss.backward()` 沿计算图计算参数梯度，不会自行更新权重。学习了 AdamW 的移动平均与自适应缩放概念；完整实现和测试在次日完成。

## 当时逐项测试回执

| 测试 | 结果 |
| --- | --- |
| 无 RoPE 的多头注意力 | 1 passed，0.09s |
| RoPE | 1 passed，0.08s |
| 无/有 RoPE 的多头注意力联合检查 | 2 passed，0.12s |
| Transformer Block | 1 passed，0.12s |
| Transformer LM、较短输入 | 2 passed，0.19s |
| 交叉熵 | 1 passed，0.07s |

这些结果验证相应前向计算，不代表模型已经完成语言训练或整个 A1 已完成。

## 当天结束位置

- 完整模型前向与交叉熵已通过对应测试。
- 已理解梯度、SGD 和 AdamW 的主要思路。
- AdamW、调度、裁剪、checkpoint 与第一次真实训练的完成情况，继续看 [10 月 27 日记录](2026-10-27-A1-TRAINING-AND-CHECKPOINT.md)。
