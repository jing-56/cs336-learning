from pathlib import Path
import torch
import numpy as np


folder = Path(__file__).resolve().parent

tokens = np.load(
    folder / "tinystories_sample_tokens.npy",
    mmap_mode="r"
)

# 用连续 9 个 token，构造长度为 8 的输入和目标
context_length = 8
chunk = tokens[:context_length + 1]

x = chunk[:-1]
y = chunk[1:]

print("原始片段：", chunk.tolist())
print("输入 x：", x.tolist())
print("目标 y：", y.tolist())
print("输入形状：", x.shape)
print("目标形状：", y.shape)
batch_size = 2

# 随机选择两段文本的起点
rng = np.random.default_rng(42)
starts = rng.integers(
    0,
    len(tokens) - context_length,
    size=batch_size
)

# 每个起点各自构造输入和目标
x_batch = np.stack([
    tokens[start:start + context_length]
    for start in starts
])

y_batch = np.stack([
    tokens[start + 1:start + context_length + 1]
    for start in starts
])

# 转成模型使用的 PyTorch 整数张量
x_batch = torch.tensor(x_batch, dtype=torch.long)
y_batch = torch.tensor(y_batch, dtype=torch.long)

print("\n随机起点：", starts)
print("批次输入：\n", x_batch)
print("批次目标：\n", y_batch)
print("批次形状：", x_batch.shape)
print("数据类型：", x_batch.dtype)
vocab_size = 10000
d_model = 4

# 为每个 token 准备一个四维向量
torch.manual_seed(42)
embedding_table = torch.randn(vocab_size, d_model)

# 用 token ID 索引对应的行
hidden = embedding_table[x_batch]

print("\nEmbedding 表形状：", embedding_table.shape)
print("输入 ID 形状：", x_batch.shape)
print("输出向量形状：", hidden.shape)

print("第一个输入 ID：", x_batch[0, 0].item())
print("它对应的向量：", hidden[0, 0])