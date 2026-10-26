import torch

from cs336_basics.nn_utils import softmax


# 一个 Query，向量长度为 2
q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# 三个位置分别提供 Key
k = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

# 三个位置分别提供 Value
v = torch.tensor([
    [10.0, 0.0],
    [0.0, 10.0],
    [5.0, 5.0]
])

# 1. 点积计算匹配分数，并进行缩放
scores = (q @ k.T) / (q.shape[-1] ** 0.5)

# 下三角为 True：允许看自己和之前的位置
mask = torch.tril(
    torch.ones(3, 3, dtype=torch.bool)
)

# 不允许看的位置，分数设为负无穷
masked_scores = scores.masked_fill(~mask, float("-inf"))

weights = softmax(masked_scores, dim=-1)
output = weights @ v

print("允许关注的位置：\n", mask)
print("遮罩后的分数：\n", masked_scores)
print("注意力权重：\n", weights)
print("输出：\n", output)
print("输出形状：", output.shape)