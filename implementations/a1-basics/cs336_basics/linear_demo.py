import torch

# 一条样本，包含两个输入特征
x = torch.tensor([[2.0, 3.0]])

# 每一行负责计算一个输出特征
weight = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

y = x @ weight.T

print("输入：", x)
print("输出：", y)
print("输入形状：", x.shape)
print("权重形状：", weight.shape)
print("输出形状：", y.shape)