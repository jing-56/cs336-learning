import torch

x = torch.tensor([
    [3.0, 4.0],
    [30.0, 40.0]
])

eps = 1e-5

# 每一行分别计算均方根
rms = torch.sqrt(
    x.square().mean(dim=-1, keepdim=True) + eps
)

y = x / rms

print("输入：\n", x)
print("每条向量的 RMS：\n", rms)
print("归一化后：\n", y)