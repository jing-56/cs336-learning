import torch

x = torch.tensor([-3.0, -1.0, 0.0, 1.0, 3.0])

# 根据输入计算倍率
gate = torch.sigmoid(x)

# 逐元素相乘
y = x * gate

print("输入：", x)
print("sigmoid 倍率：", gate)
print("SiLU 输出：", y)