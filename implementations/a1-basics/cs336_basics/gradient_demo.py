import torch

# 一个需要学习的参数，初始值为 2
w = torch.nn.Parameter(torch.tensor(2.0))

# 希望它接近 3
loss = (w - 3).square()

print("更新前的参数：", w.item())
print("更新前的损失：", loss.item())

# 自动计算损失对参数的梯度
loss.backward()

print("参数的梯度：", w.grad.item())

# 按梯度更新一次参数
learning_rate = 0.1

with torch.no_grad():
    w -= learning_rate * w.grad

print("更新后的参数：", w.item())
print("更新后的损失：", (w - 3).square().item())