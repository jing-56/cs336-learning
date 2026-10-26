import torch

from cs336_basics.linear import Linear


torch.manual_seed(42)

# 两条样本，每条八个 token，每个 token 四维
x = torch.randn(2, 8, 4)

# 演示：中间维度扩展为八维
w1 = Linear(4, 8)
w3 = Linear(4, 8)
w2 = Linear(8, 4)

# 调节分支
gate_input = w1(x)
gate = gate_input * torch.sigmoid(gate_input)

# 特征分支
value = w3(x)

# 逐维调节，再映射回原维度
gated = gate * value
output = w2(gated)

print("输入形状：", x.shape)
print("调节分支：", gate.shape)
print("特征分支：", value.shape)
print("相乘之后：", gated.shape)
print("最终输出：", output.shape)