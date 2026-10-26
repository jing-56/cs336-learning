import torch

# 原始二维向量
x = torch.tensor([3.0, 4.0])

# PyTorch 使用弧度：π / 2 就是 90°
angle = torch.tensor(torch.pi / 2)

cos_angle = torch.cos(angle)
sin_angle = torch.sin(angle)

# 按公式计算旋转后的两个分量
new_x0 = x[0] * cos_angle - x[1] * sin_angle
new_x1 = x[0] * sin_angle + x[1] * cos_angle

# 将两个分量重新组成向量
rotated = torch.stack([new_x0, new_x1])

print("原始向量：", x)
print("旋转后向量：", rotated)
print("原始长度：", torch.linalg.vector_norm(x).item())
print("旋转后长度：", torch.linalg.vector_norm(rotated).item())
head_dim = 8
theta = 10000.0

# 每一对分量的起始下标：[0, 2, 4, 6]
pair_starts = torch.arange(
    0, head_dim, 2, dtype=torch.float32
)

# 计算四对分量各自的旋转频率
frequencies = theta ** (-pair_starts / head_dim)

# 演示三个 token 的位置
positions = torch.arange(3, dtype=torch.float32)

# 每个位置 × 每一对的频率
angles = positions[:, None] * frequencies[None, :]

print("各对频率：", frequencies)
print("各位置的旋转角度：\n", angles)
print("角度表形状：", angles.shape)
# 三个位置，原始向量完全相同
features = torch.tensor([
    [1., 2., 3., 4., 5., 6., 7., 8.],
    [1., 2., 3., 4., 5., 6., 7., 8.],
    [1., 2., 3., 4., 5., 6., 7., 8.],
])

# 每一对里的第一个分量：[x0, x2, x4, x6]
first = features[..., 0::2]

# 每一对里的第二个分量：[x1, x3, x5, x7]
second = features[..., 1::2]

# angles 是上一步生成的 [3, 4] 角度表
cos_angles = torch.cos(angles)
sin_angles = torch.sin(angles)

# 同时对所有位置、所有分量对应用旋转公式
rotated_first = first * cos_angles - second * sin_angles
rotated_second = first * sin_angles + second * cos_angles

# 把每对的两个分量放回一起：[3, 4, 2]
paired = torch.stack(
    [rotated_first, rotated_second],
    dim=-1,
)

# 把四对展开，恢复为八维向量：[3, 8]
rotated_features = paired.flatten(start_dim=-2)

print("旋转后形状：", rotated_features.shape)
print("旋转后的向量：\n", rotated_features)

print(
    "旋转前各位置的长度：",
    torch.linalg.vector_norm(features, dim=-1),
)
print(
    "旋转后各位置的长度：",
    torch.linalg.vector_norm(rotated_features, dim=-1),
)