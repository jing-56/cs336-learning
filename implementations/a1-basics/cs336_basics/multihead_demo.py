import torch

# 假设这是经过线性变换后得到的 Q
q = torch.tensor([
    [
        [1.,  2.,  3.,  4.],
        [5.,  6.,  7.,  8.],
        [9., 10., 11., 12.],
    ]
])

batch_size, seq_len, d_model = q.shape

num_heads = 2
head_dim = d_model // num_heads

# 把每个位置的 4 维拆成：2 个头，每个头 2 维
q_split = q.reshape(
    batch_size, seq_len, num_heads, head_dim
)

# 把“头”放到“位置”前面，方便每个头独立计算注意力
q_heads = q_split.transpose(1, 2)

print("原始形状：", q.shape)
print("拆分后形状：", q_split.shape)
print("调整顺序后形状：", q_heads.shape)

print("第一个头：\n", q_heads[0, 0])
print("第二个头：\n", q_heads[0, 1])
from cs336_basics.attention import scaled_dot_product_attention

# 先演示形状变化，暂时让 K、V 与 Q 使用相同的数值
# 正式模块中，Q、K、V 会通过三个独立的 Linear 生成
k_heads = q_heads.clone()
v_heads = q_heads.clone()

# 因果遮罩：每个位置只能关注自己和前面的位置
mask = torch.tril(
    torch.ones(seq_len, seq_len, dtype=torch.bool)
)

# 各个头分别计算注意力
head_outputs = scaled_dot_product_attention(
    q_heads, k_heads, v_heads, mask
)

print("各头输出形状：", head_outputs.shape)

# 把“位置”换回“头”的前面
output_by_position = head_outputs.transpose(1, 2)

# 拼接同一个位置上各个头的输出
combined = output_by_position.contiguous().reshape(
    batch_size, seq_len, d_model
)

print("拼接后形状：", combined.shape)
print("拼接后结果：\n", combined)