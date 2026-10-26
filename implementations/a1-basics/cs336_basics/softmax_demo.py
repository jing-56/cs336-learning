import torch

scores = torch.tensor([1000.0, 999.0, 998.0])

# 先减最大值，避免指数溢出
shifted = scores - scores.max()

# 指数化，再归一化
exp_scores = torch.exp(shifted)
weights = exp_scores / exp_scores.sum()

print("原始分数：", scores)
print("平移后的分数：", shifted)
print("softmax 权重：", weights)
print("权重总和：", weights.sum())