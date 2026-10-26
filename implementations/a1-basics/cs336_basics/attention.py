import math

from cs336_basics.nn_utils import softmax


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]

    # 计算匹配分数
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)

    # True 表示允许关注，False 表示禁止关注
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    # 沿 Key 所在的维度分配权重
    weights = softmax(scores, dim=-1)

    return weights @ v