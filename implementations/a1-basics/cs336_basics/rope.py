import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta,
        d_k,
        max_seq_len,
        device=None,
    ):
        super().__init__()

        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k 必须是正偶数")

        # 每一对分量的起始下标：0、2、4……
        pair_starts = torch.arange(
            0, d_k, 2,
            dtype=torch.float32,
            device=device,
        )

        # 每一对分量的旋转频率
        frequencies = theta ** (-pair_starts / d_k)

        # 所有可能的位置编号
        positions = torch.arange(
            max_seq_len,
            dtype=torch.float32,
            device=device,
        )

        # [最大位置数, 分量对数]
        angles = positions[:, None] * frequencies[None, :]

        # 保存固定的查找表，不参与训练
        self.register_buffer(
            "cos_cached",
            torch.cos(angles),
            persistent=False,
        )
        self.register_buffer(
            "sin_cached",
            torch.sin(angles),
            persistent=False,
        )

    def forward(self, x, token_positions):
        token_positions = token_positions.to(
            device=self.cos_cached.device
        )

        # 根据实际位置编号查表
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 补充维度，让角度可以广播到批次和各个头
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        # 取出每对里的两个分量
        first = x[..., 0::2]
        second = x[..., 1::2]

        # 应用二维旋转公式
        rotated_first = first * cos - second * sin
        rotated_second = first * sin + second * cos

        # 配对，再恢复原来的向量形状
        paired = torch.stack(
            [rotated_first, rotated_second],
            dim=-1,
        )

        return paired.flatten(start_dim=-2).to(dtype=x.dtype)