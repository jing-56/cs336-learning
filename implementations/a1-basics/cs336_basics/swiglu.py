import math

import torch
from torch import nn

from cs336_basics.linear import Linear


def silu(x):
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff=None,
        device=None,
        dtype=None
    ):
        super().__init__()

        # 未指定时，按作业建议选择中间维度
        if d_ff is None:
            d_ff = math.ceil((8 * d_model / 3) / 64) * 64

        self.w1 = Linear(
            d_model, d_ff, device=device, dtype=dtype
        )
        self.w3 = Linear(
            d_model, d_ff, device=device, dtype=dtype
        )
        self.w2 = Linear(
            d_ff, d_model, device=device, dtype=dtype
        )

    def forward(self, x):
        gate = silu(self.w1(x))
        value = self.w3(x)

        return self.w2(gate * value)