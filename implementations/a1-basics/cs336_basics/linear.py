import torch
from torch import nn


class Linear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        device=None,
        dtype=None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype
            )
        )

        # 按作业要求初始化权重
        std = (2.0 / (in_features + out_features)) ** 0.5

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3 * std,
            b=3 * std
        )

    def forward(self, x):
        return x @ self.weight.T