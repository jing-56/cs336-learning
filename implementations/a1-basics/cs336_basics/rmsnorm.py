import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model,
        eps=1e-5,
        device=None,
        dtype=None
    ):
        super().__init__()

        self.eps = eps

        # 每个特征维度对应一个缩放参数，初始为 1
        self.weight = nn.Parameter(
            torch.ones(d_model, device=device, dtype=dtype)
        )

    def forward(self, x):
        input_dtype = x.dtype

        # 按作业要求，用 float32 完成归一化计算
        x = x.to(torch.float32)

        rms = torch.sqrt(
            x.square().mean(dim=-1, keepdim=True) + self.eps
        )

        result = (x / rms) * self.weight

        return result.to(input_dtype)