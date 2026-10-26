import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        device=None,
        dtype=None
    ):
        super().__init__()

        # 创建可训练的向量表
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype
            )
        )

        # 按作业要求初始化
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0
        )

    def forward(self, token_ids):
        return self.weight[token_ids]