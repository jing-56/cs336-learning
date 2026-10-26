import torch
from torch import nn

from cs336_basics.linear import Linear
from cs336_basics.attention import scaled_dot_product_attention


class MultiHeadSelfAttention(nn.Module):
    def __init__(
            self,
            d_model,
            num_heads,
            device=None,
            dtype=None,
            rope=None,
    ):
        super().__init__()

        if num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("头数必须为正数，且能够整除 d_model")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope = rope
        # 三个独立的线性层，分别生成 Q、K、V
        self.q_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

        # 拼接各头结果后，再做一次线性变换
        self.output_proj = Linear(
            d_model, d_model, device=device, dtype=dtype
        )

    def split_heads(self, x):
        # [..., 位置, 总维度]
        # → [..., 位置, 头, 每头维度]
        x = x.reshape(
            *x.shape[:-1],
            self.num_heads,
            self.head_dim,
        )

        # → [..., 头, 位置, 每头维度]
        return x.transpose(-3, -2)

    def forward(self, x, token_positions=None):
        seq_len = x.shape[-2]

        q = self.split_heads(self.q_proj(x))
        k = self.split_heads(self.k_proj(x))
        v = self.split_heads(self.v_proj(x))
        # 如果配置了 RoPE，就旋转 Q 和 K
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                )

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=x.device,
            )
        )

        head_outputs = scaled_dot_product_attention(
            q, k, v, mask
        )

        # 把同一个位置的各个头重新拼起来
        combined = head_outputs.transpose(-3, -2)
        combined = combined.contiguous().reshape(
            *x.shape[:-1],
            self.d_model,
        )

        return self.output_proj(combined)