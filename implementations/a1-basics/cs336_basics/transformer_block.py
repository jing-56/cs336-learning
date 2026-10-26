from torch import nn

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rope import RotaryPositionalEmbedding
from cs336_basics.multihead_attention import MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta=10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        # 两个独立的归一化模块
        self.ln1 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )
        self.ln2 = RMSNorm(
            d_model,
            device=device,
            dtype=dtype,
        )

        # 为每个头准备旋转位置编码
        rope = RotaryPositionalEmbedding(
            theta=theta,
            d_k=d_model // num_heads,
            max_seq_len=max_seq_len,
            device=device,
        )

        # 带 RoPE 的因果多头注意力
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            device=device,
            dtype=dtype,
            rope=rope,
        )

        # 前馈网络
        self.ffn = SwiGLU(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(self, x, token_positions=None):
        # 第一部分：注意力更新 + 残差连接
        attention_update = self.attn(
            self.ln1(x),
            token_positions=token_positions,
        )
        y = x + attention_update

        # 第二部分：前馈网络更新 + 残差连接
        ffn_update = self.ffn(self.ln2(y))
        output = y + ffn_update

        return output