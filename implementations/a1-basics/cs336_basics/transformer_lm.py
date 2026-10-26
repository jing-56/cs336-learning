from torch import nn

from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        context_length,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        rope_theta=10000.0,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.context_length = context_length

        # 1. 把 token ID 转换成向量
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )

        # 2. 创建多个独立的 Transformer Block
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])

        # 3. 最后的归一化
        self.ln_final = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype,
        )

        # 4. 为词表中的每个 token 计算分数
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids, token_positions=None):
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("输入序列长度超过了 context_length")

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(
                x,
                token_positions=token_positions,
            )

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits