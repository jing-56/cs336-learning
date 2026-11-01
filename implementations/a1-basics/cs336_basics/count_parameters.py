"""打印当前小模型的权重形状，并核对手算与实际参数量。"""

import torch

from cs336_basics.train_rope_ablation import MODEL_CONFIG
from cs336_basics.transformer_lm import TransformerLM


def count(module):
    return sum(parameter.numel() for parameter in module.parameters())


def main():
    torch.manual_seed(42)
    model = TransformerLM(**MODEL_CONFIG)
    vocab = MODEL_CONFIG["vocab_size"]
    dim = MODEL_CONFIG["d_model"]
    depth = MODEL_CONFIG["num_layers"]
    width = MODEL_CONFIG["d_ff"]
    if model.token_embeddings.weight is model.lm_head.weight:
        raise RuntimeError("当前公式针对输入、输出权重不共享的模型")
    rows = [
        ("输入 Embedding", vocab * dim, count(model.token_embeddings)),
        ("全部注意力投影", depth * 4 * dim * dim,
         sum(count(layer.attn) for layer in model.layers)),
        ("全部 SwiGLU", depth * 3 * dim * width,
         sum(count(layer.ffn) for layer in model.layers)),
        ("全部 RMSNorm", (2 * depth + 1) * dim,
         count(model.ln_final) + sum(count(layer.ln1) + count(layer.ln2) for layer in model.layers)),
        ("输出词表投影", vocab * dim, count(model.lm_head)),
    ]
    print(f"V={vocab}，d={dim}，L={depth}，d_ff={width}，头数={MODEL_CONFIG['num_heads']}")
    print("\n模块：手算参数量 / 程序统计")
    for name, expected, actual in rows:
        if expected != actual:
            raise RuntimeError(f"{name}：手算 {expected} 与实际 {actual} 不一致")
        print(f"{name}：{expected:,} / {actual:,}")
    manual_total = sum(row[1] for row in rows)
    actual_total = count(model)
    if manual_total != actual_total:
        raise RuntimeError("分类合计未覆盖整个模型，或存在共享参数")
    print(f"\n总参数量：{actual_total:,}")
    print("手算与实际一致：", manual_total == actual_total)
    vocab_share = (count(model.token_embeddings) + count(model.lm_head)) / actual_total
    print(f"输入、输出两张词表矩阵的占比：{vocab_share:.2%}")
    print("\n每个可学习参数的形状：")
    for name, parameter in model.named_parameters():
        print(f"{name}：{tuple(parameter.shape)}，{parameter.numel():,} 个数")
    print("\n固定 buffer 元素数：", sum(buffer.numel() for buffer in model.buffers()))
    print("本模型的这些 buffer 是 RoPE 固定查找表，不计入可学习参数。")


if __name__ == "__main__":
    main()
