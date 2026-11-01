"""AdamW 内存的纸面核算：固定状态与主要算子输出，不构建大模型。"""

import argparse
from fractions import Fraction
import json
from pathlib import Path


def integer(value):
    value = Fraction(value)
    if value.denominator != 1:
        raise ValueError("所选整体配置的存储元素数不是整数")
    return int(value)


def estimate(ffn_width):
    v, t, l, d, h = 50257, 1024, 48, 1600, 25
    f = Fraction(ffn_width)
    p = integer(2 * v * d + l * (4 * d * d + 3 * d * f + 2 * d) + d)
    # 按讲义列出的主要算子各保留一份输出；CE 计一份词表大小中间张量。
    # 元素数已合计全部 48 层，batch=1；不模拟真实 autograd 的存活期和存储共享。
    activation_elements = dict(
        block_norm_outputs=integer(l * 2 * t * d),
        qkv_outputs=integer(l * 3 * t * d),
        attention_scores_and_softmax=integer(l * 2 * h * t * t),
        attention_value_and_output=integer(l * 2 * t * d),
        swiglu_outputs=integer(l * (4 * t * f + t * d)),
        final_norm=integer(t * d),
        logits_and_ce_intermediate=integer(2 * t * v),
    )
    activations = sum(activation_elements.values()) * 4
    formula = integer(4 * t * (l * (8 * d + 4 * f + 2 * h * t) + d + 2 * v))
    assert activations == formula
    fixed = 16 * p
    capacity = 80 * 10 ** 9
    limit = (capacity - fixed) // activations
    assert fixed + limit * activations <= capacity < fixed + (limit + 1) * activations
    return dict(
        vocab_size=v, context_length=t, num_layers=l, d_model=d, num_heads=h,
        d_ff=str(f), parameters=p,
        activation_bytes_per_sequence={key: value * 4 for key, value in activation_elements.items()},
        activation_total_bytes_per_sequence=activations,
        parameter_bytes=4 * p, gradient_bytes=4 * p, optimizer_state_bytes=8 * p,
        fixed_state_bytes=fixed, capacity_bytes=capacity, max_batch_size=limit,
        batch3_bytes=fixed + 3 * activations, batch4_bytes=fixed + 4 * activations,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rounded = estimate(4288)
    continuous = estimate(Fraction(8 * 1600, 3))
    print("简化口径：主要算子各保存一份输出；CE 额外一份 BT×V 中间张量。")
    print("不是真实峰值或可运行 batch 的保证；不计运行时、临时张量、存储共享及释放时机。")
    for label, result in (("沿用具体配置 f=4288", rounded),
                          ("adamw_accounting 指定的 f=8d/3", continuous)):
        print(f"\n{label}")
        print(f"固定状态：{result['fixed_state_bytes'] / 1e9:.9f} GB")
        print(f"每序列激活：{result['activation_total_bytes_per_sequence'] / 1e9:.9f} GB")
        print("各部分（已合计全部层）：")
        for key, value in result["activation_bytes_per_sequence"].items():
            print(f"{key}: {value / 1e9:.9f} GB")
        print(f"80 GB 简化估算的最大 batch：{result['max_batch_size']}")
        print(f"B=3：{result['batch3_bytes'] / 1e9:.9f} GB；"
              f"B=4：{result['batch4_bytes'] / 1e9:.9f} GB")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(
            scope="adamw_accounting (a)-(b), explicit simplified activation-output bookkeeping",
            activation_formula_bytes="4*B*T*(L*(8*d+4*f+2*H*T)+d+2*V)",
            memory_formula_bytes="16*P + activation_formula_bytes",
            assumptions=[
                "FP32 throughout; 80 GB means 80*10^9 bytes",
                "Parameters, gradients, and m/v simultaneously present",
                "One output per listed main operator, plus one BT*V CE intermediate",
                "No checkpointing, mixed precision, FlashAttention, recomputation, or state sharding",
                "Not an autograd lifetime analysis or measured device memory; ignores aliases, extra operator internals, runtime, workspaces and buffers",
            ],
            rounded_xl=rounded, formal_continuous_ffn=continuous,
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n导出：", args.output)


if __name__ == "__main__":
    main()
