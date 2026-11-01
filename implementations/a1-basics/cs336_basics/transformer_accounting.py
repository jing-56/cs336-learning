"""讲义 transformer_accounting (a)-(e)：纯整数核算，不构建大模型。"""

import argparse
import json
import math
from pathlib import Path


def nearby_ffn_width(d_model):
    # 沿用当前 SwiGLU 实现：向上取整到 64 的倍数。
    return math.ceil(8 * d_model / (3 * 64)) * 64


def account(name, layers, d_model, num_heads, d_ff=None,
            vocab_size=50257, context_length=1024, batch_size=1):
    f = nearby_ffn_width(d_model) if d_ff is None else d_ff
    l, d, h, v, t, b = layers, d_model, num_heads, vocab_size, context_length, batch_size
    if any(value <= 0 for value in (l, d, h, f, v, t, b)) or d % h:
        raise ValueError("尺寸必须为正，且头数必须整除模型维度")
    parameter_parts = dict(
        input_embedding=v * d,
        attention_projections=l * 4 * d * d,
        swiglu=l * 3 * d * f,
        rmsnorm=(2 * l + 1) * d,
        output_head=v * d,
    )
    parameters = sum(parameter_parts.values())
    assert parameters == 2 * v * d + l * (4 * d * d + 3 * d * f + 2 * d) + d
    flops_parts = dict(
        attention_projections=l * 8 * b * t * d * d,
        attention_qk_av=l * 4 * b * t * t * d,
        swiglu=l * 6 * b * t * d * f,
        output_head=2 * b * t * d * v,
    )
    total_flops = sum(flops_parts.values())
    # 按实际矩阵形状独立求和：三投影、输出投影、两次注意力乘法、三次 FFN、词表输出。
    matrix_costs = dict(
        qkv_each=2 * b * t * d * d,
        attention_output=2 * b * t * d * d,
        qk=2 * b * h * t * (d // h) * t,
        av=2 * b * h * t * t * (d // h),
        ffn_each=2 * b * t * d * f,
        output_head=2 * b * t * d * v,
    )
    assert total_flops == l * (3 * matrix_costs["qkv_each"] + matrix_costs["attention_output"]
                               + matrix_costs["qk"] + matrix_costs["av"]
                               + 3 * matrix_costs["ffn_each"]) + matrix_costs["output_head"]
    return dict(
        name=name, num_layers=l, d_model=d, num_heads=h, head_dim=d // h,
        d_ff=f, vocab_size=v, context_length=t, batch_size=b,
        parameter_parts=parameter_parts, parameters=parameters,
        fp32_parameter_bytes=parameters * 4,
        fp32_parameter_GB=parameters * 4 / 1e9,
        fp32_parameter_GiB=parameters * 4 / 1024 ** 3,
        flops_parts=flops_parts, forward_matmul_flops=total_flops,
        flops_proportions={key: value / total_flops for key, value in flops_parts.items()},
        matrix_costs_per_layer=matrix_costs,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="可选：将核算结果导出为 JSON")
    args = parser.parse_args()
    # 与已经核验的小模型结果交叉对照。
    small_demo = account("learning_demo", 2, 64, 4, 128, 10000, 64, 4)
    assert small_demo["parameters"] == 1362240
    assert small_demo["forward_matmul_flops"] == 378011648
    models = [
        account("GPT-2 small-sized", 12, 768, 12),
        account("GPT-2 medium-sized", 24, 1024, 16),
        account("GPT-2 large-sized", 36, 1280, 20),
        account("GPT-2 XL-sized", 48, 1600, 25, 4288),
    ]
    xl = models[-1]
    longer = account("GPT-2 XL-sized, longer context", 48, 1600, 25, 4288,
                      context_length=16384)
    assert longer["parameters"] == xl["parameters"]
    assert longer["flops_parts"]["attention_qk_av"] == 256 * xl["flops_parts"]["attention_qk_av"]
    for key in ("attention_projections", "swiglu", "output_head"):
        assert longer["flops_parts"][key] == 16 * xl["flops_parts"][key]

    print("架构：作业的 RMSNorm、RoPE、SwiGLU，无偏置、输入输出权重不共享。")
    print("FLOPs 仅核算主要稠密矩阵乘法，batch=1，一次完整序列前向。")
    print("small/medium/large 的 d_ff 沿用当前实现，向上取整至 64 的倍数。\n")
    print("XL 参数分项：")
    for key, value in xl["parameter_parts"].items():
        print(f"{key}: {value:,}")
    print(f"总参数：{xl['parameters']:,}")
    print(f"仅 FP32 权重：{xl['fp32_parameter_bytes']:,} 字节，"
          f"{xl['fp32_parameter_GB']:.6f} GB，{xl['fp32_parameter_GiB']:.6f} GiB")
    print("\nXL 前向 FLOPs 分项：")
    for key, value in xl["flops_parts"].items():
        print(f"{key}: {value:,}，{xl['flops_proportions'][key]:.4%}")
    print(f"总前向：{xl['forward_matmul_flops']:,} FLOPs，"
          f"{xl['forward_matmul_flops'] / 1e12:.6f} TFLOPs")
    print("\n型号比较：投影 / QK+AV / SwiGLU / 输出占比")
    for model in models:
        parts = " / ".join(f"{value:.2%}" for value in model["flops_proportions"].values())
        print(f"{model['name']}，d_ff={model['d_ff']}，"
              f"{model['forward_matmul_flops'] / 1e12:.6f} TFLOPs：{parts}")
    print("\nXL 上下文扩展至 16384：")
    print(f"前向 {longer['forward_matmul_flops']:,} FLOPs，"
          f"{longer['forward_matmul_flops'] / 1e12:.6f} TFLOPs，"
          f"为原来的 {longer['forward_matmul_flops'] / xl['forward_matmul_flops']:.6f} 倍")
    for key, value in longer["flops_proportions"].items():
        print(f"{key}: {value:.4%}")

    report = dict(
        problem="transformer_accounting (a)-(e)",
        source="cs336_assignment1_basics.pdf, printed pages 27-28; SwiGLU rounding on page 22",
        architecture="Assignment architecture; not the original released GPT-2 architecture",
        assumptions=dict(
            batch_size=1, fp32_bytes=4, tied_embeddings=False, biases=False,
            count="dense matmul FLOPs only; multiply-accumulate counts as 2 FLOPs",
            causal_attention="full dense QK and AV; no triangular FLOPs discount",
            secondary_model_ffn_rounding="ceil(8*d_model/(3*64))*64; XL explicitly 4288",
            memory="parameter storage only; excludes buffers, activations, gradients, optimizer and runtime",
        ),
        models=models, xl_long_context=longer,
        long_context_flops_ratio=longer["forward_matmul_flops"] / xl["forward_matmul_flops"],
        small_model_crosscheck=dict(parameters=small_demo["parameters"],
                                    forward_matmul_flops=small_demo["forward_matmul_flops"]),
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n已导出：", args.output)


if __name__ == "__main__":
    main()
