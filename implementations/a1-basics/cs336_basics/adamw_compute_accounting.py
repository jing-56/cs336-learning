"""讲义 AdamW 核算 (c)-(d)：运算次数与给定 MFU 下的理论时间。"""

import argparse
from fractions import Fraction
import json
from pathlib import Path


# 系数均已按步/参数组预计算；平方、开方和除法各按一次运算记账。
PAPER_OPS = dict(weight_decay=2, first_moment=3, second_moment=4, parameter_update=5)
LOCAL_OPS = dict(first_moment=3, second_moment=4, bias_correction=2,
                 weight_decay=1, parameter_update=5)


def exact_integer(value):
    value = Fraction(value)
    if value.denominator != 1:
        raise ValueError("整体配置的核算值应为整数")
    return int(value)


def estimate(ffn_width):
    v, t, l, d = 50257, 1024, 48, 1600
    f = Fraction(ffn_width)
    p = exact_integer(2 * v * d + l * (4 * d * d + 3 * d * f + 2 * d) + d)
    forward = exact_integer(l * (8 * t * d * d + 4 * t * t * d + 6 * t * d * f) + 2 * t * d * v)
    batch, steps = 1024, 400000
    peak = 495 * 10 ** 12  # 题目给定的 TF32 算力口径，不是设备探测结果。
    mfu = Fraction(1, 2)
    training_per_step = 3 * forward * batch
    total = training_per_step * steps
    seconds = Fraction(total, 1) / (peak * mfu)
    assert sum(PAPER_OPS.values()) == 14
    assert sum(LOCAL_OPS.values()) == 15
    assert seconds * peak * mfu == total
    return dict(
        d_ff=str(f), parameters=p, forward_matmul_flops_per_sequence=forward,
        paper_adamw_flops_per_step=14 * p, local_adamw_flops_per_step=15 * p,
        effective_batch_size=batch, sequence_length=t, optimizer_steps=steps,
        processed_target_tokens=batch * t * steps,
        model_training_flops_per_step=training_per_step, model_training_flops_total=total,
        handout_peak_flops_per_second=peak, assumed_mfu=float(mfu),
        assumed_model_flops_per_second=float(peak * mfu),
        seconds_per_step=float(seconds / steps), training_seconds=float(seconds),
        training_hours=float(seconds / 3600), training_days=float(seconds / (3600 * 24)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rounded = estimate(4288)
    continuous = estimate(Fraction(8 * 1600, 3))
    assert rounded["parameters"] == 1640452800
    assert rounded["forward_matmul_flops_per_sequence"] == 3516769894400
    print("AdamW 稳态更新：讲义逐项口径 14P；当前实现逐项口径 15P，均为 O(P)。")
    print("不含反向、裁剪、初次状态创建、内存读写；标量系数计算作为低阶项。")
    for label, record in (("具体配置 f=4288", rounded), ("连续近似 f=8d/3", continuous)):
        print(f"\n{label}，参数={record['parameters']:,}")
        print(f"AdamW 讲义口径：{record['paper_adamw_flops_per_step'] / 1e9:.6f} GFLOPs/步")
        print(f"AdamW 当前实现口径：{record['local_adamw_flops_per_step'] / 1e9:.6f} GFLOPs/步")
        print(f"单序列前向：{record['forward_matmul_flops_per_sequence'] / 1e12:.6f} TFLOPs")
        print(f"训练模型 FLOPs/步：{record['model_training_flops_per_step']:,}")
        print(f"处理目标 token 数：{record['processed_target_tokens']:,}")
        print(f"假设有效算力：{record['assumed_model_flops_per_second'] / 1e12:.1f} TFLOP/s")
        print(f"每步约 {record['seconds_per_step']:.6f} 秒")
        print(f"训练约 {record['training_hours']:.6f} 小时，即 {record['training_days']:.6f} 天")
    print("\nMFU 按模型 FLOPs 定义，主时间公式不额外叠加优化器 FLOPs。")
    print("batch=1024 是本算式的有效 batch 假设，不代表物理 batch 能装入单卡。")
    print("这是纸面估算，没有实际创建模型、训练或测时。")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(
            problem="adamw_accounting (c)-(d)",
            paper_operations_per_parameter=PAPER_OPS,
            local_operations_per_parameter=LOCAL_OPS,
            assumptions=[
                "Count each arithmetic op, square root and division as one; scalar coefficients precomputed",
                "Steady-state AdamW update only; no backward, clipping, initialization, memory traffic, or latency model",
                "Handout peak 495 TFLOP/s in its specified TF32 convention; MFU fixed at 50%",
                "Backward matmul FLOPs assumed to equal twice forward matmul FLOPs",
                "Time = model FLOPs/(peak*MFU); optimizer operations not separately added to MFU time",
                "Batch 1024 is effective batch; not a feasible single-device physical batch claim",
            ],
            rounded_xl=rounded, continuous_ffn=continuous,
        )
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("导出：", args.output)


if __name__ == "__main__":
    main()
