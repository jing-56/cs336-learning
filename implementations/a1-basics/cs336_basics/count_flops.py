"""手算当前模型一次完整前向中的主要稠密矩阵乘法 FLOPs。"""

from cs336_basics.train_rope_ablation import MODEL_CONFIG


def estimate_matmul_flops(batch_size, seq_len):
    """采用 (m,n)@(n,p) 约 2mnp 的常用口径；不是实测耗时。"""
    if batch_size <= 0 or seq_len <= 0:
        raise ValueError("batch_size 和 seq_len 必须为正数")
    d = MODEL_CONFIG["d_model"]
    f = MODEL_CONFIG["d_ff"]
    heads = MODEL_CONFIG["num_heads"]
    depth = MODEL_CONFIG["num_layers"]
    vocab = MODEL_CONFIG["vocab_size"]
    if d % heads:
        raise ValueError("模型维度不能被头数整除")
    head_dim = d // heads
    b, t = batch_size, seq_len
    # 先按实际头维度计算 QK 和 AV，再核对化简后的公式。
    qk = 2 * b * heads * t * t * head_dim
    av = 2 * b * heads * t * head_dim * t
    if qk + av != 4 * b * t * t * d:
        raise RuntimeError("注意力头的维度核算不一致")
    per_layer = {
        "Q、K、V 投影": 6 * b * t * d * d,
        "注意力输出投影": 2 * b * t * d * d,
        "QK 打分": qk,
        "AV 加权求和": av,
        "SwiGLU 三次投影": 6 * b * t * d * f,
    }
    rows = {name: value * depth for name, value in per_layer.items()}
    rows["输出词表投影"] = 2 * b * t * d * vocab
    total = sum(rows.values())
    formula = depth * (8 * b * t * d * d + 4 * b * t * t * d
                       + 6 * b * t * d * f) + 2 * b * t * d * vocab
    if total != formula:
        raise RuntimeError("逐项合计与模型公式不一致")
    return rows, total


def main():
    baseline, total = estimate_matmul_flops(4, 64)
    print("统计范围：一次完整前向，主要稠密矩阵乘法，乘加记 2 FLOPs。")
    print("不含反向、优化器、归一化、softmax、RoPE、激活和其他逐元素操作。")
    print("当前 B=4、T=64，以下各层内项目已合计全部两层：")
    for name, value in baseline.items():
        print(f"{name}：{value:,} FLOPs")
    print(f"合计：{total:,} FLOPs，约 {total / 1e9:.3f} GFLOPs")
    print(f"输出词表投影占比：{baseline['输出词表投影'] / total:.2%}")
    print("\n改变输入规模的公式对比：")
    for b, t in ((4, 64), (8, 64), (4, 128)):
        rows, value = estimate_matmul_flops(b, t)
        core = rows["QK 打分"] + rows["AV 加权求和"]
        baseline_core = baseline["QK 打分"] + baseline["AV 加权求和"]
        print(f"B={b}，T={t}：合计 {value:,} FLOPs（{value / total:.4f} 倍），"
              f"QK+AV 为 {core / baseline_core:.1f} 倍")
    print("\nT=128 是扩展上下文后的假设核算，没有运行原上下文上限为 64 的模型。")
    print("本实现先计算完整 QK 分数矩阵再遮罩，因此没有把因果注意力的 FLOPs 减半。")
    print("FLOPs 是运算量，不能直接等同于实际耗时。")


if __name__ == "__main__":
    main()
