import math


def get_lr_cosine_schedule(
    it,
    max_learning_rate,
    min_learning_rate,
    warmup_iters,
    cosine_cycle_iters,
):
    if warmup_iters < 0:
        raise ValueError("预热步数不能为负数")

    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("下降结束步数必须大于预热结束步数")

    # 第一阶段：线性预热
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    # 第二阶段：余弦下降
    if it <= cosine_cycle_iters:
        progress = (
            (it - warmup_iters)
            / (cosine_cycle_iters - warmup_iters)
        )

        cosine_factor = 0.5 * (
            1 + math.cos(math.pi * progress)
        )

        return (
            min_learning_rate
            + cosine_factor
            * (max_learning_rate - min_learning_rate)
        )

    # 第三阶段：保持最低学习率
    return min_learning_rate