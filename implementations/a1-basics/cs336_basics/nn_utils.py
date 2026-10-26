import torch


def softmax(x, dim=-1):
    max_values = x.max(dim=dim, keepdim=True).values
    shifted = x - max_values

    exp_values = torch.exp(shifted)
    totals = exp_values.sum(dim=dim, keepdim=True)

    return exp_values / totals
def cross_entropy(inputs, targets):
    # 每个位置减去该位置最大的 logit，保证数值稳定
    max_values = inputs.max(dim=-1, keepdim=True).values
    shifted = inputs - max_values

    # 计算 log(sum(exp(shifted)))
    log_totals = torch.log(
        torch.exp(shifted).sum(dim=-1)
    )

    # 取出每个位置上，正确 token 对应的 shifted logit
    target_logits = torch.gather(
        shifted,
        dim=-1,
        index=targets.unsqueeze(-1),
    ).squeeze(-1)

    # 得到各位置的损失，再取平均
    losses = log_totals - target_logits

    return losses.mean()
@torch.no_grad()
def gradient_clipping(parameters, max_l2_norm):
    if max_l2_norm < 0:
        raise ValueError("梯度最大长度不能为负数")

    # 只收集实际存在的梯度
    grads = [
        p.grad
        for p in parameters
        if p.grad is not None
    ]

    if not grads:
        return

    # 把所有梯度元素的平方加起来
    total_squared_norm = sum(
        grad.float().square().sum()
        for grad in grads
    )

    # 整个模型的梯度总长度
    total_norm = torch.sqrt(total_squared_norm)

    # 超过上限时，按相同比例缩小所有梯度
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)

        for grad in grads:
            grad.mul_(scale)