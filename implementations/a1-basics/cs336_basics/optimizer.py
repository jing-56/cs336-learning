import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    ):
        if lr < 0:
            raise ValueError("学习率不能为负数")
        if eps < 0:
            raise ValueError("eps 不能为负数")
        if weight_decay < 0:
            raise ValueError("weight_decay 不能为负数")
        if not all(0 <= beta < 1 for beta in betas):
            raise ValueError("beta 必须满足 0 <= beta < 1")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        # 兼容 PyTorch 优化器的可选接口
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError("这个 AdamW 不支持稀疏梯度")

                # 获取这个参数自己的历史状态
                state = self.state[p]

                # 第一次遇到这个参数时，初始化状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                state["t"] += 1

                t = state["t"]
                m = state["m"]
                v = state["v"]

                # 更新梯度的移动平均
                m.mul_(beta1).add_(grad, alpha=1 - beta1)

                # 更新梯度平方的移动平均
                v.mul_(beta2).add_(
                    grad.square(),
                    alpha=1 - beta2,
                )

                # 修正从零开始造成的起步偏差
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)

                # 单独执行权重衰减
                p.mul_(1 - lr * weight_decay)

                # 根据梯度历史更新参数
                p.add_(
                    -lr * m_hat / (torch.sqrt(v_hat) + eps)
                )

        return loss