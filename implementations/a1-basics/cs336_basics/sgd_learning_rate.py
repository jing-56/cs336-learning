"""A1 learning_rate_tuning: the handout's decaying-learning-rate SGD."""

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if not math.isfinite(lr) or lr < 0:
            raise ValueError("lr must be finite and nonnegative")
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                t = state.get("t", 0)
                effective_lr = group["lr"] / math.sqrt(t + 1)
                parameter.add_(parameter.grad, alpha=-effective_lr)
                state["t"] = t + 1
        return loss


def run_experiment(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    generator = torch.Generator(device="cpu").manual_seed(42)
    initial_weights = 5 * torch.randn((10, 10), generator=generator)
    rows = []
    cases = []
    for lr in (1.0, 10.0, 100.0, 1000.0):
        weights = torch.nn.Parameter(initial_weights.clone())
        optimizer = SGD([weights], lr=lr)
        losses = []
        print(f"\n学习率 {lr:g}（第 0 项是更新前，第 10 项是完成十次更新后）：")
        for completed_updates in range(11):
            loss = weights.square().mean()
            value = loss.item()
            losses.append(value)
            rows.append({"lr": lr, "completed_updates": completed_updates, "loss": value})
            print(f"  已更新 {completed_updates:2d} 次，loss={value:.8g}")
            if completed_updates < 10:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        cases.append({"lr": lr, "losses": losses})

    with (output_dir / "losses.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("lr", "completed_updates", "loss"))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "problem": "learning_rate_tuning",
        "seed": 42,
        "device": "cpu",
        "dtype": str(initial_weights.dtype),
        "torch_version": str(torch.__version__),
        "shape": [10, 10],
        "updates": 10,
        "loss": "mean(weights ** 2)",
        "schedule": "lr / sqrt(t + 1), t starts at 0",
        "same_initial_weights": True,
        "recording": "losses[0:10] match the handout's pre-update prints; losses[10] is extra",
        "cases": cases,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(f"\n结果已保存：{output_dir.resolve()}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "sgd_learning_rate_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
    )
    args = parser.parse_args()
    run_experiment(args.output_dir)
