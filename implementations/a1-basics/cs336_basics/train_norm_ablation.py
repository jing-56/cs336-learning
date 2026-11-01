"""移除全部 RMSNorm，比较原学习率和两个较低学习率。"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
import torch
from torch import nn

from cs336_basics import train_rope_ablation as base
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.train_lr_sweep import batch_hash, find_baseline


SEED = 42
NO_NORM_LEARNING_RATES = (0.003, 0.001, 0.0003)


def shared_parameter_hash(model):
    """只对各组共同拥有的参数计算指纹，排除归一化的 g。"""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name == "ln_final.weight" or ".ln1." in name or ".ln2." in name:
            continue
        digest.update(name.encode("utf-8"))
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class PassThrough(nn.Module):
    """移除归一化时，保持原样传递，不引入参数。"""

    def forward(self, x):
        return x


def remove_rmsnorm(model):
    for block in model.layers:
        block.ln1 = PassThrough()
        block.ln2 = PassThrough()
    model.ln_final = PassThrough()
    if any(isinstance(module, RMSNorm) for module in model.modules()):
        raise RuntimeError("模型中仍然存在 RMSNorm")


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "lr", "train_loss", "val_loss", "event"])
        writer.writeheader()
        writer.writerows(rows)


def run_case(use_norm, max_lr, batches, val_batches, output_dir,
             expected_full_hash, expected_shared_hash, expected_batch_hash, warmup,
             *, model_transform=None, case_name=None, shared_hash_fn=None):
    torch.manual_seed(SEED)
    model = base.TransformerLM(**base.MODEL_CONFIG)
    full_hash = base.parameter_hash(model)
    if full_hash != expected_full_hash:
        raise RuntimeError("消融前模型初始化不一致")
    if not use_norm:
        remove_rmsnorm(model)
    if model_transform is not None:
        model_transform(model)
    common_hash = (shared_hash_fn or shared_parameter_hash)(model)
    if common_hash != expected_shared_hash:
        raise RuntimeError("共同参数的初始值不一致")
    actual_initial_hash = base.parameter_hash(model)
    name = case_name or (("with_norm" if use_norm else "without_norm") + f"_lr_{max_lr:g}")
    optimizer = base.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    initial_loss = base.evaluate_loss(model, val_batches)
    if not math.isfinite(initial_loss):
        raise RuntimeError("初始化验证 loss 非有限")
    rows = [dict(step=0, lr=None, train_loss=None, val_loss=initial_loss, event="initial")]
    used_digest = hashlib.sha256()
    status = "completed"
    completed_steps = 0
    final_loss = None
    max_grad_norm = 0.0
    peak_loss = 0.0
    model.train()
    start = perf_counter()
    print(f"\n===== {name} =====", flush=True)
    print(f"初始验证 loss={initial_loss:.4f}，参数量={sum(p.numel() for p in model.parameters()):,}", flush=True)
    for step, (x, y) in enumerate(batches, start=1):
        used_digest.update(x.numpy().tobytes())
        used_digest.update(y.numpy().tobytes())
        lr = base.get_lr_cosine_schedule(
            it=step, max_learning_rate=max_lr, min_learning_rate=max_lr * 0.1,
            warmup_iters=warmup, cosine_cycle_iters=len(batches),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        loss = base.cross_entropy(model(x), y)
        value = loss.item()
        if not math.isfinite(value):
            status = "nonfinite_training_loss"
        else:
            peak_loss = max(peak_loss, value)
            loss.backward()
            norm_squared = sum(p.grad.float().square().sum()
                               for p in model.parameters() if p.grad is not None)
            if not torch.isfinite(norm_squared).item():
                status = "nonfinite_gradient_norm"
            else:
                max_grad_norm = max(max_grad_norm, math.sqrt(norm_squared.item()))
        if status != "completed":
            rows.append(dict(step=step, lr=lr, train_loss=value if math.isfinite(value) else None,
                             val_loss=None, event=status))
            print(f"第 {step} 步停止：{status}", flush=True)
            break
        base.gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        completed_steps = step
        if step == 1 or step % base.EVAL_INTERVAL == 0 or step == len(batches):
            val_loss = base.evaluate_loss(model, val_batches)
            if not math.isfinite(val_loss):
                status = "nonfinite_validation_loss"
            rows.append(dict(step=step, lr=lr, train_loss=value,
                             val_loss=val_loss if math.isfinite(val_loss) else None, event=status))
            print(f"[{name}] step={step:4d}，训练 loss={value:.4f}，验证 loss={val_loss:.4f}", flush=True)
            write_rows(output_dir / f"{name}_loss.csv", rows)
            if status != "completed":
                break
            if step == len(batches):
                final_loss = val_loss
    elapsed = perf_counter() - start
    write_rows(output_dir / f"{name}_loss.csv", rows)
    checkpoint = None
    if status == "completed":
        if used_digest.hexdigest() != expected_batch_hash or completed_steps != len(batches):
            raise RuntimeError("使用的训练批次或步数不一致")
        checkpoint = output_dir / f"{name}_checkpoint.pt"
        base.save_checkpoint(model, optimizer, completed_steps, checkpoint)
    return dict(
        name=name, use_rmsnorm=use_norm, max_lr=max_lr, min_lr=max_lr * 0.1,
        status=status, completed_steps=completed_steps, initial_val_loss=initial_loss,
        final_val_loss=final_loss, peak_train_loss=peak_loss,
        peak_preclip_gradient_norm=max_grad_norm,
        pre_ablation_initial_parameter_hash=full_hash,
        actual_initial_parameter_hash=actual_initial_hash,
        shared_initial_parameter_hash=common_hash,
        used_training_batches_hash=used_digest.hexdigest(),
        parameter_count=sum(p.numel() for p in model.parameters()),
        training_and_validation_seconds=elapsed,
        curve=str(output_dir / f"{name}_loss.csv"),
        checkpoint=str(checkpoint) if checkpoint else None, reused=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="每组只运行两步，检查实验流程")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")
    np.random.seed(base.VALIDATION_SEED)
    val_batches = [base.get_batch(val_data, 4, 64, "cpu")
                   for _ in range(base.NUM_VALIDATION_BATCHES)]
    num_steps = 2 if args.smoke else 1000
    warmup = 1 if args.smoke else 50
    np.random.seed(SEED)
    batches = [base.get_batch(train_data, 4, 64, "cpu") for _ in range(num_steps)]
    data_hash = batch_hash(batches)
    torch.manual_seed(SEED)
    reference = base.TransformerLM(**base.MODEL_CONFIG)
    initial_hash = base.parameter_hash(reference)
    shared_hash = shared_parameter_hash(reference)
    initial_loss = base.evaluate_loss(reference, val_batches)
    parameter_count = sum(p.numel() for p in reference.parameters())
    norm_count = sum(isinstance(module, RMSNorm) for module in reference.modules())
    del reference
    source = baseline = None
    if not args.smoke:
        source, baseline = find_baseline(folder, len(train_data), len(val_data),
                                         initial_hash, data_hash, initial_loss)
        if not Path(baseline["checkpoint"]).is_file():
            raise RuntimeError("基线存档不存在")

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{SEED}"
    if args.smoke:
        run_name += "_smoke"
    output_dir = folder / "norm_ablation_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = dict(
        seed=SEED, validation_seed=base.VALIDATION_SEED, model_config=base.MODEL_CONFIG,
        batch_size=4, num_steps=num_steps, total_training_tokens=num_steps * 4 * 64,
        warmup_steps=warmup, min_lr_ratio=0.1, weight_decay=0.01,
        betas=[0.9, 0.999], eps=1e-8, gradient_clip_norm=1.0,
        num_validation_batches=base.NUM_VALIDATION_BATCHES,
        no_norm_max_learning_rates=list(NO_NORM_LEARNING_RATES),
        removed_rmsnorm_modules=norm_count, removal_includes_final_norm=True,
        shared_initial_parameter_hash=shared_hash, planned_training_batches_hash=data_hash,
        train_dataset_tokens=len(train_data), validation_dataset_tokens=len(val_data),
        smoke=args.smoke, torch_threads=4, torch_version=torch.__version__,
        checkpoint_loading="Instantiate TransformerLM; call remove_rmsnorm(model) before loading no-norm checkpoints.",
    )
    results = []
    if args.smoke:
        results.append(run_case(True, 0.003, batches, val_batches, output_dir,
                                initial_hash, shared_hash, data_hash, warmup))
    else:
        curve = output_dir / "with_norm_lr_0.003_loss.csv"
        shutil.copyfile(source.parent / "with_rope_loss.csv", curve)
        results.append(dict(
            name="with_norm_lr_0.003", use_rmsnorm=True, max_lr=0.003, min_lr=0.0003,
            status="completed", completed_steps=1000,
            initial_val_loss=baseline["initial_val_loss"], final_val_loss=baseline["final_val_loss"],
            pre_ablation_initial_parameter_hash=initial_hash,
            shared_initial_parameter_hash=shared_hash, used_training_batches_hash=data_hash,
            parameter_count=parameter_count, curve=str(curve),
            checkpoint=baseline["checkpoint"], source_summary=str(source), reused=True,
        ))
        print(f"复用有 RMSNorm 的基线：验证 loss={baseline['final_val_loss']:.4f}", flush=True)

    def save_summary():
        no_norm_completed = [r for r in results if not r["use_rmsnorm"] and r["status"] == "completed"]
        best = min(no_norm_completed, key=lambda r: r["final_val_loss"]) if no_norm_completed else None
        report = dict(settings=settings, results=results,
                      best_tested_no_norm_lr=best["max_lr"] if best else None)
        (output_dir / "norm_ablation_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )
    save_summary()
    for max_lr in NO_NORM_LEARNING_RATES:
        results.append(run_case(False, max_lr, batches, val_batches, output_dir,
                                initial_hash, shared_hash, data_hash, warmup))
        save_summary()
    print("\n===== RMSNorm 消融结果 =====", flush=True)
    for result in results:
        value = result["final_val_loss"]
        text = f"{value:.4f}" if value is not None else "未完成"
        print(f"{result['name']}：验证 loss={text}，状态={result['status']}", flush=True)
    print("配置、曲线及存档位置：", output_dir, flush=True)


if __name__ == "__main__":
    main()
