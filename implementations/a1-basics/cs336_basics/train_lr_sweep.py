"""固定模型和数据顺序，扫描整条余弦学习率曲线的尺度。"""

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

from cs336_basics import train_rope_ablation as base


SEED = 42
MAX_LEARNING_RATES = (0.001, 0.003, 0.01, 0.03)
# 如果上面的范围全部完成训练，再逐步探索较大的学习率。
EXTRA_LEARNING_RATES = (0.1, 0.3, 1.0)
MIN_LR_RATIO = 0.1


def write_rows(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["step", "lr", "train_loss", "val_loss", "event"],
        )
        writer.writeheader()
        writer.writerows(rows)


def batch_hash(batches):
    digest = hashlib.sha256()
    for x, y in batches:
        digest.update(x.numpy().tobytes())
        digest.update(y.numpy().tobytes())
    return digest.hexdigest()


def find_baseline(folder, train_tokens, val_tokens, initial_hash, data_hash, initial_loss):
    expected = dict(
        seed=SEED, validation_seed=base.VALIDATION_SEED,
        num_steps=base.NUM_STEPS, warmup_steps=base.WARMUP_STEPS,
        batch_size=base.BATCH_SIZE, validation_batches=base.NUM_VALIDATION_BATCHES,
        max_lr=0.003, min_lr=0.0003, weight_decay=0.01, gradient_clip_norm=1.0,
        model_config=base.MODEL_CONFIG, train_tokens=train_tokens, validation_tokens=val_tokens,
    )
    for path in sorted((folder / "rope_ablation_runs").glob("*/summary.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if any(record.get(key) != value for key, value in expected.items()):
                continue
            result = next(r for r in record["results"] if r["use_rope"] is True)
            if result["initial_parameter_hash"] != initial_hash:
                continue
            if result["training_batches_hash"] != data_hash:
                continue
            if abs(result["initial_val_loss"] - initial_loss) > 1e-6:
                continue
            with (path.parent / "with_rope_loss.csv").open(encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
            if int(rows[-1]["step"]) != base.NUM_STEPS:
                continue
            if float(rows[-1]["val_loss"]) != result["final_val_loss"]:
                continue
            return path, result
        except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
            continue
    raise RuntimeError("未找到初始化、数据顺序与配置均一致的 0.003 基线")


def run_case(max_lr, train_batches, val_batches, output_dir, expected_hash, planned_hash):
    torch.manual_seed(SEED)
    model = base.TransformerLM(**base.MODEL_CONFIG)
    if base.parameter_hash(model) != expected_hash:
        raise RuntimeError("本组初始参数与基线不一致")
    optimizer = base.AdamW(model.parameters(), lr=max_lr, weight_decay=0.01)
    initial_loss = base.evaluate_loss(model, val_batches)
    rows = [dict(step=0, lr=None, train_loss=None, val_loss=initial_loss, event="initial")]
    used_digest = hashlib.sha256()
    status = "completed"
    completed_steps = 0
    failure_step = None
    last_finite_val_loss = initial_loss
    final_val_loss = None
    peak_train_loss = 0.0
    start = perf_counter()
    model.train()
    print(f"\n===== max_lr={max_lr:g}，min_lr={max_lr * MIN_LR_RATIO:g} =====", flush=True)

    for step, (x, y) in enumerate(train_batches, start=1):
        used_digest.update(x.numpy().tobytes())
        used_digest.update(y.numpy().tobytes())
        lr = base.get_lr_cosine_schedule(
            it=step, max_learning_rate=max_lr,
            min_learning_rate=max_lr * MIN_LR_RATIO,
            warmup_iters=base.WARMUP_STEPS, cosine_cycle_iters=base.NUM_STEPS,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        loss = base.cross_entropy(model(x), y)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            status = "nonfinite_training_loss"
        else:
            peak_train_loss = max(peak_train_loss, loss_value)
            loss.backward()
            # 与现有裁剪实现使用相同的平方和，提前检查是否发生溢出。
            grad_squared = sum(
                p.grad.float().square().sum()
                for p in model.parameters() if p.grad is not None
            )
            if not torch.isfinite(grad_squared).item():
                status = "nonfinite_gradient_norm"

        if status != "completed":
            failure_step = step
            rows.append(dict(
                step=step, lr=lr,
                train_loss=loss_value if math.isfinite(loss_value) else None,
                val_loss=None, event=status,
            ))
            print(f"学习率 {max_lr:g} 在第 {step} 步停止：{status}", flush=True)
            break

        base.gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        completed_steps = step

        if step == 1 or step % base.EVAL_INTERVAL == 0 or step == base.NUM_STEPS:
            val_loss = base.evaluate_loss(model, val_batches)
            if not math.isfinite(val_loss):
                status = "nonfinite_validation_loss"
                failure_step = step
            else:
                last_finite_val_loss = val_loss
                if step == base.NUM_STEPS:
                    final_val_loss = val_loss
            rows.append(dict(
                step=step, lr=lr, train_loss=loss_value,
                val_loss=val_loss if math.isfinite(val_loss) else None,
                event=status if status != "completed" else "evaluation",
            ))
            print(
                f"[max_lr={max_lr:g}] step={step:4d}，训练 loss={loss_value:.4f}，"
                f"验证 loss={val_loss:.4f}", flush=True,
            )
            # 每次检查都落盘，保留训练中途及失稳前的记录。
            write_rows(output_dir / f"lr_{max_lr:g}_loss.csv", rows)
            if status != "completed":
                break

    elapsed = perf_counter() - start
    curve_path = output_dir / f"lr_{max_lr:g}_loss.csv"
    write_rows(curve_path, rows)
    checkpoint_path = None
    if status == "completed":
        if completed_steps != base.NUM_STEPS or used_digest.hexdigest() != planned_hash:
            raise RuntimeError("训练步数或输入批次与计划不一致")
        checkpoint_path = output_dir / f"lr_{max_lr:g}_checkpoint.pt"
        base.save_checkpoint(model, optimizer, completed_steps, checkpoint_path)

    return dict(
        max_lr=max_lr, min_lr=max_lr * MIN_LR_RATIO, status=status,
        completed_steps=completed_steps, failure_step=failure_step,
        initial_val_loss=initial_loss, final_val_loss=final_val_loss,
        last_finite_val_loss=last_finite_val_loss, peak_train_loss=peak_train_loss,
        initial_parameter_hash=expected_hash, planned_training_batches_hash=planned_hash,
        used_training_batches_hash=used_digest.hexdigest(),
        training_and_validation_seconds=elapsed, curve=str(curve_path),
        checkpoint=str(checkpoint_path) if checkpoint_path else None, reused=False,
    )


def main():
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")
    np.random.seed(base.VALIDATION_SEED)
    val_batches = [
        base.get_batch(val_data, base.BATCH_SIZE, base.CONTEXT_LENGTH, "cpu")
        for _ in range(base.NUM_VALIDATION_BATCHES)
    ]
    np.random.seed(SEED)
    train_batches = [
        base.get_batch(train_data, base.BATCH_SIZE, base.CONTEXT_LENGTH, "cpu")
        for _ in range(base.NUM_STEPS)
    ]
    data_hash = batch_hash(train_batches)
    torch.manual_seed(SEED)
    reference_model = base.TransformerLM(**base.MODEL_CONFIG)
    initial_hash = base.parameter_hash(reference_model)
    initial_loss = base.evaluate_loss(reference_model, val_batches)
    del reference_model
    source, baseline = find_baseline(
        folder, len(train_data), len(val_data), initial_hash, data_hash, initial_loss,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = folder / "lr_sweep_runs" / f"{timestamp}_seed{SEED}"
    output_dir.mkdir(parents=True, exist_ok=False)
    results = []

    settings = dict(
        seed=SEED, validation_seed=base.VALIDATION_SEED,
        model_config=base.MODEL_CONFIG, use_rope=True,
        num_steps=base.NUM_STEPS, warmup_steps=base.WARMUP_STEPS,
        batch_size=base.BATCH_SIZE, validation_batches=base.NUM_VALIDATION_BATCHES,
        min_lr_ratio=MIN_LR_RATIO, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-8,
        gradient_clip_norm=1.0, train_tokens=len(train_data), validation_tokens=len(val_data),
        initial_parameter_hash=initial_hash, planned_training_batches_hash=data_hash,
    )

    def save_summary():
        completed = [r for r in results if r["status"] == "completed"]
        best = min(completed, key=lambda r: r["final_val_loss"]) if completed else None
        report = dict(
            settings=settings, results=results,
            best_max_lr=best["max_lr"] if best else None,
            best_final_val_loss=best["final_val_loss"] if best else None,
        )
        (output_dir / "lr_sweep_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

    for max_lr in MAX_LEARNING_RATES:
        if max_lr == 0.003:
            curve = output_dir / "lr_0.003_loss.csv"
            shutil.copyfile(source.parent / "with_rope_loss.csv", curve)
            result = dict(
                max_lr=max_lr, min_lr=0.0003, status="completed",
                completed_steps=base.NUM_STEPS, failure_step=None,
                initial_val_loss=baseline["initial_val_loss"],
                final_val_loss=baseline["final_val_loss"],
                last_finite_val_loss=baseline["final_val_loss"],
                initial_parameter_hash=initial_hash, planned_training_batches_hash=data_hash,
                used_training_batches_hash=data_hash, curve=str(curve),
                checkpoint=baseline["checkpoint"], source_summary=str(source), reused=True,
            )
            print(f"\n复用 0.003 基线，最终验证 loss={result['final_val_loss']:.4f}", flush=True)
        else:
            result = run_case(max_lr, train_batches, val_batches, output_dir, initial_hash, data_hash)
        results.append(result)
        save_summary()

    if all(r["status"] == "completed" for r in results):
        print("\n起始范围内均完成训练，继续探索较大的学习率。", flush=True)
        for max_lr in EXTRA_LEARNING_RATES:
            result = run_case(max_lr, train_batches, val_batches, output_dir, initial_hash, data_hash)
            results.append(result)
            save_summary()
            if result["status"] != "completed":
                break

    print("\n===== 学习率扫描结果 =====")
    for result in results:
        value = result["final_val_loss"]
        text = f"{value:.4f}" if value is not None else "未完成"
        print(f"max_lr={result['max_lr']:g}：最终验证 loss={text}，状态={result['status']}")
    print("完整配置、曲线和存档已保存：", output_dir)


if __name__ == "__main__":
    main()
