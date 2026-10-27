"""对比有 RoPE 和无 RoPE：相同初始参数、训练批次与验证批次。"""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cs336_basics.checkpoint import save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer_lm import TransformerLM


SEED = 42
VALIDATION_SEED = 12345
NUM_STEPS = 1000
WARMUP_STEPS = 50
EVAL_INTERVAL = 100
BATCH_SIZE = 4
CONTEXT_LENGTH = 64
NUM_VALIDATION_BATCHES = 10

MODEL_CONFIG = dict(
    vocab_size=10000,
    context_length=CONTEXT_LENGTH,
    d_model=64,
    num_layers=2,
    num_heads=4,
    d_ff=128,
    rope_theta=10000.0,
    device="cpu",
)


@torch.no_grad()
def evaluate_loss(model, batches):
    was_training = model.training
    model.eval()
    losses = [cross_entropy(model(x), y).item() for x, y in batches]
    model.train(was_training)
    return sum(losses) / len(losses)


def parameter_hash(model):
    digest = hashlib.sha256()
    for parameter in model.parameters():
        digest.update(parameter.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def run_case(use_rope, train_data, val_batches, output_dir):
    # 每个版本都重置种子，从相同参数、相同训练批次顺序开始。
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    model = TransformerLM(**MODEL_CONFIG)

    # 仅改变这一项：是否在注意力中旋转 Q、K。
    # 因果遮罩、Q/K/V 投影、前馈网络等保持相同。
    if not use_rope:
        for block in model.layers:
            block.attn.rope = None

    name = "with_rope" if use_rope else "without_rope"
    initial_hash = parameter_hash(model)
    batch_digest = hashlib.sha256()

    optimizer = AdamW(
        model.parameters(), lr=3e-3, weight_decay=0.01,
    )
    initial_loss = evaluate_loss(model, val_batches)
    rows = [dict(step=0, lr=None, train_loss=None, val_loss=initial_loss)]

    print(f"\n开始 {name}：从头训练 {NUM_STEPS} 步", flush=True)
    print(f"初始验证 loss={initial_loss:.4f}", flush=True)
    model.train()
    start = perf_counter()

    for step in range(1, NUM_STEPS + 1):
        x, y = get_batch(
            train_data, BATCH_SIZE, CONTEXT_LENGTH, "cpu",
        )
        batch_digest.update(x.numpy().tobytes())
        batch_digest.update(y.numpy().tobytes())

        lr = get_lr_cosine_schedule(
            it=step,
            max_learning_rate=3e-3,
            min_learning_rate=3e-4,
            warmup_iters=WARMUP_STEPS,
            cosine_cycle_iters=NUM_STEPS,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        optimizer.zero_grad()
        loss = cross_entropy(model(x), y)
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()

        if step == 1 or step % EVAL_INTERVAL == 0 or step == NUM_STEPS:
            val_loss = evaluate_loss(model, val_batches)
            rows.append(dict(
                step=step, lr=lr, train_loss=loss.item(), val_loss=val_loss,
            ))
            print(
                f"[{name}] step={step:4d}，lr={lr:.6f}，"
                f"训练 loss={loss.item():.4f}，验证 loss={val_loss:.4f}",
                flush=True,
            )

    elapsed = perf_counter() - start
    checkpoint_path = output_dir / f"{name}_checkpoint.pt"
    save_checkpoint(model, optimizer, NUM_STEPS, checkpoint_path)

    with (output_dir / f"{name}_loss.csv").open(
        "w", encoding="utf-8", newline="",
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=["step", "lr", "train_loss", "val_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)

    return dict(
        use_rope=use_rope,
        initial_parameter_hash=initial_hash,
        training_batches_hash=batch_digest.hexdigest(),
        initial_val_loss=initial_loss,
        final_val_loss=rows[-1]["val_loss"],
        training_and_validation_seconds=elapsed,
        checkpoint=str(checkpoint_path),
    )


def main():
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")

    # 只抽取一次验证批次，两组实验使用同一组检查材料。
    # 每组训练开始时会另行重置训练用的随机数种子。
    np.random.seed(VALIDATION_SEED)
    val_batches = [
        get_batch(val_data, BATCH_SIZE, CONTEXT_LENGTH, "cpu")
        for _ in range(NUM_VALIDATION_BATCHES)
    ]

    run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{SEED}"
    output_dir = folder / "rope_ablation_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    results = [
        run_case(True, train_data, val_batches, output_dir),
        run_case(False, train_data, val_batches, output_dir),
    ]
    same_initial_parameters = (
        results[0]["initial_parameter_hash"] == results[1]["initial_parameter_hash"]
    )
    same_training_batches = (
        results[0]["training_batches_hash"] == results[1]["training_batches_hash"]
    )

    summary = dict(
        seed=SEED,
        validation_seed=VALIDATION_SEED,
        num_steps=NUM_STEPS,
        warmup_steps=WARMUP_STEPS,
        batch_size=BATCH_SIZE,
        validation_batches=NUM_VALIDATION_BATCHES,
        max_lr=3e-3,
        min_lr=3e-4,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        model_config=MODEL_CONFIG,
        train_tokens=len(train_data),
        validation_tokens=len(val_data),
        same_initial_parameters=same_initial_parameters,
        same_training_batches=same_training_batches,
        results=results,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if not same_initial_parameters or not same_training_batches:
        raise RuntimeError("两组实验的初始参数或训练批次不一致，请检查实验设置")

    print("\n初始参数一致：", same_initial_parameters)
    print("训练批次及顺序一致：", same_training_batches)
    for result in results:
        name = "有 RoPE" if result["use_rope"] else "无 RoPE"
        print(f"{name}：最终验证 loss={result['final_val_loss']:.4f}")
    difference = results[1]["final_val_loss"] - results[0]["final_val_loss"]
    print(f"无 RoPE − 有 RoPE 的验证 loss 差值：{difference:+.4f}")
    print("结果、曲线数据和存档已保存：", output_dir)


if __name__ == "__main__":
    main()
