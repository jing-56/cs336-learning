"""固定训练 token 预算，比较每批 2、4、8 段文本。可在 PyCharm 直接运行。"""

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cs336_basics import train_rope_ablation as base


SEED = 42
BATCH_SIZES = (2, 4, 8)
NUM_EXAMPLES = 4000
CONTEXT_LENGTH = base.CONTEXT_LENGTH
TOTAL_TOKENS = NUM_EXAMPLES * CONTEXT_LENGTH
WARMUP_TOKENS = 50 * 4 * CONTEXT_LENGTH
EVAL_EVERY_EXAMPLES = 400
MAX_LR = 0.003
MIN_LR = 0.0003


def update_example_hash(digest, x, y):
    # 每段输入后接它的目标；指纹不受怎样划分批次影响。
    for x_row, y_row in zip(x, y):
        digest.update(x_row.numpy().tobytes())
        digest.update(y_row.numpy().tobytes())


def write_curve(path, rows):
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "step", "processed_tokens", "lr", "train_loss", "val_loss",
        ])
        writer.writeheader()
        writer.writerows(rows)


def run_case(batch_size, x_stream, y_stream, val_batches, output_dir,
             expected_initial_hash, expected_stream_hash, warmup_tokens,
             eval_every_examples):
    num_examples = len(x_stream)
    if num_examples % batch_size or eval_every_examples % batch_size:
        raise ValueError("例子总数和检查间隔必须能被 batch size 整除")
    total_tokens = num_examples * CONTEXT_LENGTH
    num_steps = num_examples // batch_size
    torch.manual_seed(SEED)
    model = base.TransformerLM(**base.MODEL_CONFIG)
    initial_hash = base.parameter_hash(model)
    if initial_hash != expected_initial_hash:
        raise RuntimeError("初始参数不一致")
    optimizer = base.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.01)

    evaluation_seconds = 0.0

    def evaluate():
        nonlocal evaluation_seconds
        start = perf_counter()
        value = base.evaluate_loss(model, val_batches)
        evaluation_seconds += perf_counter() - start
        if not math.isfinite(value):
            raise RuntimeError("验证 loss 出现非有限数值")
        return value

    initial_loss = evaluate()
    rows = [dict(step=0, processed_tokens=0, lr=None,
                 train_loss=None, val_loss=initial_loss)]
    stream_digest = hashlib.sha256()
    batch_digest = hashlib.sha256()
    training_seconds = 0.0
    model.train()
    print(f"\n===== batch_size={batch_size}，共 {num_steps} 次更新 =====", flush=True)

    for step, start_row in enumerate(range(0, num_examples, batch_size), start=1):
        end_row = start_row + batch_size
        x, y = x_stream[start_row:end_row], y_stream[start_row:end_row]
        update_example_hash(stream_digest, x, y)
        batch_digest.update(x.numpy().tobytes())
        batch_digest.update(y.numpy().tobytes())
        processed_tokens = end_row * CONTEXT_LENGTH

        start = perf_counter()
        # 调度函数接收训练进度：这里把单位从“更新次数”换成“token 数”。
        lr = base.get_lr_cosine_schedule(
            it=processed_tokens, max_learning_rate=MAX_LR,
            min_learning_rate=MIN_LR, warmup_iters=warmup_tokens,
            cosine_cycle_iters=total_tokens,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad()
        loss = base.cross_entropy(model(x), y)
        train_loss = loss.item()
        if not math.isfinite(train_loss):
            raise RuntimeError(f"batch_size={batch_size} 第 {step} 步 loss 非有限")
        loss.backward()
        base.gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        training_seconds += perf_counter() - start

        if step == 1 or end_row % eval_every_examples == 0 or end_row == num_examples:
            val_loss = evaluate()
            rows.append(dict(step=step, processed_tokens=processed_tokens,
                             lr=lr, train_loss=train_loss, val_loss=val_loss))
            print(f"[batch={batch_size}] step={step:4d}，token={processed_tokens:6,d}，"
                  f"训练 loss={train_loss:.4f}，验证 loss={val_loss:.4f}", flush=True)
            write_curve(output_dir / f"batch_{batch_size}_loss.csv", rows)

    if stream_digest.hexdigest() != expected_stream_hash:
        raise RuntimeError("实际使用的文本及顺序与计划不一致")
    checkpoint = output_dir / f"batch_{batch_size}_checkpoint.pt"
    base.save_checkpoint(model, optimizer, num_steps, checkpoint)
    return dict(
        batch_size=batch_size, completed_steps=num_steps,
        processed_tokens=total_tokens, initial_parameter_hash=initial_hash,
        used_example_stream_hash=stream_digest.hexdigest(),
        used_batch_grouping_hash=batch_digest.hexdigest(),
        initial_val_loss=initial_loss, final_val_loss=rows[-1]["val_loss"],
        training_seconds=training_seconds, evaluation_seconds=evaluation_seconds,
        training_tokens_per_second=total_tokens / training_seconds,
        curve=str(output_dir / f"batch_{batch_size}_loss.csv"),
        checkpoint=str(checkpoint),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="仅用 8 段文本检查实验流程")
    args = parser.parse_args()
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")
    num_examples = 8 if args.smoke else NUM_EXAMPLES
    warmup_tokens = int(num_examples * CONTEXT_LENGTH * 0.05)
    eval_every = 8 if args.smoke else EVAL_EVERY_EXAMPLES
    if not args.smoke:
        assert num_examples * CONTEXT_LENGTH == TOTAL_TOKENS
        assert warmup_tokens == WARMUP_TOKENS

    # 验证批次保持原来的大小 4；它不随训练 batch size 改变。
    np.random.seed(base.VALIDATION_SEED)
    val_batches = [base.get_batch(val_data, 4, CONTEXT_LENGTH, "cpu")
                   for _ in range(base.NUM_VALIDATION_BATCHES)]
    np.random.seed(SEED)
    # 先按旧基线的方式抽样，再把 4000 段文本按原顺序排好。
    original_batches = [base.get_batch(train_data, 4, CONTEXT_LENGTH, "cpu")
                        for _ in range(num_examples // 4)]
    x_stream = torch.cat([x for x, _ in original_batches])
    y_stream = torch.cat([y for _, y in original_batches])
    stream_digest = hashlib.sha256()
    update_example_hash(stream_digest, x_stream, y_stream)
    stream_hash = stream_digest.hexdigest()
    original_digest = hashlib.sha256()
    for x, y in original_batches:
        original_digest.update(x.numpy().tobytes())
        original_digest.update(y.numpy().tobytes())
    del original_batches
    torch.manual_seed(SEED)
    reference_model = base.TransformerLM(**base.MODEL_CONFIG)
    initial_hash = base.parameter_hash(reference_model)
    del reference_model

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{timestamp}_seed{SEED}" + ("_smoke" if args.smoke else "")
    output_dir = folder / "batch_size_runs" / run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    settings = dict(
        seed=SEED, validation_seed=base.VALIDATION_SEED,
        model_config=base.MODEL_CONFIG, batch_sizes=list(BATCH_SIZES),
        num_examples=num_examples, total_tokens=num_examples * CONTEXT_LENGTH,
        warmup_tokens=warmup_tokens, eval_every_examples=eval_every,
        max_lr=MAX_LR, min_lr=MIN_LR, weight_decay=0.01,
        gradient_clip_norm=1.0, betas=[0.9, 0.999], eps=1e-8,
        validation_batch_size=4, validation_batches=base.NUM_VALIDATION_BATCHES,
        train_dataset_tokens=len(train_data), val_dataset_tokens=len(val_data),
        torch_threads=4, torch_version=torch.__version__, smoke=args.smoke,
        initial_parameter_hash=initial_hash, example_stream_hash=stream_hash,
        reference_batch4_grouping_hash=original_digest.hexdigest(),
        timing_scope="training_seconds: LR, zero_grad, forward, backward, clipping, optimizer; excludes validation, hashing, checkpoint IO",
    )
    results = []
    for batch_size in BATCH_SIZES:
        result = run_case(batch_size, x_stream, y_stream, val_batches, output_dir,
                          initial_hash, stream_hash, warmup_tokens, eval_every)
        if batch_size == 4 and result["used_batch_grouping_hash"] != original_digest.hexdigest():
            raise RuntimeError("batch=4 与旧基线的分组不同")
        results.append(result)
        report = dict(settings=settings, results=results,
                      same_initial_parameters=len({r["initial_parameter_hash"] for r in results}) == 1,
                      same_examples_and_order=len({r["used_example_stream_hash"] for r in results}) == 1)
        (output_dir / "batch_size_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8",
        )

    print("\n===== 批次大小实验结果 =====", flush=True)
    for result in results:
        print(f"batch={result['batch_size']}，更新 {result['completed_steps']} 次，"
              f"验证 loss={result['final_val_loss']:.4f}，"
              f"训练计算耗时={result['training_seconds']:.2f} 秒", flush=True)
    print("初始参数、文本及顺序一致；每组训练 token 数：", settings["total_tokens"], flush=True)
    print("配置、曲线及存档位置：", output_dir, flush=True)


if __name__ == "__main__":
    main()
