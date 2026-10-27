"""复用已完成的 RoPE 实验，补齐种子 42、43、44，并汇总配对差值。"""

import csv
from datetime import datetime
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch

from cs336_basics import train_rope_ablation as experiment


SEEDS = (42, 43, 44)


def expected_settings(train_data, val_data):
    return dict(
        validation_seed=experiment.VALIDATION_SEED,
        num_steps=experiment.NUM_STEPS,
        warmup_steps=experiment.WARMUP_STEPS,
        batch_size=experiment.BATCH_SIZE,
        validation_batches=experiment.NUM_VALIDATION_BATCHES,
        max_lr=3e-3,
        min_lr=3e-4,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        model_config=experiment.MODEL_CONFIG,
        train_tokens=len(train_data),
        validation_tokens=len(val_data),
    )


def find_completed_run(root, seed, settings):
    # 只复用配置一致、配对检查通过、曲线记录完整的结果。
    for path in sorted(root.glob("*/summary.json"), reverse=True):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["seed"] != seed:
                continue
            if any(record.get(key) != value for key, value in settings.items()):
                continue
            results = record["results"]
            if len(results) != 2 or [r["use_rope"] for r in results] != [True, False]:
                continue
            if results[0]["initial_parameter_hash"] != results[1]["initial_parameter_hash"]:
                continue
            if results[0]["training_batches_hash"] != results[1]["training_batches_hash"]:
                continue
            for result, name in zip(results, ("with_rope", "without_rope")):
                with (path.parent / f"{name}_loss.csv").open(
                    encoding="utf-8", newline="",
                ) as file:
                    rows = list(csv.DictReader(file))
                if int(rows[-1]["step"]) != settings["num_steps"]:
                    raise ValueError("曲线没有完成全部训练步数")
                if float(rows[-1]["val_loss"]) != result["final_val_loss"]:
                    raise ValueError("曲线与汇总记录不一致")
            return record, path
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            continue
    return None


def main():
    torch.set_num_threads(4)
    folder = Path(__file__).resolve().parent
    root = folder / "rope_ablation_runs"
    train_data = np.load(folder / "tinystories_5m_train.npy", mmap_mode="r")
    val_data = np.load(folder / "tinystories_5m_val.npy", mmap_mode="r")
    settings = expected_settings(train_data, val_data)

    # 所有种子共用同一组验证批次；训练随机数在每个版本开始时重置。
    np.random.seed(experiment.VALIDATION_SEED)
    val_batches = [
        experiment.get_batch(
            val_data, experiment.BATCH_SIZE, experiment.CONTEXT_LENGTH, "cpu",
        )
        for _ in range(experiment.NUM_VALIDATION_BATCHES)
    ]

    records = []
    for seed in SEEDS:
        completed = find_completed_run(root, seed, settings)
        if completed is not None:
            record, source = completed
            print(f"\n复用种子 {seed} 的已完成记录：{source}", flush=True)
        else:
            if seed == 42:
                raise RuntimeError("未找到配置一致的种子 42 记录，请先检查原实验")
            experiment.SEED = seed
            name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + f"_seed{seed}"
            output_dir = root / name
            output_dir.mkdir(parents=True, exist_ok=False)
            print(f"\n===== 开始种子 {seed} 的两组实验 =====", flush=True)
            results = [
                experiment.run_case(True, train_data, val_batches, output_dir),
                experiment.run_case(False, train_data, val_batches, output_dir),
            ]
            same_parameters = (
                results[0]["initial_parameter_hash"] == results[1]["initial_parameter_hash"]
            )
            same_batches = (
                results[0]["training_batches_hash"] == results[1]["training_batches_hash"]
            )
            record = dict(
                settings, seed=seed, same_initial_parameters=same_parameters,
                same_training_batches=same_batches, results=results,
            )
            source = output_dir / "summary.json"
            source.write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            if not same_parameters or not same_batches:
                raise RuntimeError(f"种子 {seed} 的配对条件不一致，停止汇总")

        with_loss = record["results"][0]["final_val_loss"]
        without_loss = record["results"][1]["final_val_loss"]
        records.append(dict(
            seed=seed, with_rope=with_loss, without_rope=without_loss,
            gap=without_loss - with_loss, source_summary=str(source),
        ))

    gaps = [row["gap"] for row in records]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = root / f"multiseed_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(
        seeds=list(SEEDS), settings=settings, results=records,
        mean_with_rope=mean(row["with_rope"] for row in records),
        mean_without_rope=mean(row["without_rope"] for row in records),
        mean_paired_gap=mean(gaps), sample_std_paired_gap=stdev(gaps),
        positive_gap_count=sum(gap > 0 for gap in gaps),
    )
    (output_dir / "multi_seed_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    with (output_dir / "multi_seed_results.csv").open(
        "w", encoding="utf-8", newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\n===== 三个种子的配对结果 =====")
    for row in records:
        print(
            f"seed={row['seed']}：有 RoPE={row['with_rope']:.4f}，"
            f"无 RoPE={row['without_rope']:.4f}，差值={row['gap']:+.4f}"
        )
    print(f"平均差值：{mean(gaps):+.4f}")
    print(f"差值的样本标准差：{stdev(gaps):.4f}")
    print(f"RoPE 的验证 loss 更低：{report['positive_gap_count']}/{len(SEEDS)} 个种子")
    print("标准差描述这几个种子之间的波动，不是置信区间。")
    print("汇总已保存：", output_dir)


if __name__ == "__main__":
    main()
