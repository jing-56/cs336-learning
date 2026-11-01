"""Fork a train_cli checkpoint into a new directory on another device."""

import argparse
import csv
from datetime import datetime
import hashlib
from pathlib import Path

import torch

from cs336_basics.train_cli import ARCHITECTURE, COLUMNS, TrainConfig, write_json


def migrate(src, device, output_dir):
    src = Path(src).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    payload = torch.load(src, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("architecture") != ARCHITECTURE:
        raise ValueError("Requires a baseline checkpoint created by train_cli")
    original_device = payload["config"]["device"]
    config = TrainConfig(**{**payload["config"], "device": device})
    config.validate()
    iteration = payload["iteration"]
    metrics_path = src.parent / "metrics.csv"
    with metrics_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != COLUMNS:
            raise ValueError("Unexpected metrics schema")
        rows = [row for row in reader if int(row["step"]) <= iteration]
    if not rows or int(rows[-1]["step"]) != iteration:
        raise ValueError("No metrics row matching the selected checkpoint")
    new_cuda_rng = False
    if device == "cuda":
        if not payload["rng"]["cuda"]:
            torch.cuda.manual_seed_all(config.seed)
            payload["rng"]["cuda"] = torch.cuda.get_rng_state_all()
            new_cuda_rng = True
        elif len(payload["rng"]["cuda"]) != torch.cuda.device_count():
            raise ValueError("Saved CUDA RNG count differs from the available GPU count")
    else:
        payload["rng"]["cuda"] = []
    payload["config"]["device"] = device
    migration = {
        "source_checkpoint": str(src),
        "source_checkpoint_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
        "source_device": original_device,
        "destination_device": device,
        "completed_steps_at_migration": iteration,
        "model_and_optimizer_preserved": True,
        "python_numpy_cpu_torch_rng_preserved": True,
        "cuda_rng_initialized_from_seed": config.seed if new_cuda_rng else None,
        "timing_policy": "CSV includes source history; subsequent elapsed_seconds accumulates active training time across devices, excluding migration downtime",
        "cross_device_bitwise_equality_expected": False,
        "torch_version": str(torch.__version__),
    }
    payload["device_migration"] = migration
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_json(output_dir / "config.json", payload["config"])
    destination = output_dir / f"checkpoint_migrated_step_{iteration:06d}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    # CPU serialized tensors are moved to model/optimizer devices by train_cli.
    torch.save(payload, temporary)
    temporary.replace(destination)
    write_json(output_dir / "device_migration.json", migration)
    write_json(output_dir / "summary.json", dict(status="ready_to_resume_on_new_device", completed_steps=iteration,
               total_steps=config.steps, processed_tokens=iteration*config.batch_size*config.context_length,
               elapsed_seconds=payload["elapsed_seconds"], best_val_loss=payload["best_val_loss"], checkpoint=str(destination)))
    provenance = src.parent / "tokenizer_provenance.json"
    if provenance.exists():
        (output_dir / provenance.name).write_bytes(provenance.read_bytes())
    print(f"新运行目录：{output_dir}\n迁移存档：{destination}\n进度：第 {iteration} 步，{original_device} → {device}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent/"training_runs"/(datetime.now().strftime("%Y%m%d_%H%M%S_%f")+"_device_migration"))
    args = parser.parse_args()
    migrate(args.checkpoint, args.device, args.output_dir)
