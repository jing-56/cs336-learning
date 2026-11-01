"""可配置的基线训练入口；直接运行默认先训练 100 步，整个计划为 1000 步。"""

import argparse
import csv
from dataclasses import asdict, dataclass, fields
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
from time import perf_counter

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.lr_schedule import get_lr_cosine_schedule
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW
from cs336_basics.transformer_lm import TransformerLM


ARCHITECTURE = "pre_norm_swiglu_rope"
COLUMNS = ["step", "processed_tokens", "elapsed_seconds", "lr", "train_loss", "val_loss", "grad_norm"]


@dataclass
class TrainConfig:
    train_data: str = ""
    val_data: str = ""
    vocab_size: int = 10000
    context_length: int = 64
    d_model: int = 64
    num_layers: int = 2
    num_heads: int = 4
    d_ff: int = 128
    rope_theta: float = 10000.0
    batch_size: int = 4
    steps: int = 1000
    warmup_steps: int = 50
    max_lr: float = 0.003
    min_lr: float = 0.0003
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 20
    eval_batches: int = 10
    eval_batch_size: int = 4
    save_every: int = 100
    seed: int = 42
    val_seed: int = 12345
    threads: int = 4
    device: str = "cpu"

    def model_config(self):
        return {key: getattr(self, key) for key in (
            "vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff", "rope_theta",
        )}

    def validate(self):
        positive = ("vocab_size", "context_length", "d_model", "num_layers", "num_heads", "d_ff",
                    "batch_size", "steps", "eval_every", "eval_batches", "eval_batch_size", "save_every", "threads")
        if any(type(getattr(self, key)) is not int or getattr(self, key) <= 0 for key in positive):
            raise ValueError("尺寸、次数和线程数必须为正整数")
        if self.d_model % self.num_heads or (self.d_model // self.num_heads) % 2:
            raise ValueError("d_model 必须整除头数，且 RoPE 的每头维度必须为偶数")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps 必须在 [0, steps) 内")
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{field.name} 必须为有限数")
        if not 0 <= self.min_lr <= self.max_lr or self.max_lr <= 0:
            raise ValueError("学习率必须满足 0 <= min_lr <= max_lr 且 max_lr > 0")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("beta 必须在 [0,1) 内")
        if self.eps <= 0 or self.rope_theta <= 0 or self.weight_decay < 0 or self.grad_clip < 0:
            raise ValueError("eps/theta 必须为正，weight_decay/grad_clip 不能为负")
        if self.device not in ("cpu", "cuda") or (self.device == "cuda" and not torch.cuda.is_available()):
            raise ValueError("device 应为 cpu 或当前可用的 cuda")


def fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def rng_state():
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return dict(python=random.getstate(), torch=torch.get_rng_state(),
                numpy=dict(name=name, keys=torch.tensor(keys.astype(np.int64)), pos=pos,
                           has_gauss=has_gauss, cached=cached),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    n = state["numpy"]
    np.random.set_state((n["name"], n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["cached"]))
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


@torch.no_grad()
def evaluate(model, batches):
    training = model.training
    model.eval()
    try:
        return sum(cross_entropy(model(x), y).item() for x, y in batches) / len(batches)
    finally:
        model.train(training)


def load_tokens(path, config):
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    if data.ndim != 1 or not np.issubdtype(data.dtype, np.integer) or len(data) <= config.context_length:
        raise ValueError(f"{path} 必须是一维整数 .npy，长度大于上下文长度")
    if data.min() < 0 or data.max() >= config.vocab_size:
        raise ValueError(f"{path} 的 token ID 超出词表范围")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="JSON 配置，可被显式命令行参数覆盖")
    parser.add_argument("--resume", type=Path, help="恢复本入口生成的存档，读取存档中的训练计划")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-steps", type=int, default=100, help="本次最多执行的步数，默认 100")
    for field in fields(TrainConfig):
        parser.add_argument("--" + field.name.replace("_", "-"), dest=field.name,
                            type=type(field.default), default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    folder = Path(__file__).resolve().parent
    checkpoint = None
    if args.resume:
        if args.config:
            raise ValueError("恢复时直接使用存档配置，不同时指定 --config")
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint.get("format_version") != 1 or checkpoint.get("architecture") != ARCHITECTURE:
            raise ValueError("恢复训练需要本入口生成的基线存档；旧存档仍使用原来的脚本")
        config = TrainConfig(**checkpoint["config"])
        for field in fields(config):
            override = getattr(args, field.name)
            if override is not None and override != getattr(config, field.name):
                raise ValueError(f"恢复时不能改变 {field.name}；新实验应新建运行目录")
        run_dir = args.resume.resolve().parent
        if args.output_dir and args.output_dir.resolve() != run_dir:
            raise ValueError("本入口在原运行目录续写日志，output-dir 必须与存档目录一致")
    else:
        values = dict(train_data=str(folder / "tinystories_5m_train.npy"),
                      val_data=str(folder / "tinystories_5m_val.npy"))
        if args.config:
            values.update(json.loads(args.config.read_text(encoding="utf-8")))
        values.update({field.name: getattr(args, field.name) for field in fields(TrainConfig)
                       if getattr(args, field.name) is not None})
        config = TrainConfig(**values)
        run_dir = args.output_dir or folder / "training_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_dir = run_dir.resolve()
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("新运行的输出目录必须为空；已有运行请用 --resume")
    if args.run_steps <= 0:
        raise ValueError("run-steps 必须为正数")
    config.train_data = str(Path(config.train_data).resolve())
    config.val_data = str(Path(config.val_data).resolve())
    config.validate()
    if config.train_data == config.val_data:
        raise ValueError("训练集和验证集不能是同一个文件")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_data, val_data = load_tokens(config.train_data, config), load_tokens(config.val_data, config)
    data_hashes = {"train": fingerprint(config.train_data), "val": fingerprint(config.val_data)}
    if checkpoint and checkpoint["data_hashes"] != data_hashes:
        raise ValueError("数据内容与存档不一致")
    if config.device == "cuda":
        # Use a writable project-local cache for CUDA JIT elementwise kernels.
        # Respect an explicitly configured cache directory.
        kernel_cache = Path(os.environ.get("PYTORCH_KERNEL_CACHE_PATH", str(folder.parent / ".cuda_kernel_cache")))
        kernel_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(kernel_cache))
    model = TransformerLM(**config.model_config(), device=config.device, dtype=torch.float32)
    optimizer = AdamW(model.parameters(), lr=config.max_lr, betas=(config.beta1, config.beta2),
                      eps=config.eps, weight_decay=config.weight_decay)
    # 固定验证材料，抽取过程不改变训练数据的随机流。
    previous_np_state = np.random.get_state()
    np.random.seed(config.val_seed)
    val_batches = [get_batch(val_data, config.eval_batch_size, config.context_length, config.device)
                   for _ in range(config.eval_batches)]
    np.random.set_state(previous_np_state)
    start_step, previous_elapsed, best_val = 0, 0.0, float("inf")
    metrics_path = run_dir / "metrics.csv"
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["iteration"]
        previous_elapsed = checkpoint["elapsed_seconds"]
        best_val = checkpoint["best_val_loss"]
        with metrics_path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
            if reader.fieldnames != COLUMNS or not rows or int(rows[-1]["step"]) != start_step:
                raise ValueError("日志末尾与存档步数不一致，请使用最新匹配存档，保留现有日志")
        restore_rng(checkpoint["rng"])
    run_dir.mkdir(parents=True, exist_ok=True)
    if not checkpoint:
        write_json(run_dir / "config.json", asdict(config))
    end_step = min(config.steps, start_step + args.run_steps)
    print(f"运行目录：{run_dir}\n本次：{start_step} → {end_step} 步；总计划 {config.steps} 步", flush=True)
    if start_step >= config.steps:
        print("已完成整个计划，保留现有结果。", flush=True)
        return
    session_start = perf_counter()

    def elapsed():
        return previous_elapsed + perf_counter() - session_start

    mode = "a" if checkpoint else "w"
    with metrics_path.open(mode, encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        if not checkpoint:
            writer.writeheader()

        def log(step, lr, train_loss, grad_norm):
            nonlocal best_val
            val_loss = evaluate(model, val_batches)
            if not math.isfinite(val_loss):
                raise RuntimeError(f"第 {step} 步验证 loss 非有限，停止以保留先前存档")
            best_val = min(best_val, val_loss)
            writer.writerow(dict(step=step, processed_tokens=step * config.batch_size * config.context_length,
                                 elapsed_seconds=elapsed(), lr=lr, train_loss=train_loss,
                                 val_loss=val_loss, grad_norm=grad_norm))
            file.flush()
            print(f"step={step:5d}，累计秒={elapsed():.2f}，lr={lr:.6f}，验证 loss={val_loss:.4f}", flush=True)

        if not checkpoint:
            log(0, 0.0, None, None)
        model.train()
        for step in range(start_step + 1, end_step + 1):
            x, y = get_batch(train_data, config.batch_size, config.context_length, config.device)
            lr = get_lr_cosine_schedule(step, config.max_lr, config.min_lr, config.warmup_steps, config.steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad()
            loss = cross_entropy(model(x), y)
            if not torch.isfinite(loss).item():
                raise RuntimeError(f"第 {step} 步训练 loss 非有限")
            loss.backward()
            norm_squared = sum(p.grad.float().square().sum() for p in model.parameters() if p.grad is not None)
            if not torch.isfinite(norm_squared).item():
                raise RuntimeError(f"第 {step} 步梯度范数非有限")
            grad_norm = math.sqrt(norm_squared.item())
            gradient_clipping(model.parameters(), config.grad_clip)
            optimizer.step()
            save_now = step % config.save_every == 0 or step == end_step
            if step == 1 or step % config.eval_every == 0 or save_now:
                log(step, lr, loss.item(), grad_norm)
            if save_now:
                destination = run_dir / f"checkpoint_step_{step:06d}.pt"
                if destination.exists():
                    raise FileExistsError(f"保留已有存档，拒绝覆盖：{destination}")
                payload = dict(format_version=1, architecture=ARCHITECTURE,
                               config=asdict(config), model_config=config.model_config(),
                               model=model.state_dict(), optimizer=optimizer.state_dict(), iteration=step,
                               elapsed_seconds=elapsed(), best_val_loss=best_val, rng=rng_state(),
                               data_hashes=data_hashes)
                temporary = destination.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                temporary.replace(destination)
                write_json(run_dir / "summary.json", dict(
                    status="completed" if step == config.steps else "paused_between_sessions",
                    completed_steps=step, total_steps=config.steps,
                    processed_tokens=step * config.batch_size * config.context_length,
                    elapsed_seconds=elapsed(), best_val_loss=best_val, checkpoint=str(destination),
                ))
                print(f"已保存：{destination}", flush=True)


if __name__ == "__main__":
    main()
