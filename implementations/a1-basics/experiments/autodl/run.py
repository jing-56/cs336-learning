"""Run the archived CS336 A1 trainer on already encoded data."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST = json.loads((ROOT / "data-manifest.json").read_text(encoding="utf-8"))
PLAN = ROOT / "formal.json"


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_file(name, metadata, verify_hash):
    path = DATA / name
    if not path.is_file():
        raise FileNotFoundError(f"缺少数据文件 {path}")
    if path.stat().st_size != metadata["bytes"]:
        raise ValueError(f"文件长度与清单不符：{name}")
    if "tokens" in metadata:
        import numpy as np

        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.shape != (metadata["tokens"],) or array.dtype != np.dtype("<u2"):
            raise ValueError(f"数组形状或类型与清单不符：{name}")
    if verify_hash and sha256(path) != metadata["sha256"]:
        raise ValueError(f"SHA-256 与清单不符：{name}")
    return path


def check_gpu():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("当前 PyTorch 无法使用 CUDA，请检查 AutoDL 镜像和 nvidia-smi。")
    sample = torch.ones((64, 64), device="cuda")
    torch.testing.assert_close((sample @ sample)[0, 0].cpu(), torch.tensor(64.0))
    properties = torch.cuda.get_device_properties(0)
    result = dict(
        gpu=properties.name, capability=list(torch.cuda.get_device_capability(0)),
        vram_gib=round(properties.total_memory / 2**30, 2),
        torch=str(torch.__version__), cuda=torch.version.cuda,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def config_for(dataset):
    spec = MANIFEST[dataset]
    return dict(
        json.loads(PLAN.read_text(encoding="utf-8")),
        vocab_size=spec["vocab_size"],
        train_data=str(DATA / spec["train"]),
        val_data=str(DATA / spec["validation"]),
    )


@contextmanager
def run_lock(directory):
    # The production target is AutoDL Linux. The same wrapper can be smoke-tested on Windows.
    lock_dir = directory.parent / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / directory.name).open("a", encoding="utf-8") as lock:
        if sys.platform != "win32":
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def latest_checkpoint(directory):
    paths = sorted(directory.glob("checkpoint_step_*.pt"))
    for path in reversed(paths):
        match = re.fullmatch(r"checkpoint_step_(\d{6})\.pt", path.name)
        if match:
            return int(match[1]), path
    return 0, None


def repair_metrics(directory, saved_step, columns):
    path = directory / "metrics.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if reader.fieldnames != columns:
            raise ValueError("训练日志表头有误，请保留存档和日志检查。")
    boundary = next((i for i, row in enumerate(rows) if row["step"] == str(saved_step)), None)
    if boundary is None or any(value is None for row in rows[:boundary + 1] for value in row.values()):
        raise ValueError("找不到与存档步数匹配的完整日志，请保留原文件检查。")
    if boundary < len(rows) - 1:
        backup = directory / f"metrics.interrupted-{time.time_ns()}.csv"
        shutil.copyfile(path, backup)
        temporary = path.with_suffix(".csv.recover")
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows[:boundary + 1])
        temporary.replace(path)
        print(f"未存档的日志已另存：{backup.name}；从第 {saved_step} 步继续。", flush=True)


def invoke(main, arguments):
    old = sys.argv
    try:
        sys.argv = [old[0], *map(str, arguments)]
        main()
    finally:
        sys.argv = old


def training(dataset, directory, until):
    from cs336_basics import train_cli
    import torch

    spec = MANIFEST[dataset]
    config = config_for(dataset)
    until = config["steps"] if until is None else until
    if not 1 <= until <= config["steps"]:
        raise ValueError(f"--until 必须在 1 到 {config['steps']} 之间。")
    for name in (spec["train"], spec["validation"]):
        check_file(name, spec["files"][name], verify_hash=False)
    torch.set_float32_matmul_precision("highest")
    hardware = check_gpu() if config["device"] == "cuda" else {"gpu": "CPU"}
    os.environ.setdefault("PYTORCH_KERNEL_CACHE_PATH", str(ROOT / ".cuda_kernel_cache"))
    with run_lock(directory):
        step, checkpoint = latest_checkpoint(directory)
        if checkpoint:
            if json.loads((directory / "config.json").read_text(encoding="utf-8")) != config:
                raise ValueError("本地训练计划与已有运行不同，请指定新 --run-name。")
            if step >= until:
                print(f"已存档到第 {step} 步，达到本次目标。", flush=True)
                return
            repair_metrics(directory, step, train_cli.COLUMNS)
            command = ["--resume", checkpoint]
        else:
            if directory.exists() and any(directory.iterdir()):
                raise ValueError("第一次存档前训练中断。保留旧目录，再指定新 --run-name 重跑。")
            command = ["--config", PLAN, "--vocab-size", config["vocab_size"], "--train-data",
                       config["train_data"], "--val-data", config["val_data"], "--output-dir", directory]
        command.extend(["--run-steps", until - step])
        if dataset == "owt":
            from train_owt import get_batch

            train_cli.get_batch = get_batch
        original_fingerprint = train_cli.fingerprint

        def verified_fingerprint(path):
            actual = original_fingerprint(path)
            expected = spec["files"][Path(path).name]["sha256"]
            if actual != expected:
                raise ValueError(f"文件校验不通过：{Path(path).name}")
            return actual

        train_cli.fingerprint = verified_fingerprint
        if config["device"] == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            invoke(train_cli.main, command)
        finally:
            train_cli.fingerprint = original_fingerprint
        if until >= 200:
            with (directory / "metrics.csv").open(encoding="utf-8", newline="") as stream:
                rows = {int(row["step"]): row for row in csv.DictReader(stream)}
            if 100 in rows and 200 in rows and step <= 100:
                seconds = float(rows[200]["elapsed_seconds"]) - float(rows[100]["elapsed_seconds"])
                rate = 100 * config["batch_size"] * config["context_length"] / seconds
                report = dict(
                    hardware, from_step=100, to_step=200, seconds=seconds, tokens_per_second=rate,
                    estimated_total_training_minutes=config["steps"] * seconds / 100 / 60,
                    estimated_remaining_minutes=(config["steps"] - until) * seconds / 100 / 60,
                    peak_cuda_allocated_gib=torch.cuda.max_memory_allocated() / 2**30 if config["device"] == "cuda" else None,
                )
                train_cli.write_json(directory / "benchmark.json", report)
                print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def generation(dataset, directory, prompt):
    import torch
    from cs336_basics.decoding import generate
    from cs336_basics.tokenizer_demo import Tokenizer
    from cs336_basics.train_cli import ARCHITECTURE, write_json
    from cs336_basics.transformer_lm import TransformerLM

    spec = MANIFEST[dataset]
    step, path = latest_checkpoint(directory)
    if path is None:
        raise FileNotFoundError("请先完成至少一次训练存档。")
    tok_paths = [check_file(name, spec["files"][name], True) for name in spec["tokenizer"]]
    marker = "<|endoftext|>"
    if dataset == "tiny":
        tokenizer = Tokenizer.from_files(*tok_paths, special_tokens=[marker])
        prompt = prompt or "Once upon a time"
    else:
        payload = pickle.loads(tok_paths[0].read_bytes())
        tokenizer = Tokenizer(payload["vocab"], payload["merges"], special_tokens=[marker])
        prompt = prompt or "The city council announced on Tuesday that"
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["architecture"] != ARCHITECTURE or len(tokenizer.vocab) != checkpoint["model_config"]["vocab_size"]:
        raise ValueError("模型与词表不匹配。")
    device = checkpoint["config"]["device"]
    model = TransformerLM(**checkpoint["model_config"], device=device, dtype=torch.float32)
    model.load_state_dict(checkpoint["model"])
    del checkpoint
    result = generate(model, tokenizer.encode(prompt), 256, 0.8, 0.9, tokenizer.encode(marker)[0], 42)
    result.update(prompt=prompt, checkpoint_step=step, seed=42, temperature=0.8, top_p=0.9,
                  full_text=tokenizer.decode(result["token_ids"]))
    output = directory / f"generation-{step:06d}-{time.time_ns()}.json"
    write_json(output, result)
    print(result["full_text"], "\n", output, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "verify", "train", "generate", "evaluate"))
    parser.add_argument("dataset", nargs="?", choices=tuple(MANIFEST))
    parser.add_argument("--until", type=int, help="训练到指定总步数；省略则继续到计划末尾。")
    parser.add_argument("--run-name", help="另建运行时指定名称，默认使用数据集名称。")
    parser.add_argument("--prompt", help="覆盖生成提示词。")
    args = parser.parse_args()
    if args.command == "check":
        check_gpu()
        return
    if not args.dataset:
        parser.error("请选择 tiny 或 owt。")
    spec = MANIFEST[args.dataset]
    if args.command == "verify":
        for name, metadata in spec["files"].items():
            check_file(name, metadata, True)
            print(f"已核对 {name}", flush=True)
        return
    name = args.run_name or args.dataset
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
        parser.error("--run-name 只允许字母、数字、下划线和连字符，并以字母或数字开头。")
    directory = ROOT / "runs" / name
    if args.command == "train":
        training(args.dataset, directory, args.until)
    elif args.command == "generate":
        generation(args.dataset, directory, args.prompt)
    else:
        from evaluate_full import main as evaluate

        step, checkpoint = latest_checkpoint(directory)
        if checkpoint is None:
            raise FileNotFoundError("请先完成至少一次训练存档。")
        invoke(evaluate, ["--checkpoint", checkpoint, "--validation", DATA / spec["validation"],
                          "--output", directory / f"full-eval-{step:06d}.json", "--batch-size", 16,
                          "--device", config_for(args.dataset)["device"]])


if __name__ == "__main__":
    main()
