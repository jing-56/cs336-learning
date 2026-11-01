"""Measure the existing BPE trainer; defaults to TinyStories validation for debugging."""

import argparse
import cProfile
import hashlib
import io
import json
import pickle
import pstats
import threading
from datetime import datetime
from pathlib import Path
from time import perf_counter

import psutil

from cs336_basics.bpe import train_bpe
from cs336_basics.tokenizer_demo import Tokenizer


def monitored_training(input_path, vocab_size, output_dir):
    process = psutil.Process()
    stop = threading.Event()
    started = perf_counter()
    io_before = process.io_counters().read_bytes
    log_path = output_dir / "progress.jsonl"

    def report(status):
        memory = process.memory_info()
        cpu = process.cpu_times()
        row = {
            "status": status,
            "elapsed_seconds": perf_counter() - started,
            "process_read_bytes_since_start": process.io_counters().read_bytes - io_before,
            "process_cpu_seconds": cpu.user + cpu.system,
            "rss_bytes": memory.rss,
            "windows_peak_working_set_bytes": getattr(memory, "peak_wset", None),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"运行 {row['elapsed_seconds'] / 60:.1f} 分钟，当前内存 {memory.rss / 2**20:.1f} MiB，进程读取量 {row['process_read_bytes_since_start'] / 1e9:.2f} GB", flush=True)

    def monitor():
        while not stop.wait(30):
            report("running")

    report("started")
    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        result = train_bpe(input_path, vocab_size, ["<|endoftext|>"])
    except BaseException:
        stop.set()
        thread.join()
        report("failed_or_interrupted")
        raise
    stop.set()
    thread.join()
    report("completed")
    return result


def run(input_path, output_dir, vocab_size, split, profile=True):
    input_path = Path(input_path).resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    process = psutil.Process()
    before = process.memory_info().rss
    print(f"语料：{input_path}，{input_path.stat().st_size:,} 字节", flush=True)
    print(f"用途：{split}；开始不带 profiler 的计时运行……", flush=True)
    started = perf_counter()
    vocab, merges = monitored_training(input_path, vocab_size, output_dir)
    elapsed = perf_counter() - started
    memory = process.memory_info()
    # Windows process lifetime peak, sampled immediately after the unprofiled run.
    peak = getattr(memory, "peak_wset", None)
    special = b"<|endoftext|>"
    longest = max((value for value in vocab.values() if value != special), key=lambda value: (len(value), value))
    tokenizer = Tokenizer(vocab, merges, ["<|endoftext|>"])
    example = "A robot opened the window and watched the stars. 你好！"
    ids = tokenizer.encode(example)
    roundtrip = tokenizer.decode(ids) == example
    if not roundtrip:
        raise AssertionError("编码未能完整还原")
    for name, value in (("vocab.pkl", vocab), ("merges.pkl", merges)):
        with (output_dir / name).open("wb") as stream:
            pickle.dump(value, stream)
    print(f"词表 {len(vocab)}，合并 {len(merges)}，耗时 {elapsed:.3f} 秒", flush=True)
    print(f"最长普通 token：{longest!r}，{len(longest)} 字节", flush=True)
    profiled_elapsed = None
    profile_matches = None
    if profile:
        print("开始另一次带 profiler 的运行（不混入上方耗时和内存）……", flush=True)
        profiler = cProfile.Profile()
        started = perf_counter()
        profiled_vocab, profiled_merges = profiler.runcall(train_bpe, input_path, vocab_size, ["<|endoftext|>"])
        profiled_elapsed = perf_counter() - started
        if (profiled_vocab, profiled_merges) != (vocab, merges):
            raise AssertionError("两次 BPE 训练输出不一致")
        profiler.dump_stats(str(output_dir / "bpe.prof"))
        buffer = io.StringIO()
        pstats.Stats(profiler, stream=buffer).strip_dirs().sort_stats("cumulative").print_stats(15)
        (output_dir / "profile.txt").write_text(buffer.getvalue(), encoding="utf-8")
        profile_matches = True
    digest = hashlib.sha256()
    with input_path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    result = {
        "split_usage": split,
        "input_filename": input_path.name,
        "input_bytes": input_path.stat().st_size,
        "input_sha256": digest.hexdigest(),
        "target_vocab_size": vocab_size,
        "actual_vocab_size": len(vocab),
        "merges": len(merges),
        "special_tokens": ["<|endoftext|>"],
        "training_seconds_without_profiler": elapsed,
        "training_seconds_with_profiler": profiled_elapsed,
        "profiler_enabled": profile,
        "bpe_source_sha256": hashlib.sha256(Path(__file__).with_name("bpe.py").read_bytes()).hexdigest(),
        "process_rss_before_bytes": before,
        "process_rss_after_bytes": memory.rss,
        "windows_process_lifetime_peak_before_profiling_bytes": peak,
        "longest_normal_token_hex": longest.hex(),
        "longest_normal_token_repr": repr(longest),
        "longest_normal_token_bytes": len(longest),
        "roundtrip_example": example,
        "roundtrip": roundtrip,
        "profiled_and_unprofiled_vocab_merges_identical": profile_matches,
    }
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"结果已保存：{output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=root / "data/TinyStoriesV2-GPT4-valid.txt")
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--split", default="validation_debug_only")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "bpe_corpus_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--skip-profile", action="store_true", help="Run training once without an additional cProfile run")
    args = parser.parse_args()
    run(args.input, args.output_dir, args.vocab_size, args.split, profile=not args.skip_profile)
