"""Profile one full BPE run and check it against the saved normal-run output."""

import argparse
import builtins
import cProfile
import hashlib
import json
import pickle
import pstats
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import psutil

from cs336_basics import bpe


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True, help="Archived JSON with expected source and output hashes.")
    parser.add_argument("--baseline", type=Path, required=True, help="Directory with trusted local vocab.pkl and merges.pkl.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = json.loads(args.reference.read_text(encoding="utf-8"))
    original = reference["summary"]
    source_sha = sha256(Path(bpe.__file__))
    input_sha = sha256(args.input)
    if source_sha != original["bpe_source_sha256"] or input_sha != original["input_sha256"]:
        raise ValueError("Input or BPE code differs from the saved run.")
    expected = []
    for name in ("vocab.pkl", "merges.pkl"):
        path = args.baseline / name
        if sha256(path) != reference["artifact_sha256"][name]:
            raise ValueError(f"Baseline artifact mismatch: {name}")
        with path.open("rb") as stream:
            expected.append(pickle.load(stream))
    args.output.mkdir(parents=True, exist_ok=False)

    process = psutil.Process()
    stop = Event()
    state = dict(phase="pretokenization", merges=0, expected_merges=original["merges"])
    phases = dict(pretokenization=0.0, prepare_pairs=0.0, select_pair=0.0, apply_merge=0.0)
    pretokenize = bpe.pretokenize_file_counts
    last_selection_end = None
    pretokenization_end = None
    started = perf_counter()

    def timed_pretokenize(*args, **kwargs):
        nonlocal pretokenization_end
        begin = perf_counter()
        counts = pretokenize(*args, **kwargs)
        pretokenization_end = perf_counter()
        phases["pretokenization"] = pretokenization_end - begin
        state["phase"] = "prepare_pairs"
        return counts

    def timed_max(*args, **kwargs):
        nonlocal last_selection_end
        begin = perf_counter()
        if last_selection_end is None:
            phases["prepare_pairs"] = begin - pretokenization_end
            state["phase"] = "merge"
        else:
            phases["apply_merge"] += begin - last_selection_end
            state["merges"] += 1
        value = builtins.max(*args, **kwargs)
        last_selection_end = perf_counter()
        phases["select_pair"] += last_selection_end - begin
        return value

    def report(status):
        memory = process.memory_info()
        row = dict(
            state, status=status, elapsed_seconds=perf_counter() - started,
            rss_bytes=memory.rss, peak_working_set_bytes=getattr(memory, "peak_wset", None),
        )
        with (args.output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row) + "\n")
        print(f"{row['elapsed_seconds']:.1f}s {row['phase']} {row['merges']}/{row['expected_merges']} "
              f"RSS={row['rss_bytes'] / 2**20:.1f} MiB", flush=True)

    def monitor():
        while not stop.wait(45):
            report("running")

    thread = Thread(target=monitor, daemon=True)
    report("started")
    thread.start()
    profiler = cProfile.Profile()
    bpe.pretokenize_file_counts = timed_pretokenize
    bpe.max = timed_max  # Local to bpe.py; the original selection still uses builtins.max.
    try:
        vocab, merges = profiler.runcall(
            bpe.train_bpe, args.input, original["target_vocab_size"], original["special_tokens"]
        )
        finished = perf_counter()
        if last_selection_end is not None:
            phases["apply_merge"] += finished - last_selection_end
            state["merges"] += 1
    finally:
        bpe.pretokenize_file_counts = pretokenize
        del bpe.max
        stop.set()
        thread.join()

    assert vocab == expected[0] and merges == expected[1], "Profiled training changed the BPE output."
    assert state["merges"] == len(merges)
    state["phase"] = "completed"
    report("completed")
    profiler.dump_stats(str(args.output / "bpe.prof"))
    with (args.output / "profile.txt").open("w", encoding="utf-8") as stream:
        stats = pstats.Stats(profiler, stream=stream).strip_dirs()
        stats.sort_stats("cumulative").print_stats(30)
        stats.sort_stats("tottime").print_stats(30)

    result = dict(
        input_file=args.input.name, input_bytes=args.input.stat().st_size,
        input_sha256=input_sha, bpe_source_sha256=source_sha,
        vocab_size=len(vocab), merges=len(merges), vocab_and_merges_match_baseline=True,
        seconds_with_profiler=finished - started, previous_seconds_without_profiler=original["training_seconds_without_profiler"],
        phase_seconds=phases, peak_working_set_bytes=getattr(process.memory_info(), "peak_wset", None),
    )
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
