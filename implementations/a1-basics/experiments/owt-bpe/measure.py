import hashlib
from importlib import import_module
import json
from pathlib import Path
import pickle
import sys
from threading import Event, Thread
from time import perf_counter

import psutil

root = Path(__file__).resolve().parent
pilot = root.parent / "owt-tokenizer-pilot" / "20261104-202038"
case = sys.argv[1]
variant = sys.argv[2] if len(sys.argv) > 2 else "heap"
assert variant in ("heap", "compact")
bpe = import_module(f"bpe_{variant}")
paths = {
    "5m": pilot / "owt_train_pilot.txt",
    "50m": root / "owt_train_50m.txt",
    "500m": root / "owt_train_500m.txt",
}
path = paths[case]
pretokenize = bpe.pretokenize_file_counts
phase = {}
process = psutil.Process()
available_before = psutil.virtual_memory().available


def timed_pretokenization(*args, **kwargs):
    start = perf_counter()
    counts = pretokenize(*args, **kwargs)
    phase["pretokenization_seconds"] = perf_counter() - start
    phase["unique_pretokens"] = len(counts)
    print(f"Pre-tokenization complete: {phase}", flush=True)
    return counts


bpe.pretokenize_file_counts = timed_pretokenization
print(f"Training {case} sample with {variant}: {path.stat().st_size:,} bytes", flush=True)
start = perf_counter()
finished = Event()


def show_progress():
    while not finished.wait(20):
        stage = "building pair index and merging" if phase else "pre-tokenizing"
        rss = process.memory_info().rss / 2**20
        available = psutil.virtual_memory().available / 2**20
        print(f"{perf_counter()-start:.0f}s | {stage} | RSS {rss:.0f} MiB | available {available:.0f} MiB",
              flush=True)


monitor = Thread(target=show_progress, daemon=True)
monitor.start()
try:
    vocab, merges = bpe.train_bpe(path, 32_000, ["<|endoftext|>"])
finally:
    finished.set()
    monitor.join()
elapsed = perf_counter() - start
peak = process.memory_info().peak_wset
assert len(vocab) == 32_000 and len(merges) == 32_000 - 257
matched = None
if case == "5m" or variant == "compact":
    reference = pilot / "owt_pilot" if case == "5m" else root / case
    separator = "_" if case == "5m" else "-"
    with reference.with_name(reference.name + separator + "vocab.pkl").open("rb") as stream:
        assert vocab == pickle.load(stream)
    with reference.with_name(reference.name + separator + "merges.pkl").open("rb") as stream:
        assert merges == pickle.load(stream)
    matched = True

digest = hashlib.sha256()
with path.open("rb") as stream:
    while block := stream.read(4 * 1024**2):
        digest.update(block)
report = dict(
    case=case, variant=variant, input_bytes=path.stat().st_size,
    input_sha256=digest.hexdigest(),
    vocabulary_size=len(vocab), merge_count=len(merges),
    training_seconds=elapsed, windows_peak_working_set_bytes=peak,
    system_available_memory_before_bytes=available_before,
    build_and_merge_seconds=elapsed - phase["pretokenization_seconds"],
    matches_reference_vocab_and_merges=matched,
    **phase,
)
if case in ("50m", "500m") and variant == "heap":
    for name, value in (("vocab", vocab), ("merges", merges)):
        with (root / f"{case}-{name}.pkl").open("xb") as stream:
            pickle.dump(value, stream)
suffix = "-compact" if variant == "compact" else ""
(root / f"benchmark-{case}{suffix}.json").write_text(
    json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8",
)
print(json.dumps(report, indent=2), flush=True)
