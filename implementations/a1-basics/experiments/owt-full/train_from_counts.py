"""Train the OpenWebText BPE from saved frequencies with a RAM reserve."""

import argparse
import json
import msvcrt
from pathlib import Path
import pickle
from time import monotonic, sleep

import psutil

from bpe_compact import train_bpe_from_counts
from count_owt_pretokens import MARKER, load_complete_counts


ROOT = Path(__file__).resolve().parent
MIB = 1024**2
MIN_AVAILABLE = 1536 * MIB
VOCAB_SIZE = 32_000


class MemoryMonitor:
    def __init__(self):
        self.process = psutil.Process()
        self.stage = None
        self.next_check = 0.0
        self.next_report = 0.0
        self.stages = []
        self.available_before = psutil.virtual_memory().available
        self.min_available = self.available_before

    def __call__(self, stage, done, total):
        now = monotonic()
        changed = stage != self.stage
        if not changed and now < self.next_check:
            return
        self.next_check = now + 0.25
        available = psutil.virtual_memory().available
        self.min_available = min(self.min_available, available)
        if available < MIN_AVAILABLE:
            raise MemoryError(f"{stage}, {done:,}/{total:,}: available RAM is "
                              f"{available/MIB:.0f} MiB (reserve: {MIN_AVAILABLE/MIB:.0f} MiB)")
        if changed or now >= self.next_report:
            memory = self.process.memory_info()
            print(f"{stage}: {done:,}/{total:,} | working set {memory.rss/MIB:.0f} MiB "
                  f"| available {available/MIB:.0f} MiB", flush=True)
            self.next_report = now + 30
            if changed:
                self.stages.append(dict(stage=stage, working_set_bytes=memory.rss,
                                        peak_working_set_bytes=memory.peak_wset,
                                        available_bytes=available))
        self.stage = stage


def wait_for_full_counts():
    summary = ROOT / "pretoken-counts-full.json"
    if summary.is_file():
        return
    print("Waiting for the full frequency table", flush=True)
    with (ROOT / "full-count.lock").open("a+b") as lock:
        while not summary.is_file():
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                if error.errno != 13:
                    raise
                sleep(10)
            else:
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                if not summary.is_file():
                    raise RuntimeError("The count runner stopped before producing a full frequency table")


def train(case):
    output = ROOT / f"bpe-from-counts-{case}.json"
    if output.is_file():
        print(output.read_text(encoding="utf-8"), flush=True)
        return
    if case == "full":
        wait_for_full_counts()

    monitor = MemoryMonitor()
    monitor("load frequencies", 0, 1)
    started = monotonic()
    vocab, merges = train_bpe_from_counts(
        load_complete_counts(ROOT / f"pretoken-counts-{case}.pkl"),
        VOCAB_SIZE, [MARKER], progress=monitor,
    )
    monitor("complete", len(merges), VOCAB_SIZE - 257)
    if len(vocab) != VOCAB_SIZE or len(merges) != VOCAB_SIZE - 257:
        raise ValueError("The corpus did not yield the requested vocabulary size")
    if case == "full":
        tokenizer = ROOT / "owt-full-32k.pkl"
        partial = tokenizer.with_suffix(".pkl.tmp")
        with partial.open("wb") as file:
            pickle.dump(dict(vocab=vocab, merges=merges), file, protocol=pickle.HIGHEST_PROTOCOL)
        partial.replace(tokenizer)
    else:
        for name, value in (("vocab", vocab), ("merges", merges)):
            with (ROOT / f"{case}-{name}.pkl").open("rb") as reference:
                if value != pickle.load(reference):
                    raise ValueError(f"The {case} {name} differs from the previous result")

    result = dict(case=case, vocabulary_size=len(vocab), merge_count=len(merges),
                  elapsed_seconds=round(monotonic() - started, 3),
                  peak_working_set_bytes=monitor.process.memory_info().peak_wset,
                  available_before_bytes=monitor.available_before,
                  lowest_observed_available_bytes=monitor.min_available, stages=monitor.stages)
    if case != "full":
        result["matches_reference"] = True
    payload = json.dumps(result, indent=2) + "\n"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(output)
    print(payload, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("50m", "500m", "full"))
    args = parser.parse_args()
    with (ROOT / f"bpe-from-counts-{args.case}.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        train(args.case)
