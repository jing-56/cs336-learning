"""Count BPE pre-tokens without building an index or performing merges."""

import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import pickle
from time import perf_counter

import psutil
import regex

from bpe_compact import PAT


MARKER = "<|endoftext|>"
PATTERN = regex.compile(PAT)
CHUNK_CHARS = 1024**2
SAVE_EVERY = 512 * CHUNK_CHARS
MIN_AVAILABLE = 768 * 1024**2


def count_segments(counts, text):
    segments = text.split(MARKER)
    for segment in segments:
        counts.update(PATTERN.findall(segment))
    return len(segments) - 1


def save(path, state):
    partial = path.with_name(path.name + ".tmp")
    with partial.open("wb") as output:
        pickle.dump(state, output, protocol=pickle.HIGHEST_PROTOCOL)
    partial.replace(path)


def source_identity(source):
    stat = source.stat()
    return (str(source.resolve()), stat.st_size, stat.st_mtime_ns, PAT, MARKER)


def count(source, output, save_every=SAVE_EVERY, chunk_chars=CHUNK_CHARS):
    started = perf_counter()
    identity = source_identity(source)
    if source.resolve() == output.resolve():
        raise ValueError("Input and checkpoint must be different files")
    process = psutil.Process()
    if output.exists():
        # Only load checkpoints produced locally by this script.
        with output.open("rb") as file:
            state = pickle.load(file)
        if state.get("identity") != identity:
            raise ValueError("Checkpoint belongs to a different input or counting rule")
        if state["complete"]:
            return state["summary"]
    else:
        state = dict(identity=identity, counts=Counter(), complete=False,
                     position=0, pending="", chars=0, markers=0, seconds=0.0,
                     peak_working_set_bytes=0)

    counts = state["counts"]
    chars, markers, pending = state["chars"], state["markers"], state["pending"]
    previous_seconds = state["seconds"]
    next_save = chars + save_every
    next_report = chars + 128 * CHUNK_CHARS
    print(f"Counting from {chars:,} decoded characters in {source.name}", flush=True)

    def snapshot(stream, complete=False):
        memory = process.memory_info()
        peak = max(state["peak_working_set_bytes"], getattr(memory, "peak_wset", memory.rss))
        seconds = previous_seconds + perf_counter() - started
        state.update(complete=complete, position=stream.tell(), pending=pending,
                     chars=chars, markers=markers, seconds=seconds,
                     peak_working_set_bytes=peak)
        if complete:
            state["summary"] = dict(
                source=identity[0], source_bytes=identity[1], decoded_chars=chars,
                end_markers=markers, unique_pretokens=len(counts),
                pretoken_occurrences=counts.total(),
                peak_working_set_bytes=peak, working_set_bytes=memory.rss,
                system_available_bytes=psutil.virtual_memory().available,
                count_seconds=round(seconds, 3),
            )
        save(output, state)

    open_text = gzip.open if source.suffix == ".gz" else Path.open
    with open_text(source, "rt", encoding="utf-8", newline=None) as stream:
        if state["position"]:
            stream.seek(state["position"])
        while chunk := stream.read(chunk_chars):
            chars += len(chunk)
            pending += chunk
            boundary = pending.rfind(MARKER)
            if boundary >= 0:
                boundary += len(MARKER)
                markers += count_segments(counts, pending[:boundary])
                pending = pending[boundary:]

            if chars >= next_report:
                rss = process.memory_info().rss / 1024**2
                print(f"{chars:,} characters, {len(counts):,} distinct, {rss:.0f} MiB RAM", flush=True)
                next_report = chars + 128 * CHUNK_CHARS
            low_memory = psutil.virtual_memory().available < MIN_AVAILABLE
            if chars >= next_save or low_memory:
                snapshot(stream)
                next_save = chars + save_every
                if low_memory:
                    raise MemoryError(f"Saved counts to {output}; less than 768 MiB system RAM is available")

        markers += count_segments(counts, pending)
        pending = ""
        if source_identity(source) != identity:
            raise ValueError("Input changed during counting")
        snapshot(stream, complete=True)

    return state["summary"]


def load_complete_counts(checkpoint):
    with checkpoint.open("rb") as file:
        state = pickle.load(file)
    if not state["complete"]:
        raise ValueError(f"Counting is not finished: {checkpoint}")
    return state["counts"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = count(args.input, args.output)
    print(json.dumps(result, indent=2), flush=True)
