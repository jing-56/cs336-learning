"""Stream OpenWebText into a resumable, memory-mappable uint16 array."""

import argparse
import gzip
import hashlib
import io
import json
import msvcrt
import os
from pathlib import Path
import pickle
import shutil
from time import monotonic

import numpy as np
import tiktoken

from bpe_compact import PAT


MARKER = "<|endoftext|>"
EOS_ID = 256
DTYPE = np.dtype("<u2")
CHUNK_CHARS = 4 * 1024**2
SAVE_EVERY = 16
DISK_RESERVE = 1536 * 1024**2


def load_encoder(model_data):
    model = pickle.loads(model_data)
    vocab, merges = model["vocab"], model["merges"]
    if (set(vocab) != set(range(32000)) or len(set(vocab.values())) != 32000
            or vocab[EOS_ID] != MARKER.encode() or len(merges) != 31743
            or any(vocab[257 + i] != left + right for i, (left, right) in enumerate(merges))):
        raise ValueError("Expected the complete 32K OpenWebText tokenizer")
    return tiktoken.Encoding(
        name="owt_cs336_32k", pat_str=PAT,
        mergeable_ranks={token: index for index, token in vocab.items() if index != EOS_ID},
        special_tokens={MARKER: EOS_ID},
    )


def npy_header(count):
    buffer = io.BytesIO()
    np.lib.format.write_array_header_2_0(
        buffer, {"descr": DTYPE.str, "fortran_order": False, "shape": (count,)},
    )
    return buffer.getvalue()


def save_state(path, state):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def encode(source, model_path, target, chunk_chars=CHUNK_CHARS, save_every=SAVE_EVERY):
    source, target = source.resolve(), target.resolve()
    if target.suffix != ".npy" or target == source or target == model_path.resolve():
        raise ValueError("Use a separate .npy destination")
    partial = target.with_suffix(".npy.part")
    state_path = target.with_suffix(".json")
    model_data = model_path.read_bytes()
    tokenizer = load_encoder(model_data)
    stat = source.stat()
    identity = dict(source=str(source), source_bytes=stat.st_size,
                    source_mtime_ns=stat.st_mtime_ns, dtype=DTYPE.str,
                    tokenizer_sha256=hashlib.sha256(model_data).hexdigest(),
                    pretoken_pattern=PAT, newline_policy="preserve original bytes")
    header = npy_header(0)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["identity"] != identity:
            raise ValueError("Checkpoint belongs to another source or tokenizer")
    else:
        if target.exists() or partial.exists():
            raise FileExistsError(f"Existing output without checkpoint: {target}")
        state = dict(identity=identity, complete=False, position=0, pending="",
                     chars=0, decoded_bytes=0, tokens=0, eos=0, seconds=0.0)
        save_state(state_path, state)

    def publish():
        expected_size = len(header) + state["tokens"] * DTYPE.itemsize
        if not target.exists():
            if partial.stat().st_size != expected_size:
                raise ValueError("Output size differs from its checkpoint")
            partial.replace(target)
        array = np.load(target, mmap_mode="r", allow_pickle=False)
        if array.shape != (state["tokens"],) or array.dtype != DTYPE or target.stat().st_size != expected_size:
            raise ValueError("Saved NPY array disagrees with its checkpoint")
        print(f"Complete: {state['tokens']:,} tokens, {state['eos']:,} EOS, "
              f"{state['seconds']:.1f} seconds | {target}", flush=True)

    if state["complete"]:
        publish()
        return
    if target.exists():
        raise FileExistsError(target)

    started = monotonic()
    previous_seconds = state["seconds"]
    print(f"Encoding from {state['chars']:,} characters in {source.name}", flush=True)
    open_text = gzip.open if source.suffix == ".gz" else Path.open
    with (open_text(source, "rt", encoding="utf-8", newline="") as stream,
          partial.open("r+b" if partial.exists() else "x+b") as output):
        if state["position"] == 0:
            output.write(header)
        size = len(header) + state["tokens"] * DTYPE.itemsize
        if output.seek(0, os.SEEK_END) < size:
            raise ValueError("Partial output is shorter than its checkpoint")
        output.truncate(size)
        output.seek(size)
        if state["position"]:
            stream.seek(state["position"])

        def append(text):
            ids = tokenizer.encode_to_numpy(text, allowed_special={MARKER}, disallowed_special=())
            raw = text.encode("utf-8")
            eos = text.count(MARKER)
            if tokenizer.decode_bytes(ids) != raw or int(np.count_nonzero(ids == EOS_ID)) != eos:
                raise ValueError("Encoded chunk failed byte or EOS verification")
            if shutil.disk_usage(target.parent).free < DISK_RESERVE + ids.size * DTYPE.itemsize:
                raise OSError("Disk reserve reached; resume from the last saved checkpoint")
            output.write(ids.astype(DTYPE).tobytes())
            state["tokens"] += int(ids.size)
            state["decoded_bytes"] += len(raw)
            state["eos"] += eos

        def checkpoint():
            output.flush()
            os.fsync(output.fileno())
            state["position"] = stream.tell()
            state["seconds"] = previous_seconds + monotonic() - started
            save_state(state_path, state)

        blocks = 0
        while block := stream.read(chunk_chars):
            pending = state["pending"] + block
            state["chars"] += len(block)
            boundary = pending.rfind(MARKER)
            if boundary >= 0:
                boundary += len(MARKER)
                append(pending[:boundary])
                pending = pending[boundary:]
            state["pending"] = pending
            blocks += 1
            if blocks % save_every == 0:
                checkpoint()
                print(f"{state['chars']:,} chars | {state['tokens']:,} tokens | "
                      f"{state['seconds']:.0f} seconds", flush=True)

        append(state["pending"])
        state["pending"] = ""
        final_header = npy_header(state["tokens"])
        if len(final_header) != len(header):
            raise ValueError("Final NPY header would overlap the token data")
        output.seek(0)
        output.write(final_header)
        state["complete"] = True
        checkpoint()

    publish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.with_suffix(".lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        encode(args.source, args.tokenizer, args.output)
