"""Encode complete TinyStories with the trained tokenizer, using bounded buffers."""

import argparse
import hashlib
import json
import os
import pickle
import shutil
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import regex

from cs336_basics.bpe import PAT, merge_pair
from cs336_basics.tokenizer_demo import Tokenizer


EOS = "<|endoftext|>"


class CachedTokenizer(Tokenizer):
    """Same BPE merge algorithm; cache exact pre-token results and fixed maps."""

    def __init__(self, vocab, merges, special_tokens=None, cache_size=100000):
        super().__init__(vocab, merges, special_tokens)
        if cache_size < 0:
            raise ValueError("cache_size must be nonnegative")
        self.token_to_id = {token: index for index, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.byte_tokens = tuple(bytes([value]) for value in range(256))
        self.normal_pattern = regex.compile(PAT)
        self.special_to_id = {token: self.token_to_id[token.encode("utf-8")] for token in self.special_tokens}
        self.special_pattern = regex.compile("|".join(regex.escape(token) for token in sorted(self.special_tokens, key=len, reverse=True))) if self.special_tokens else None
        self.encode_piece = lru_cache(maxsize=cache_size)(self._encode_piece)

    def _encode_piece(self, piece):
        tokens = tuple(self.byte_tokens[value] for value in piece.encode("utf-8"))
        while len(tokens) > 1:
            best = min(
                (pair for pair in zip(tokens, tokens[1:]) if pair in self.merge_ranks),
                key=self.merge_ranks.__getitem__, default=None,
            )
            if best is None:
                break
            tokens = next(iter(merge_pair({tokens: 1}, best)))
        return tuple(self.token_to_id[token] for token in tokens)

    def _append_normal(self, text, result):
        for match in self.normal_pattern.finditer(text):
            result.extend(self.encode_piece(match.group()))

    def encode(self, text):
        result = []
        if self.special_pattern is None:
            self._append_normal(text, result)
            return result
        position = 0
        for match in self.special_pattern.finditer(text):
            self._append_normal(text[position:match.start()], result)
            result.append(self.special_to_id[match.group()])
            position = match.end()
        self._append_normal(text[position:], result)
        return result


def iter_eos_chunks(path, chunk_chars=1024 * 1024):
    if chunk_chars <= 0:
        raise ValueError("chunk_chars must be positive")
    pending = ""
    # Preserve CRLF and all other source bytes through strict UTF-8 decoding.
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        while block := stream.read(chunk_chars):
            pending += block
            end = pending.rfind(EOS)
            if end >= 0:
                end += len(EOS)
                yield pending[:end]
                pending = pending[end:]
    if pending:
        yield pending


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def encode_split(path, tokenizer, output_dir, split, expected_sha256=None, chunk_chars=1024 * 1024):
    path = Path(path)
    output_dir = Path(output_dir)
    target = output_dir / f"{split}.npy"
    raw_path = output_dir / f"{split}.uint16.part"
    npy_path = output_dir / f"{split}.npy.part"
    for item in (target, raw_path, npy_path):
        if item.exists():
            raise FileExistsError(item)
    if not tokenizer.vocab or min(tokenizer.vocab) < 0 or max(tokenizer.vocab) > 65535:
        raise ValueError("token IDs do not fit uint16")
    dtype = np.dtype("<u2")
    started = last_report = perf_counter()
    source_hash, token_hash = hashlib.sha256(), hashlib.sha256()
    bytes_processed = total_tokens = eos_count = batches = 0
    min_id, max_id = 65535, 0
    process = psutil.Process()
    cache_before = tokenizer.encode_piece.cache_info()
    print(f"开始编码 {split}：{path.name}，{path.stat().st_size:,} 字节", flush=True)
    with raw_path.open("xb") as stream:
        for text in iter_eos_chunks(path, chunk_chars):
            ids = tokenizer.encode(text)
            original = text.encode("utf-8")
            restored = b"".join(tokenizer.vocab[index] for index in ids)
            if restored != original:
                raise AssertionError(f"{split}: byte-exact roundtrip failed")
            array = np.asarray(ids, dtype=dtype)
            current_eos = text.count(EOS)
            if int(np.count_nonzero(array == tokenizer.special_to_id[EOS])) != current_eos:
                raise AssertionError("EOS ID count differs from source marker count")
            if len(array):
                min_id = min(min_id, int(array.min()))
                max_id = max(max_id, int(array.max()))
            data = array.tobytes()
            stream.write(data)
            source_hash.update(original)
            token_hash.update(data)
            bytes_processed += len(original)
            total_tokens += len(ids)
            eos_count += current_eos
            batches += 1
            now = perf_counter()
            if now - last_report >= 30:
                row = {"split": split, "source_bytes_processed": bytes_processed, "source_bytes_total": path.stat().st_size, "tokens": total_tokens, "elapsed_seconds": now-started, "rss_bytes": process.memory_info().rss}
                with (output_dir / "progress.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(row) + "\n")
                print(f"  {split} {bytes_processed/path.stat().st_size:.1%}，{total_tokens:,} tokens，{(now-started)/60:.1f} 分钟，内存 {row['rss_bytes']/2**20:.1f} MiB", flush=True)
                last_report = now
    encoding_seconds = perf_counter() - started
    if bytes_processed != path.stat().st_size:
        raise AssertionError("source byte count mismatch")
    if expected_sha256 and source_hash.hexdigest() != expected_sha256:
        raise AssertionError("complete source SHA-256 differs from official metadata")

    print(f"  {split} 编码完成，封装 .npy 并校验全部 token 数据……", flush=True)
    with npy_path.open("xb") as destination, raw_path.open("rb") as source:
        np.lib.format.write_array_header_2_0(destination, {"descr": np.lib.format.dtype_to_descr(dtype), "fortran_order": False, "shape": (total_tokens,)})
        shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    array = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    if array.shape != (total_tokens,) or array.dtype != dtype:
        raise AssertionError("saved NPY shape/dtype mismatch")
    data_offset = array.offset
    del array
    saved_hash = hashlib.sha256()
    with npy_path.open("rb") as stream:
        stream.seek(data_offset)
        while block := stream.read(8 * 1024 * 1024):
            saved_hash.update(block)
    if saved_hash.hexdigest() != token_hash.hexdigest():
        raise AssertionError("saved NPY token data differs from encoded stream")
    os.replace(npy_path, target)
    assert raw_path.resolve().parent == output_dir.resolve()
    raw_path.unlink()  # Only this run's verified temporary token stream.
    elapsed = perf_counter() - started
    cache_after = tokenizer.encode_piece.cache_info()
    memory = process.memory_info()
    result = {
        "split": split, "source_filename": path.name, "source_bytes": bytes_processed,
        "source_sha256": source_hash.hexdigest(), "output_filename": target.name,
        "token_count": total_tokens, "dtype": dtype.str, "array_data_bytes": total_tokens*2,
        "npy_file_bytes": target.stat().st_size, "token_data_sha256": saved_hash.hexdigest(),
        "min_token_id": min_id if total_tokens else None, "max_token_id": max_id if total_tokens else None,
        "eos_count": eos_count, "eos_id": tokenizer.special_to_id[EOS],
        "batches": batches, "full_byte_exact_roundtrip": True,
        "saved_token_data_hash_matches": True, "bytes_per_token": bytes_processed/total_tokens if total_tokens else None,
        "encoding_roundtrip_raw_write_seconds": encoding_seconds,
        "npy_finalize_and_verify_seconds": elapsed-encoding_seconds, "total_seconds": elapsed,
        "end_to_end_source_mb_per_second": bytes_processed/1e6/elapsed,
        "cache_hits_this_split": cache_after.hits-cache_before.hits,
        "cache_misses_this_split": cache_after.misses-cache_before.misses,
        "cache_entries_after_split": cache_after.currsize,
        "process_rss_after_bytes": memory.rss,
        "windows_process_lifetime_peak_working_set_bytes": getattr(memory, "peak_wset", None),
    }
    write_json(output_dir / f"{split}_summary.json", result)
    print(f"{split} 已保存：{total_tokens:,} tokens，{total_tokens*2/2**20:.2f} MiB，{elapsed:.2f} 秒，完整还原=True", flush=True)
    return result


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project / "data")
    parser.add_argument("--tokenizer-dir", type=Path, default=Path(__file__).resolve().parent / "bpe_corpus_runs/20261101_190151_430986")
    parser.add_argument("--output-dir", type=Path, default=project / "data/tokenized_tinystories_full" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--cache-size", type=int, default=100000)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    vocab_path, merges_path = args.tokenizer_dir / "vocab.pkl", args.tokenizer_dir / "merges.pkl"
    tokenizer = CachedTokenizer.from_files(vocab_path, merges_path, [EOS]) if args.cache_size == 100000 else CachedTokenizer(pickle.loads(vocab_path.read_bytes()), pickle.loads(merges_path.read_bytes()), [EOS], args.cache_size)
    if len(tokenizer.vocab) != 10000 or set(tokenizer.vocab) != set(range(10000)):
        raise ValueError("Expected the complete TinyStories 10k vocabulary")
    source = json.loads((args.data_dir / "tinystories_manifest.json").read_text(encoding="utf-8"))
    entries = {item["filename"]: item for item in source["files"]}
    config = {
        "source_dataset_revision": source["revision"], "vocab_size": len(tokenizer.vocab),
        "tokenizer_directory": str(args.tokenizer_dir.resolve()),
        "vocab_sha256": hashlib.sha256(vocab_path.read_bytes()).hexdigest(),
        "merges_sha256": hashlib.sha256(merges_path.read_bytes()).hexdigest(),
        "encoder_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cache_size": args.cache_size, "chunk_chars": 1024*1024,
        "numpy_version": np.__version__, "newline_policy": "preserve exact source bytes",
        "split_order": ["valid", "train"], "splits": [],
    }
    write_json(args.output_dir / "metadata.json", config)
    for split in ("valid", "train"):
        name = f"TinyStoriesV2-GPT4-{split}.txt"
        config["splits"].append(encode_split(args.data_dir/name, tokenizer, args.output_dir, split, entries[name]["sha256"]))
        write_json(args.output_dir / "metadata.json", config)
    config["completed"] = True
    write_json(args.output_dir / "metadata.json", config)
    print(f"完整训练集与验证集编码完成：{args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
