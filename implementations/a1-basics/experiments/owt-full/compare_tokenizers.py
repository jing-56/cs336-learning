"""Compare the full-data tokenizers on frozen validation documents and measure encoding speed."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import pickle
from statistics import median
from time import perf_counter

from cs336_basics.tokenizer_demo import Tokenizer
from encode_owt import EOS_ID, MARKER, load_encoder


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_sample(path):
    raw = path.read_bytes()
    docs = json.loads(raw)
    if len(docs) != 10 or any(not isinstance(doc, str) or not doc for doc in docs):
        raise ValueError("Expected ten nonempty validation documents.")
    return docs, sha256(raw)


def measure(encode, text, expected, repeats):
    seconds = []
    for _ in range(repeats):
        start = perf_counter()
        tokens = encode(text)
        seconds.append(perf_counter() - start)
        if tokens != expected:
            raise RuntimeError("Token IDs differ from the Python implementation.")
    rate = len(text.encode("utf-8")) / median(seconds)
    return dict(seconds=seconds, bytes_per_second=rate, estimated_825_decimal_gb_hours=825e9 / rate / 3600)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("owt-model", "tiny-vocab", "tiny-merges", "tiny-sample", "owt-sample", "owt-validation-gzip", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    model_bytes = args.owt_model.read_bytes()
    tiny_vocab_bytes, tiny_merges_bytes = args.tiny_vocab.read_bytes(), args.tiny_merges.read_bytes()
    tiny_vocab, tiny_merges = pickle.loads(tiny_vocab_bytes), pickle.loads(tiny_merges_bytes)
    fast = load_encoder(model_bytes)
    model = pickle.loads(model_bytes)
    if len(tiny_vocab) != 10000 or len(tiny_merges) != 9743 or tiny_vocab[EOS_ID] != MARKER.encode():
        raise ValueError("Expected the 10K TinyStories tokenizer.")
    tiny = Tokenizer(tiny_vocab, tiny_merges, special_tokens=[MARKER])
    owt = Tokenizer(model["vocab"], model["merges"], special_tokens=[MARKER])

    def fast_encode(text):
        return fast.encode(text, allowed_special={MARKER})

    results = {}
    sample_hashes = {}
    for domain, path in (("tinystories", args.tiny_sample), ("openwebtext", args.owt_sample)):
        docs, sample_hashes[domain] = load_sample(path)
        byte_count = sum(len(doc.encode("utf-8")) for doc in docs)
        row = {}
        for name, tokenizer in (("tinystories_10k", tiny), ("openwebtext_32k", owt)):
            encoded = [tokenizer.encode(doc) for doc in docs]
            if any(tokenizer.decode(ids) != doc for ids, doc in zip(encoded, docs)):
                raise RuntimeError(f"Roundtrip failed for {name} on {domain}.")
            if tokenizer is owt and any(fast_encode(doc) != ids for ids, doc in zip(encoded, docs)):
                raise RuntimeError(f"OpenWebText encoders disagree on {domain}.")
            count = sum(map(len, encoded))
            row[name] = dict(tokens=count, utf8_bytes=byte_count, bytes_per_token=byte_count / count, roundtrip=True)
        results[domain] = row

    # Stop at a complete document; read/decompress before starting the timer.
    with gzip.open(args.owt_validation_gzip, "rb") as stream:
        prefix = stream.read(1024 * 1024)
    prefix = prefix[:prefix.rindex(MARKER.encode()) + len(MARKER)]
    text = prefix.decode("utf-8")
    expected = owt.encode(text)
    if owt.decode(expected) != text or fast_encode(text) != expected:
        raise RuntimeError("Benchmark encoders must match and roundtrip before timing.")
    example = " legislation"
    longest = max((value for key, value in model["vocab"].items() if key != EOS_ID), key=lambda value: (len(value), value))
    output = dict(
        tokenizer_sha256=dict(tinystories_vocab=sha256(tiny_vocab_bytes), tinystories_merges=sha256(tiny_merges_bytes),
                              openwebtext_model=sha256(model_bytes)),
        sample_json_sha256=sample_hashes, documents_per_domain=10, results=results,
        example=dict(text=example, tinystories_pieces=[tiny.decode([i]) for i in tiny.encode(example)],
                     openwebtext_pieces=[owt.decode([i]) for i in owt.encode(example)]),
        longest_openwebtext_token=dict(bytes_repr=repr(longest), bytes=len(longest)),
        throughput=dict(utf8_bytes=len(prefix), sha256=sha256(prefix), repeats=3,
                        scope="encoding only, OpenWebText validation prefix through last EOS, median of 3",
                        python=measure(owt.encode, text, expected, 3), accelerated=measure(fast_encode, text, expected, 3)),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
