from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from time import perf_counter, sleep
from urllib.request import Request, urlopen
import zlib


ROOT = Path(__file__).resolve().parent
URL = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/488afad/owt_train.txt.gz"
TOTAL_COMPRESSED_BYTES = 4_591_240_837
MIB = 1024**2
BLOCK_SIZE = 4 * MIB
TARGET = 500 * MIB
MARKER = b"<|endoftext|>"


def fetch_block(start):
    end = start + BLOCK_SIZE - 1
    expected_range = f"bytes {start}-{end}/{TOTAL_COMPRESSED_BYTES}"
    for attempt in range(3):
        try:
            request = Request(
                URL + f"?download=true&part={start}&attempt={attempt}",
                headers={"Range": f"bytes={start}-{end}"},
            )
            with urlopen(request, timeout=30) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                    raise OSError(f"Unexpected range at offset {start}")
                data = response.read(BLOCK_SIZE + 1)
                if len(data) != BLOCK_SIZE:
                    raise OSError(f"Incomplete range at offset {start}")
                return data
        except OSError:
            if attempt == 2:
                raise
            sleep(attempt + 1)


def digest_file(path, limit=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = limit
        while remaining is None or remaining > 0:
            block = stream.read(MIB if remaining is None else min(MIB, remaining))
            if not block:
                if remaining is not None:
                    raise ValueError("File ended before the expected prefix")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def main():
    destination = ROOT / "owt_train_500m.txt"
    if destination.exists():
        raise FileExistsError(destination)
    partial = destination.with_suffix(".txt.part")
    prior = json.loads((ROOT / "data-summary.json").read_text(encoding="utf-8"))
    decoder = zlib.decompressobj(31)
    tail = b""
    written = 0
    last_document_end = 0
    documents = 0
    next_progress = 50 * MIB
    started = perf_counter()

    with partial.open("wb") as output, ThreadPoolExecutor(max_workers=8) as pool:
        offset = 0
        while written < TARGET:
            offsets = range(offset, offset + 8 * BLOCK_SIZE, BLOCK_SIZE)
            for block in pool.map(fetch_block, offsets):
                while block and written < TARGET:
                    decoded = decoder.decompress(block, min(MIB, TARGET - written))
                    block = decoder.unconsumed_tail
                    combined = tail + decoded
                    end = combined.rfind(MARKER)
                    if end >= 0:
                        last_document_end = written - len(tail) + end + len(MARKER)
                        documents += combined.count(MARKER)
                    tail = combined[-(len(MARKER) - 1):]
                    output.write(decoded)
                    written += len(decoded)
                    if written >= next_progress:
                        print(f"Decoded {written/MIB:.1f} / 500 MiB", flush=True)
                        next_progress += 50 * MIB
                if written == TARGET:
                    break
            offset += 8 * BLOCK_SIZE
        if last_document_end <= prior["bytes"]:
            raise ValueError("Missing complete documents in the larger prefix")
        output.truncate(last_document_end)

    if digest_file(partial, prior["bytes"]) != prior["sha256"]:
        raise ValueError("500 MiB data does not match the verified 50 MiB prefix")
    with partial.open("r", encoding="utf-8") as stream:
        while stream.read(MIB):
            pass
    summary = dict(url=URL, bytes=last_document_end, documents=documents,
                   sha256=digest_file(partial), matches_verified_50m_prefix=True,
                   preparation_seconds=perf_counter() - started)
    partial.rename(destination)
    (ROOT / "data-summary-500m.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
