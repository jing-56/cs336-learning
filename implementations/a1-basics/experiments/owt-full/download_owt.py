"""Download a pinned Stanford OpenWebText archive with resumable HTTP ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from http.client import HTTPException
from pathlib import Path
import shutil
import time
from urllib.request import Request, urlopen


URL = "https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/488afad/owt_train.txt.gz"
SIZE = 4_591_240_837
SHA256 = "b19ae88cfbc4016b304c348522455fe38ebac48fffed955adcc7191a89e38ccf"
VALID_URL = URL.replace("owt_train.txt.gz", "owt_valid.txt.gz")
VALID_SIZE = 111_785_382
VALID_SHA256 = "bc73db5da2f19c360836b9c8a88094e13346ec83798c2f40060be39135768c80"
BLOCK = 4 * 1024**2
RUN_ID = time.time_ns()


class DownloadRangeError(OSError):
    """A requested range remained unavailable after its immediate retries."""


def fetch(start, url=URL, total=SIZE):
    stop = min(start + BLOCK, total)
    expected = f"bytes {start}-{stop - 1}/{total}"
    for attempt in range(5):
        try:
            request = Request(
                f"{url}?download=true&part={start}&attempt={RUN_ID}-{attempt}",
                headers={"Range": f"bytes={start}-{stop - 1}"},
            )
            with urlopen(request, timeout=30) as response:
                if response.status != 206 or response.headers.get("Content-Range") != expected:
                    raise OSError(f"Unexpected HTTP range for offset {start}")
                data = response.read(stop - start + 1)
                if len(data) != stop - start:
                    raise OSError(f"Incomplete HTTP range at offset {start}")
                return data
        except (OSError, HTTPException) as error:
            if attempt == 4:
                raise DownloadRangeError(f"Range {expected} failed after five attempts") from error
            time.sleep(attempt + 1)


def verify(path, sha256=SHA256):
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != sha256:
        raise ValueError(f"Archive checksum mismatch: {actual}")


def download(destination, workers, *, url=URL, total=SIZE, sha256=SHA256):
    destination = destination.resolve()
    if destination.exists():
        verify(destination, sha256)
        print(f"Archive already verified: {destination}", flush=True)
        return

    partial = destination.with_name(destination.name + ".part")
    size = partial.stat().st_size if partial.exists() else 0
    if size > total:
        raise ValueError(f"Partial file is larger than expected: {partial}")
    offset = size if size == total else size - size % BLOCK
    if shutil.disk_usage(destination.parent).free < total - offset + 1024**3:
        raise OSError("Not enough free disk space for archive plus a 1 GiB reserve")

    initial = offset
    started = time.monotonic()
    print(f"Downloading from {offset:,} / {total:,} bytes", flush=True)
    with partial.open("r+b" if partial.exists() else "x+b") as output:
        output.truncate(offset)
        output.seek(offset)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while offset < total:
                offsets = range(offset, min(offset + workers * BLOCK, total), BLOCK)
                for block in pool.map(lambda start: fetch(start, url, total), offsets):
                    output.write(block)
                    offset += len(block)
                output.flush()
                elapsed = time.monotonic() - started
                print(f"{offset:,} / {total:,} bytes ({offset/total:.1%}), "
                      f"{(offset-initial)/elapsed/1024**2:.2f} MiB/s", flush=True)

    verify(partial, sha256)
    partial.rename(destination)
    print(f"Archive verified: {destination}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "valid"), default="train")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--workers", type=int, default=12, choices=range(1, 25), metavar="1..24")
    args = parser.parse_args()
    output = args.output or Path(__file__).with_name(f"owt_{args.split}.txt.gz")
    if args.split == "valid":
        download(output, args.workers, url=VALID_URL, total=VALID_SIZE, sha256=VALID_SHA256)
    else:
        download(output, args.workers)
