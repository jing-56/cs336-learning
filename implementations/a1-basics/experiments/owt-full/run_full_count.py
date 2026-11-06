"""Finish the Stanford download, then count the full corpus with checkpoints."""

import _winapi
import json
import msvcrt
from pathlib import Path
import time

from count_owt_pretokens import count
from download_owt import SIZE, DownloadRangeError, download


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "owt_train.txt.gz"
CHECKPOINT = ROOT / "pretoken-counts-full.pkl"
SUMMARY = ROOT / "pretoken-counts-full.json"
FILE_SHARE_READ = 0x00000001


def wait_for_download():
    partial = ARCHIVE.with_name(ARCHIVE.name + ".part")
    last_report = -60.0
    started = time.monotonic()
    while not ARCHIVE.exists():
        try:
            info = partial.stat()
            size = info.st_size
            # Request a handle which allows other readers but excludes writers.
            handle = _winapi.CreateFile(str(partial), _winapi.GENERIC_READ,
                                        FILE_SHARE_READ, 0,
                                        _winapi.OPEN_EXISTING, 0, 0)
        except FileNotFoundError:
            return
        except OSError as error:
            if error.winerror != 32:  # ERROR_SHARING_VIOLATION
                raise
        else:
            _winapi.CloseHandle(handle)
            if size == SIZE and time.time() - info.st_mtime < 120:
                # The downloader may be verifying the completed partial file.
                time.sleep(5)
                continue
            return
        elapsed = time.monotonic() - started
        if elapsed - last_report >= 60:
            print(f"Current download: {size:,} / {SIZE:,} bytes ({size/SIZE:.1%})", flush=True)
            last_report = elapsed
        time.sleep(5)


def main():
    # Windows releases the byte-range lock even if the runner is interrupted.
    with (ROOT / "full-count.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        wait_for_download()
        for attempt in range(3):
            try:
                download(ARCHIVE, workers=12)
                break
            except DownloadRangeError as error:
                if attempt == 2:
                    raise
                delay = 30 * (attempt + 1)
                print(f"{error}; resuming after {delay} seconds", flush=True)
                time.sleep(delay)
        result = count(ARCHIVE, CHECKPOINT)
        payload = json.dumps(result, indent=2) + "\n"
        temporary = SUMMARY.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(SUMMARY)
        print(payload, flush=True)


if __name__ == "__main__":
    main()
