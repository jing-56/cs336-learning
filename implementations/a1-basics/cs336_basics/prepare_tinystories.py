"""Download the two TinyStories text files named in the assignment README."""

import argparse
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DATASET = "roneneldan/TinyStories"
FILES = ("TinyStoriesV2-GPT4-valid.txt", "TinyStoriesV2-GPT4-train.txt")


def fetch_metadata():
    request = urllib.request.Request(
        f"https://huggingface.co/api/datasets/{DATASET}?blobs=true",
        headers={"User-Agent": "cs336-personal-learning/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.load(response)
    entries = {item["rfilename"]: item for item in data["siblings"]}
    result = {"dataset": DATASET, "revision": data["sha"], "files": {}}
    for filename in FILES:
        item = entries[filename]
        lfs = item["lfs"]
        result["files"][filename] = {"size": lfs["size"], "sha256": lfs["sha256"]}
    return result


def digest_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download_file(directory, filename, spec, revision):
    target = directory / filename
    if target.exists():
        if target.stat().st_size != spec["size"] or digest_file(target) != spec["sha256"]:
            raise ValueError(f"已有文件与官方元数据不符，请先检查：{target}")
        print(f"已存在且 SHA-256 正确：{filename}", flush=True)
    else:
        partial = directory / (filename + ".part")
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec["size"]:
            raise ValueError(f"临时文件大小超出预期：{partial}")
        if offset < spec["size"]:
            url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{filename}"
            headers = {"User-Agent": "cs336-personal-learning/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"开始下载 {filename}，总计 {spec['size']:,} 字节", flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=45) as response:
                if offset and response.status == 206:
                    expected = f"bytes {offset}-"
                    if not response.headers.get("Content-Range", "").startswith(expected):
                        raise ValueError("服务器返回的断点位置不符，停止写入")
                    mode = "ab"
                elif response.status == 200:
                    offset = 0
                    mode = "wb"
                else:
                    raise ValueError(f"意外 HTTP 状态：{response.status}")
                started = last_report = time.perf_counter()
                received = offset
                with partial.open(mode) as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                        received += len(block)
                        if received > spec["size"]:
                            raise ValueError("下载字节数超出官方文件大小")
                        now = time.perf_counter()
                        if now - last_report >= 10:
                            speed = (received - offset) / (now - started) / 1e6
                            print(f"  {received / spec['size']:.1%}，{speed:.2f} MB/s", flush=True)
                            last_report = now
        if partial.stat().st_size != spec["size"]:
            raise ValueError(f"下载未完整结束；保留 .part，重新运行可尝试续传：{partial}")
        print(f"正在校验 SHA-256：{filename}", flush=True)
        if digest_file(partial) != spec["sha256"]:
            raise ValueError(f"SHA-256 不符，文件未启用，请检查临时文件：{partial}")
        os.replace(partial, target)
    return {"filename": filename, **spec, "sha256_verified": True}


def prepare(data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = data_dir / "tinystories_source.json"
    # Reuse the pinned revision on reruns, including partial-download recovery.
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    else:
        metadata = fetch_metadata()
        write_json(metadata_path, metadata)
    results = []
    for filename in FILES:
        results.append(download_file(data_dir, filename, metadata["files"][filename], metadata["revision"]))
    manifest = {
        "dataset": DATASET,
        "revision": metadata["revision"],
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Assignment README's TinyStoriesV2-GPT4 train/valid text files",
        "files": results,
    }
    write_json(data_dir / "tinystories_manifest.json", manifest)
    print(f"数据下载及校验完成：{data_dir.resolve()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = parser.parse_args()
    prepare(args.data_dir)
