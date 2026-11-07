# TinyStories 完整训练集 BPE 性能分析

本实验在同一份 2,227,753,162 字节的训练集上调用现有 `train_bpe` 一次。启动前核对输入、BPE 源码和历史词表产物的 SHA-256。运行结束后，逐项比较新词表的 ID 到字节映射和有序合并规则。

本次耗时 1,231.85 秒，词表与规则逐项匹配。预分词占总时间 80.27%，每轮选对占 18.65%，其余是建表和应用合并。[阶段计时与核对结果](summary.json) · [原始 cProfile 文本](profile.txt)

脚本记录四段经过时间。预分词从进入 `pretokenize_file_counts` 到返回。建表从预分词返回到第一次选对。选对累加每次 `max` 调用。应用合并累加每次选对结束后到下一次选对或函数返回的间隔，包含词表写入、命中片段更新与循环开销。四段计时都包含 profiler 开销。

从学习仓库根目录运行以下命令。需要本地 A1 Python 环境和可信的历史 pickle 产物。输出目录需为新目录。

```powershell
$project = 'E:/CS336/assignments/a1-basics'
$python = Join-Path $project '.venv/Scripts/python.exe'
$env:PYTHONPATH = $project
$runs = Join-Path $project 'cs336_basics/bpe_corpus_runs'
$profileArgs = @(
    '--input', (Join-Path $project 'data/TinyStoriesV2-GPT4-train.txt'),
    '--reference', 'notes/experiments/2026-11-01/tinystories-full-bpe/full-training-result.json',
    '--baseline', (Join-Path $runs '20261101_190151_430986'),
    '--output', (Join-Path $runs 'full-training-profile')
)
& $python -B -X utf8 'notes/experiments/2026-11-07/tinystories-bpe-profile/profile_full_bpe.py' @profileArgs
```

脚本将 `summary.json`、`profile.txt`、`progress.jsonl` 和二进制 `bpe.prof` 写到输出目录。`profile.txt` 按累计时间和函数自身时间分别排序。本次原始表包含进度监控线程的等待，出现累计等待时间超过整次经过时间的情况。用 `summary.json` 中独立记录的阶段经过时间计算占比。
