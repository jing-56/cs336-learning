# OpenWebText 全量本地实验

本目录归档下载、预分词、32K BPE、数据编码、模型训练、全量验证和词表对照的源码。[学习记录](../../../../notes/2026-11-07-A1-OPENWEBTEXT-FULL.md)与[实验结果](../../../../notes/experiments/2026-11-07/owt/)保存配置、耗时、指标、生成样例和产物校验值。[源码清单](SOURCE_MANIFEST.json)

使用已恢复个人实现的 CS336 A1 环境。Python 环境需要 `regex`、`psutil`、`numpy`、`tiktoken` 和可用 CUDA 的 PyTorch。以下从工作区根目录运行，归档脚本把下载和生成的数据写到脚本目录。完整复现需要为压缩包、编码数组和周期存档预留磁盘空间。

```powershell
$root = (Get-Location).Path
$owt = Join-Path $root 'implementations/a1-basics/experiments/owt-full'
$project = Join-Path $root 'assignments/a1-basics'
$py = Join-Path $project '.venv/Scripts/python.exe'
$env:PYTHONPATH = $project
$env:PYTORCH_KERNEL_CACHE_PATH = Join-Path $owt 'cuda_kernel_cache'

& $py (Join-Path $owt 'run_full_count.py')
& $py (Join-Path $owt 'train_from_counts.py') full
& $py (Join-Path $owt 'download_owt.py') --split valid
& $py (Join-Path $owt 'encode_owt.py') (Join-Path $owt 'owt_train.txt.gz') (Join-Path $owt 'owt-full-32k.pkl') (Join-Path $owt 'owt-train-full.npy')
& $py (Join-Path $owt 'encode_owt.py') (Join-Path $owt 'owt_valid.txt.gz') (Join-Path $owt 'owt-full-32k.pkl') (Join-Path $owt 'owt-valid-full.npy')

$config = Join-Path $root 'notes/experiments/2026-11-07/owt/training-config.json'
& $py (Join-Path $owt 'train_owt.py') --config $config --train-data (Join-Path $owt 'owt-train-full.npy') --val-data (Join-Path $owt 'owt-valid-full.npy') --output-dir (Join-Path $owt 'training-run') --run-steps 20000
& $py (Join-Path $owt 'evaluate_full.py') --checkpoint (Join-Path $owt 'training-run/checkpoint_step_020000.pt') --validation (Join-Path $owt 'owt-valid-full.npy') --output (Join-Path $owt 'full-eval/summary.json') --batch-size 16
```

下载、计数和编码可以从落盘的断点续跑。BPE 合并中断后可以重新使用全量频率表。训练入口沿用个人实现中的配置、固定验证批次和存档格式。

全量验证核对存档中的验证集哈希，按位置累计交叉熵，每约 100 万个位置更新一次进度文件。中断后用同一命令从最后一次记录续跑；试跑可用 `--max-targets` 和另一个输出路径。全量评估只读取模型权重，不更新模型。

`compare_tokenizers.py --help` 可查看词表对照和编码测速的输入。对照使用之前固定的两组验证文档，每组一个含 10 条字符串的 JSON 数组。原文保留在本地，归档结果包含样本哈希和汇总数字。
