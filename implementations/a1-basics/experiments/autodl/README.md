# AutoDL 单卡正式训练

本目录归档云端运行入口、训练计划和数据指纹。正式实验在 RTX PRO 6000 Blackwell Server Edition 上完成，单卡运行。TinyStories 和 OpenWebText 各 20,000 步、327,680,000 个目标位置。使用预先编码的 NumPy 数组，文件名、大小和 SHA-256 见 `data-manifest.json`。

在 Linux GPU 实例上，从仓库根目录或解压后的代码包根目录运行：

```bash
cd implementations/a1-basics/experiments/autodl
export PYTHONPATH="$(cd ../.. && pwd):$(cd ../owt-full && pwd)"
mkdir -p data
bash setup.sh
```

按清单放入本地准备好的对应数据和词表，再运行：

```bash
.venv/bin/python run.py verify tiny
.venv/bin/python -u run.py train tiny --until 200
.venv/bin/python -u run.py train tiny
.venv/bin/python run.py evaluate tiny
.venv/bin/python run.py generate tiny
```

把 `tiny` 换成 `owt` 可运行第二组。前 200 步使用完整训练的学习率计划并存档，后续命令自动从最近一次完整存档继续。目录内的 `runs/` 保存日志、生成、评估和模型存档。PyTorch 镜像须支持实例 GPU，初始化会执行 CUDA 矩阵乘法。

[实际配置、日志和生成](../../../../notes/experiments/2026-11-07/autodl-formal/) · [A1 交付](../../../../deliverables/a1-basics/README.md)
