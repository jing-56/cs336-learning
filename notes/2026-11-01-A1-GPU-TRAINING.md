# 2026-11-01：从 CPU 切换到 RTX 3050 GPU 训练

已安装并验证 PyTorch 2.11.0+cu126，CUDA 12.6 可用，显卡为 NVIDIA GeForce RTX 3050 Laptop GPU（4 GB 显存）。当前 NVIDIA 驱动 546.92 保持原样；真实 CUDA 前向、反向、AdamW 和存档续训均成功。

> 后续已[完成 20,000 步训练与故事生成](2026-11-01-A1-FULL-TRAINING-AND-GENERATION.md)，最新为 checkpoint_step_020000.pt。下文第 200 步命令仅保留为迁移过程记录，不用于已完成运行的再次续训。

## 当时的 PyCharm 续训设置

解释器仍选 `E:/CS336/assignments/a1-basics/.venv/Scripts/python.exe`；打开 `E:/CS336/assignments/a1-basics/cs336_basics/train_cli.py`，Run → Edit Configurations → Parameters 填入：

```text
--resume "E:\CS336\assignments\a1-basics\cs336_basics\training_runs\20261101_gpu_from_cpu100\checkpoint_step_000200.pt" --run-steps 800
```

这会从第 201 步继续到第 1,000 步，使用 GPU。想一次跑完剩余计划，可将 800 改为 19800。下次继续使用对应运行目录里最新的存档；不要同时传 --config，也不要用原 CPU 目录的存档冒充 GPU 最新进度。

如果要从头新开 GPU 实验，使用：

```text
--config "E:\CS336\assignments\a1-basics\configs\tinystories_full_gpu_40m.json" --run-steps 100
```

默认无参数运行仍是原 5 MB/CPU 教学配置；本次没有偷偷更改默认模型或默认数据路径。

## 当前实测

| 项目 | 实测 |
| --- | --- |
| 同模型 CPU 8 线程短基准 | 0.337377 秒/步 |
| GPU 同步计时短基准 | 0.078774 秒/步 |
| 短基准提速比 | 4.28 倍 |
| GPU 目标 token 吞吐 | 25,998 tokens/s |
| PyTorch 峰值分配显存 | 460.30 MiB |
| PyTorch 峰值保留显存 | 520.00 MiB |
| 实际 GPU 续训 | 第 101–200 步，100 次更新 |
| 续训训练/验证活跃耗时 | 8.454 秒 |
| 第 100 步验证 loss（CPU） | 6.0035 |
| 第 200 步验证 loss（GPU） | 4.3970 |

GPU 基准使用同一第 100 步模型和 AdamW 状态，batch=8、context=256、FP32、不使用 AMP，前 2 步热身、后 10 步计时；每步前后 `torch.cuda.synchronize()`，包含取 batch、前后向、有限值检查、梯度范数/裁剪及 AdamW，排除验证、模型初始化、文件哈希和存档。

峰值分配/保留显存是 PyTorch 统计，不包含桌面、其他程序、全部驱动上下文占用。CPU 基准和 GPU 基准不是同时运行，负载/温度会影响结果。按短基准 20,000 步纯更新约 26.3 分钟，完整计划先预留约 30–40 分钟，随实际日志修正；首次大包下载不属于训练耗时。

全计划仍是 20,000 步、40,960,000 个目标 token；**实际只累计完成 200 步、409,600 个目标 token**，其中前 100 步 CPU、后 100 步 GPU。模型仍是 4 层、d=128、4 头、FFN=384 的学习配置，未改成课程 d=512 原始模型，不保证达到验证 loss≤2.00。

## 如何保留 CPU 进度

新增 `cs336_basics/migrate_training_device.py`，在新目录复制存档与截至选定步数的历史日志，将 config.device 改为 cuda。原 CPU 目录 `20261101_194227_274998` 和第 100 步存档未改动；新 GPU 目录为 `20261101_gpu_from_cpu100`。

迁移前后逐项核对模型、AdamW、数据哈希、步数、累计活跃时间、Python/NumPy/CPU Torch RNG 均完全相同。CPU 存档没有 GPU RNG，因此以原 seed=42 初始化新的 CUDA RNG，并记录在 device_migration.json；不宣称跨设备逐位复现。

同一模型/批次 CPU 与 GPU loss 均为 5.9453502，最大 logit 绝对差 2.38e-06，通过容差核验；实际优化器 m/v 均在 CUDA 上。迁移工具不修改原训练入口的严格配置校验，通过显式的新存档保留硬件切换记录。

GPU 目录的 metrics.csv 包含原 CPU 0/1/100 步历史和 GPU 200 步结果；elapsed_seconds 是跨设备累计活跃时间，不是纯 GPU 时长。GPU-only 续训耗时用第 200 步与源存档累计时间之差估算，仍不包含启动/加载和所有存档 I/O。

## 环境安装与缓存

使用 [PyTorch 官方 Windows/Python 3.12 CUDA 12.6 wheel](https://download.pytorch.org/whl/cu126/torch/)，文件 `torch-2.11.0+cu126-cp312-cp312-win_amd64.whl`，2,596,453,924 字节，SHA-256 `1b7d728ddcf84b12f64ca10ec5f6ce91dd9edcc5e62ed5f0a7260ca00f029e8b`。

索引形式安装未能解析指定版本；官方精确文件的在线整包下载很慢并发生重试，已在替换环境前停止。随后从同一官方地址按 Content-Range 分段下载，合并后校验完整官方 SHA-256，才安装本地文件。安装只替换 torch，其他已核验依赖不变，pyproject.toml/uv.lock 未改动；目前 PyCharm 直接使用这份 .venv。若以后用锁文件重新同步环境，应重新确认实际 torch 版本和 CUDA 可用性。

下载/校验信息保存在 `E:/CS336/.tools/downloads/torch-2.11.0-cu126/verified_wheel.json`。未安装额外系统 CUDA Toolkit，未更新驱动或重启机器；[NVIDIA 的 minor-version compatibility 文档](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)有适用条件，本次结论以真实运行成功为依据。

首次 GPU 模型初始化曾提示默认 C 盘临时缓存目录无法创建。训练入口现仅在 device=cuda 时，将 `PYTORCH_KERNEL_CACHE_PATH` 缺省设为作业内 `.cuda_kernel_cache` 并创建目录，同时尊重用户显式配置；独立一步 GPU 初始化/训练已验证不再出现该提示。未对跨进程缓存命中收益作性能承诺。参见 [PyTorch CUDA JIT 缓存说明](https://docs.pytorch.org/docs/main/notes/cuda.html#just-in-time-compilation)。

## 验证与记录

安装后重跑当前整套测试：**46 passed、2 skipped，17.06 秒**。跳过的仍是两项 Linux 内存限制测试。官方套件主要验证 CPU 路径，CUDA 由独立同批次比较、12 步基准和真实 100 步续训补充；不把 46 项都称为 GPU 测试。

[环境、迁移、速度/显存与续训结果](experiments/2026-11-01/gpu-switch/gpu-switch-result.json)、[跨设备曲线 CSV](experiments/2026-11-01/gpu-switch/metrics.csv)、[本次全套测试 XML](experiments/2026-11-01/gpu-switch/test-results.xml)。

本次新增迁移模块、GPU 配置，并修改训练入口的 CUDA 缓存路径；已更新笔记/进度。本次历史记录随后连同完整训练结果整理进学习仓库，数据/模型/环境产物不上传。
