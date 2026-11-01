# 2026-11-01：正式词表与完整数据上的新模型训练起步

本次给本机安排 CPU 学习配置，总计划 20,000 步（40,960,000 目标 token），**实际只运行并完成 100 步**，没有将计划预算算作已经完成。模型从随机参数开始，没有恢复旧词表训练的模型。

## PyCharm 下一步怎么运行

解释器继续使用 `E:/CS336/assignments/a1-basics/.venv/Scripts/python.exe`，打开 `E:/CS336/assignments/a1-basics/cs336_basics/train_cli.py`，在 Run → Edit Configurations 的 Parameters 中填入：

```text
--resume "E:\CS336\assignments\a1-basics\cs336_basics\training_runs\20261101_194227_274998\checkpoint_step_000100.pt" --run-steps 900
```

这会从第 101 步接着训练到第 1,000 步，沿用原来的模型、AdamW 状态、RNG 和学习率进度。预计先用几分钟再看验证曲线；第 1,000 步后再使用同目录的 `checkpoint_step_001000.pt` 继续。续训不要同时传 `--config`。

如果要从头重做一组实验，才使用下面参数：

```text
--config "E:\CS336\assignments\a1-basics\configs\tinystories_full_cpu_40m.json" --run-steps 100
```

直接点运行且 Parameters 为空仍会使用旧 5 MB 默认设置；必须填入上面的新配置或新存档。`--run-steps` 控制本次执行多少步，`steps=20000` 才是完整学习率计划的终点。

## 本次实际结果

- 模型参数 3,413,120，结构为 pre-norm + RoPE + SwiGLU，无权重共享。
- 固定 10 个验证批次，每批 8 条、每条 256 token，独立 val_seed=12345。
- 第 0 步验证 loss=9.2332；第 100 步=6.0035。
- 本次训练/验证日志累计约 38.52 秒；不包含数据初始全量检查、哈希、建模等准备，也不把存档 I/O 计时视为完整端到端上界。
- 完成目标 token 数 204,800；计划目标 40,960,000。随机抽取窗口可能重叠，不等于看过同样数量的不同原文位置，也不要求每步读取整个数组。
- 第 100 步 lr=0.000500，处于前 200 步 warmup 的中间；它是有意安排，不是调度未到最大值的错误。

运行目录：`E:\CS336\assignments\a1-basics\cs336_basics\training_runs\20261101_194227_274998`。包含 config.json、metrics.csv、summary.json、checkpoint_step_000100.pt；另保存 tokenizer_provenance.json，指明本次正式分词器的哈希与数组出处。checkpoint 本身已有完整数据文件哈希和模型配置。

## 为什么选这组配置

| 配置项 | CPU 学习配置 | 讲义原始模型配置 |
| --- | ---: | ---: |
| 词表 | 10,000 | 10,000 |
| 上下文 | 256 | 256 |
| 层数 | 4 | 4 |
| 隐藏维度 | 128 | 512 |
| 头数 | 4 | 16 |
| FFN 宽度 | 384 | 1,344 |

当前 PyTorch 为 2.11.0+cpu，CUDA 不可用；本机可见 8 个物理核、16 个逻辑核。新配置使用 8 线程、batch_size=8；AdamW beta=(0.9,0.999)、eps=1e-8、weight_decay=0.01、裁剪 1.0；max_lr=0.001、min_lr=0.0001、warmup=200，余弦终点为第 20,000 步。这些是起步候选，不宣称已调到最优学习率。

先用真实训练数据测量，每组 7 次更新，前 2 次热身、后 5 次计算时间；包含取 batch、前后向、裁剪和优化器，不包含验证、存档、完整数据扫描：

- 小配置 4 线程：每步约 0.545 秒，约 3,759 tokens/s。
- 小配置 8 线程：每步约 0.337 秒，约 6,070 tokens/s；20,000 步纯更新约 1.87 小时。
- 课程原始配置 4 线程：每步约 2.527 秒，同样 batch/context 的 20,000 步纯更新约 14.04 小时；没有测其 8 线程最优速度，所以不把这个数当作原配置的性能极限。

因此本机本轮先选较小配置，完整运行先预留 2–3 小时，实际依赖负载与验证/存档开销。短基准不是长期运行性能保证；基准内存峰值是跨配置进程生命周期累计量，不拿来比较单模型峰值。

40,960,000 = 8 × 256 × 20,000，接近讲义 CPU/MPS 的约 40M token 建议。本次还额外缩小了模型，**并不等同于只减少 token 预算的讲义原配置**；不能保证达到验证 loss≤2.00，也不能把 100 步下降当成全部正式实验已完成。

每 100 步验证；每 1,000 步或本次结束保存存档，所以本次第 100 步也有可恢复存档。之后先观察第 1,000 步结果，再决定学习率及后续预算。

[配置/基准/100步结果](experiments/2026-11-01/tinystories-full-training-start/training-start-result.json)、[曲线 CSV](experiments/2026-11-01/tinystories-full-training-start/metrics.csv)。本次未修改训练核心、未重跑整套测试；核验了新配置、数据来源、模型参数量、有限训练 loss/梯度、验证下降及存档可加载。没有提交或推送 Git。
