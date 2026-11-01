# 2026-10-30：可配置训练、断点续训、耗时日志和生成入口

这次完成[交付核对](2026-10-30-A1-COMPLETION-AUDIT.md)安排的第一步。新增入口复用已有自写模型、AdamW、调度、数据采样和 loss，没有替换这些核心实现。

## 在 PyCharm 怎样使用

解释器继续选 `E:\CS336\assignments\a1-basics\.venv\Scripts\python.exe`。

### 训练

打开 `E:\CS336\assignments\a1-basics\cs336_basics\train_cli.py`。直接运行会使用现有 5 MB token 数组，创建独立的 `training_runs/日期时间` 目录，总计划 1000 步，本次先执行 100 步。

已经实际运行了一次默认设置，结果位置：

```text
E:\CS336\assignments\a1-basics\cs336_basics\training_runs\20261030_192617_854057
```

初始固定验证 loss 为 9.2105；第 100 步为 5.4833。它复现了此前严格基线第 100 步的数值，仍是小模型、短预算运行，不表示正式训练目标已完成。

下一次若要接着这一份训练，打开 PyCharm 的 Run → Edit Configurations，在 `train_cli.py` 的 Parameters 栏填写下面这一行，然后运行：

```text
--resume "E:\CS336\assignments\a1-basics\cs336_basics\training_runs\20261030_192617_854057\checkpoint_step_000100.pt" --run-steps 100
```

这会从第 101 步训练到第 200 步，沿用原计划、模型配置和学习率进度；不设置 `--resume` 则创建一份新实验。

| 参数 | 含义 |
| --- | --- |
| `--steps 1000` | 整个训练计划的更新次数，也决定余弦下降的终点 |
| `--run-steps 100` | 这个进程本次最多执行多少次更新 |
| `--resume 实际存档路径` | 从新入口的存档恢复，读取其原计划 |
| `--train-data`、`--val-data` | 一维整数 `.npy` 文件，通过 mmap 读取 |
| `--batch-size`、`--context-length`、`--d-model` 等 | 模型与训练尺寸 |
| `--eval-every`、`--save-every` | 验证记录与周期存档间隔 |
| `--output-dir` | 新实验输出目录，要求为空，避免覆盖结果 |

新运行也可以在 Parameters 中使用：

```text
--config "E:\CS336\assignments\a1-basics\configs\pilot_5m.json" --run-steps 100
```

这份 JSON 集中列出基线超参数；显式命令行参数覆盖 JSON。默认数据路径按脚本所在文件夹解析；用户自定义的相对路径按运行工作目录解析，跨目录使用时可填写绝对路径。

### 生成

打开 `E:\CS336\assignments\a1-basics\cs336_basics\generate_cli.py`。直接运行会加载原有 **1000 步** `train_5m_checkpoint.pt`，默认提示 `Once upon a time`、temperature=0.8、top-p=0.9、seed=42，生成最多 256 个新 token，遇 EOS 提前结束。

要使用新训练入口的 100 步模型，在该生成脚本的 Parameters 栏填写：

```text
--checkpoint "E:\CS336\assignments\a1-basics\cs336_basics\training_runs\20261030_192617_854057\checkpoint_step_000100.pt" --max-new-tokens 256 --seed 42
```

新存档自动提供模型配置。生成参数还包括 `--prompt`、`--temperature`、`--top-p`、`--vocab`、`--merges`、`--output-dir`。使用与训练 token ID 对应的同一套词表/merges；词表大小相同并不能证明语义映射相同，生成 JSON 会记录这些文件的校验和供复核。

生成结果单独保存在 `generation_runs/日期时间`，包括可读的 `sample.txt` 和记录 token IDs、模型/采样配置、实际长度及停止原因的 `generation.json`。

## 新增文件的分工

| 文件 | 作用 |
| --- | --- |
| `cs336_basics/train_cli.py` | 配置、数据校验、固定验证材料、训练、逐点日志、周期存档与恢复 |
| `cs336_basics/decoding.py` | 可复用的 `generate(...)` 函数，局部采样生成器、温度、top-p、EOS/长度停止 |
| `cs336_basics/generate_cli.py` | 读取模型和分词器、接收提示和采样参数、保存文本及元数据 |
| `configs/pilot_5m.json` | 当前学习用基线的集中配置示例 |
| `train_norm_ablation.py` 的 `PassThrough` | 替换原来的 `nn.Identity`，用自写 Module 原样传递输入 |

训练/生成入口当前支持原 pre-norm、SwiGLU、RoPE 基线；各结构消融仍使用此前对应的实验脚本。生成按原来的滑动上下文方式运行，没有实现 KV cache。

## 存档与日志保存什么

新存档保留模型、优化器、已完成步数、完整训练/模型配置、架构标记、数据文件 SHA256、累计计时和最佳验证 loss，还保留 Python、NumPy、PyTorch 的随机数状态。CUDA 状态在存在时也会保存，但本次只验证 CPU。

恢复前检查配置和数据内容，避免把另一份数据或训练计划接到旧结果上。调度由已完成步数和保存的总计划决定，不在每次启动时重新预热。只支持用新入口存档恢复训练；缺配置/RNG 的旧存档不伪装为可精确恢复，原旧脚本仍可使用。已知的旧 5 MB 基线可以直接在新生成入口读取；其他旧基线存档需显式提供 `--model-config`。

`metrics.csv` 每个记录点含：

```text
step, processed_tokens, elapsed_seconds, lr, train_loss, val_loss, grad_norm
```

训练 loss 是该次单批次 loss，验证 loss 是固定验证批次平均值；step=0 没有训练 loss。默认验证批次大小固定为 4，与训练 batch 可分别设置。累计 token 是更新次数×训练 batch×上下文长度，不是独立原始语料的大小。

elapsed_seconds 使用真实计时，覆盖运行段内的训练、验证和之前存档等耗时，续训会加上旧运行段计时，不包含停机等待或各次启动准备时间，最后一次存档写入未完整纳入存档内部的时间值。没有用总耗时按比例伪造各点时间。

存档先写临时文件再替换为目标文件。恢复要求 CSV 末尾与存档步数一致；如果异常中断导致日志超前，会拒绝续写，保留日志供先行核对，而不是自动回滚历史。新入口验证数据范围与校验和会额外扫描数据文件，完整语料上的启动成本还需评估。

## 实际验证结果

1. 当前 CPU、4 线程、同环境下，连续 4 步与 2 步+恢复2步的参数、优化器、RNG、数据指纹、迭代数和最佳验证值完全一致；除时间外的日志字段逐项相同，累计耗时非递减。
2. 固定种子重复采样输出一致；函数恢复模型原训练/评估模式，不改变全局 PyTorch 随机数状态。核对 EOS 首步停止、零生成长度、长提示窗口裁剪，以及非法温度、top-p、长度的拒绝行为。
3. 新存档自动读取配置的 CLI 生成通过；旧 1000 步模型两次独立 CLI 生成的 256 个 token IDs 完全一致。
4. JSON 配置和命令行覆盖通过；恢复时试图更改总步数被拒绝，原 CSV 未改变。
5. 换为自写 PassThrough 后，重新载入原去归一化实验的三份存档及归一化基线，重算的验证 loss 均与原记录一致。
6. 新默认入口已实际完成 100 步，验证 loss=5.483281135559082，与旧严格基线对应点一致。

这些是针对新增入口的行为核验。本轮未重复运行整套 A1 测试；此前核对时为 46 passed、2 个 Linux 内存测试 skipped，核心模型/优化器文件本轮未修改。CUDA、跨平台/跨硬件逐位一致性及异常断电恢复没有验证。

## 已保存的生成样例

本次使用原 1000 步样本模型，生成 **256 个新 token**，停止原因是 `max_new_tokens`，没有遇到 EOS。格式上满足“至少 256 token 或直到首次 EOS”的长度要求；训练模型本身仍未达到正式训练预算/验证目标。

[样例全文](experiments/2026-10-30/training-tools/sample.txt) · [生成配置与 token IDs](experiments/2026-10-30/training-tools/generation.json) · [100 步耗时曲线数据](experiments/2026-10-30/training-tools/metrics.csv) · [验证摘要](experiments/2026-10-30/training-tools/verification.json)

文本有局部英文句式，但人物、事件和指代不连贯，末尾因长度上限截断。影响因素至少包括：训练数据/更新预算较小、64-token 上下文窗口限制、模型容量，以及 temperature/top-p 的随机采样设置。本次曲线与文本只支持观察，不单独证明每个因素的因果贡献。

旧的 1000 步训练存档仍保留原内容。本次没有下载完整语料、启动长训练或 push；Unicode/SGD 书面题、全量分词、正式训练、发散边界、OWT 和最终报告等仍待补齐。
