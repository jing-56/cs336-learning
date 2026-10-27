# A1 本人实现源码归档

这是记录 **2026-10-27** 学习内容、于 **2026-10-28** 整理同步的实现快照。活动开发目录仍为 `E:/CS336/assignments/a1-basics`；本目录让个人源码可以随学习仓库一起保存，不把整个上游作业仓库嵌进 Git。

## 包含什么

- `cs336_basics/`：本人学习过程中实现的 BPE、Tokenizer、数据采样、模型、优化器、训练支持组件和演示脚本。首次同步也包含此前写好的依赖，具体来源与 SHA-256 见 `SOURCE_MANIFEST.json`。
- `adapter_functions.py`：从本地测试适配器中提取的已实现函数。保留函数体，去掉上游说明文字和类型注解，便于恢复接口；不包含测试断言、fixtures、上游未实现占位。
- `cs336_basics/__init__.py`：本次归档新增的最小包标识；没有复制上游的版本查询实现。
- `.gitignore`：防止在归档目录运行演示时误加入数据、pickle、模型存档和性能文件。
- `.gitattributes`：保留 Python 快照的原始换行字节，避免 Git 自动换行转换使清单哈希失效。

上游自带的 `pretokenization_example.py`、测试文件、课件、语料、虚拟环境、`.npy/.pkl/.pt/.prof` 等不在归档范围。活动目录中的代码没有被本次归档改写。

## 实现导航

| 文件 | 用途 |
| --- | --- |
| `bpe.py`、`tokenizer_demo.py` | BPE 训练、编解码、特殊 token、逐行编码、文件加载 |
| `data.py`、`prepare_tokens.py` | 随机 batch 与 token 数组准备 |
| `embedding.py`、`linear.py`、`rmsnorm.py`、`swiglu.py` | 基础层 |
| `attention.py`、`multihead_attention.py`、`rope.py` | 缩放点积注意力、多头注意力、旋转位置编码 |
| `transformer_block.py`、`transformer_lm.py` | pre-norm Block 与完整 LM 前向 |
| `nn_utils.py` | 稳定 softmax、交叉熵、全局梯度裁剪 |
| `optimizer.py`、`lr_schedule.py`、`checkpoint.py` | AdamW、学习率调度、训练状态存取 |
| `train_demo.py` | 固定 32 个预测位置、CPU 100 步的拟合实验 |
| `train_loop.py`、`train_5m.py` | 随机批次、固定验证、调度与分段存档恢复；小样本及 5 MB 两种入口 |
| `prepare_5m_tokens.py` | 按故事边界划分约 90/10 文本，编码训练/验证数组 |
| `generate_demo.py`、`generate_5m.py` | 加载存档，温度/top-p 采样、EOS 停止和上下文截取 |
| `train_rope_ablation.py`、`run_rope_multiseed.py` | 有/无 RoPE 严格配对与三个种子汇总 |
| `train_lr_sweep.py` | 相同初始参数和批次的学习率尺度扫描，复用已核对的基线 |
| `*_demo.py`、`experiment_*.py`、`view_training_sample.py` | 分步理解与样本实验 |

部分早期文件名保留了 `demo`，其中也包含正在使用的类；本次同步没有为改名而重构源码。

## 恢复到官方作业环境

推荐先按仓库的 [环境说明](../../plan/01-SETUP.md) 准备官方 A1 与依赖，再在 `E:/CS336` 下执行：

```powershell
Copy-Item -Path .\implementations\a1-basics\cs336_basics\*.py -Destination .\assignments\a1-basics\cs336_basics\ -Force
```

这会替换目标目录中的同名 Python 文件；若目标有尚未归档的新改动，先保存它们。将 `adapter_functions.py` 中相应同名函数的实现接回官方 `tests/adapters.py`；不要用这个提取文件覆盖整份上游适配器。

使用已有的官方环境运行：

```powershell
cd E:\CS336\assignments\a1-basics
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_model.py tests/test_nn_utils.py tests/test_optimizer.py tests/test_data.py tests/test_serialization.py -v
```

### 数据与训练演示

源码归档不自带语料和已生成的模型文件。`experiment_scale.py` 依赖官方 `tests/fixtures/tinystories_sample_5M.txt`，生成 10k 词表/合并规则；随后 `prepare_tokens.py` 使用官方小故事样本生成 `tinystories_sample_tokens.npy`。`train_demo.py` 依赖这个数组，运行后自行生成 checkpoint。

在准备好官方 fixtures 的活动作业目录下，可依次运行：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cs336_basics.experiment_scale
.\.venv\Scripts\python.exe -X utf8 -m cs336_basics.prepare_tokens
.\.venv\Scripts\python.exe -X utf8 -m cs336_basics.train_demo
```

`experiment_scale.py` 的峰值内存读取使用 Windows `peak_wset`；上述样本实验按当前 Windows 环境验证。早期 BPE 小演示还依赖自行准备的 `corpus_demo.txt`，该文本未归档。

## 已验证与未完成

具体测试与训练数值见 [10 月 26 日记录](../../notes/2026-10-26-A1-TRANSFORMER-AND-LOSS.md)、[10 月 27 日记录](../../notes/2026-10-27-A1-TRAINING-AND-CHECKPOINT.md)。

当前已跑通样本规模训练/验证/生成与恢复流程；checkpoint 不保存 RNG/采样进度。生成脚本未显式固定采样 RNG；还没有统一 CLI。Tokenizer 任意切块边界、Linux 内存限额与完整语料实验仍未完成。5 MB 小模型与正式实验要求不同。

2026-10-27 累计归档现有 38 个活动模块、最小包标识和提取的接入函数。同步复查逐文件哈希、Python 语法及 20 项模型/训练支持测试；详细范围与记录见 SOURCE_MANIFEST.json。20 项不是 A1 全套测试。

### 继续复现今天的样本实验

在官方环境中恢复源码，准备好原来的 10k 词表后依次运行：

```powershell
./.venv/Scripts/python.exe -X utf8 -m cs336_basics.prepare_5m_tokens
./.venv/Scripts/python.exe -X utf8 -m cs336_basics.train_5m
./.venv/Scripts/python.exe -X utf8 -m cs336_basics.generate_5m
./.venv/Scripts/python.exe -X utf8 -m cs336_basics.train_rope_ablation
./.venv/Scripts/python.exe -X utf8 -m cs336_basics.run_rope_multiseed
./.venv/Scripts/python.exe -X utf8 -m cs336_basics.train_lr_sweep
```

`train_5m.py` 当前每次最多训练 900 步，历史实验先设 100、再改 900。新机器从头运行会先完成 900 步，再次运行补齐 100 步；有旧存档时自动恢复，不会自动重新从头训练。不要把旧存档用于不同结构。

严格对照脚本各从头连续训练 1000 步并新建结果文件夹；多种子脚本需要先有配置一致的种子 42 结果，学习率扫描也需要同初始化及批次指纹的 0.003 基线。无 RoPE 存档加载后生成时同样需要关闭 RoPE，配置记录中的 use_rope 不能忽略。

已提交的[实验 CSV/配置](../../notes/experiments/2026-10-27/README.md)可直接查看，无需下载模型。生成样例、数值解释及尚未完成项见[当天晚间记录](../../notes/2026-10-27-A1-TRAINING-GENERATION-AND-EXPERIMENTS.md)。
