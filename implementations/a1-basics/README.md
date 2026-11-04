# A1 本人实现源码归档

11 月 4 日的堆版、紧凑索引版和计时工具单独归档在 [OpenWebText BPE 实验](experiments/owt-bpe/)。下文仍描述 11 月 1 日的活动源码快照。

本目录为 **2026-11-01** 累计快照，包含 56 个本人实现的活动模块、最小包标识和已实现接入函数，共 58 个 Python 文件，另附三个训练配置。活动开发目录仍是 `E:/CS336/assignments/a1-basics`。上游 starter、测试、fixtures、数据、模型和环境未复制。

## 本次同步改了什么

- 比 10 月 27 日快照新增 18 个模块：batch/结构对照、资源核算、训练与解码 CLI、SGD 实验、完整数据准备/BPE 实验/缓存编码、CPU/GPU 存档迁移。
- `bpe.py` 同步按文档边界分块预分词；保持原有预分词、全局统计和合并规则。
- `train_cli.py` 保存配置、模型/优化器、数据指纹及 Python/NumPy/Torch RNG；同设备恢复已有短运行逐位核验。显式设备迁移使用新目录，不宣称 CPU/GPU 逐位一致。
- `generate_cli.py` 自动读取基线存档的模型配置，支持温度、top-p、局部采样 RNG、EOS 停止及结果记录。完整语料训练必须显式使用匹配的 vocab/merges。
- `configs/` 包含小样本试运行、完整数据 CPU 与 GPU 学习配置；其中绝对路径是原机器布局，换机器需要调整。

逐文件来源、SHA-256、18 个新增文件及修改清单见 [SOURCE_MANIFEST.json](SOURCE_MANIFEST.json)。`.gitattributes` 保留 Python 和训练配置的原始换行字节；`adapter_functions.py` 只提取实现函数，不包含上游测试或说明文字。`__init__.py` 是最小包标识，不复制上游版本查询代码。

## 模块导航

| 模块 | 用途 |
| --- | --- |
| bpe.py、tokenizer_demo.py | BPE 训练、特殊 token、编解码和文件接口 |
| embedding/linear/rmsnorm/swiglu/attention/multihead_attention/rope | 基础层、注意力和位置编码 |
| transformer_block.py、transformer_lm.py | pre-norm/SwiGLU/RoPE 语言模型 |
| data.py、nn_utils.py、optimizer.py、lr_schedule.py、checkpoint.py | 批次、loss、裁剪、AdamW、调度及基本存档 |
| train_cli.py、decoding.py、generate_cli.py、migrate_training_device.py | 配置、训练、RNG 续训、生成、显式设备迁移 |
| prepare_tinystories.py、experiment_bpe_corpus.py、encode_full_tinystories.py | 完整数据校验、BPE 实验、缓存编码及全量还原检查 |
| train_*ablation.py、train_batch_size.py、train_post_norm.py、run_rope_multiseed.py、train_lr_sweep.py | 小配置对照、多个种子和学习率实验 |
| count_*.py、*accounting.py | 参数量、FLOPs、内存、AdamW 计算与时间核算 |
| sgd_learning_rate.py、*_demo.py、experiment_*.py | 逐题实验和分步教学演示 |

## 恢复与运行

先按[环境说明](../../plan/01-SETUP.md)准备官方 A1，再将本目录 `cs336_basics/*.py` 复制到官方包目录，将 `configs/*.json` 复制到配置目录；先保存目标目录中尚未归档的改动。将 adapter_functions.py 的同名实现接回官方 tests/adapters.py，不用它覆盖整份上游文件。

在作业根目录运行自己的解释器，例如：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m cs336_basics.train_cli --config configs/tinystories_full_gpu_40m.json --run-steps 100
```

三个配置引用的数据和存档不在 GitHub，需要先重新准备。训练入口无参数仍是旧 5 MB/CPU 教学默认值；完整数据模型要使用对应配置或带元数据的最新存档。旧式 checkpoint.py 不记录 RNG；统一 train_cli.py 的新格式记录 RNG，两者不能混为一谈。不同模型结构的消融存档不能直接交给只支持基线结构的生成入口。

## 验证与结果边界

最近完整官方套件记录为 2026-11-01：46 passed、2 skipped、17.06 秒；两个跳过项为 Linux 内存限制。官方套件主要走 CPU，CUDA 前后向、AdamW、设备迁移和真实续训另有实测。归档额外核对 56 个模块与三个配置的原始 SHA-256、58 个 Python 文件语法及 adapter 实现一致性。

[完整训练与生成记录](../../notes/2026-11-01-A1-FULL-TRAINING-AND-GENERATION.md)中，3.41M 参数学习模型完成 40.96M 目标 token，固定批次验证 loss=2.0210，生成 200 个新 token 后 EOS。此配置小于课程原始模型；OWT、完整 BPE profile、正式实验范围、Linux 内存验证和最终 writeup/code 交付仍未全部完成。数值、CSV、生成文本可随笔记查看，原数据和模型不上传。
