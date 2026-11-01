# 2026-11-01：用正式 10k 词表编码完整 TinyStories

完整训练集和验证集均已重新编码，保存为独立的 uint16 一维 NumPy 数组。使用的是[完整训练集训练的 10k 分词器](2026-11-01-A1-TINYSTORIES-FULL-BPE.md)，没有重训词表，也没有用验证文本更新任何 BPE 规则。

## 实际结果

| 数据 | 原始字节数 | token 数 | token 数据 MiB | bytes/token | 编码及全部检查秒数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 训练集 | 2,227,753,162 | 541,229,347 | 1032.31 | 4.116 | 689.01 |
| 验证集 | 22,502,601 | 5,465,883 | 10.43 | 4.117 | 7.64 |

两份数组均完成全量逐字节还原检查，EOS 数分别为训练集 2,717,699、验证集 27,630，与原始文本一致。ID 范围均在 0–9,999 内，EOS ID 为 256。上表数组大小不含 .npy 头部；实际文件大小及数据哈希见 JSON。

本次进程生命周期工作集峰值为 94.41 MiB（Windows 工作集口径，包含两个 split、缓存和检查；不是算法净分配量）。程序先编码 valid，再编码 train，两份使用同一固定词表和同一片段缓存；训练集计时包含验证集已经暖好的缓存效应。

## 文件在哪里、怎样接入训练

目录：`E:/CS336/assignments/a1-basics/data/tokenized_tinystories_full/20261101_192402_611757`。

- `train.npy`：完整训练集。
- `valid.npy`：官方独立验证集，没有重新按 90/10 切分训练文本。
- `metadata.json`：源 revision、词表/merges 哈希、编码程序哈希、两份源 SHA-256、token 数据 SHA-256、dtype/shape/计时/缓存统计。
- `train_summary.json`、`valid_summary.json`、`progress.jsonl`：逐份结果及过程记录。

后续使用已有 `train_cli.py` 时，将 Parameters 中的数据路径指向这两份新数组。例如：

```text
--train-data "E:\CS336\assignments\a1-basics\data\tokenized_tinystories_full\20261101_192402_611757\train.npy" --val-data "E:\CS336\assignments\a1-basics\data\tokenized_tinystories_full\20261101_192402_611757\valid.npy"
```

这会沿用训练入口的默认小模型配置和本次 100 步限制，不自动达到正式训练预算；更大的模型/批次/步数需要另行安排。新词表与旧词表同为 10,000 条，但 ID 的含义有大量变化，因此新数组应开启新的模型训练实验，不能直接接旧样本模型存档续训。生成文本时也必须使用本次正式词表和 merges。

旧 `tinystories_5m_train.npy`、`tinystories_5m_val.npy`、旧词表和旧存档没有替换。

## 实现改动

新增 `cs336_basics/encode_full_tinystories.py`，PyCharm 直接运行会用本次正式分词器，创建独立输出目录并编码两份完整文本。可通过 `--tokenizer-dir`、`--data-dir`、`--output-dir`、`--cache-size` 指定路径和缓存上限；目录已存在时拒绝覆盖。

其中 `CachedTokenizer` 继承现有分词器，沿用同一正则、按 merge rank 选择相邻对与 `merge_pair` 算法，只将固定反向词表/合并顺序表预先构建，并缓存重复预分词片段的 token ID。缓存键是完整片段（包含前导空白），最大 100,000 个条目，不用近似规则或第三方分词器替代编码；原 `tokenizer_demo.py` 和磁盘上的 tests/adapters 均未修改。

按 EOS 边界分块，每块保留完整 UTF-8 字符，读取时设置 `newline=""`，保留 CRLF 等原始换行。缓存、频率表和最大文档仍会占内存，不能把整体内存称为严格常数。

程序先顺序写入本次运行独有的临时 uint16 数据流，在总 token 数已知后添加合法 .npy 头部并流式复制。全部检查通过后才启用正式数组并删除本次临时文件；失败时不将未检查文件冒充成品。不需要把整个文本或数亿 token 放进内存。

## 如何验证

- 49 次分块/编码等价核验：中文、CRLF、空白、EOS 跨块、连续 EOS、无 EOS、空文本，均与原分词器 token ID 逐项一致；小语料保存后的 .npy 也与原编码完全一致。
- 临时在测试进程中将官方测试 factory 指向 `CachedTokenizer`，结果 **23 passed、2 skipped，4.88 秒**。两项跳过仍是 Linux 内存限制测试。最初测试没有设置本地参考数据缓存，触发了参考分词器的联网失败；设置已有 `DATA_GYM_CACHE_DIR` 后重跑通过，没有改动测试断言。
- 每一块将 token ID 对应的字节拼接，与原始 UTF-8 字节逐项比较；累计源 SHA-256 与已固定版本的官方校验值相同。这覆盖完整文件，不只抽查样本。
- 将编码流 token 数据的 SHA-256 与保存后的整个 .npy 数据区重新计算出的 SHA-256 比较；重新打开数组检查形状与 uint16 类型，避免仅验证写盘前的内存结果。
- 两份最终数组都能用 `mmap_mode="r"` 加载，并通过现有 `get_batch` 生成形状 `[2,16]`、类型 int64 的输入/目标批次，目标正确右移一个位置。
- 本次没有重新运行整套模型训练测试，也没有开始语言模型训练。完整字节还原与 dtype/shape 合法不等于模型已经学会了文本。

[完整编码结果与验证](experiments/2026-11-01/tinystories-full-encoding/encoding-result.json)、[进程记录](experiments/2026-11-01/tinystories-full-encoding/progress.jsonl)。吞吐是编码、字节还原检查、磁盘写入和数组校验的端到端管道吞吐，不是独立纯编码基准；这里没有据此提前完成 OWT/跨域比较或 825 GB 标准吞吐外推题。

## 当前范围

本次完成完整 TinyStories 的新词表配套编码；下一步可安排基于新数据的模型训练。完整训练集 BPE profile、OWT/跨域实验、正式语言模型训练预算、发散边界和最终 writeup/code 仍未完成。本次新增一个活动模块并更新笔记/进度，没有提交或推送 Git；数组与数据不上传。
