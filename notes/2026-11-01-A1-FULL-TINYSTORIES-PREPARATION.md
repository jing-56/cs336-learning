# 2026-11-01：完整 TinyStories 数据准备与 BPE 分块读取

本次完成完整数据下载/校验、验证集调试训练，以及按文档边界分块预分词。**尚未在完整训练集上完成正式 10k BPE 训练**；验证集调试结果不能替代 `train_bpe_tinystories` 的训练集结果。

## 数据来源与实际文件

使用本地作业 README 指定的 [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) 中 `TinyStoriesV2-GPT4-train.txt` 和 `TinyStoriesV2-GPT4-valid.txt`，不是其他版本的 JSON/Parquet；下载时固定 Hugging Face revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，以该 revision 的 LFS SHA-256 和字节数校验。

| 文件 | 实际字节数 | EOS 出现次数 | SHA-256 |
| --- | ---: | ---: | --- |
| train | 2,227,753,162 | 2,717,699 | `6418d412de72888f52b5142c761ac21a582f7d1166f0bfbdb5f03ccfdec90443` |
| valid | 22,502,601 | 27,630 | `6874bae9a4c1a4e7edcf0e53b86c17817e9cf881fc75ff2368da457b80c0585d` |

两个文件均通过 SHA-256、完整字节数和严格 UTF-8 解码检查。EOS 出现次数是直接计数，不假定等于所有情况下的完整文档数。数据保存在 `E:\CS336\assignments\a1-basics\data`，不上传 Git。

新增 `cs336_basics/prepare_tinystories.py`：默认下载两份指定文件；保存官方源元数据和校验 manifest；已存在文件先核验；使用 `.part` 并支持 HTTP Range 续传；固定版本避免续传拼接不同数据。下载失败可重新运行，已有正式文件校验不符时停止，不覆盖它。

## 为什么现在改读取方式

检查时系统总内存约 15.2 GiB、可用约 5.2 GiB。旧 `train_bpe` 先 `read_text()` 读完整文本，再 `regex.split()` 生成分段副本；2.23 GB 输入在这些副本及统计结构叠加时会有较大内存压力，因此先处理读取方式，未冒险直接整份读入。

新增 `bpe.py::pretokenize_file_counts`，`train_bpe` 调用它取得相同的频率表；单个非空特殊标记时按约 1 Mi 字符读取，处理到最后一个完整标记，未完成文档留到下一块。TextIO 保持 UTF-8 字符完整，换行归一化与旧 `read_text` 相同；多个特殊标记或不提供标记时仍保留旧路径。

这改变读取与统计方式，保留 BPE 的计数加权、频率并列排序、倒排索引和合并更新。临时文本内存与块大小/最长文档有关，全局 Counter 仍随不同片段的数量增长；不能称总内存恒定。尚未添加多进程或优化每轮寻找最大频率的 `max()`。

## 验证集调试运行

新增 `cs336_basics/experiment_bpe_corpus.py`，PyCharm 直接运行默认使用完整验证集作为调试语料，目标词表 10,000，特殊标记 `<|endoftext|>`。每次保存独立的词表、merges、summary、profile.txt 和原始 .prof；**正常计时和 profile 分成两次运行**，不混用其时间。

这次基准在修改分块读取之前运行，目录为 `cs336_basics/bpe_corpus_runs/20261101_validation_debug`：

- 正常训练 37.065 秒；带 profiler 的另一次运行 75.909 秒，两次输出完全一致。
- 进程训练前 RSS 26.42 MiB，训练后当前 RSS 39.20 MiB；截至训练结束、运行 profiler 之前的 Windows 进程生命周期峰值 96.18 MiB。这不是纯算法分配量，也不是完整训练集峰值；并行下载会影响机器负载和计时。
- 实际词表 10,000，合并规则 9,743；最长普通 token `b' accomplishment'`，15 字节，是带前导空格的完整英文单词，符合预分词规则。
- 英中混合示例能完整还原。特殊标记作为独立词条，不参与普通合并。
- profiler 显示 `max` 累计约 60.16 秒，其中比较 key 的 lambda 约 27.57 秒；预分词累计约 12.02 秒。累计时间有包含关系，不应把 `max` 与 lambda 相加。

修改后再次在该验证集训练，词表与 merges 的内容和顺序均与上面完全一致；这次等价检查没有另报独立峰值内存，不拿旧测量冒充改进后的内存测量。

## 词表大小相同，不代表 token ID 含义相同

与旧 5 MB 样本的词表相比，10,000 个 ID 中 9,640 个 ID 对应的字节内容不同。例如 ID 257 从 `b'he'` 变为 `b' t'`，ID 258 则反过来。不同语料改变频率和合并顺序；基础 256 字节与一个特殊标记之后的 ID 由本次训练决定。

因此新词表不能直接配旧 token 数组或旧模型使用。旧 `tinystories_sample_10k_*.pkl`、5 MB `.npy` 和模型存档保留各自配套关系；此次验证集词表仅作调试材料，不作为之后正式语言模型的训练词表。

## 本次核验与下一步

- 7 类边界案例、51 次频率表等价比较：含中文、CRLF、空白、标记跨块、自重叠标记、多标记回退、无标记和空文本。
- 完整验证集在 64 Ki / 1 Mi 字符块下得到的频率表均与旧方式相同；完整 10k 词表与 merges 再训练也与修改前完全相同。
- 官方 `tests/test_train_bpe.py` **3 passed，4.83 秒**，包含速度测试。此次没有重跑全套模型/Tokenizer 测试；最近全套结果仍是 2026-10-30 的 46 passed、2 skipped。
- [源版本、完整数据检查、训练结果和等价核验](experiments/2026-11-01/full-tinystories/data-and-bpe-report.json)。
- [修改前验证集 profiler 摘要](experiments/2026-11-01/full-tinystories/profile-validation-before-streaming.txt)。

下一步用完整训练集进行 BPE 的预分词与合并资源检查，再完成正式 10k 训练。若单进程预分词或最大值扫描超出预算，再按实测结果选择多进程/优先队列。OWT 数据尚未下载，全量 BPE、跨域编码、正式语言模型训练预算、LM 发散边界、最终 writeup/code 仍待完成。

本次新增两个活动模块，并修改已有 `bpe.py`；笔记及学习进度已更新。本次未做 Git 提交或推送；源码快照仍待统一同步。
