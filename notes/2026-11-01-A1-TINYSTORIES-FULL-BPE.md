# 2026-11-01：完整 TinyStories 10k BPE 训练

完整训练集已实际训练完成。输入是官方 `TinyStoriesV2-GPT4-train.txt`，2,227,753,162 字节，revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`，不是之前 5 MB 样本，也不是验证集调试词表。

## 实际结果

| 项目 | 实测 |
| --- | --- |
| 运行设备/实现 | CPU、单进程、自写 BPE；按文档边界分块预分词 |
| 正常训练时间 | 841.456 秒（14.02 分钟） |
| 训练前进程 RSS | 26.64 MiB |
| 训练后当前进程 RSS | 72.47 MiB |
| 截至训练结束的 Windows 进程生命周期工作集峰值 | 97.57 MiB |
| 实际词表大小 | 10,000 |
| 合并规则数 | 9,743 |
| 特殊 token | `<|endoftext|>`，ID 256 |
| 最长普通 token | `b' responsibility'` |
| 最长普通 token 字节数 | 15 |

计时范围是一次 `train_bpe`（包含每 30 秒低频进程监控），不包含后面的 pickle 保存、完整输入 SHA-256 复查和示例编码。内存是 Windows 进程工作集口径，不是算法净分配量，也不是系统总内存占用。此前 30–60 分钟是未运行前的粗略预算，本表是实际结果。

**train_bpe_tinystories(a) 回答草稿：** 在完整 TinyStories 训练集上训练 10k 词表，CPU 单进程正常训练耗时 841.46 秒，Windows 进程工作集峰值 97.57 MiB。最长普通 token 是 `b' responsibility'`，15 字节，是带前导空格的完整英文单词，符合预分词允许前导空格且不跨片段合并的规则。

如需在 PyCharm 再跑，打开 `E:/CS336/assignments/a1-basics/cs336_basics/experiment_bpe_corpus.py`，Parameters 填入：

```text
--input "E:\CS336\assignments\a1-basics\data\TinyStoriesV2-GPT4-train.txt" --split full_training_corpus --skip-profile
```

不填参数仍使用验证集调试模式，并会额外运行一次 profiler。

## 为什么没有再跑一次 profiler

实验入口原本会在正常训练之后自动再运行一次 cProfile。本次新增 `--skip-profile`，完整语料只训练一次；对应 JSON 中 profiler 字段为 false，profile 耗时及输出一致性字段为 null，不用“未运行”冒充“已验证”。

已有[验证集 profiling](2026-11-01-A1-FULL-TINYSTORIES-PREPARATION.md)说明最大频率扫描是该调试运行的主要瓶颈；完整训练集的 profiler 尚未执行，不能将验证集的时间比例当作完整集实测比例。`progress.jsonl` 记录每 30 秒进程状态，但不能据此精确分解预分词与合并各阶段耗时。

## 保存与核验

产物目录：`E:/CS336/assignments/a1-basics/cs336_basics/bpe_corpus_runs/20261101_190151_430986`。

包含 `vocab.pkl`、`merges.pkl`、`summary.json`、`progress.jsonl`。旧样本词表、旧 token 数组和旧模型没有替换。重新运行该入口会创建新的独立目录。

- 完整输入再次计算 SHA-256，与固定版本官方元数据一致。
- 检查 ID 连续、10,000 个词条的字节值唯一、9,743 条规则的左右 token 均已存在于更早词表；合并结果与对应新增 ID 一致，EOS 不参与普通合并。
- 固定 seed=42，从验证集完整文档中 reservoir 抽取 10 个进行编码/还原，全部正确，末尾 EOS 均对应 ID 256；首个明显不完整的片段排除。这里只是 TinyStories 健全性检查，未完成 OWT 跨域对照。
- 新词表与旧 5 MB 词表共有 9,656 个 ID 映射不同；与验证集调试词表共有 9,590 个不同。因此之后应以新词表重新编码原始训练/验证数据，再开始相配套的模型训练。
- [完整结果及核验](experiments/2026-11-01/tinystories-full-bpe/full-training-result.json)、[低频进程日志](experiments/2026-11-01/tinystories-full-bpe/progress.jsonl)。

入口只增加监控与跳过 profile 的选项，BPE 核心没有改动；先用小语料验证单次运行可保存结果且不生成 .prof。最近 BPE 官方回执仍是本日分块读取修改后的 3 passed，未把它当成本次重跑或全套测试。

## 后续范围

本次完成 TinyStories 完整训练集的正常 BPE 训练与产物核验；完整训练集 profile、OWT BPE/跨域对照、完整数据编码、正式语言模型训练预算与最终 writeup/code 仍待完成。数据和词表产物不上传 Git；本次更新笔记及进度，未提交或推送 Git。
