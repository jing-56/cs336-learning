# 2026-10-25｜A1：分词器接口、实验记录、数据准备与模型组件

## 记录范围与证据

本篇记录从前两天的 BPE 训练继续到今天结束的实际工作。依据为聊天中的运行回执、本地当前源码、文件时间、已保存的数组及实验记录；这是按阶段回顾，不是当天源码的完整 Git diff。

测试表中的通过结果来自本次学习过程中逐项运行的回执；本次整理文档时没有重新运行 A1 全套测试，也不据此宣称全部作业通过。数组规格与现有源码在整理时另外进行了只读核对。

所有本地文件路径相对于 `E:/CS336/assignments/a1-basics`。本笔记仓库原有 `.gitignore` 忽略 `assignments/`，因此本文描述的本地源码、二进制产物不会随本次文档提交上传。

## 一、完成独立编码、特殊标记与 Tokenizer 类

### 编码与解码分别做什么

- 解码：ID 查词表，先拼接字节串，再整体按 UTF-8 解码；不能先单独解码每个 token，因为单个 token 可能只含字符的部分字节。非法字节序列使用替换字符处理。
- 普通编码：预分词、拆为基础字节、按已有合并规则的优先级合并，再通过反向词表查 ID。编码不重新统计频率，也不新增普通合并规则。
- 特殊标记编码：使用带捕获组的正则拆分，保留特殊标记的位置；特殊标记直接输出专用 ID，普通片段交给普通编码函数。
- 多个特殊标记按长度降序匹配，处理短标记被长标记包含的情况。

小词表示例中，特殊标记 ID 为 256，`ca` 为 257，`cat` 为 258，因此：

```text
cat<|endoftext|>cat! -> [258, 256, 258, 33]
```

这些是该小词表的编号，不是所有词表通用的固定编号。

### 将配置与操作封装为对象

在 `cs336_basics/tokenizer_demo.py` 增加 `Tokenizer`，保存词表副本、规则和特殊标记。用户指定的特殊标记若未在词表中，分配新 ID；已存在则复用。

| 接口 | 具体作用 |
| --- | --- |
| `Tokenizer(...)` | 在内存中保存分词器配置 |
| `encode(text)` | 普通文本及特殊标记编码 |
| `decode(ids)` | ID 转回文本 |
| `encode_iterable(iterable)` | 逐个字符串编码，使用 `yield` 陆续交出 ID |
| `from_files(vocab_filepath, merges_filepath, ...)` | 从两份 pickle 文件读取结果并创建对象 |

理解了对象属性 `self`、类方法 `cls`、`@classmethod` 与保存到磁盘的区别。创建对象只是在内存中保存配置；持久化需要另外写文件。

`encode_iterable` 当前把每个输入字符串当作独立编码单位：适用于已经验证的文件逐行测试，不保证任意切块后与整段编码等价，也不支持被拆在两块之间的特殊标记。输出生成器不会在内部收集全量结果，但当前段的编码列表仍会暂存。

把演示部分放入 `if __name__ == '__main__'`，导入路径改为 `cs336_basics...`，避免测试导入时自动读取小模型、打印或执行实验。

### 保存格式和测试接入

- `bpe.py` 增加分别保存 `bpe_vocab.pkl`、`bpe_merges.pkl` 的流程，供 `from_files` 使用。
- 在 `tests/adapters.py` 中接通 `get_tokenizer`。
- `tests/test_tokenizer.py` 的无条件 `import resource` 在 Windows 导致测试收集失败，改为尝试导入，缺失时设为 `None`。原有仅限 Linux 的内存测试跳过标记保持不变，没有伪造资源限制或改动正确性断言。
- 使用 `python -X utf8` 执行测试，保证语料按 UTF-8 读取。

验证回执：普通编码、解码和特殊标记共 **21 passed, 4 deselected in 23.12s**；随后两项流式测试 **2 passed, 23 deselected in 2.42s**。两项 Linux 内存测试未验证，不能记为通过。文件加载后的小实验也能完整还原。

## 二、词表大小与评估文本分离实验

新增 `cs336_basics/experiment_tokenizer.py`。先在约 3.8 KB 的故事样本上训练、编码同一文本，再用固定的新文本评估，分开观察训练文本上的切分与未参与训练文本上的效果。

### 同一份训练文本上的结果

| 实际词表 | 训练耗时 | 编码 token 数 | bytes/token |
| ---: | ---: | ---: | ---: |
| 300 | 0.015 秒 | 2,463 | 1.54 |
| 500 | 0.042 秒 | 1,308 | 2.90 |
| 839（目标 1,000） | 0.055 秒 | 889 | 4.27 |

文本读取后有 3,794 个 UTF-8 字节，其中包含 5 个 13 字节的 EOS 标记。若每个 EOS 整体算一个特殊 token，未执行普通 BPE 合并时共有 `3794 - 5 × 13 + 5 = 3734` 个 token。

目标 1,000 而停在 839，是因为没有可合并的片段内部相邻对；不能跨预分词边界继续合并。839 是词表条目数，规则数是 `839 - 256 - 1 = 582`，889 则是编码后的序列长度。三者不能混淆，也不能把饱和点当作通用最佳词表大小。

### 固定新文本上的结果

评估文本为 98 个 UTF-8 字节，只用于编码，不传给训练函数：

```text
A robot opened the window and watched the stars. It wanted to learn why the sky was dark at night.
```

| 实际词表 | 评估 token 数 | bytes/token | 完整还原 |
| ---: | ---: | ---: | --- |
| 300 | 63 | 1.56 | True |
| 500 | 54 | 1.81 | True |
| 839 | 49 | 2.00 | True |

扩大词表在这段新文本上仍有收益，但比训练文本上的提升小。这里只是一段短文本的观察，不能代表普遍泛化表现。

### 编码吞吐量

将固定评估文本加换行后重复 1,000 次，连续编码三次，计算总字节数除以总耗时：

| 实际词表 | 编码 MB/s（十进制） | tokens/s |
| ---: | ---: | ---: |
| 300 | 0.32 | 209,035 |
| 500 | 0.29 | 159,236 |
| 839 | 0.28 | 142,369 |

这是重复短文本上的简单基准，不是完整语料吞吐量。不同词表会产生不同 token 数，比较处理同一份原文的速度时优先观察 MB/s。词表增大、输出变短，不保证当前实现的分词速度更快。

## 三、约 5 MB 样本：规模、最长 token、时间与内存

新增 `cs336_basics/experiment_scale.py`，使用 `tests/fixtures/tinystories_sample_5M.txt`，固定前述评估文本：

| 目标/实际词表 | 合并规则 | 训练时间 | 评估 token 数 | bytes/token |
| --- | ---: | ---: | ---: | ---: |
| 1,000 / 1,000 | 743 | 2.945 秒 | 30 | 3.27 |
| 10,000 / 10,000 | 9,743 | 约 14.65 秒 | 22 | 4.45 |

10,000 词表多次普通运行记录约为 14.444–14.675 秒，完整还原均为 True。较小语料中目标 1,000 的结果为 49 个评估 token，扩大语料后降到 30；此后扩大词表至 10,000，又降至 22。结论只适用于本次评估样本。

最长普通 token：`b' congratulations'`，共 16 字节，即一个空格加 15 个 ASCII 字母。它符合允许词前空格的预分词边界；最长并不意味着最常见，更不意味着算法理解其语义。

保存两份独立产物：

- `cs336_basics/tinystories_sample_10k_vocab.pkl`
- `cs336_basics/tinystories_sample_10k_merges.pkl`

### 性能分析

使用 cProfile 保存 `bpe_training.prof`，按累计时间查看。该次实验总计 29.121 秒，`train_bpe` 累计约 28.978 秒；性能记录有额外开销，不拿这一时间直接替代正常运行的约 14.65 秒。

| 项目 | 累计耗时 | 解释 |
| --- | ---: | --- |
| `max` 相关调用 | 24.822 秒 | 主要开销是逐轮扫描候选相邻对以选出最高频对 |
| 选择函数 `lambda` | 10.735 秒 | 约 53,695,350 次调用；已包含在上方 `max` 累计时间内 |
| `pretokenize_counts` | 2.169 秒 | 文本预分词与统计 |
| `merge_pair` | 0.508 秒 | 局部序列修正 |

`tottime` 是函数自身耗时，`cumtime` 包括调用的下层函数；不能重复相加父子函数的累计时间。瓶颈由原来的全量片段重建转移为最高频对的全量选择。优先队列是可继续研究的方向，目前尚未实现。

### Windows 进程内存测量

使用 psutil 读取训练前 RSS，并在训练返回后立即读取当前 RSS 和 Windows 的 `peak_wset`：

| 指标 | 数值 |
| --- | ---: |
| 训练前 | 23.86 MiB |
| 训练后当前 | 32.23 MiB |
| 截至训练结束的进程峰值 | 48.99 MiB |

相对训练前的峰值差约 25.13 MiB，只是进程读数之差，不等于精确的算法对象分配量。峰值从进程启动开始累计，包含解释器和依赖；它也不是流式编码的 Linux 限额测试。

本节小样本结论此前已单独提交为 [BPE 实验记录](A1-BPE-SAMPLE-EXPERIMENT.md)，对应提交 `cceb847`。

## 四、从文本走到训练 batch

| 新增本地文件 | 具体实现与检查 |
| --- | --- |
| `cs336_basics/prepare_tokens.py` | 加载样本 10k 分词器，将小故事文件流式编码为 `uint16` 数组；检查 ID 范围、保存 `.npy`、解码核对 |
| `cs336_basics/tinystories_sample_tokens.npy` | 整理时读取到形状 `(892,)`，`uint16`，数组数据 1,784 字节，含格式头的文件共 1,912 字节 |
| `cs336_basics/view_training_sample.py` | 演示内存映射读取、输入目标错一位、随机批次、转换 `torch.long`，随后加入 Embedding 查表演示 |
| `cs336_basics/data.py` | 将采样封装为 `get_batch(dataset, batch_size, context_length, device)` |
| `tests/adapters.py` | 接通 `run_get_batch` |

`uint16` 范围为 0–65,535，能存当前 10,000 词表的 ID，每项 2 字节。进入 Embedding 前转为 `torch.long`。`np.fromiter` 避免先收集一个完整 Python 列表，但最终数组本身仍占内存。

读取连续 9 个 token，可构造长度为 8 的输入和目标：`x=chunk[:-1]`、`y=chunk[1:]`。每个位置的目标是下一个 token；模型可利用该位置之前的完整可见前文，而非只看一个输入 ID。

批次演示采用 B=2、T=8，得到 `torch.Size([2, 8])` 和 `torch.int64`。正式采样不在函数内部重设固定种子，避免每次取到同一批；范围保留最后目标需要的一个额外 token，并支持指定设备。

注意：这里的 892 个 token 使用在 5 MB 语料上训练的 10k 词表；前面 889 的结果使用小语料自身训练出的 839 词表，两者不是同一个模型。

## 五、正式模型组件：逐项做了什么

| 本地文件 | 具体变化 | 对应适配器 |
| --- | --- | --- |
| `cs336_basics/embedding.py` | 自定义 `Embedding(nn.Module)`；以 Parameter 保存 `[V,d]` 表；截断正态初始化；通过 ID 索引 | `run_embedding` |
| `cs336_basics/linear.py` | 无偏置 `Linear`；权重形状 `[d_out,d_in]`；按讲义标准差初始化；执行 `x @ weight.T` | `run_linear` |
| `cs336_basics/rmsnorm.py` | 最后维度求 RMS；加 epsilon；float32 计算后恢复 dtype；乘以初始全 1 的逐维可训练参数 | `run_rmsnorm` |
| `cs336_basics/swiglu.py` | `silu(x)=x*sigmoid(x)`；复用三个 Linear，执行门控特征乘法并映射回 d_model；默认 d_ff 约 8/3 倍并向上取 64 的倍数 | `run_silu`、`run_swiglu` |
| `cs336_basics/nn_utils.py` | 稳定 softmax；沿指定维度减最大值，指数化、求和归一化，保留计算维度 | `run_softmax` |
| `cs336_basics/attention.py` | 缩放点积注意力；只转置 K 最后两维；按可选布尔掩码屏蔽，再对 Key 维 softmax 并乘 V | `run_scaled_dot_product_attention` |

各模块适配器把测试给定的固定权重装入自定义模块，再比较输出；不是依靠随机初始化凑出参考结果。

### 通过的新增测试

| 测试 | 回执 |
| --- | --- |
| `tests/test_data.py::test_get_batch` | 1 passed in 0.43s |
| `tests/test_model.py::test_embedding` | 1 passed in 0.10s |
| `tests/test_model.py::test_linear` | 1 passed in 0.08s |
| `tests/test_model.py::test_rmsnorm` | 用户确认通过 |
| `test_swiglu`、`test_silu_matches_pytorch` | 合计 2 passed in 0.08s |
| `tests/test_nn_utils.py::test_softmax_matches_pytorch` | 1 passed in 0.05s |
| `test_scaled_dot_product_attention`、`test_4d_scaled_dot_product_attention` | 合计 2 passed in 0.09s |

### 配套演示文件与形成的理解

| 演示文件 | 具体说明 |
| --- | --- |
| `linear_demo.py` | `[2,3]` 经三组权重得到 `[2,3,5]`；线性层变换最后一个特征维度 |
| `rmsnorm_demo.py` | `[3,4]` 与 `[30,40]` 归一化后几乎相同；g 是长度 d_model 的共享可训练向量 |
| `silu_demo.py` | 输入决定 sigmoid 倍率；非线性与固定缩放系数不同 |
| `swiglu_demo.py` | 两路投影、SiLU 门控、逐元素乘法，再降回原维度；不混合 token 位置 |
| `softmax_demo.py` | 大分数先减最大值，避免指数溢出；权重总和为 1 |
| `attention_demo.py` | 先单 Query，再三个 Query；下三角 True 表示可见，未来分数设为负无穷；Value 加权验证 |
| `multihead_demo.py` | 仅演示 `[1,3,4] -> [1,3,2,2] -> [1,2,3,2]` 的拆头与换轴 |

Embedding 演示中 `[2,8]` 个 ID 查表后成为 `[2,8,4]`：token 数没变，只是每个编号换成一个四维向量。

Attention 演示的权重为 `[1,0,0]`、`[0.3302,0.6698,0]`、`[0.2483,0.2483,0.5035]`，未来位置不参与加权。禁止位置不能仅把分数置零，因为指数后仍有非零权重。函数本身不自动决定因果规则，掩码由调用者传入。

## 六、当前未完成事项与下次起点

- **多头注意力只完成拆头演示**。还未把 Q/K/V 投影、拆头、注意力、拼接和输出投影组成正式模块；适配器仍为未实现。
- RoPE、带 RoPE 的多头注意力、Transformer Block 和完整 Transformer LM 尚未实现。
- 交叉熵、梯度裁剪、AdamW、学习率调度、checkpoint 与完整训练循环尚未实现。
- 完整 TinyStories、OpenWebText 训练及更充分的独立评估实验尚未完成；不能用 5 MB 样本结果替代正式全量实验。
- 流式编码的任意切块边界问题、Linux 内存约束测试、大规模编码性能仍需处理或验证。
- 没有训练完成的语言模型，也没有生成能力或整体 A1 完成率的正式证据。

下次从多头注意力继续：先确认每个头保留全部位置，再实现合并输出与投影，接着学习位置编码。今天在此结束，不额外推进实现。

## 三天记录导航

- [10 月 23 日：字节表示、预分词、单轮合并](2026-10-23-A1-BPE-FOUNDATIONS.md)
- [10 月 24 日：训练循环、特殊标记、局部更新优化](2026-10-24-A1-BPE-TRAINING.md)
- [10 月 25 日：本篇](2026-10-25-A1-TOKENIZER-AND-MODEL.md)
