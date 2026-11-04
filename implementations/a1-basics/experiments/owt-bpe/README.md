# OpenWebText BPE 样本实验源码

- `bpe_heap.py` 用堆选下一次合并，维持原版的频率和字节对排序。
- `bpe_compact.py` 在堆版基础上缩小倒排索引，并提前释放预分词频率表。
- `prepare_500m.py` 从课程数据固定版本分段读取 gzip，生成约 500 MiB 的完整文档前缀。
- `measure.py` 记录 32K 词表训练的时间和 Windows 进程峰值，紧凑版会逐项核对堆版输出。

这些是独立实验版本，正式包的 BPE 入口仍位于 `cs336_basics/bpe.py`。运行依赖 Python 3.12、`regex` 和 `psutil`。先将[实验的 data-summary.json](../../../../notes/experiments/2026-11-04/owt-scale/data-summary.json)放到脚本目录，再运行 `python prepare_500m.py`。脚本使用其中的 50 MiB 前缀哈希确认数据版本。

同目录依次运行 `python measure.py 500m` 和 `python measure.py 500m compact`，后者会比较前者保存的词表和合并文件。50 MiB 用同样的两个命令并将参数换成 `50m`，输入文件名是 `owt_train_50m.txt`，长度和哈希见上面的数据摘要。5 MiB 复核模式使用原实验文件夹的语料及参考词表。

仓库保存了源码、校验值和测量结果。语料和 pickle 需要在本地准备。 [源码清单](SOURCE_MANIFEST.json) · [实验说明](../../../../notes/2026-11-04-A1-EVALUATION-AND-OWT-BPE.md)
