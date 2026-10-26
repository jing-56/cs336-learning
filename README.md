# CS336 学习工作区

> Stanford **CS336: Language Modeling from Scratch**（2026 春季版，Percy Liang + Tatsunori Hashimoto）
> 官方主页：<https://cs336.stanford.edu> ｜ 视频：[YouTube 播放列表](https://www.youtube.com/watch?v=JuoVZkPBiKk&list=PLoROMvodv4rMqXOcazWaTUHhq-yembLCV)

一句话：这门课让你用 PyTorch **从零**写出 tokenizer、Transformer、训练循环、Triton kernel、分布式训练、数据流水线和对齐算法，最后每一块都能跑出真实数字。它不教"怎么调包"，它教"这套东西为什么长这样、慢在哪、钱花在哪"。

---

## 课程速览（2026 版）

| 项目 | 内容 |
| --- | --- |
| 讲次 | 19 讲，约 80 分钟/讲（第 18、19 讲是客座） |
| 作业 | 5 个：Basics / Systems / Scaling / Data / Alignment & Reasoning RL |
| 代码量 | 官方说法：比其他 AI 课多**一个数量级**，脚手架极少 |
| 学分强度 | 5 units；官方建议全职学生也要留足时间 |
| 先修 | Python 熟练、PyTorch 熟、微积分 + 线代、概率统计、ML 基础 |

每一讲对应的课件都在 `lectures/cs336-lectures/` 里：`.pdf` 是幻灯片，`.py` 是 **live-coding 脚本**（讲课时现场跑的代码，比 PDF 更适合边看边跑）。

---

## 目录地图

```
E:\CS336\
├─ README.md              ← 你在这里
├─ plan\
│  ├─ 00-ROADMAP.md       13 周计划：每周看什么、写什么、怎么算过关
│  ├─ 01-SETUP.md         环境：uv / Python 3.12 / 国内镜像 / Windows 坑
│  ├─ 02-COMPUTE.md       算力与预算：本地 3050 能干什么、云端花多少钱
│  ├─ 03-ASSIGNMENTS.md   A1–A5 拆解：模块清单、分值、依赖、推荐顺序
│  └─ 04-PITFALLS.md      踩坑手册：症状 → 原因 → 怎么查
├─ notes\
│  ├─ INDEX.md            19 讲笔记索引（带进度勾选）
│  └─ _TEMPLATE.md        单讲笔记模板
├─ implementations\a1-basics\  本人实现源码快照、接入函数与文件哈希
├─ lectures\cs336-lectures\   官方课件仓库（19 讲 pdf/py）
├─ assignments\
│  ├─ a1-basics\          tokenizer + 模型 + 优化器 + 训练
│  ├─ a2-systems\         性能分析 + Triton FlashAttention + DDP/FSDP
│  ├─ a3-scaling\         缩放定律（需要 Stanford 网络 + 学号 API）
│  ├─ a4-data\            Common Crawl → 可训练语料
│  └─ a5-alignment\       SFT / GRPO / DPO 对齐
└─ .tools\                我用的工具（pypdf、uv、解压出的 handout 文本），不用管
```

---

## 当前进度（2026-10-27）

- 完整 Transformer 前向与训练支持组件已通过对应测试；BPE 和 Tokenizer 的此前验证范围见每日记录。
- 首次 CPU 单批次训练：100 步，loss **9.2533 → 0.0124**，固定 32 个预测位置准确率 **100%**；没有独立验证集结果。
- 今天逐文件改动、排错过程与实验配置：[10 月 27 日记录](notes/2026-10-27-A1-TRAINING-AND-CHECKPOINT.md)。补齐的前一天模型记录：[10 月 26 日](notes/2026-10-26-A1-TRANSFORMER-AND-LOSS.md)。
- [本人实现源码](implementations/a1-basics/README.md)包含此前依赖与今天的新模块，不含数据、模型存档和上游测试。
- 下次继续：训练/验证划分、随机批次、训练中学习率调度、正式训练脚本与生成。
- 这不是 A1 全部完成：完整语料实验、资源核算、调参与消融、writeup 和部分分词边界/内存约束仍待处理。

完整导航见 [笔记索引](notes/INDEX.md)与[作业进度](plan/03-ASSIGNMENTS.md)。

---

## 你的硬件，说实话

本机：**RTX 3050 Laptop / 4 GB 显存**，Windows。

| 能本地做 | 必须上云（或 WSL2 + 云 GPU） |
| --- | --- |
| A1 全部**代码与单元测试**（CPU 就能跑） | A1 正式训练实验（B200 上 20–30 分钟/次） |
| A1 的 TinyStories 缩小版训练（d_model 256–512、ctx 256，几十分钟） | A2 的 Triton kernel、nsys、2 卡 DDP/FSDP |
| A4 的数据流水线代码（CPU 密集，磁盘要 10 GB+） | A4 的最终训练评测（8×B200，约 2 小时） |
| 读代码、写 writeup、跑 pytest | A5 的 GRPO（vLLM + 训练，需要 2 张卡） |

结论：**本地当"开发机"，云端当"跑分机"**。具体预算见 `plan/02-COMPUTE.md`，整套 5 个作业的云端花费大约 **100–400 美元**，取决于你怎么省。

---

## 当前学习方式与记录约定

已有基于 nanoGPT 的 Transformer 理论基础，按“具体例子 → 小段代码 → 逐步解释 → 运行测试 → 实验结果”的节奏学习。编程不是主业，采用代码演示配合讲解，理解每个模块如何连接和为什么这样计算。

- 每次记录实际修改的文件、测试回执、实验配置和数值。
- 将单项测试、单批次拟合与独立验证集结果分开，不用测试通过代替完整实验和理解。
- 本仓库的 `implementations/` 保存个人实现快照，活动开发仍在官方作业目录；上游课件、测试、数据和环境单独保留。

---

## 这门课怎么学才不白学

1. **先写后看**。每讲先自己写 20 行，再看 live-coding 脚本。看懂了 ≠ 写得出。
2. **测试即规格**。A1/A2/A4/A5 都给了 `tests/`，测试文件比讲义更精确地定义了正确性；先读测试，再写实现。
3. **每个实验留一条曲线**。A1 后半段、A4、A5 的分数大半在"你能不能说清为什么"。`plan/03-ASSIGNMENTS.md` 里给了每份 writeup 该有什么图。

---

## 换一台机器怎么复现

这个仓库保存**文档、工具脚本与本人实现的源码快照**。课件、整个上游作业仓库、虚拟环境、数据和模型产物不上传。到新机器上先准备官方环境，再按[源码恢复说明](implementations/a1-basics/README.md)接回自己的实现：

```powershell
git clone https://github.com/jing-56/cs336-learning.git CS336
cd CS336
.\tools\setup.ps1        # 浅克隆 lectures\ 与 assignments\ 共 6 个官方仓库
. .\tools\env.ps1        # 设好 uv / Python / 镜像（若报策略错误，见 plan\01-SETUP.md 第 1 节）
cd assignments\a1-basics
uv sync
uv run pytest -p no:cacheprovider
```

尚未接回个人实现时，未实现接口会报 `NotImplementedError`；测试数量随上游版本而变化。恢复源码后按本仓库记录的测试范围验证。

> 上游更新很快（作业 handout 会带版本号，如 A2 的 `26.1.4`）。进作业目录 `git pull` 就能同步，冲突通常只在 `tests/adapters.py`。

开始吧 → [`plan/00-ROADMAP.md`](plan/00-ROADMAP.md)
