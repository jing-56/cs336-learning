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

## 当前进度（2026-11-07）

- [11 月 7 日 OpenWebText 实验](notes/2026-11-07-A1-OPENWEBTEXT-FULL.md)完成全量 32K BPE、完整训练集与验证集编码，并在 RTX 3050 上跑完 **20,000 步**小模型训练。固定验证批次 loss 为 **4.9594**，已记录 **256 个新 token** 的生成样例。
- [11 月 4 日实验](notes/2026-11-04-A1-EVALUATION-AND-OWT-BPE.md)补齐 TinyStories 完整验证（loss **1.9715**）、采样对照、高学习率失稳样例与 OpenWebText 分词样本对照。

- 2026-11-01 GPU 版 PyTorch 安装后本地核验：**46 passed、2 skipped，17.06 秒**；两项跳过均为 Linux 专属内存限制测试。测试通过不代替完整数据实验。
- 已跑通 BPE/Tokenizer、完整 Transformer、训练与生成闭环；5 MB 样本的训练/验证数组为 1,143,921 / 127,340 token，主要单组对照预算为 25.6 万目标 token。
- 学习率七候选、RoPE 三种子、B=2/4/8、去 RMSNorm、post-norm、参数量配平的 SiLU 对照均已有记录。当前小配置里 pre-norm/SwiGLU 基线验证 loss 为 3.6541；结论限于对应设置。
- 参数量、前向 FLOPs、XL 正式配置和 AdamW 时间核算已有脚本与推导；激活内存采用明确的简化记账口径，不是实测峰值。
- [A1 逐项交付核对与下一步](notes/2026-10-30-A1-COMPLETION-AUDIT.md)记录了原始核对清单。完整 TinyStories 训练集 profile、课程原始尺寸实验、精确稳定边界及最终打包仍待补。
- [训练/生成入口与操作说明](notes/2026-10-30-A1-TRAINING-TOOLS.md)已新增：配置、周期存档、累计耗时日志、CPU 精确续训检查及固定种子 256-token 样例。[Unicode/SGD 逐题回答及十步实验](notes/2026-11-01-A1-UNICODE-AND-SGD.md)已补齐，[完整 TinyStories 数据已下载校验，BPE 已支持按文档边界分块读取](notes/2026-11-01-A1-FULL-TINYSTORIES-PREPARATION.md)；[完整训练集 10k BPE 已完成](notes/2026-11-01-A1-TINYSTORIES-FULL-BPE.md)，[完整训练/验证编码已完成](notes/2026-11-01-A1-TINYSTORIES-FULL-ENCODING.md)，[新数据 CPU 起步训练](notes/2026-11-01-A1-FULL-DATA-TRAINING-START.md)后已[切到 RTX 3050](notes/2026-11-01-A1-GPU-TRAINING.md)，并[完成 20,000 步、4,096 万目标 token 的学习模型训练与 EOS 故事生成](notes/2026-11-01-A1-FULL-TRAINING-AND-GENERATION.md)；最终固定批次验证 loss=2.0210，模型尺寸小于课程原始配置。[本人源码快照](implementations/a1-basics/README.md)已同步 56 个活动模块、接入函数及三个配置，数据和模型存档不上传。

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
