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

## 今天先做三件事（约 1 小时）

1. **读** `plan/00-ROADMAP.md` 的第一周，别通读全文。
2. **跑** 环境：按 `plan/01-SETUP.md` 在 `assignments/a1-basics` 里 `uv sync`，然后 `uv run pytest`。
   正确的初始状态是：**环境装好、测试全部报 `NotImplementedError` 失败**。全绿说明你抄到答案了，全红但跑不起来说明环境没通。
3. **记** 打开 `notes/INDEX.md`，把第 1 讲标成 `[~]`，看完第一讲（tokenization）再来填 `notes/L01.md`。

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

## AI 使用约定（重要，别跳过）

课程仓库自带 `AGENTS.md`，明确要求 AI 扮演**助教**而不是代写：

- ✅ 可以：解释概念、解读报错、帮你设计断言和玩具例子、review 你写的代码、指出该看哪一讲
- ❌ 不做：替你写 tokenizer / Transformer / Triton kernel / 分布式 / 对齐算法，不填 `TODO`，不直接把作业要求翻译成可运行代码

这不是客套。这门课的价值全在"你自己撞墙再爬出来"的过程里，一旦被代写，作业分有了、面试题答不出。所以在本目录里，我按 `AGENTS.md` 的角色行事：**给方向、给检查手段、给 review，不给实现**。

另外两条实战规则：
- 做作业时关掉 AI 自动补全（Cursor Tab / Copilot），官方明确说它显著降低理解深度。
- 卡住超过 90 分钟再问人/AI，但问的时候带上：你试了什么、期望什么、实际什么、最小复现。

---

## 这门课怎么学才不白学

1. **先写后看**。每讲先自己写 20 行，再看 live-coding 脚本。看懂了 ≠ 写得出。
2. **测试即规格**。A1/A2/A4/A5 都给了 `tests/`，测试文件比讲义更精确地定义了正确性；先读测试，再写实现。
3. **每个实验留一条曲线**。A1 后半段、A4、A5 的分数大半在"你能不能说清为什么"。`plan/03-ASSIGNMENTS.md` 里给了每份 writeup 该有什么图。

---

## 换一台机器怎么复现

这个仓库只放**文档与脚本**，不含课件和作业代码（那是上游仓库，117 MB + 带 `.venv`，不该进版本库）。到新机器上：

```powershell
git clone https://github.com/<你的用户名>/<仓库名>.git CS336
cd CS336
.\tools\setup.ps1        # 浅克隆 lectures\ 与 assignments\ 共 6 个官方仓库
. .\tools\env.ps1        # 设好 uv / Python / 镜像（若报策略错误，见 plan\01-SETUP.md 第 1 节）
cd assignments\a1-basics
uv sync
uv run pytest -p no:cacheprovider
```

初始状态应当是 **13 failed, 全是 `NotImplementedError`**。

> 上游更新很快（作业 handout 会带版本号，如 A2 的 `26.1.4`）。进作业目录 `git pull` 就能同步，冲突通常只在 `tests/adapters.py`。

开始吧 → [`plan/00-ROADMAP.md`](plan/00-ROADMAP.md)
