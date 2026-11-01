# 2026-10-30：A1 交付清单核对

> 后续进展：本次核对之后，已补统一训练/生成入口、逐点耗时、CPU RNG 续训验证、256-token 样例，并处理 Identity 规范问题，见[工具记录](2026-10-30-A1-TRAINING-TOOLS.md)。2026-11-01 又补 [Unicode/SGD 逐题回答与十步实测](2026-11-01-A1-UNICODE-AND-SGD.md)。随后已[下载校验完整 TinyStories、完成验证集 BPE 调试并验证分块读取](2026-11-01-A1-FULL-TINYSTORIES-PREPARATION.md)。完整训练集的 [10k BPE 正常训练及核验](2026-11-01-A1-TINYSTORIES-FULL-BPE.md)也已完成。随后[完整训练/验证集的正式词表编码及全量还原检查](2026-11-01-A1-TINYSTORIES-FULL-ENCODING.md)也已完成。随后已[切 GPU 续训到第 200 步，并重跑全套 46 passed/2 skipped](2026-11-01-A1-GPU-TRAINING.md)。其后又[完成 20,000 步、4,096 万目标 token 的学习模型训练与 EOS 故事生成](2026-11-01-A1-FULL-TRAINING-AND-GENERATION.md)，最终固定批次验证 loss=2.0210，仍小于课程原始模型尺寸。下表保留核对时的状态和原始证据。

依据本地 `cs336_assignment1_basics.pdf` **Version 26.0.3**，逐项核对活动源码、笔记、实验 CSV/JSON、存档是否存在，并重新运行当前 Windows 环境下的整套 tests。判断以本地讲义为主，README/旧计划中的日期和“可选”等说法不能代替当前题目要求。

结论：**主要实现通过当前可运行测试，样本训练闭环与四类结构对照已有证据；完整数据实验、正式训练目标、若干文字题和最终交付仍未完成。** 不给出简单百分比，以免把学习用小实验视为完整题目已达标。

## 本次新验证

- **46 passed，2 skipped，18.39 秒**，共收集 48 项，未失败。torch CPU 线程数 4，pytest 禁用缓存，单项超时设置为 30 秒，临时目录和原始 XML 保存在 Codex 工作目录。
- 测试分布：模型 13、nn_utils 3、优化器 2、数据 1、checkpoint 1、Tokenizer 25、BPE 3；Tokenizer 中两项 Linux 专属内存测试在 Windows 跳过。因此本次实际通过 Tokenizer 23 项。
- BPE 速度测试通过，该 testcase 记录耗时约 0.629 秒。这里是 testcase 的时间，不宣称等同于函数内部计时。
- 现有 adapter 接口未发现未实现占位。
- 复核学习率 7 条、batch 3 条、去 RMSNorm 4 条、pre/post 2 条、SwiGLU/SiLU 2 条曲线的最终步数、最终 loss 与汇总一致，所引用存档均存在。含复用基线，不能把曲线数当成独立训练次数。
- RoPE 三种子的六条曲线均到 1000 步，配对初始化和批次检查通过；本次未重新训练或逐一重新载入所有存档。
- 可机读依据见 [audit-evidence.json](experiments/2026-10-30/audit-evidence.json)。原始测试 XML 在 `C:\Users\96244\Documents\Codex\2026-10-03\jian\work\a1-audit\test-results.xml`。

## 逐项交付状态

| 讲义题目 / 内容 | 当前证据 | 还需要做什么 |
| --- | --- | --- |
| unicode1、unicode2，页 3–5 | 已讲字符、UTF-8 和字节；未找到逐小问成稿 | 补 chr(0)、UTF 编码选择、逐字节错误解码反例、非法两字节序列的书面回答 |
| train_bpe，页 8–9 | 3 项当前测试通过，5 MB 词表和 merges 已保存 | 完整语料训练的时间/内存可行性仍需检查 |
| train_bpe_tinystories，页 9–10 | 5 MB、10k 词表、最长 token、计时/内存及 profiling 记录 | 用课程完整 TinyStories 数据完成相同记录；样本不能代替全量 |
| train_bpe_expts_owt，页 10 | 未找到 OWT 数据、32k 词表或结果 | 完成 OWT BPE 训练、最长 token 和与 TinyStories 的比较 |
| tokenizer，页 11–12 | Windows 可运行的 23 项全部通过 | 在合适 Linux 环境核验内存限制；不必为了未知额外目标修改官方测试 |
| tokenizer_experiments，页 12–13 | 已有小样本 bytes/token、吞吐和 uint16 编码说明 | 两域各抽 10 文档、跨域编码对比、825 GB 吞吐外推、完整 train/dev 编码 |
| Linear、Embedding、RMSNorm、SwiGLU、RoPE、softmax、SDPA、MHA、Block、LM | 相关测试本次通过 | 保留规范与接口，不把前向测试通过等同于所有训练行为已验证 |
| transformer_accounting，页 27–28 | a–e 推导、各尺寸及长上下文 JSON 已记录 | 汇入正式 writeup，注明 f 取整、FP32 存储和稠密 FLOPs 口径 |
| cross_entropy、AdamW、调度、裁剪、数据采样、checkpoint | 对应测试本次通过 | 增强通用训练入口；完整 RNG 恢复属于复现增强，勿误写成 checkpoint 单测未通过 |
| learning_rate_tuning，页 31 | 有其他梯度演示；未找到题设 SGD 的三个学习率结果 | 按 10、100、1000 各运行 10 步并写观察；与语言模型 LR 扫描是两道题 |
| adamw_accounting，页 32–33 | a–d 均有带假设的推导、脚本和结果 | 激活部分是明确的简化输出记账，需在报告校核并说明，不能当作真实峰值；14P/15P 与 MFU 口径需保留说明 |
| training_together，页 36–37 | memmap、训练/验证、调度和存档已跑通 | 路径、超参数和输出位置改为通用可配置入口；增加周期存档。当前 train_5m.py 多项写死，结束时保存 |
| decoding，页 38 | 温度、top-p、EOS 停止和上下文窗口裁剪已有实现 | 提取可复用函数，支持用户给定 prompt/长度/采样参数，并补边界检查及固定采样种子的复现方式 |
| experiment_log，页 38–39 | CSV/JSON 有步数、loss、配置、总耗时 | 各曲线点缺累计 wall-clock 时间；增加逐点耗时字段，不能用整次总时间伪造各点时间 |
| learning_rate，页 40–41 | 7 个学习率、每组 25.6 万目标 token；最好 0.003 | 达到正式/低资源训练预算与验证目标；仍需至少一次可解释的发散运行及稳定边界分析 |
| batch_size_experiment，页 41 | 固定 25.6 万 token 的 B=2、4、8 对照 | 讲义范围含 B=1、64、128 到显存上限，当前只完成小范围学习实验；按资源条件扩展并必要时调 LR |
| generate，页 42 | 已有故事样例，脚本最多 120 个新 token | 保存至少 256 新 token 或直到首次 EOS 的样例，记录实际长度/停止原因；补流畅性和至少两个影响因素。120 上限本身不能证明满足要求 |
| 四类结构消融，页 42–44 | 去 RMSNorm、post-norm、NoPE、SiLU 都已有样本对照 | 整理成报告，明确小模型、小预算、单种子/三种子等限制。SiLU 宽 192 匹配当前 SwiGLU 宽 128，不是讲义默认配置的直接复现 |
| main_experiment，页 45 | 未找到 OWT LM 曲线/存档/文本样例 | 同架构和同训练迭代预算进行 OWT 对照，解释 loss 与流畅性的差别 |
| leaderboard，页 45–46 | 未做 | 讲义列为 6 分题，未明确标为可选；自学可按学习目标后置，但不能声称整份作业无此要求 |
| 最终 writeup.pdf / code.zip，页 2 | 有分散笔记与源码快照，未找到最终交付文件 | 汇总全部书面题、图表、实验日志、局限；检查打包内容及可复现入口 |

## 预算差距与数据范围

讲义常规配置为 d=512、f=1344、4 层、16 头、T=256，总目标 token 约 327,680,000。CPU/MPS 低资源提示约 **40,000,000 token**，验证 loss 目标可放宽至 2.00。

现有主要对照每组仅 4×64×1000=256,000 个目标 token。达到约 4000 万是其 156.25 倍；不能把不同模型/学习率的多组训练相加，作为某一个模型完成目标的证据。用户分段训练验证 loss 3.6202 与严格对照基线 3.6541 的验证/RNG 设置不同，不应混为一组结果。

在本次检查的 `E:\CS336` 数据位置中，只找到官方 fixtures 的约 5 MB TinyStories 文本及其派生数组/词表，没有找到课程全量 TinyStories/OWT；没有检查其他磁盘或下载数据。5 MB 词表训练用过后来划为验证集的文本，当前 LM 验证集不参与梯度更新，但不是 tokenizer 也严格独立的端到端验证。

## 两处规范/范围澄清

1. `train_norm_ablation.py` 当前移除模块使用 `nn.Identity()`。它行为简单，但讲义页 1 的允许范围不包含 Identity；严格提交前应换为自写的 `nn.Module` 原样返回模块。本次只记录发现，没有改实验源码或重跑训练。
2. 旧计划把“任意切块后与整串编码完全一致”当成待补问题。本地讲义只明确要求 iterable 惰性编码与内存效率，提供的逐行测试现已通过；不将任意切块等价擅自升级为新的必交要求。该行为可作为额外质量范围注明。

## 后续执行顺序

1. **先完善通用训练/生成入口和逐点日志。** 路径、模型/优化器配置、输出目录可配置；记录 step、processed_tokens、elapsed_seconds、train/val loss、LR、种子、停止原因，并支持周期存档。处理 Identity 的规范问题，避免后续大实验反复改脚本或缺日志。
2. **用现成数据补低成本交付。** Unicode 与 SGD 文字题；固定种子的 256-token/EOS 生成样例；正式 writeup 大纲。已有模型生成样例可以独立完成，不代表新的训练目标已达到。
3. **准备完整数据与分词实验。** 先核对存储、内存和 BPE 效率，再准备 TinyStories/OWT、训练词表、做域内/跨域和吞吐实验、编码 train/dev。
4. **按可承受资源完成扩大训练。** 优先制定 TinyStories 低资源方案，定期保存、按累计 token 和耗时跟踪验证；接着补 LR 边界、batch、结构对照及 OWT。需要耗费长时间或付费的具体运行，先形成有预算的可评估配置。
5. **统一报告与打包。** 检查图表来源、配置一致性、复现命令、测试记录与样例停止条件；最后再检查是否做 leaderboard。

## 本次实际改动

只新增本清单、可机读核对证据和测试记录摘要，并更新 README 当前进度、笔记索引及 plan/03-ASSIGNMENTS.md 的 A1 进度；没有改核心实现、没有生成新训练存档、没有启动大规模训练、没有下载语料或 push。

源码归档仍是 10 月 27 日的 38 个活动模块快照，这 38 个文件与当前活动文件字节一致；另有 **10 个后来新增的本人模块**尚未纳入快照。上游的 pretokenization_example.py 按原归档说明排除，不计入待同步的本人源码。以后同步 GitHub 时再统一更新源码清单与结果导出。
