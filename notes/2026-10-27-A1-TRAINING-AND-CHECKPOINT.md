# 2026-10-27｜A1：优化器、训练支持组件与首次单批次拟合

## 今天完成到哪里

完成 AdamW、预热与余弦学习率调度、全局梯度裁剪、checkpoint 保存恢复，并将已有 Transformer 连成可运行的小训练脚本。同一固定批次训练 100 步后，loss 从 `9.2533` 降到 `0.0124`，32 个预测位置的最高分 token 全部正确。

这是固定训练批次的拟合验证，不是独立验证集结果，也不表示模型已经具有通顺生成故事的能力。正式随机批次训练、验证集评估与文本生成还没实现。

## 逐文件说明

文件名相对于本地 `assignments/a1-basics/cs336_basics/`；[源码归档](../implementations/a1-basics/README.md)中保留同名文件。

| 文件 | 新增或修改的内容 | 结果与边界 |
| --- | --- | --- |
| `optimizer.py` | 新增 `AdamW(Optimizer)`；参数组设置、每个参数的 `m/v/t`、偏差修正、独立权重衰减、原地更新；跳过无梯度参数 | 连续 1000 步的小模型测试通过；当前不支持稀疏梯度 |
| `lr_schedule.py` | 新增 `get_lr_cosine_schedule`；线性预热、余弦下降、结束后保持最低值；检查时间段边界 | 返回当前步的学习率，函数自身不更新优化器；已单独测试，尚未接入演示训练 |
| `nn_utils.py` | 在 softmax、交叉熵后追加 `gradient_clipping` | 全部参数梯度的平方求和后开方，超过上限按同一比例原地缩放；无梯度参数跳过 |
| `checkpoint.py` | 新增 `save_checkpoint` 与 `load_checkpoint`，保存模型、优化器、已完成步数 | 恢复前先创建相同结构的模型和对应优化器；不是保存模型类定义 |
| `train_demo.py` | 新增 CPU 单批次拟合实验：固定种子、内存映射数据、建模、计算损失、反传、裁剪、AdamW 更新、打印指标、保存存档 | 数据只抽取一次；固定学习率 0.003，训练 100 步；不包含验证、调度调用或恢复训练入口 |
| 本地 `tests/adapters.py` | 接通 `get_adamw_cls`、学习率调度、梯度裁剪、保存与加载 checkpoint | 归档为独立的已实现函数集合，不上传上游测试与断言 |

本次首次归档源码，也包含 23—26 日已实现的依赖与演示文件，保证模型源码不是缺少 Embedding/Linear 等依赖的孤立片段。这些旧文件不记为今天新写。

## AdamW：代码与公式怎样对应

对每个有梯度的参数保存 m、v 和步数 t；第一次更新时 t=1：

```text
m = beta1*m + (1-beta1)*g
v = beta2*v + (1-beta2)*g²
m_hat = m / (1-beta1**t)
v_hat = v / (1-beta2**t)
w_new = (1-lr*weight_decay)*w_old - lr*m_hat/(sqrt(v_hat)+eps)
```

m 保留梯度方向的历史，v 记录梯度平方的历史。二者从零开始，因此修正起步偏差。权重衰减独立执行，不加入用于累计 m、v 的梯度。

实现采用显式修正 m_hat、v_hat 的形式。讲义将修正因子合入学习率的形式在 epsilon 的位置上有细节差别；本地官方测试明确接受与 PyTorch AdamW 一致的结果，本实现已通过。实际默认 `betas=(0.9,0.999)`、`eps=1e-8`。

## 学习率、裁剪与梯度清零

- 调度器的参数 `cosine_cycle_iters` 是余弦下降的结束步数，不是预热之后额外下降的步数。
- 裁剪使用整个模型的梯度总 L2 长度；缩放比例为 `max_norm/(total_norm+1e-6)`，方向保持不变。
- `optimizer.zero_grad()` 清除旧的 `.grad`，不会清空 AdamW 的 m、v、t。
- 裁剪放在 `loss.backward()` 之后、`optimizer.step()` 之前。

## Checkpoint 排错与验证

第一次运行 `test_checkpointing` 失败，报错定位到本地 `tests/adapters.py` 的 `run_load_checkpoint` 仍是 `raise NotImplementedError`。测试停在适配器入口，还没执行到读取存档。

将入口接到 `load_checkpoint(src, model, optimizer)` 后，测试通过；没有修改上游测试断言，也没有绕过比较。

当前存档包含 `model`、`optimizer`、`iteration` 三个键。`torch.load(..., map_location="cpu", weights_only=True)` 读取后，把状态装回已有对象，返回此前完成的更新次数。当前不保存随机数生成器与数据采样进度，因此没有验证逐位一致的中断续训。

## 首次训练实验

### 配置

| 项目 | 值 |
| --- | --- |
| 数据 | 之前保存的 `tinystories_sample_tokens.npy`，892 个 uint16 token |
| 设备 | CPU，4 个计算线程 |
| 随机种子 | NumPy / PyTorch 均为 42 |
| 词表大小 | 10,000 |
| 批次 / 上下文 | 2 / 16，共 32 个预测位置 |
| d_model / 层数 / 头数 | 64 / 2 / 4，每头 16 维 |
| SwiGLU d_ff | 128 |
| RoPE theta | 10,000 |
| 参数量 | 1,362,240 |
| 优化器 | 自实现 AdamW，lr=0.003，weight_decay=0.01 |
| 梯度裁剪上限 | 1.0 |
| 更新次数 | 100；始终使用同一批 x、y |

### 用户运行回执

| 已完成步数 | 本步更新前的 loss |
| ---: | ---: |
| 1 | 9.2533 |
| 20 | 3.2556 |
| 40 | 0.2832 |
| 60 | 0.0356 |
| 80 | 0.0174 |
| 100 | 0.0126 |

最后一次更新后重新前向计算：**loss=0.0124，固定批次准确率=100.00%**。准确率表示最高分 token 是否正确；不意味着正确 token 的概率已经为 1。每个 token 等概率时 loss 为 `ln(10000)≈9.2103`，与起步时接近。

同步前，助教也曾用相同配置做一次内存内检查，得到 loss `9.2533416748 → 0.0124474186`、准确率 `1.0`，与用户回执一致。此次整理没有覆盖或重跑写入用户的存档。

### 产物只读核对

本地存档 `overfit_demo_checkpoint.pt`：

- 文件大小 16,367,926 字节。
- 保存的 iteration 为 100。
- 模型状态合计 1,362,240 个参数元素。
- AdamW 有 21 组参数张量状态，保存的 lr 为 0.003。
- 存档、词表 pickle、token 数组、数据文件均不上传 GitHub；只提交源码与文字结果。

## 测试证据

| 项目 | 当时回执 |
| --- | --- |
| `test_adamw` | 1 passed in 3.95s |
| `test_get_lr_cosine_schedule` | 用户确认通过 |
| `test_gradient_clipping` | 用户确认通过 |
| `test_checkpointing` | 1 passed in 2.15s（修复适配器之后） |

同步时，直接导入归档目录中的源码，并用提取的接入函数连接现有官方测试，组合复查 **20 passed，11.14s**：

- `test_model.py`：13 项。
- `test_nn_utils.py`：3 项。
- `test_optimizer.py`：2 项。
- `test_data.py`、`test_serialization.py`：各 1 项。

归档的 30 个原始模块逐字节 SHA-256 与活动源码一致，32 个 Python 文件通过语法检查。本次复查使用工作区临时目录，并复制逐字节一致的参考快照来满足测试的相对路径要求，限定 4 个 CPU 计算线程。没有修改官方测试、原始参考快照或活动源码。

上述 20 项不是 A1 全套测试；BPE、分词器和 Linux 专属内存约束不属于此次组合复查范围。

## 下次从哪里继续

1. 划分训练集和验证集；目前只讨论了概念，没有写入训练脚本。
2. 每步重新采样训练批次，定期计算不参与反传的验证 loss。
3. 接入已实现的预热与余弦调度。
4. 增加配置参数、周期存档与加载后继续训练入口。
5. 实现 temperature / top-p 采样与文本生成。
6. 再扩展语料、资源核算、学习率/batch 扫描、消融和正式 writeup。

固定批次拟合通过是训练链路检查，不替代泛化评估、正式训练实验或 A1 全部交付。
