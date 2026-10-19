# 环境搭建（Windows 实测版）

本机已经装好并验证过的部分标了 ✅，剩下的都是需要你自己决策的（WSL2、云 GPU）。

---

## 0. 现状（2026-10-19 实测）

| 项 | 状态 |
| --- | --- |
| `uv` | ✅ 0.13.1，在 `E:\CS336\.tools\uvpkg\bin\uv.exe` |
| Python | ✅ 3.12.15（uv 托管，装在 `E:\CS336\.tools\pythons`） |
| PyTorch | ✅ 2.11.0+cpu（CPU 版，`torch.cuda.is_available()` 为 False） |
| A1 环境 | ✅ `assignments\a1-basics\.venv` 已同步，`pytest` 能收集 23 个测试 |
| A1 初始状态 | ✅ 全部 `NotImplementedError` —— 这是**正确**的起点 |
| WSL2 | ❌ 未安装（A2 的 Triton / nsys 需要它，见第 4 节） |
| GPU | RTX 3050 Laptop 4 GB（能跑小模型，跑不了 B200 级实验） |

---

## 1. 每开一个新终端，先做这一步

```powershell
. E:\CS336\tools\env.ps1
```

这个脚本做三件事：把 `uv` 加进 PATH、把 Python/缓存目录指到 `E:\CS336\.tools\`、把包索引换成本地最快的镜像（默认清华源）。

**如果报"running scripts is disabled on this system"**：本机 PowerShell 是 `Restricted` 策略。两种解法，选一个：

```powershell
# 解法 A（一次性，之后 .ps1 都能跑；不需要管理员）
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 解法 B（不想改策略，直接内联设置）
$env:Path="E:\CS336\.tools\uvpkg\bin;$env:Path"; $env:UV_PYTHON_INSTALL_DIR="E:\CS336\.tools\pythons"; $env:UV_PYTHON_BIN_DIR="E:\CS336\.tools\bin"; $env:UV_CACHE_DIR="E:\CS336\.tools\uv-cache"; $env:UV_DEFAULT_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"; $env:HF_ENDPOINT="https://hf-mirror.com"
```

其实这些环境变量只影响"下载到哪、从哪下"，`uv run` 在已有 `.venv` 的情况下不设也能跑；但设了能避免往 `C:\Users` 里写东西（会失败）。

> 为什么要指到 E: ？因为 `C:` 只剩 17 GB，uv 的 Python 和缓存动辄几个 GB。而且本会话的沙箱只允许写 `E:\CS336`，写 `C:\Users\...\.local` 会被拒绝。

---

## 2. 已经装好的东西（不用重装）

```powershell
# uv 本体（通过 pip 装到一个独立目录，取出 exe）
E:\CS336\.tools\uvpkg\bin\uv.exe --version

# Python 3.12
$env:UV_PYTHON_INSTALL_DIR='E:\CS336\.tools\pythons'
uv python list --only-installed

# A1 环境
cd E:\CS336\assignments\a1-basics
uv sync          # 已执行过，重复执行是幂等的
```

如果哪天要换机器，从零装的完整命令：

```powershell
# 1) 装 uv：下载官方安装器，或 pip 装到独立目录（本机用的后者）
python -m pip install --target E:\CS336\.tools\uvpkg -i https://pypi.tuna.tsinghua.edu.cn/simple uv

# 2) 装 Python 3.12
$env:UV_PYTHON_INSTALL_DIR='E:\CS336\.tools\pythons'
$env:UV_PYTHON_BIN_DIR='E:\CS336\.tools\bin'
uv python install 3.12

# 3) 同步某个作业的依赖
cd E:\CS336\assignments\a1-basics
$env:UV_DEFAULT_INDEX='https://pypi.tuna.tsinghua.edu.cn/simple'
uv sync
```

---

## 3. 跑测试（本机特有的两条注意事项）

**A1 是唯一能在 Windows 上完整跑测试的作业。** 正确的命令：

```powershell
cd E:\CS336\assignments\a1-basics
uv run pytest -p no:cacheprovider
```

两个坑，都已在实测中确认：

1. **`-p no:cacheprovider` 不能省。** 沙箱禁止删除目录，而 pytest 的缓存插件会"建临时目录再改名"，于是报 `WinError 5 拒绝访问`。加上这个参数就好了（只是没有增量缓存，不影响正确性）。
2. **`tests/test_tokenizer.py` 在 Windows 上无法收集。** 它 `import resource`，这是 Unix 专有模块，用来测分词器内存。报错长这样：

   ```
   tests\test_tokenizer.py:5: in <module>
       import resource
   E   ModuleNotFoundError: No module named 'resource'
   ```

   三条出路，按推荐度排：
   - **进 WSL2 跑全量测试**（最终正解，A2 反正也要 Linux）
   - 本地只跑其它文件：`uv run pytest tests/test_model.py tests/test_optimizer.py -q -p no:cacheprovider`
   - 自己写个 `resource` 桩（不推荐，测内存的那条会失真，等于自欺欺人）

3. **`a1-basics\` 下可能残留 `pytest-cache-files-*` 目录**。这是第一次没加 `-p no:cacheprovider` 时 pytest 留下的，本机沙箱不允许删目录，所以删不掉。无害，忽略即可（进 WSL2 后 `rm -rf` 掉）。

**初始状态应当是：环境装好、测试全红且原因全是 `NotImplementedError`。** 出现 `ImportError`、`AssertionError`、`RuntimeError` 才是你写错了。

---

## 4. WSL2：A2 绕不过去的坎

A2 要写 Triton kernel、跑 Nsight Systems、做 2 卡分布式。这三样在原生 Windows 上都别扭：

- Triton 官方不支持 Windows（只有社区 fork `triton-windows`，版本滞后）
- `nsys` 的 GPU 指标采集在 WSL2 上更完整
- 分布式测试脚本是 bash + torchrun，直接跑在 Linux 更省事

装 WSL2（需要管理员 + 重启）：

```powershell
wsl --install -d Ubuntu-24.04
# 重启后进 Ubuntu，装 CUDA 驱动（Windows 侧驱动会自动透传，Linux 内不要再装 display driver）
```

WSL2 里再做一遍 `uv` + `uv sync`，把项目放在 `/mnt/e/CS336/...`（跨文件系统 IO 慢，训练可接受；数据建议放 WSL 内部 `~/data`）。

**注意**：WSL2 能吃满你的 3050（4 GB），但吃不出更多显存。A2 的 2 卡实验仍然要上云。

---

## 5. 数据下载（本机网络有坑）

实测：`curl` 和 PowerShell 的 `Invoke-WebRequest` 在这台机器上 **HTTPS 全部失败**（`schannel: SEC_E_NO_CREDENTIALS`），但 **git 和 Python 的 urllib 正常**。所以：

| 场景 | 能不能用 | 办法 |
| --- | --- | --- |
| `git clone` | ✅ | 但必须加 `-c http.sslBackend=openssl`，否则同样的 schannel 报错 |
| `pip install` | ✅ | 换清华源 |
| `curl` / `wget` / `irm` | ❌ | 用 Python 的 `urllib` 代替 |
| HuggingFace 下载 | ⚠️ | 加镜像 `HF_ENDPOINT=https://hf-mirror.com` |

用 Python 下载大文件的模板：

```powershell
cd E:\CS336\assignments\a1-basics
mkdir data -Force
uv run python -c "import urllib.request; urllib.request.urlretrieve('https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt','data/TinyStoriesV2-GPT4-train.txt')"
```

A1 需要的数据（合计约 3 GB）：

- `TinyStoriesV2-GPT4-train.txt` / `-valid.txt`（约 2 GB）
- `owt_train.txt.gz` / `owt_valid.txt.gz`（解压后约 11 GB，磁盘留足）

> 国内网络建议先把 `HF_ENDPOINT` 设成 `https://hf-mirror.com`，否则 hugface 大概率断流。

---

## 6. 编辑器与效率配置

- **关掉 AI 自动补全**（Cursor Tab / Copilot inline）：官方明确说它会显著降低理解深度，且 A1–A5 的代码量正好在"补全帮你跳过思考"的区间。
- 装 `ruff` + `ty`（已在依赖里），格式化和类型检查能省掉大量低级错误：
  ```powershell
  uv run ruff check .     # 官方 pyproject 已经配好 line-length=120
  uv run ty check         # 类型检查，能抓到 jaxtyping 形状不匹配
  ```
- 用 `wandb` 记实验（依赖里也有）。A1 的 `experiment_log` 那题要的就是可复现的实验记录，越早用越省事。
- VS Code 装 Jupyter + Python 插件，`lecture_XX.py` 是带 `# %%` 分块的脚本，可以当 notebook 逐块跑。

---

## 7. 自检清单

```powershell
. E:\CS336\tools\env.ps1
cd E:\CS336\assignments\a1-basics
uv run python -c "import torch,sys;print(sys.version.split()[0], torch.__version__)"
# 期望：3.12.15 2.11.0+cpu

uv run pytest tests/test_model.py -q -p no:cacheprovider
# 期望：13 failed，全是 NotImplementedError

uv run pytest tests/test_tokenizer.py -q -p no:cacheprovider
# 期望：Windows 上 ImportError: No module named 'resource'（见第 3 节）
```

三条都对上，环境就算通了，去 `plan/00-ROADMAP.md` 第 1 周开工。
