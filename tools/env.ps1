# CS336 环境变量（在 PowerShell 里 dot-source 本文件）
#   . E:\CS336\tools\env.ps1
#
# 若报 "running scripts is disabled on this system"，先执行一次：
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
#
# 为什么这么设：
#   - uv / Python / 缓存都放 E:，因为 C: 只剩十几 GB，而且本会话只允许写 E:\CS336
#   - 默认索引换清华源，国内直连 pypi 会卡住
#   - HF 走镜像，否则大文件大概率断流

$cs336Root = 'E:\CS336'

$env:UV_PYTHON_INSTALL_DIR = Join-Path $cs336Root '.tools\pythons'
$env:UV_PYTHON_BIN_DIR     = Join-Path $cs336Root '.tools\bin'
$env:UV_CACHE_DIR          = Join-Path $cs336Root '.tools\uv-cache'
$env:UV_DEFAULT_INDEX      = 'https://pypi.tuna.tsinghua.edu.cn/simple'
$env:HF_ENDPOINT           = 'https://hf-mirror.com'

$uvBin = Join-Path $cs336Root '.tools\uvpkg\bin'
if (-not ($env:Path -split ';' -contains $uvBin)) {
    $env:Path = "$uvBin;$env:Path"
}

Write-Host "CS336 env ready" -ForegroundColor Green
Write-Host ("  root   : " + $cs336Root)
Write-Host ("  uv     : " + (Get-Command uv -ErrorAction SilentlyContinue).Source)
Write-Host ("  python : " + $env:UV_PYTHON_INSTALL_DIR)
Write-Host "  测试   : cd E:\CS336\assignments\a1-basics; uv run pytest -p no:cacheprovider"
