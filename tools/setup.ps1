# 一键拉取课程官方仓库（课件 + 5 个作业）
#
#   .\tools\setup.ps1          # 在 E:\CS336 下补齐 lectures\ 与 assignments\
#
# 说明：
#   - 本机 HTTPS 走 schannel 会报 SEC_E_NO_CREDENTIALS，所以 git 必须加 -c http.sslBackend=openssl
#   - 用 --depth 1 浅克隆，只要最新版，省几百 MB

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot

$repos = [ordered]@{
    'lectures\cs336-lectures'  = 'lectures'
    'assignments\a1-basics'    = 'assignment1-basics'
    'assignments\a2-systems'   = 'assignment2-systems'
    'assignments\a3-scaling'   = 'assignment3-scaling'
    'assignments\a4-data'      = 'assignment4-data'
    'assignments\a5-alignment' = 'assignment5-alignment'
}

$env:GIT_TERMINAL_PROMPT = '0'

foreach ($k in $repos.Keys) {
    $dst = Join-Path $root $k
    if (Test-Path $dst) {
        Write-Host "skip  $k (已存在)" -ForegroundColor DarkGray
        continue
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dst) | Out-Null
    Write-Host "clone $k ..." -ForegroundColor Cyan
    git -c http.sslBackend=openssl clone --depth 1 "https://github.com/stanford-cs336/$($repos[$k]).git" $dst
}

Write-Host ""
Write-Host "完成。下一步：" -ForegroundColor Green
Write-Host "  . .\tools\env.ps1"
Write-Host "  cd .\assignments\a1-basics; uv sync; uv run pytest -p no:cacheprovider"
