# 创建 jcc-yolo conda 环境（YOLO 训练 / 检测专用）
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== 创建 YOLO conda 环境: jcc-yolo ===" -ForegroundColor Cyan

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "未找到 conda，请先安装 Anaconda/Miniconda"
}

$envExists = conda env list | Select-String "^\s*jcc-yolo\s"
if ($envExists) {
    Write-Host "环境 jcc-yolo 已存在，尝试更新依赖..." -ForegroundColor Yellow
    conda env update -f environment-yolo.yml --prune -y
} else {
    Write-Host "正在创建 GPU 版环境（首次约 5~15 分钟）..." -ForegroundColor Green
    conda env create -f environment-yolo.yml -y
    if ($LASTEXITCODE -ne 0) {
        Write-Host "GPU 版创建失败，改用 CPU 版..." -ForegroundColor Yellow
        conda env create -f environment-yolo-cpu.yml -y
    }
}

Write-Host ""
Write-Host "验证安装..." -ForegroundColor Cyan
conda run -n jcc-yolo python -c "import torch; from ultralytics import YOLO; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('ultralytics OK')"

Write-Host ""
Write-Host "=== 完成 ===" -ForegroundColor Green
Write-Host "训练 YOLO:  powershell -File scripts/train_yolo.ps1"
Write-Host "推理检测:   conda run -n jcc-yolo python inference.py 截图.png"
Write-Host "激活环境:   conda activate jcc-yolo"
