# 在 jcc-yolo conda 环境中训练 YOLO
$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$envExists = conda env list | Select-String "^\s*jcc-yolo\s"
if (-not $envExists) {
    Write-Host "环境 jcc-yolo 不存在，先创建..." -ForegroundColor Yellow
    & "$PSScriptRoot\setup_yolo_env.ps1"
}

Write-Host "=== 在 jcc-yolo 环境中训练 YOLO ===" -ForegroundColor Cyan
conda run --no-capture-output -n jcc-yolo python train_yolo.py @args
