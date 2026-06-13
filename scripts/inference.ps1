# 端到端推理（检测在 jcc-yolo 环境，需该环境含 torch+ultralytics）
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Image,
    [string]$Output = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

$envExists = conda env list | Select-String "^\s*jcc-yolo\s"
if (-not $envExists) {
    & "$PSScriptRoot\setup_yolo_env.ps1"
}

$args = @("inference.py", $Image)
if ($Output) { $args += @("-o", $Output) }

conda run --no-capture-output -n jcc-yolo python @args
