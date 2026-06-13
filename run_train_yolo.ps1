[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING="utf-8"
$env:PYTHONUTF8="1"
Set-Location "e:/金铲铲课题"
conda run -n jcc-yolo python train_yolo.py
