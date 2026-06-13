@echo off
powershell -ExecutionPolicy Bypass -File "%~dp0train_yolo.ps1" %*
pause
