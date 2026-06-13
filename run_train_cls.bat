@echo off
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
call conda activate jcc-yolo
python "%~dp0train_classifier.py"
