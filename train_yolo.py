"""
训练 YOLO 单类棋子检测（模型由 config.YOLO_MODEL 指定）

请在 jcc-yolo conda 环境中运行:
    powershell -File scripts/train_yolo.ps1
或:
    conda run -n jcc-yolo python train_yolo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from config import ROOT, YOLO_BATCH, YOLO_CONDA_ENV, YOLO_DATA, YOLO_EPOCHS, YOLO_IMGSZ, YOLO_MODEL


def _check_env():
    if YOLO_CONDA_ENV not in sys.prefix.replace("\\", "/"):
        print(f"[提示] 建议在 conda 环境 {YOLO_CONDA_ENV} 中运行 YOLO 训练")
        print(f"       powershell -File scripts/train_yolo.ps1")


def train():
    _check_env()
    if not YOLO_DATA.exists():
        raise FileNotFoundError("请先运行 prepare_data.py 生成 yolo_detect 数据集")

    from ultralytics import YOLO

    save_dir = ROOT / "runs" / "detect"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(YOLO_MODEL)
    model_stem = Path(YOLO_MODEL).stem  # e.g. yolo11s
    results = model.train(
        data=str(YOLO_DATA),
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=YOLO_BATCH,
        patience=20,
        project=str(save_dir),
        name=f"{model_stem}_piece",
        exist_ok=True,
        rect=True,
        augment=True,
        workers=2,
        device=0 if _has_cuda() else "cpu",
    )
    print(f"训练完成，权重目录: {results.save_dir}")


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    train()
