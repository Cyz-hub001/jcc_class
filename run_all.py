"""
一键流程：预处理 → 训练分类器 → 训练检测 → 在 test 集上试跑
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from config import ROOT
from utils.conda_yolo import ensure_yolo_env, run_yolo_script


def run(cmd: list[str]):
    print(f"\n>>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    # 默认环境：数据预处理 + ResNet 分类器
    run([sys.executable, "prepare_data.py"])
    run([sys.executable, "train_classifier.py"])

    # jcc-yolo conda 环境：YOLO 检测训练
    ensure_yolo_env()
    run_yolo_script("train_yolo.py")

    # 端到端推理也在 jcc-yolo 环境（含 ultralytics + torch）
    test_txt = ROOT / "dataset" / "yolo_detect" / "test.txt"
    if test_txt.exists():
        first = test_txt.read_text(encoding="utf-8").strip().splitlines()[0]
        img = ROOT / "dataset" / "yolo_detect" / first
        if img.exists():
            run_yolo_script(
                "inference.py",
                str(img),
                "-o",
                str(ROOT / "runs" / "demo_result.png"),
            )


if __name__ == "__main__":
    main()
