"""在独立 conda 环境 jcc-yolo 中运行 YOLO 相关命令。"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from config import YOLO_CONDA_ENV

ROOT = Path(__file__).parent.parent


def conda_available() -> bool:
    return shutil.which("conda") is not None


def in_yolo_env() -> bool:
    return sys.prefix and Path(sys.prefix).name.replace("\\", "/").endswith(YOLO_CONDA_ENV)


def yolo_python_cmd() -> list[str]:
    """返回用于执行 YOLO 脚本的 python 命令前缀。"""
    if in_yolo_env():
        return [sys.executable]
    if not conda_available():
        raise RuntimeError(
            f"当前不在 {YOLO_CONDA_ENV} 环境且未找到 conda。"
            f"请先运行 scripts/setup_yolo_env.ps1 创建环境。"
        )
    return ["conda", "run", "--no-capture-output", "-n", YOLO_CONDA_ENV, "python"]


def run_yolo_script(script: str, *args: str, cwd: Path | None = None):
    cmd = yolo_python_cmd() + [script, *args]
    print(f">>> {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=cwd or ROOT)


def ensure_yolo_env():
    if in_yolo_env():
        return
    if not conda_available():
        raise RuntimeError("未找到 conda，无法使用 YOLO 环境")
    result = subprocess.run(
        ["conda", "env", "list"],
        capture_output=True,
        text=True,
        check=False,
    )
    if YOLO_CONDA_ENV not in result.stdout:
        raise RuntimeError(
            f"conda 环境 {YOLO_CONDA_ENV} 不存在，请先执行:\n"
            f"  powershell -ExecutionPolicy Bypass -File scripts/setup_yolo_env.ps1"
        )
