"""金铲铲棋子识别 — 共享配置"""
from pathlib import Path

ROOT = Path(__file__).parent
RAW_DATASET = ROOT / "标注数据集"
DATASET = ROOT / "dataset"

HEROES = [
    "Aphelios",
    "Fiddlesticks",
    "Janna",
    "Leona",
    "Mordekaiser",
    "Nunu",
    "Syndra",
    "Urgot",
]

NUM_HEROES = len(HEROES)
NUM_STARS = 3

# 无标注图片：保留为 YOLO 负样本
NEGATIVE_IMAGES: set[str] = set()

SPLIT_RATIO = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

# YOLO（独立 conda 环境 jcc-yolo）
YOLO_CONDA_ENV = "jcc-yolo"
YOLO_MODEL = "yolo11s.pt"  # 预训练权重文件，升级到 s 版本以提升稀疏场景检测能力
YOLO_DATA = DATASET / "yolo_detect" / "data_detect.yaml"
YOLO_IMGSZ = 640
YOLO_EPOCHS = 100
YOLO_BATCH = 4

# ResNet
CLS_INPUT_SIZE = 224  # ResNet18 标准输入尺寸，特征图从 4×4 恢复到 7×7
CLS_EPOCHS = 50
CLS_BATCH = 64
STAR_CLASS_WEIGHTS = [1.0, 1.5, 5.0]  # ★=1.0  ★★=1.5(轻微提升)  ★★★=5.0(三星极少，大幅加权)

# 推理
CROP_PADDING = 0.08
DETECT_CONF = 0.25
DETECT_IOU = 0.45
