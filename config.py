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
YOLO_DATA = DATASET / "yolo_detect" / "data_detect.yaml"
YOLO_IMGSZ = 640
YOLO_EPOCHS = 100
YOLO_BATCH = 4

# ResNet
CLS_INPUT_SIZE = 128
CLS_EPOCHS = 50
CLS_BATCH = 64
STAR_CLASS_WEIGHTS = [1.0, 1.0, 2.5]

# 推理
CROP_PADDING = 0.08
DETECT_CONF = 0.25
DETECT_IOU = 0.45
