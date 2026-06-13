"""
端到端推理：游戏截图 → 检测框 → 英雄 + 星级

用法:
    python inference.py 截图.png
    python inference.py 截图.png --output result.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms

from config import (
    CLS_INPUT_SIZE,
    CROP_PADDING,
    DETECT_CONF,
    DETECT_IOU,
    HEROES,
    ROOT,
)
from models.classifier import PieceClassifier


def find_default_weights() -> tuple[Path | None, Path | None]:
    yolo_candidates = [
        ROOT / "runs" / "detect" / "yolo11n_piece" / "weights" / "best.pt",
        ROOT / "runs" / "detect" / "yolo11n_piece" / "weights" / "last.pt",
    ]
    cls_candidates = [
        ROOT / "runs" / "classifier" / "best.pt",
        ROOT / "runs" / "classifier" / "last.pt",
    ]
    yolo_w = next((p for p in yolo_candidates if p.exists()), None)
    cls_w = next((p for p in cls_candidates if p.exists()), None)
    return yolo_w, cls_w


def load_classifier(ckpt_path: Path, device: torch.device) -> PieceClassifier:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    heroes = ckpt.get("heroes", HEROES)
    model = PieceClassifier(num_heroes=len(heroes)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, heroes, ckpt.get("input_size", CLS_INPUT_SIZE)


def crop_with_padding(img: Image.Image, box, padding: float) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1 = max(0, int(x1 - bw * padding))
    y1 = max(0, int(y1 - bh * padding))
    x2 = min(w, int(x2 + bw * padding))
    y2 = min(h, int(y2 + bh * padding))
    return img.crop((x1, y1, x2, y2))


@torch.no_grad()
def classify_crop(model, crop: Image.Image, input_size: int, device: torch.device):
    tfm = transforms.Compose(
        [
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    x = tfm(crop.convert("RGB")).unsqueeze(0).to(device)
    hero_logits, star_logits = model(x)
    hero_id = hero_logits.argmax(1).item()
    star_id = star_logits.argmax(1).item()
    hero_conf = torch.softmax(hero_logits, 1)[0, hero_id].item()
    star_conf = torch.softmax(star_logits, 1)[0, star_id].item()
    return hero_id, star_id + 1, hero_conf, star_conf


def draw_results(img: Image.Image, results: list[dict]) -> Image.Image:
    out = img.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()

    colors = ["#00FF88", "#00BFFF", "#FFD700", "#FF6B6B", "#C084FC"]
    for i, r in enumerate(results):
        x1, y1, x2, y2 = r["box"]
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{r['hero']} {r['star']}★ ({r['det_conf']:.0%})"
        draw.rectangle([x1, max(0, y1 - 26), x1 + len(label) * 10, y1], fill=color)
        draw.text((x1 + 2, max(0, y1 - 24)), label, fill="black", font=font)
    return out


def predict(
    image_path: str | Path,
    yolo_weights: str | Path | None = None,
    cls_weights: str | Path | None = None,
    output_path: str | Path | None = None,
    conf: float = DETECT_CONF,
    iou: float = DETECT_IOU,
) -> list[dict]:
    from ultralytics import YOLO

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    default_yolo, default_cls = find_default_weights()
    yolo_weights = Path(yolo_weights) if yolo_weights else default_yolo
    cls_weights = Path(cls_weights) if cls_weights else default_cls

    if not yolo_weights or not yolo_weights.exists():
        raise FileNotFoundError("未找到 YOLO 权重，请先运行 train_yolo.py")
    if not cls_weights or not cls_weights.exists():
        raise FileNotFoundError("未找到分类器权重，请先运行 train_classifier.py")

    detector = YOLO(str(yolo_weights))
    classifier, heroes, input_size = load_classifier(cls_weights, device)

    img = Image.open(image_path)
    det_results = detector.predict(
        source=str(image_path),
        conf=conf,
        iou=iou,
        verbose=False,
        device=0 if device.type == "cuda" else "cpu",
    )[0]

    parsed = []
    if det_results.boxes is not None and len(det_results.boxes):
        for box in det_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            det_conf = float(box.conf[0])
            crop = crop_with_padding(img, (x1, y1, x2, y2), CROP_PADDING)
            hero_id, star, hero_conf, star_conf = classify_crop(classifier, crop, input_size, device)
            parsed.append(
                {
                    "box": [int(x1), int(y1), int(x2), int(y2)],
                    "hero": heroes[hero_id],
                    "hero_id": hero_id,
                    "star": star,
                    "det_conf": det_conf,
                    "hero_conf": hero_conf,
                    "star_conf": star_conf,
                }
            )

    print(f"识别到 {len(parsed)} 个棋子:")
    for i, r in enumerate(parsed, 1):
        print(
            f"  [{i}] {r['hero']} {r['star']}★  "
            f"检测={r['det_conf']:.0%} 英雄={r['hero_conf']:.0%} 星级={r['star_conf']:.0%}  "
            f"框={r['box']}"
        )

    if output_path:
        vis = draw_results(img, parsed)
        vis.save(output_path)
        print(f"结果图已保存: {output_path}")

    return parsed


def main():
    parser = argparse.ArgumentParser(description="金铲铲棋子识别")
    parser.add_argument("image", help="输入截图路径")
    parser.add_argument("--output", "-o", help="输出标注图路径")
    parser.add_argument("--yolo", help="YOLO 权重路径")
    parser.add_argument("--cls", help="分类器权重路径")
    parser.add_argument("--conf", type=float, default=DETECT_CONF)
    parser.add_argument("--iou", type=float, default=DETECT_IOU)
    args = parser.parse_args()

    out = args.output or str(Path(args.image).with_stem(Path(args.image).stem + "_result"))
    predict(args.image, args.yolo, args.cls, out, args.conf, args.iou)


if __name__ == "__main__":
    main()
