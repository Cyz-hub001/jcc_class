"""
数据预处理：
1. 修复路径
2. 按原图划分 train/val/test
3. 生成单类 YOLO 检测集
4. 导出分类裁剪图 cls_crops
"""
from __future__ import annotations

import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

from config import (
    DATASET,
    HEROES,
    NEGATIVE_IMAGES,
    NUM_STARS,
    RANDOM_SEED,
    RAW_DATASET,
    SPLIT_RATIO,
)


def parse_class_id(cid: int) -> tuple[int, int]:
    hero_id = cid // 3
    star = cid % 3 + 1
    return hero_id, star


def yolo_to_xyxy(cx: float, cy: float, w: float, h: float, img_w: int, img_h: int):
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2)


def load_image_records() -> list[dict]:
    train_txt = RAW_DATASET / "train.txt"
    lines = train_txt.read_text(encoding="utf-8").strip().splitlines()

    records = []
    for line in lines:
        rel = line.strip().replace("data/", "", 1) if line.startswith("data/") else line.strip()
        img_path = RAW_DATASET / rel
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {img_path}")

        stem = img_path.stem
        label_path = RAW_DATASET / "labels" / "train" / "chanzi" / f"{stem}.txt"
        boxes = []
        if label_path.exists():
            content = label_path.read_text(encoding="utf-8").strip()
            if content:
                for row in content.splitlines():
                    parts = row.split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        boxes.append(
                            {
                                "class_id": cid,
                                "hero_id": cid // 3,
                                "star": cid % 3 + 1,
                                "cx": float(parts[1]),
                                "cy": float(parts[2]),
                                "w": float(parts[3]),
                                "h": float(parts[4]),
                            }
                        )

        records.append(
            {
                "stem": stem,
                "img_path": img_path,
                "rel_path": f"images/train/chanzi/{img_path.name}",
                "boxes": boxes,
                "is_negative": stem in NEGATIVE_IMAGES or len(boxes) == 0,
            }
        )
    return records


def stratified_split(records: list[dict]) -> dict[str, list[dict]]:
    """按原图划分，尽量让各英雄在 val/test 有代表。"""
    rng = random.Random(RANDOM_SEED)

    # 以每张图出现最多的英雄作为分层键
    buckets: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        if not rec["boxes"]:
            buckets[-1].append(rec)
            continue
        hero_counts = defaultdict(int)
        for b in rec["boxes"]:
            hero_counts[b["hero_id"]] += 1
        dominant = max(hero_counts, key=hero_counts.get)
        buckets[dominant].append(rec)

    splits = {"train": [], "val": [], "test": []}
    for hero_id, group in buckets.items():
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * SPLIT_RATIO[2])) if n >= 3 else 0
        n_val = max(1, round(n * SPLIT_RATIO[1])) if n >= 2 else 0
        if n - n_val - n_test < 1:
            n_test = 0
            n_val = min(1, n - 1) if n > 1 else 0

        test_part = group[:n_test]
        val_part = group[n_test : n_test + n_val]
        train_part = group[n_test + n_val :]
        splits["test"].extend(test_part)
        splits["val"].extend(val_part)
        splits["train"].extend(train_part)

    for key in splits:
        rng.shuffle(splits[key])
    return splits


def write_yolo_split(split_name: str, items: list[dict], yolo_root: Path):
    img_out = yolo_root / "images" / split_name / "chanzi"
    lbl_out = yolo_root / "labels" / split_name / "chanzi"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for rec in items:
        dst_img = img_out / rec["img_path"].name
        if not dst_img.exists():
            shutil.copy2(rec["img_path"], dst_img)

        lbl_lines = []
        for b in rec["boxes"]:
            lbl_lines.append(f"0 {b['cx']:.6f} {b['cy']:.6f} {b['w']:.6f} {b['h']:.6f}")

        (lbl_out / f"{rec['stem']}.txt").write_text(
            "\n".join(lbl_lines) + ("\n" if lbl_lines else ""),
            encoding="utf-8",
        )
        abs_path = (img_out / rec["img_path"].name).as_posix()
        manifest.append(abs_path)

    (yolo_root / f"{split_name}.txt").write_text("\n".join(manifest) + "\n", encoding="utf-8")


def export_crops(split_name: str, items: list[dict], crops_root: Path) -> int:
    count = 0
    meta = []
    for rec in items:
        with Image.open(rec["img_path"]) as im:
            im = im.convert("RGB")
            w, h = im.size
            for idx, b in enumerate(rec["boxes"]):
                x1, y1, x2, y2 = yolo_to_xyxy(b["cx"], b["cy"], b["w"], b["h"], w, h)
                pad_w = int((x2 - x1) * 0.05)
                pad_h = int((y2 - y1) * 0.05)
                x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
                x2, y2 = min(w, x2 + pad_w), min(h, y2 + pad_h)
                crop = im.crop((x1, y1, x2, y2))

                hero = HEROES[b["hero_id"]]
                star = b["star"]
                out_dir = crops_root / split_name / hero / str(star)
                out_dir.mkdir(parents=True, exist_ok=True)
                fname = f"{rec['stem']}_{idx:02d}.jpg"
                crop.save(out_dir / fname, quality=95)
                meta.append(
                    {
                        "file": str(out_dir.relative_to(crops_root) / fname),
                        "hero": hero,
                        "hero_id": b["hero_id"],
                        "star": star,
                        "source_image": rec["stem"],
                    }
                )
                count += 1

    (crops_root / f"{split_name}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return count


def write_yolo_yaml(yolo_root: Path):
    yaml_content = f"""path: {yolo_root.as_posix()}
train: train.txt
val: val.txt
test: test.txt

names:
  0: piece
"""
    (yolo_root / "data_detect.yaml").write_text(yaml_content, encoding="utf-8")


def main():
    if DATASET.exists():
        shutil.rmtree(DATASET)
    DATASET.mkdir(parents=True)

    records = load_image_records()
    splits = stratified_split(records)

    yolo_root = DATASET / "yolo_detect"
    crops_root = DATASET / "cls_crops"
    yolo_root.mkdir(parents=True)
    crops_root.mkdir(parents=True)

    split_map = {}
    crop_total = 0
    for split_name, items in splits.items():
        write_yolo_split(split_name, items, yolo_root)
        crop_total += export_crops(split_name, items, crops_root)
        for rec in items:
            split_map[rec["stem"]] = split_name

    write_yolo_yaml(yolo_root)

    summary = {
        "total_images": len(records),
        "splits": {k: len(v) for k, v in splits.items()},
        "total_crops": crop_total,
        "negative_images": list(NEGATIVE_IMAGES),
        "heroes": HEROES,
        "split_map": split_map,
    }
    (DATASET / "split.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=== 数据预处理完成 ===")
    print(f"图片总数: {len(records)}")
    for k, v in splits.items():
        boxes = sum(len(r["boxes"]) for r in v)
        print(f"  {k}: {len(v)} 张图, {boxes} 个框")
    print(f"裁剪图总数: {crop_total}")
    print(f"YOLO 配置: {yolo_root / 'data_detect.yaml'}")
    print(f"分类数据: {crops_root}")


if __name__ == "__main__":
    main()
