"""Quick dataset analysis for 标注数据集."""
from pathlib import Path
from collections import Counter, defaultdict

base = Path(__file__).parent / "标注数据集"
train_txt = (base / "train.txt").read_text(encoding="utf-8").strip().splitlines()

actual_img_dir = base / "images" / "train" / "chanzi"
actual_imgs = set(p.name for p in actual_img_dir.glob("*.png")) if actual_img_dir.exists() else set()

print("=== 路径检查 ===")
print(f"train.txt 行数: {len(train_txt)}")
print(f"首条路径: {train_txt[0]}")
print(f"实际图片目录存在: {actual_img_dir.exists()}")
if actual_img_dir.exists():
    print(f"实际图片数: {len(actual_imgs)}")

missing_imgs = []
for line in train_txt:
    rel = line.strip()
    p1 = base / rel
    p2 = base / rel.replace("data/", "", 1) if rel.startswith("data/") else base / rel
    if not p1.exists() and not p2.exists():
        missing_imgs.append(rel)
print(f"找不到的图片: {len(missing_imgs)}")

# path prefix analysis
with_data = sum(1 for l in train_txt if l.startswith("data/"))
print(f"带 data/ 前缀的路径: {with_data}/{len(train_txt)}")

label_dir = base / "labels" / "train" / "chanzi"
labels = sorted(label_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)

print("\n=== 标注统计 ===")
print(f"标注文件数: {len(labels)}")

empty_labels = []
missing_label_for_img = []
box_counts = []
class_counter = Counter()
hero_star = defaultdict(lambda: Counter())

heroes = ["Yone", "Draven", "Ornn", "Thresh", "Lee Sin", "Azir", "Sett", "Swain"]

for lf in labels:
    content = lf.read_text(encoding="utf-8").strip()
    if not content:
        empty_labels.append(lf.stem)
        continue
    lines = [l for l in content.splitlines() if l.strip()]
    box_counts.append(len(lines))
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cid = int(parts[0])
            class_counter[cid] += 1
            hero_id = cid // 3
            star = cid % 3 + 1
            if hero_id < 8:
                hero_star[heroes[hero_id]][star] += 1

# images in train.txt without labels
txt_stems = set()
for line in train_txt:
    stem = Path(line.strip()).stem
    txt_stems.add(stem)

label_stems = set(lf.stem for lf in labels)
no_label = sorted(txt_stems - label_stems, key=lambda x: int(x) if x.isdigit() else x)
empty_or_missing = no_label + empty_labels

print(f"无标注文件(图片在train.txt但无txt): {no_label}")
print(f"空标注文件: {empty_labels}")
print(f"总框数: {sum(class_counter.values())}")
print(f"有标注的图片数: {len(box_counts)}")
if box_counts:
    print(f"每图框数: min={min(box_counts)}, max={max(box_counts)}, avg={sum(box_counts)/len(box_counts):.1f}")

print("\n=== 类别分布 (hero x star) ===")
for h in heroes:
    c = hero_star[h]
    print(f"{h:10s}  1星={c[1]:4d}  2星={c[2]:4d}  3星={c[3]:4d}  total={sum(c.values())}")

print("\n=== 星级汇总 ===")
star_total = Counter()
for h in heroes:
    for s in [1, 2, 3]:
        star_total[s] += hero_star[h][s]
for s in [1, 2, 3]:
    print(f"{s}星: {star_total[s]}")

# image size sample
try:
    from PIL import Image
    sample_path = base / "images" / "train" / "chanzi" / "1.png"
    if sample_path.exists():
        with Image.open(sample_path) as im:
            print(f"\n=== 图片尺寸样例 ===")
            print(f"1.png: {im.size}")
except ImportError:
    pass

# bbox size analysis (normalized w,h)
wh_list = []
for lf in labels:
    content = lf.read_text(encoding="utf-8").strip()
    if not content:
        continue
    for line in content.splitlines():
        parts = line.split()
        if len(parts) >= 5:
            w, h = float(parts[3]), float(parts[4])
            wh_list.append((w, h))

if wh_list:
    ws = [x[0] for x in wh_list]
    hs = [x[1] for x in wh_list]
    print(f"\n=== BBox 归一化尺寸 ===")
    print(f"w: min={min(ws):.4f}, max={max(ws):.4f}, avg={sum(ws)/len(ws):.4f}")
    print(f"h: min={min(hs):.4f}, max={max(hs):.4f}, avg={sum(hs)/len(hs):.4f}")
    # approx pixels on 2688x1216
    avg_px_w = sum(ws) / len(ws) * 2688
    avg_px_h = sum(hs) / len(hs) * 1216
    print(f"约合像素(2688x1216): avg_w={avg_px_w:.0f}px, avg_h={avg_px_h:.0f}px")
