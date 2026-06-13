"""
训练 ResNet18 多任务分类器（英雄 8 类 + 星级 3 类）
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from config import (
    CLS_BATCH,
    CLS_EPOCHS,
    CLS_INPUT_SIZE,
    DATASET,
    HEROES,
    NUM_HEROES,
    NUM_STARS,
    ROOT,
    STAR_CLASS_WEIGHTS,
)
from models.classifier import PieceClassifier


class CropDataset(Dataset):
    def __init__(self, meta_path: Path, transform):
        self.items = json.loads(meta_path.read_text(encoding="utf-8"))
        self.root = meta_path.parent
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(self.root / item["file"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, item["hero_id"], item["star"] - 1


def build_transforms(train: bool):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(CLS_INPUT_SIZE, scale=(0.8, 1.0)),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((CLS_INPUT_SIZE, CLS_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def make_star_sampler(meta_path: Path) -> WeightedRandomSampler:
    items = json.loads(meta_path.read_text(encoding="utf-8"))
    star_counts = [0, 0, 0]
    for it in items:
        star_counts[it["star"] - 1] += 1
    weights = []
    for it in items:
        w = STAR_CLASS_WEIGHTS[it["star"] - 1] / star_counts[it["star"] - 1]
        weights.append(w)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    hero_correct = hero_total = 0
    star_correct = star_total = 0
    both_correct = both_total = 0

    for imgs, hero_y, star_y in loader:
        imgs = imgs.to(device)
        hero_y = hero_y.to(device)
        star_y = star_y.to(device)
        hero_logits, star_logits = model(imgs)
        hero_pred = hero_logits.argmax(1)
        star_pred = star_logits.argmax(1)

        hero_correct += (hero_pred == hero_y).sum().item()
        star_correct += (star_pred == star_y).sum().item()
        both_correct += ((hero_pred == hero_y) & (star_pred == star_y)).sum().item()
        n = imgs.size(0)
        hero_total += n
        star_total += n
        both_total += n

    return {
        "hero_acc": hero_correct / max(hero_total, 1),
        "star_acc": star_correct / max(star_total, 1),
        "joint_acc": both_correct / max(both_total, 1),
    }


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    crops_root = DATASET / "cls_crops"
    train_meta = crops_root / "train_meta.json"
    val_meta = crops_root / "val_meta.json"
    if not train_meta.exists():
        raise FileNotFoundError("请先运行 prepare_data.py")

    train_ds = CropDataset(train_meta, build_transforms(train=True))
    val_ds = CropDataset(val_meta, build_transforms(train=False))

    sampler = make_star_sampler(train_meta)
    train_loader = DataLoader(train_ds, batch_size=CLS_BATCH, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=CLS_BATCH, shuffle=False, num_workers=0)

    model = PieceClassifier(NUM_HEROES, NUM_STARS).to(device)
    star_weight = torch.tensor(STAR_CLASS_WEIGHTS, dtype=torch.float32, device=device)
    hero_crit = nn.CrossEntropyLoss()
    star_crit = nn.CrossEntropyLoss(weight=star_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CLS_EPOCHS)

    save_dir = ROOT / "runs" / "classifier"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_joint = 0.0
    history = []

    for epoch in range(1, CLS_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, hero_y, star_y in tqdm(train_loader, desc=f"Epoch {epoch}/{CLS_EPOCHS}"):
            imgs = imgs.to(device)
            hero_y = hero_y.to(device)
            star_y = star_y.to(device)

            optimizer.zero_grad()
            hero_logits, star_logits = model(imgs)
            loss = hero_crit(hero_logits, hero_y) + 1.2 * star_crit(star_logits, star_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        metrics["loss"] = total_loss / len(train_loader)
        metrics["epoch"] = epoch
        history.append(metrics)

        print(
            f"Epoch {epoch}: loss={metrics['loss']:.4f} "
            f"hero_acc={metrics['hero_acc']:.3f} star_acc={metrics['star_acc']:.3f} "
            f"joint_acc={metrics['joint_acc']:.3f}"
        )

        if metrics["joint_acc"] > best_joint:
            best_joint = metrics["joint_acc"]
            ckpt = {
                "model_state": model.state_dict(),
                "heroes": HEROES,
                "input_size": CLS_INPUT_SIZE,
                "metrics": metrics,
            }
            torch.save(ckpt, save_dir / "best.pt")

    torch.save({"model_state": model.state_dict(), "heroes": HEROES, "input_size": CLS_INPUT_SIZE}, save_dir / "last.pt")
    (save_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"最佳 joint_acc={best_joint:.3f}, 模型保存在 {save_dir / 'best.pt'}")


if __name__ == "__main__":
    train()
