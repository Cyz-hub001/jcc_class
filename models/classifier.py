"""ResNet18 多任务分类：英雄 + 星级"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


class PieceClassifier(nn.Module):
    def __init__(self, num_heroes: int = 8, num_stars: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.hero_head = nn.Linear(in_features, num_heroes)
        self.star_head = nn.Linear(in_features, num_stars)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone(x)
        return self.hero_head(feat), self.star_head(feat)
