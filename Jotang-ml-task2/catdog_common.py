from pathlib import Path
import random

import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms


CLASS_NAMES = ["猫", "狗"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def label_from_path(path):
    name = path.name.lower()
    parent = path.parent.name.lower()
    if name.startswith("cat.") or parent in {"cat", "cats"}:
        return 0
    if name.startswith("dog.") or parent in {"dog", "dogs"}:
        return 1
    return None


def collect_images(data_dir, verify=True):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"找不到数据目录：{data_dir}")
    items, invalid = [], []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_from_path(path)
        if label is None:
            continue
        if verify:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                invalid.append(path)
                continue
        items.append((path, label))
    if not items:
        raise RuntimeError("没有找到 cat.* / dog.* 图片，或 Cat、Dog 子文件夹。")
    return items, invalid


def stratified_split(items, seed=42, train_ratio=0.7, val_ratio=0.15):
    rng = random.Random(seed)
    groups = {0: [], 1: []}
    for item in items:
        groups[item[1]].append(item)
    splits = {"train": [], "val": [], "test": []}
    for group in groups.values():
        rng.shuffle(group)
        n_train = int(len(group) * train_ratio)
        n_val = int(len(group) * val_ratio)
        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train : n_train + n_val])
        splits["test"].extend(group[n_train + n_val :])
    for split in splits.values():
        rng.shuffle(split)
    return splits


def limit_per_class(items, max_per_class, seed=42):
    if not max_per_class:
        return items
    rng = random.Random(seed)
    result = []
    for label in (0, 1):
        group = [item for item in items if item[1] == label]
        rng.shuffle(group)
        result.extend(group[:max_per_class])
    rng.shuffle(result)
    return result


def build_transform(training=False, augment=False, image_size=160):
    if training and augment:
        geometry = [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        ]
    else:
        geometry = [transforms.Resize(image_size + 16), transforms.CenterCrop(image_size)]
    return transforms.Compose(
        geometry + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


class CatDogDataset(Dataset):
    def __init__(self, items, transform):
        self.items = list(items)
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        path, label = self.items[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image)
        return tensor, label, str(path)


class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = self._block(3, 32)
        self.block2 = self._block(32, 64)
        self.block3 = self._block(64, 128)
        self.block4 = self._block(128, 128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, 2))

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward_features(self, x):
        features = {}
        for name in ("block1", "block2", "block3", "block4"):
            x = getattr(self, name)(x)
            features[name] = x
        return x, features

    def forward(self, x):
        x, _ = self.forward_features(x)
        return self.classifier(self.pool(x))


def denormalize(tensor):
    mean = torch.tensor(MEAN, device=tensor.device).view(3, 1, 1)
    std = torch.tensor(STD, device=tensor.device).view(3, 1, 1)
    return (tensor * std + mean).clamp(0, 1)
