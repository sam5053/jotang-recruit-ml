import argparse
import copy
import json
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from catdog_common import CatDogCNN, collect_images, limit_per_class, stratified_split
from train_classifier import make_loaders, run_epoch, set_seed


class FullyConnectedNet(nn.Module):
    def __init__(self, image_size=160):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * image_size * image_size, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.layers(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--max-per-class", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    set_seed(args.seed)
    items, _ = collect_images(data_dir)
    splits = stratified_split(limit_per_class(items, args.max_per_class, args.seed), args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaders = make_loaders(splits, args.image_size, False, args.batch_size, 0, device)
    criterion = nn.CrossEntropyLoss()
    results = {}

    for name, factory in (
        ("fully_connected", lambda: FullyConnectedNet(args.image_size)),
        ("cnn", CatDogCNN),
    ):
        set_seed(args.seed)
        model = factory().to(device)
        optimizer = Adam(model.parameters(), lr=1e-3)
        best_val, best_state = -1, None
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
            val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
            print(f"{name} epoch {epoch}: train={train_acc:.2%}, val={val_acc:.2%}", flush=True)
            if val_acc > best_val:
                best_val, best_state = val_acc, copy.deepcopy(model.state_dict())
        model.load_state_dict(best_state)
        _, test_acc = run_epoch(model, loaders["test"], criterion, device)
        results[name] = {
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "best_val_accuracy": best_val,
            "test_accuracy": test_acc,
        }

    results["setup"] = {
        "max_per_class": args.max_per_class,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "split_sizes": {name: len(items) for name, items in splits.items()},
    }
    path = root / "outputs" / "practice2" / "fc_baseline_metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(path, flush=True)


if __name__ == "__main__":
    main()
