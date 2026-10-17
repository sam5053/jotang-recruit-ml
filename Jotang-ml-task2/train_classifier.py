import argparse
import copy
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from catdog_common import (
    CLASS_NAMES,
    CatDogCNN,
    CatDogDataset,
    build_transform,
    collect_images,
    denormalize,
    limit_per_class,
    stratified_split,
)


def parse_args():
    parser = argparse.ArgumentParser(description="训练猫狗分类 CNN")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["no_aug", "augmented"],
        default=["no_aug", "augmented"],
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def count_labels(items):
    return {label: sum(item[1] == label for item in items) for label in (0, 1)}


def save_data_overview(items, output_dir, seed):
    rng = random.Random(seed)
    sample = rng.sample(items, min(12, len(items)))
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (path, label) in zip(axes.flat, sample):
        with Image.open(path) as image:
            ax.imshow(image.convert("RGB"))
            ax.set_title(f"{'cat' if label == 0 else 'dog'}  {image.width}x{image.height}")
        ax.axis("off")
    fig.suptitle("Random samples and original sizes")
    fig.tight_layout()
    fig.savefig(output_dir / "data_overview.png", dpi=160)
    plt.close(fig)


def tensor_image(tensor):
    return denormalize(tensor).permute(1, 2, 0).cpu().numpy()


def save_preprocessing_check(items, output_dir, image_size, seed):
    rng = random.Random(seed)
    sample = rng.sample(items, min(4, len(items)))
    plain = build_transform(False, False, image_size)
    augmented = build_transform(True, True, image_size)
    fig, axes = plt.subplots(len(sample), 3, figsize=(10, 3 * len(sample)))
    axes = np.atleast_2d(axes)
    for row, (path, label) in enumerate(sample):
        with Image.open(path) as image:
            image = image.convert("RGB")
            axes[row, 0].imshow(image)
            axes[row, 1].imshow(tensor_image(plain(image)))
            axes[row, 2].imshow(tensor_image(augmented(image)))
        axes[row, 0].set_ylabel("cat" if label == 0 else "dog")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0, 0].set_title("Original")
    axes[0, 1].set_title("Resize + center crop")
    axes[0, 2].set_title("Random augmentation")
    fig.tight_layout()
    fig.savefig(output_dir / "preprocessing_check.png", dpi=160)
    plt.close(fig)

    path, label = sample[0]
    with Image.open(path) as image:
        image = image.convert("RGB")
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        for ax in axes.flat:
            ax.imshow(tensor_image(augmented(image)))
            ax.axis("off")
        fig.suptitle(f"Eight random augmentations of one {'cat' if label == 0 else 'dog'} image")
        fig.tight_layout()
        fig.savefig(output_dir / "augmentation_samples.png", dpi=160)
        plt.close(fig)


def save_model_structure(model, output_dir, image_size, device):
    x = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        features, activations = model.forward_features(x)
        output = model.classifier(model.pool(features))
    rows = [("Input", tuple(x.shape))]
    rows.extend((name, tuple(value.shape)) for name, value in activations.items())
    rows.extend([("AdaptiveAvgPool", (1, 128, 1, 1)), ("Linear", tuple(output.shape))])
    text = str(model) + "\n\n" + "\n".join(f"{name}: {shape}" for name, shape in rows)
    (output_dir / "model_structure.txt").write_text(text, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    xs = np.linspace(0.06, 0.94, len(rows))
    colors = ["#DCEEFF", "#C8E6C9", "#A5D6A7", "#81C784", "#66BB6A", "#FFE0B2", "#FFCCBC"]
    for index, ((name, shape), x_pos) in enumerate(zip(rows, xs)):
        label = name + "\n" + "×".join(map(str, shape))
        ax.text(
            x_pos,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "fc": colors[index], "ec": "#455A64"},
        )
        if index < len(rows) - 1:
            ax.annotate("", xy=(xs[index + 1] - 0.055, 0.5), xytext=(x_pos + 0.055, 0.5), arrowprops={"arrowstyle": "->"})
    ax.set_title("CatDogCNN structure and tensor shapes", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "model_structure.png", dpi=180)
    plt.close(fig)
    return rows


def make_loaders(splits, image_size, augment, batch_size, num_workers, device):
    datasets = {
        "train": CatDogDataset(splits["train"], build_transform(True, augment, image_size)),
        "val": CatDogDataset(splits["val"], build_transform(False, False, image_size)),
        "test": CatDogDataset(splits["test"], build_transform(False, False, image_size)),
    }
    return {
        name: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        for name, dataset in datasets.items()
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = total_correct = total = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


def train_experiment(name, loaders, args, device, output_dir, model_dir):
    model = CatDogCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_state, best_val, best_epoch = None, -1.0, 0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        print(
            f"[{name}] {epoch:02d}/{args.epochs} "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} "
            f"val loss={val_loss:.4f} acc={val_acc:.4f}"
        )

    model.load_state_dict(best_state)
    test_loss, test_acc = run_epoch(model, loaders["test"], criterion, device)
    checkpoint = {
        "model_state": {key: value.cpu() for key, value in best_state.items()},
        "image_size": args.image_size,
        "class_names": CLASS_NAMES,
        "experiment": name,
        "best_epoch": best_epoch,
        "val_accuracy": best_val,
        "test_accuracy": test_acc,
    }
    torch.save(checkpoint, model_dir / f"{name}_best.pt")
    plot_history(history, output_dir / f"curves_{name}.png", name)
    return model, {
        "experiment": name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "seconds": time.time() - start,
        "history": history,
        "checkpoint": checkpoint,
    }


def plot_history(history, path, name):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], label="train")
    axes[1].plot(epochs, history["val_acc"], label="validation")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(alpha=0.25)
    fig.suptitle(name)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def predict_all(model, loader, device):
    model.eval()
    records = []
    with torch.no_grad():
        for images, labels, paths in loader:
            probabilities = model(images.to(device)).softmax(1).cpu()
            predictions = probabilities.argmax(1)
            for path, true, pred, probs in zip(paths, labels, predictions, probabilities):
                records.append(
                    {
                        "path": path,
                        "true": int(true),
                        "pred": int(pred),
                        "confidence": float(probs[pred]),
                    }
                )
    return records


def save_errors(records, output_dir, limit=12):
    errors = [record for record in records if record["true"] != record["pred"]]
    errors.sort(key=lambda record: record["confidence"], reverse=True)
    shown = errors[:limit]
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for ax in axes.flat:
        ax.axis("off")
    for ax, record in zip(axes.flat, shown):
        with Image.open(record["path"]) as image:
            ax.imshow(image.convert("RGB"))
        ax.set_title(
            f"true={'cat' if record['true'] == 0 else 'dog'} "
            f"pred={'cat' if record['pred'] == 0 else 'dog'}\n"
            f"confidence={record['confidence']:.1%}"
        )
        ax.axis("off")
    fig.suptitle("Highest-confidence mistakes")
    fig.tight_layout()
    fig.savefig(output_dir / "errors.png", dpi=160)
    plt.close(fig)
    return errors


def save_feature_maps(model, dataset, output_dir, device):
    model.eval()
    selected = []
    for label in (0, 1, 0):
        for index, (_, item_label) in enumerate(dataset.items):
            if item_label == label and index not in selected:
                selected.append(index)
                break
    fig, axes = plt.subplots(len(selected), 9, figsize=(16, 3 * len(selected)))
    axes = np.atleast_2d(axes)
    for ax in axes.flat:
        ax.axis("off")
    for row, item_index in enumerate(selected):
        image, label, path = dataset[item_index]
        with torch.no_grad():
            _, features = model.forward_features(image.unsqueeze(0).to(device))
        first = features["block1"][0].cpu()
        last = features["block4"][0].cpu()
        axes[row, 0].imshow(tensor_image(image))
        axes[row, 0].set_title(f"{'cat' if label == 0 else 'dog'}: {Path(path).name}", fontsize=9)
        for channel in range(4):
            axes[row, channel + 1].imshow(first[channel], cmap="viridis")
            axes[row, channel + 1].set_title(f"first {channel}")
            axes[row, channel + 5].imshow(last[channel], cmap="viridis")
            axes[row, channel + 5].set_title(f"last {channel}")
    fig.suptitle("Test images and feature maps from different channels")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_maps.png", dpi=170)
    plt.close(fig)


def write_results_report(path, full_counts, invalid, splits, shape_rows, results, best, errors, args):
    rows = "\n".join(
        f"| {item['experiment']} | {item['best_epoch']} | {item['best_val_accuracy']:.2%} | {item['test_accuracy']:.2%} | {item['seconds']:.1f} |"
        for item in results
    )
    shape_text = "\n".join(f"- `{name}`：`{shape}`" for name, shape in shape_rows)
    error_examples = "\n".join(
        f"- `{Path(item['path']).name}`：真实为{CLASS_NAMES[item['true']]}，预测为{CLASS_NAMES[item['pred']]}，置信度 {item['confidence']:.1%}。"
        for item in errors[:5]
    ) or "- 测试集没有错误样本。"
    content = f"""# 实践 2 运行结果

## 数据

- 完整数据：猫 {full_counts[0]} 张，狗 {full_counts[1]} 张。
- 无法读取：{len(invalid)} 张。
- 本次训练划分：训练 {len(splits['train'])}，验证 {len(splits['val'])}，测试 {len(splits['test'])}。
- 随机种子：{args.seed}。

随机样本见 `outputs/practice2/data_overview.png`，预处理对比见 `preprocessing_check.png`，增强样本见 `augmentation_samples.png`。

## Batch 和预处理

每张图片先按比例缩放，再裁剪为 {args.image_size}×{args.image_size}，转成 CHW Tensor 并归一化。DataLoader 把 {args.batch_size} 张图片组成 `(N, 3, {args.image_size}, {args.image_size})` 的 batch。按比例缩放后再裁剪，避免直接改成长宽相等造成明显拉伸。

增强实验使用随机水平翻转、轻度随机裁剪和轻度颜色扰动。左右翻转不会改变猫狗标签；轻度裁剪和颜色变化也符合现实拍摄。垂直翻转、旋转过大、裁掉大部分动物或很强的颜色变化可能产生失真，因此没有使用。

## CNN 各层形状

{shape_text}

卷积把通道数从 3 增加到 32、64、128，让模型学习更多类型的特征；每次 2×2 池化让宽高减半，减少计算量并保留主要响应。

![模型结构](outputs/practice2/model_structure.png)

## 对照实验

| 实验 | 最佳 epoch | 验证准确率 | 测试准确率 | 用时/秒 |
| --- | ---: | ---: | ---: | ---: |
{rows}

最终选择 `{best['experiment']}`，依据是验证集准确率最高。最佳模型保存为 `models/best_cat_dog_cnn.pt`，没有用测试集挑选模型。

## 特征图

![特征图](outputs/practice2/feature_maps.png)

同一层的不同通道出现不同响应，是因为每个卷积核学到的权重不同。第一层通常更容易看到颜色、明暗和边缘；最后一层更抽象，常对毛发纹理、耳朵、脸部轮廓或背景组合产生响应。亮的位置表示该通道对此区域响应较强，但不能只凭一张特征图确定它唯一代表什么。

## 错误分析

测试集共有 {len(errors)} 个错误样本，置信度较高的例子如下：

{error_examples}

![错误样本](outputs/practice2/errors.png)

常见原因包括动物太小、遮挡、画面同时出现猫狗、光线差、姿态特殊，或背景比动物更明显。可以增加相似困难样本、使用更合适的数据增强、训练更多轮并加入学习率调整，也可以换更强的预训练网络。改进时仍应只根据验证集选择方案，测试集留到最后评估。
"""
    path.write_text(content, encoding="utf-8")


def main():
    args = parse_args()
    set_seed(args.seed)
    root = Path(__file__).resolve().parent
    data_dir = args.data_dir if args.data_dir.is_absolute() else root / args.data_dir
    output_dir = root / "outputs" / "practice2"
    model_dir = root / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    full_items, invalid = collect_images(data_dir)
    full_counts = count_labels(full_items)
    print(f"完整数据：猫 {full_counts[0]}，狗 {full_counts[1]}，损坏 {len(invalid)}")
    save_data_overview(full_items, output_dir, args.seed)
    save_preprocessing_check(full_items, output_dir, args.image_size, args.seed)

    selected = limit_per_class(full_items, args.max_per_class, args.seed)
    splits = stratified_split(selected, args.seed)
    print({name: count_labels(items) for name, items in splits.items()})

    structure_model = CatDogCNN().to(device)
    shape_rows = save_model_structure(structure_model, output_dir, args.image_size, device)
    results = []
    for experiment in args.experiments:
        loaders = make_loaders(
            splits,
            args.image_size,
            experiment == "augmented",
            args.batch_size,
            args.num_workers,
            device,
        )
        _, result = train_experiment(experiment, loaders, args, device, output_dir, model_dir)
        results.append(result)

    best = max(results, key=lambda item: item["best_val_accuracy"])
    torch.save(best["checkpoint"], model_dir / "best_cat_dog_cnn.pt")
    best_model = CatDogCNN().to(device)
    best_model.load_state_dict(best["checkpoint"]["model_state"])
    final_loaders = make_loaders(
        splits,
        args.image_size,
        best["experiment"] == "augmented",
        args.batch_size,
        args.num_workers,
        device,
    )
    records = predict_all(best_model, final_loaders["test"], device)
    errors = save_errors(records, output_dir)
    save_feature_maps(best_model, final_loaders["test"].dataset, output_dir, device)

    serializable = [{key: value for key, value in item.items() if key != "checkpoint"} for item in results]
    (output_dir / "metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_results_report(
        root / "实践2运行结果.md",
        full_counts,
        invalid,
        splits,
        shape_rows,
        results,
        best,
        errors,
        args,
    )
    print("最佳模型：", model_dir / "best_cat_dog_cnn.pt")
    print("结果报告：", root / "实践2运行结果.md")


if __name__ == "__main__":
    main()
