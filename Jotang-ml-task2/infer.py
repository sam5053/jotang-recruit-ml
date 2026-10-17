import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from catdog_common import CLASS_NAMES, CatDogCNN, build_transform, denormalize


def parse_args():
    parser = argparse.ArgumentParser(description="用训练好的模型判断一张图片是猫还是狗")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=Path("models/best_cat_dog_cnn.pt"))
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/practice2/inference_preprocessed.png")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parent
    image_path = args.image if args.image.is_absolute() else root / args.image
    model_path = args.model if args.model.is_absolute() else root / args.model
    output_path = args.output if args.output.is_absolute() else root / args.output
    if not image_path.exists():
        raise FileNotFoundError(f"找不到图片：{image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"找不到模型：{model_path}，请先运行训练程序。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = CatDogCNN().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image_size = checkpoint.get("image_size", 160)
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = build_transform(False, False, image_size)(image)

    with torch.no_grad():
        probabilities = model(tensor.unsqueeze(0).to(device)).softmax(1)[0].cpu()
    predicted = int(probabilities.argmax())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    to_pil_image(denormalize(tensor)).save(output_path)

    print(f"预测：{class_names[predicted]}")
    print(f"概率：{probabilities[predicted].item():.2%}")
    print(f"猫概率：{probabilities[0].item():.2%}")
    print(f"狗概率：{probabilities[1].item():.2%}")
    print(f"处理后的图片：{output_path}")


if __name__ == "__main__":
    main()
