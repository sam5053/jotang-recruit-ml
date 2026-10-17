from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

try:
    import torch
except ModuleNotFoundError:
    torch = None

ROOT = Path(__file__).resolve().parent
IMAGE_PATH = ROOT / "assets" / "road_scene.png"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

pil_image = Image.open(IMAGE_PATH).convert("RGB")
image_hwc = np.asarray(pil_image)
print("Pillow mode:", pil_image.mode)
print("NumPy:", image_hwc.shape, image_hwc.dtype, image_hwc.min(), image_hwc.max())
row, col = 270, 480
print(f"RGB [{row}, {col}]:", image_hwc[row, col].tolist())

if torch is None:
    print("PyTorch 未安装，请先安装 torch。")
else:
    image_chw = torch.from_numpy(image_hwc.copy()).permute(2, 0, 1)
    print("Tensor:", tuple(image_chw.shape), image_chw.dtype)
    print("Tensor min/max:", image_chw.min().item(), image_chw.max().item())


def conv2d_numpy(image, kernel, padding="valid", stride=1):
    array = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    if array.ndim not in (2, 3) or kernel.ndim != 2 or stride < 1:
        raise ValueError("输入图片、卷积核或步长不合法")
    is_gray = array.ndim == 2
    if is_gray:
        array = array[..., None]
    height, width, channels = array.shape
    kh, kw = kernel.shape
    if padding == "valid":
        pad_top = pad_bottom = pad_left = pad_right = 0
        out_h, out_w = (height - kh) // stride + 1, (width - kw) // stride + 1
    elif padding == "same":
        out_h, out_w = int(np.ceil(height / stride)), int(np.ceil(width / stride))
        pad_h = max((out_h - 1) * stride + kh - height, 0)
        pad_w = max((out_w - 1) * stride + kw - width, 0)
        pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
        pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    else:
        raise ValueError("padding 只能是 'valid' 或 'same'")
    if out_h <= 0 or out_w <= 0:
        raise ValueError("卷积核大于输入图片")
    padded = np.pad(
        array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="constant"
    )
    output = np.empty((out_h, out_w, channels), dtype=np.float32)
    for out_y in range(out_h):
        for out_x in range(out_w):
            y, x = out_y * stride, out_x * stride
            patch = padded[y : y + kh, x : x + kw, :]
            output[out_y, out_x, :] = np.sum(patch * kernel[..., None], axis=(0, 1))
    return output[..., 0] if is_gray else output


def rgb_to_gray(rgb):
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def clip_uint8(array):
    return np.clip(array, 0, 255).astype(np.uint8)


def normalize_uint8(array):
    low, high = float(array.min()), float(array.max())
    if np.isclose(low, high):
        return np.zeros_like(array, dtype=np.uint8)
    return ((array - low) / (high - low) * 255).astype(np.uint8)


mean_3x3 = np.ones((3, 3), dtype=np.float32) / 9
coords = np.arange(5, dtype=np.float32) - 2
xx, yy = np.meshgrid(coords, coords)
gaussian_5x5 = np.exp(-(xx**2 + yy**2) / 2)
gaussian_5x5 /= gaussian_5x5.sum()
sharpen_3x3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

mean_blur = conv2d_numpy(image_hwc, mean_3x3, "same")
gaussian_blur = conv2d_numpy(image_hwc, gaussian_5x5, "same")
sharpened = conv2d_numpy(image_hwc, sharpen_3x3, "same")
gray = rgb_to_gray(image_hwc)
sobel_x_response = conv2d_numpy(gray, sobel_x, "same")
sobel_y_response = conv2d_numpy(gray, sobel_y, "same")
edge_magnitude = np.hypot(sobel_x_response, sobel_y_response)
valid_mean = conv2d_numpy(image_hwc, mean_3x3, "valid")
same_mean = conv2d_numpy(image_hwc, mean_3x3, "same")
print("valid:", valid_mean.shape, "same:", same_mean.shape)

panels = [
    (image_hwc, "Original RGB", None),
    (clip_uint8(mean_blur), "Mean blur (3×3)", None),
    (clip_uint8(gaussian_blur), "Gaussian blur (5×5)", None),
    (clip_uint8(sharpened), "Sharpen (3×3)", None),
    (normalize_uint8(sobel_x_response), "Sobel-X", "gray"),
    (normalize_uint8(sobel_y_response), "Sobel-Y", "gray"),
    (normalize_uint8(edge_magnitude), "Sobel magnitude", "gray"),
    (clip_uint8(valid_mean), "Mean blur: valid", None),
]
fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
for ax, (array, title, cmap) in zip(axes.flat, panels):
    ax.imshow(array, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
fig.suptitle("Handwritten NumPy 2D convolution", fontsize=16)
fig.savefig(OUTPUT_DIR / "convolution_comparison.png", dpi=180, bbox_inches="tight")
plt.close(fig)
