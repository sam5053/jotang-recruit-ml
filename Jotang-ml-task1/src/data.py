from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


SEED = 4
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def split_data(X, y, seed=SEED):

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=seed,
        stratify=y,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    return {
        "X_all": X,
        "y_all": y,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def prepare_data(seed=SEED, n_samples=1000, noise=0.2):
    X, y = make_moons(
        n_samples=n_samples,
        noise=noise,
        random_state=seed,
    )

    return split_data(X, y, seed)


def prepare_imbalanced_data(
    seed=SEED,
    majority_samples=950,
    minority_samples=50,
    noise=0.3,
):
    X, y = make_moons(
        n_samples=(majority_samples, minority_samples),
        noise=noise,
        random_state=seed,
    )

    return split_data(X, y, seed)


def prepare_overfit_data(
    seed=SEED,
    train_size=80,
    noise=0.3,
    label_flip_ratio=0.3,
):
    data = prepare_data(seed=seed, noise=noise)
    X_small, _, y_small, _ = train_test_split(
        data["X_train"],
        data["y_train"],
        train_size=train_size,
        random_state=seed,
        stratify=data["y_train"],
    )

    corrupted_y = y_small.copy()
    rng = np.random.default_rng(seed)
    flipped_indices = rng.choice(
        len(corrupted_y),
        size=round(train_size * label_flip_ratio),
        replace=False,
    )
    corrupted_y[flipped_indices] = 1 - corrupted_y[flipped_indices]

    data["X_train"] = X_small
    data["y_train"] = corrupted_y
    data["y_train_clean"] = y_small
    data["flipped_train_indices"] = flipped_indices
    return data


def plot_raw_data(X, y):
    plt.figure(figsize=(6, 5))
    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        cmap="coolwarm",
        s=25,
        edgecolors="black",
        linewidths=0.2,
    )
    plt.title("Original make_moons Dataset")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "raw_data.png", dpi=200)
    plt.close()


def plot_data_splits(X_train, y_train, X_val, y_val, X_test, y_test):
    datasets = [
        ("Train", X_train, y_train),
        ("Validation", X_val, y_val),
        ("Test", X_test, y_test),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, (title, X_part, y_part) in zip(axes, datasets):
        axis.scatter(
            X_part[:, 0],
            X_part[:, 1],
            c=y_part,
            cmap="coolwarm",
            s=20,
            edgecolors="black",
            linewidths=0.2,
        )
        axis.set_title(title)
        axis.set_xlabel("Feature 1")
        axis.set_ylabel("Feature 2")

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "data_split.png", dpi=200)
    plt.close(fig)


def main():
    set_seed(SEED)
    data = prepare_data(SEED)

    print("X shape:", data["X_all"].shape)
    print("y shape:", data["y_all"].shape)
    print("train:", data["X_train"].shape)
    print("validation:", data["X_val"].shape)
    print("test:", data["X_test"].shape)

    plot_raw_data(data["X_all"], data["y_all"])
    plot_data_splits(
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
    )


if __name__ == "__main__":
    main()
