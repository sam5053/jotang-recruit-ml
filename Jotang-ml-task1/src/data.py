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


def prepare_data(seed=SEED):
    X, y = make_moons(
        n_samples=1000,
        noise=0.2,
        random_state=seed,
    )

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