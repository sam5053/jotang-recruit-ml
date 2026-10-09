import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import torch

from src.data import PROJECT_ROOT, SEED, prepare_data
from src.train import MODEL_PATH, load_model


FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


def predict(model, X, device):
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X_tensor)
        probabilities = torch.sigmoid(logits)

    probabilities = probabilities.cpu().numpy().ravel()
    predictions = (probabilities >= 0.5).astype(int)

    return probabilities, predictions


def plot_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Class 0", "Class 1"],
    )

    display.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    try:
        plt.savefig(
            FIGURE_DIR / "confusion_matrix.png",
            dpi=200,
        )
    except PermissionError:
        print(
            "confusion_matrix.png is open in another program; "
            "keeping the existing file."
        )
    plt.close()


def plot_decision_boundary(model, data, device):
    X_all = data["X_all"]
    y_all = data["y_all"]
    X_test = data["X_test"]
    y_test = data["y_test"]

    probabilities, predictions = predict(model, X_test, device)

    x_min = X_all[:, 0].min() - 0.5
    x_max = X_all[:, 0].max() + 0.5
    y_min = X_all[:, 1].min() - 0.5
    y_max = X_all[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    with torch.no_grad():
        grid_logits = model(grid_tensor)
        grid_probabilities = torch.sigmoid(grid_logits)

    grid_probabilities = (
        grid_probabilities.cpu().numpy().reshape(xx.shape)
    )

    plt.figure(figsize=(7, 6))

    filled = plt.contourf(
        xx,
        yy,
        grid_probabilities,
        levels=20,
        cmap="coolwarm",
        alpha=0.35,
    )
    plt.colorbar(
        filled,
        label="Probability of class 1",
    )
    plt.contour(
        xx,
        yy,
        grid_probabilities,
        levels=[0.5],
        colors="black",
        linewidths=2,
    )

    plt.scatter(
        X_all[:, 0],
        X_all[:, 1],
        c=y_all,
        cmap="coolwarm",
        s=15,
        alpha=0.65,
        edgecolors="white",
        linewidths=0.2,
    )

    wrong_indices = np.where(predictions != y_test)[0]

    plt.scatter(
        X_test[wrong_indices, 0],
        X_test[wrong_indices, 1],
        c="black",
        marker="x",
        s=90,
        linewidths=2,
        label="Wrong prediction",
    )

    plt.title("Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    try:
        plt.savefig(
            FIGURE_DIR / "decision_boundary.png",
            dpi=200,
        )
    except PermissionError:
        fallback_path = FIGURE_DIR / "decision_boundary_new.png"
        plt.savefig(fallback_path, dpi=200)
        print(
            "decision_boundary.png is open in another program; "
            f"saved to {fallback_path.name} instead."
        )
    plt.close()


def save_error_analysis(data, probabilities, predictions):
    y_test = data["y_test"]

    error_table = pd.DataFrame(
        {
            "sample_index": np.arange(len(y_test)),
            "feature_1": data["X_test"][:, 0],
            "feature_2": data["X_test"][:, 1],
            "true_label": y_test,
            "predicted_label": predictions,
            "probability_class_1": probabilities,
            "distance_to_boundary": np.abs(probabilities - 0.5),
        }
    )

    wrong_samples = error_table[
        error_table["true_label"] != error_table["predicted_label"]
    ].copy()

    wrong_samples = wrong_samples.sort_values(
        "distance_to_boundary"
    )

    wrong_samples.to_csv(
        TABLE_DIR / "misclassified_samples.csv",
        index=False,
    )

    selected_errors = wrong_samples.head(5)
    selected_errors.to_csv(
        TABLE_DIR / "selected_errors.csv",
        index=False,
    )

    return selected_errors


def save_metrics(y_test, predictions):
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    with open(
        TABLE_DIR / "test_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=2)

    return metrics


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    data = prepare_data(SEED)

    model = load_model(MODEL_PATH, device)

    probabilities, predictions = predict(
        model,
        data["X_test"],
        device,
    )

    plot_confusion_matrix(data["y_test"], predictions)
    plot_decision_boundary(model, data, device)

    save_error_analysis(
        data,
        probabilities,
        predictions,
    )

    metrics = save_metrics(
        data["y_test"],
        predictions,
    )

    print("Test metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
