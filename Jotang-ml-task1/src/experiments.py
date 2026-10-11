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
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data import (
    PROJECT_ROOT,
    SEED,
    prepare_data,
    prepare_imbalanced_data,
    prepare_overfit_data,
)
from src.evaluate import predict
from src.train import plot_accuracy_curve, plot_loss_curve, run_training, save_history


FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLE_DIR = PROJECT_ROOT / "outputs" / "tables"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

for directory in (FIGURE_DIR, TABLE_DIR, LOG_DIR):
    directory.mkdir(parents=True, exist_ok=True)

BASELINE = {
    "hidden_dims": (32,),
    "activation": "relu",
    "optimizer_name": "adam",
    "learning_rate": 0.01,
    "batch_size": 64,
    "epochs": 100,
    "seed": SEED,
}


def result_row(factor, setting, result):
    history = result["history"]
    return {
        "factor": factor,
        "setting": str(setting),
        "hidden_layers": "-".join(map(str, result["model"].hidden_dims)),
        "activation": result["model"].activation,
        "optimizer": "adam",
        "best_epoch": history["best_epoch"],
        "best_validation_loss": history["best_val_loss"],
        "test_loss": history["test_loss"],
        "test_accuracy": history["test_accuracy"],
        "training_time_seconds": history["training_time_seconds"],
        "parameter_count": history["parameter_count"],
        "peak_gpu_memory_mb": history["peak_gpu_memory_mb"],
        "device": history["device"],
    }


def run_controlled_experiments():
    rows = []
    for width in (8, 32, 128):
        config = {**BASELINE, "hidden_dims": (width,)}
        result = run_training(prepare_data(SEED), **config)
        rows.append(result_row("hidden_layer_width", width, result))

    for learning_rate in (0.001, 0.01, 0.1):
        config = {**BASELINE, "learning_rate": learning_rate}
        result = run_training(prepare_data(SEED), **config)
        rows.append(result_row("learning_rate", learning_rate, result))

    table = pd.DataFrame(rows)
    table.to_csv(TABLE_DIR / "controlled_experiments.csv", index=False)
    plot_controlled_experiments(table)
    return table


def plot_controlled_experiments(table):
    metrics = [
        ("test_accuracy", "Test accuracy"),
        ("best_validation_loss", "Best validation loss"),
        ("training_time_seconds", "Training time (s)"),
        ("parameter_count", "Parameter count"),
    ]
    factors = ["hidden_layer_width", "learning_rate"]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))

    for row_index, factor in enumerate(factors):
        subset = table[table["factor"] == factor]
        for column_index, (metric, title) in enumerate(metrics):
            axis = axes[row_index, column_index]
            axis.plot(subset["setting"], subset[metric], marker="o")
            axis.set_title(f"{factor}: {title}")
            axis.set_xlabel("Setting")
            axis.set_ylabel(title)
            axis.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "controlled_experiments.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(y_true, predictions, path, title):
    display = ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        display_labels=["Class 0", "Class 1"],
        cmap="Blues",
        values_format="d",
    )
    display.ax_.set_title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_imbalanced_data(data):
    plt.figure(figsize=(6, 5))
    plt.scatter(
        data["X_all"][:, 0],
        data["X_all"][:, 1],
        c=data["y_all"],
        cmap="coolwarm",
        s=18,
        alpha=0.7,
        edgecolors="white",
        linewidths=0.2,
    )
    plt.title("Imbalanced data: 950 class-0 vs 50 class-1 samples")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "imbalanced_data.png", dpi=200)
    plt.close()


def run_imbalance_experiment():
    data = prepare_imbalanced_data()
    result = run_training(data, **BASELINE)
    probabilities, predictions = predict(
        result["model"], data["X_test"], result["device"]
    )

    metrics = {
        "class_counts_all": np.bincount(data["y_all"]).tolist(),
        "class_counts_test": np.bincount(data["y_test"]).tolist(),
        "accuracy": float(accuracy_score(data["y_test"], predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(data["y_test"], predictions)
        ),
        "minority_precision": float(
            precision_score(data["y_test"], predictions, zero_division=0)
        ),
        "minority_recall": float(
            recall_score(data["y_test"], predictions, zero_division=0)
        ),
        "minority_f1": float(
            f1_score(data["y_test"], predictions, zero_division=0)
        ),
        "mean_probability_class_1": float(probabilities.mean()),
    }
    with open(TABLE_DIR / "imbalance_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    plot_imbalanced_data(data)
    save_confusion_matrix(
        data["y_test"],
        predictions,
        FIGURE_DIR / "imbalance_confusion_matrix.png",
        "Imbalanced-data confusion matrix",
    )
    return metrics


def plot_overfit_data(data):
    flipped = data["flipped_train_indices"]
    plt.figure(figsize=(6, 5))
    plt.scatter(
        data["X_all"][:, 0],
        data["X_all"][:, 1],
        c=data["y_all"],
        cmap="coolwarm",
        s=10,
        alpha=0.2,
        label="Full clean dataset",
    )
    plt.scatter(
        data["X_train"][:, 0],
        data["X_train"][:, 1],
        c=data["y_train"],
        cmap="coolwarm",
        edgecolors="black",
        s=45,
        label="Small noisy training set",
    )
    plt.scatter(
        data["X_train"][flipped, 0],
        data["X_train"][flipped, 1],
        facecolors="none",
        edgecolors="gold",
        linewidths=1.5,
        s=95,
        label="Flipped labels",
    )
    plt.title("Deliberately overfit training data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "overfit_data.png", dpi=200)
    plt.close()


def run_overfit_experiment():
    data = prepare_overfit_data()
    config = {
        **BASELINE,
        "hidden_dims": (128, 128, 128, 128),
        "learning_rate": 0.003,
        "epochs": 1600,
        "restore_best": False,
    }
    result = run_training(data, **config)
    history = result["history"]
    save_history(history, LOG_DIR / "overfit_history.json")
    plot_loss_curve(history, FIGURE_DIR / "overfit_loss_curve.png")
    plot_accuracy_curve(history, FIGURE_DIR / "overfit_accuracy_curve.png")
    plot_overfit_data(data)

    final_train_accuracy = history["train_accuracy"][-1]
    final_val_accuracy = history["val_accuracy"][-1]
    metrics = {
        "train_samples": len(data["y_train"]),
        "label_flips": len(data["flipped_train_indices"]),
        "final_train_accuracy": final_train_accuracy,
        "final_validation_accuracy": final_val_accuracy,
        "final_test_accuracy": history["test_accuracy"],
        "final_train_loss": history["train_loss"][-1],
        "final_validation_loss": history["val_loss"][-1],
        "accuracy_gap_train_minus_validation": (
            final_train_accuracy - final_val_accuracy
        ),
        "best_validation_epoch": history["best_epoch"],
        "parameter_count": history["parameter_count"],
    }
    with open(TABLE_DIR / "overfit_metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    return metrics


def main():
    controlled = run_controlled_experiments()
    imbalance = run_imbalance_experiment()
    overfit = run_overfit_experiment()

    print("Controlled experiments:")
    print(controlled.to_string(index=False))
    print("\nImbalance metrics:")
    print(json.dumps(imbalance, indent=2))
    print("\nOverfit metrics:")
    print(json.dumps(overfit, indent=2))


if __name__ == "__main__":
    main()
