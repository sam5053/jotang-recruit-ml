import json
import time
from copy import deepcopy
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import SEED, prepare_data, set_seed
from src.model import MLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
LOG_DIR = PROJECT_ROOT / "outputs" / "logs"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pt"

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01


def set_all_seeds(seed):
    set_seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(X, y, batch_size, shuffle):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    dataset = TensorDataset(X_tensor, y_tensor)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = loss_fn(logits, y_batch)

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (predictions == y_batch).sum().item()
        total_samples += X_batch.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)

        predictions = (torch.sigmoid(logits) >= 0.5).float()
        total_correct += (predictions == y_batch).sum().item()
        total_samples += X_batch.size(0)

    average_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    return average_loss, accuracy


def save_history(history):
    history_path = LOG_DIR / "training_history.json"

    with open(history_path, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)


def plot_loss_curve(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "loss_curve.png", dpi=200)
    plt.close()


def plot_accuracy_curve(history):
    epochs = range(1, len(history["train_accuracy"]) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(
        epochs,
        history["train_accuracy"],
        label="Train Accuracy",
    )
    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation Accuracy",
    )
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "accuracy_curve.png", dpi=200)
    plt.close()


def save_model(model, hidden_dim, input_dim):
    cpu_state = {
        key: value.cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": cpu_state,
        "hidden_dim": hidden_dim,
        "input_dim": input_dim,
    }

    torch.save(checkpoint, MODEL_PATH)


def load_model(model_path, device):
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=False,
    )

    model = MLP(
        hidden_dim=checkpoint["hidden_dim"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def train_model():
    set_all_seeds(SEED)

    data = prepare_data(SEED)

    train_loader = make_loader(
        data["X_train"],
        data["y_train"],
        BATCH_SIZE,
        shuffle=True,
    )
    val_loader = make_loader(
        data["X_val"],
        data["y_val"],
        BATCH_SIZE,
        shuffle=False,
    )
    test_loader = make_loader(
        data["X_test"],
        data["y_test"],
        BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = MLP(hidden_dim=32).to(device)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    best_model_state = None

    start_time = time.perf_counter()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            loss_fn,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = deepcopy(model.state_dict())

        if epoch == 1 or epoch % 20 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"train loss {train_loss:.4f} | "
                f"val loss {val_loss:.4f} | "
                f"val accuracy {val_accuracy:.4f}"
            )

    training_time = time.perf_counter() - start_time

    model.load_state_dict(best_model_state)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_fn,
        device,
    )

    history["best_val_loss"] = best_val_loss
    history["test_loss"] = test_loss
    history["test_accuracy"] = test_accuracy
    history["training_time_seconds"] = training_time
    history["parameter_count"] = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    save_model(
        model,
        hidden_dim=32,
        input_dim=2,
    )
    save_history(history)
    plot_loss_curve(history)
    plot_accuracy_curve(history)

    print()
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Training time: {training_time:.2f} seconds")

    return history, data, device


def main():
    history, data, device = train_model()

    loss_fn = nn.BCEWithLogitsLoss()

    test_loader = make_loader(
        data["X_test"],
        data["y_test"],
        BATCH_SIZE,
        shuffle=False,
    )

    loaded_model = load_model(MODEL_PATH, device)

    loaded_loss, loaded_accuracy = evaluate(
        loaded_model,
        test_loader,
        loss_fn,
        device,
    )

    print()
    print(f"Loaded model test loss: {loaded_loss:.4f}")
    print(f"Loaded model test accuracy: {loaded_accuracy:.4f}")


if __name__ == "__main__":
    main()