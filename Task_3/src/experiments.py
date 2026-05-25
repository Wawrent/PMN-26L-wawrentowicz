import torch
import torch.nn as nn
import torch.optim as optim

from src.model import CNN
from src.train import train_model
from src.data import load_data


def run_experiment(
    kernel_size=3,
    pooling="max",
    lr=0.001,
    batch_size=64,
    epochs=10
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, _, _ = load_data(batch_size)

    model = CNN(kernel_size=kernel_size, pooling=pooling).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = train_model(
        model,
        train_loader,
        test_loader,
        criterion,
        optimizer,
        device,
        epochs
    )

    best_acc = max(history["test_acc"])

    return model, history, best_acc


def kernel_size_experiments(epochs=10):
    results = []

    for k in [3, 5, 7]:
        model, history, acc = run_experiment(
            kernel_size=k,
            epochs=epochs
        )

        results.append({
            "kernel_size": k,
            "accuracy": acc,
            "history": history
        })

    return results


def pooling_experiments(epochs=10):
    results = []

    for pooling in ["max", "avg", "none"]:
        model, history, acc = run_experiment(
            pooling=pooling,
            epochs=epochs
        )

        results.append({
            "pooling": pooling,
            "accuracy": acc,
            "history": history
        })

    return results