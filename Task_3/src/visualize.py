import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(history, title="Training"):

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)

    plt.plot(epochs, history["train_loss"], label="Train")
    plt.plot(epochs, history["test_loss"], label="Test")

    plt.title(f"{title} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.subplot(1, 2, 2)

    plt.plot(epochs, history["train_acc"], label="Train")
    plt.plot(epochs, history["test_acc"], label="Test")

    plt.title(f"{title} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.show()


def plot_confusion_matrix(cm, classes):

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )

    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.title("Confusion Matrix")

    plt.show()