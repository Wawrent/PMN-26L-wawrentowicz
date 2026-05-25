import torch
import numpy as np

from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

def get_predictions(model, loader, device):

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def generate_report(model, loader, classes, device):

    y_true, y_pred = get_predictions(
        model,
        loader,
        device
    )

    cm = confusion_matrix(y_true, y_pred)

    report = classification_report(
        y_true,
        y_pred,
        target_names=classes
    )

    return cm, report