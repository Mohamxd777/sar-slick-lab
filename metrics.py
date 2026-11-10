"""Pixel-level loss and metrics for mutually exclusive segmentation classes."""
import torch
from torch.nn import functional as F


def objective(logits, target, weights=None):
    cross_entropy = F.cross_entropy(logits, target, weight=weights)
    if weights is None:
        return cross_entropy
    probabilities = logits.float().softmax(dim=1)
    truth = F.one_hot(target, 5).movedim(-1, 1).float()
    dimensions = (0, 2, 3)
    intersection = (probabilities * truth).sum(dimensions)
    support = truth.sum(dimensions)
    overlap = (2 * intersection + 1e-6) / (probabilities.sum(dimensions) + support + 1e-6)
    return cross_entropy + .5 * (1 - overlap[support > 0].mean())


def confusion(predicted, target):
    return torch.bincount((5 * target + predicted).reshape(-1), minlength=25).reshape(5, 5)


def scores(matrix):
    matrix = matrix.double()
    correct = matrix.diag()
    actual = matrix.sum(1)
    predicted = matrix.sum(0)
    union = actual + predicted - correct
    iou = [float(a / b) if b else None for a, b in zip(correct, union)]
    dice = [float(2 * a / b) if b else None for a, b in zip(correct, actual + predicted)]
    return {
        "miou": sum(value for value in iou if value is not None) / int((union > 0).sum()),
        "iou": iou, "dice": dice, "accuracy": float(correct.sum() / matrix.sum()),
        "confusion": matrix.long().tolist(),
    }
