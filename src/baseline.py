"""Standard cross-entropy baseline used for comparison in the report."""

import copy

import torch
from torch import nn

from .cost_mat import COST_MATRIX
from .train_coral import (
    DROPOUT,
    GRADE_ORDER,
    HIDDEN,
    LR,
    MAX_EPOCHS,
    MIN_DELTA,
    PATIENCE,
    WEIGHT_DECAY,
    macro_f1,
)


class BaselineMLP(nn.Module):
    def __init__(self, n_features, n_classes=len(GRADE_ORDER)):
        super().__init__()
        self.pipeline = nn.Sequential(
            nn.Linear(n_features, HIDDEN),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, n_classes),
        )

    def forward(self, x):
        return self.pipeline(x)


def train_baseline(model, x_train, y_train, x_val, y_val):
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_checks = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        loss_train = loss_fn(model(x_train), y_train)
        loss_train.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x_val), y_val)

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_checks = 0
        else:
            bad_checks += 1

        if bad_checks >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return best_epoch, epoch, best_val_loss


def predict(model, x):
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        preds = probs.argmax(dim=1)
    return preds, probs


def evaluate_baseline(model, x, y):
    model.eval()
    with torch.no_grad():
        preds, _ = predict(model, x)
        cost_matrix = COST_MATRIX.to(y.device)
    return {
        "accuracy": (preds == y).float().mean().item(),
        "within_one": ((preds - y).abs() <= 1).float().mean().item(),
        "macro_f1": macro_f1(preds, y),
        "avg_cost": cost_matrix[y, preds].mean().item(),
        "preds": preds,
    }
