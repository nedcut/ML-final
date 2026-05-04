from pathlib import Path
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F


DATA_PATH = Path("data/processed/vt_condition_weather_aligned.csv")
CHECKPOINT_PATH = Path("data/processed/best_ski_cnn.pt")
SEED = 0

TEST_SEASONS = 5
VAL_SEASONS = 3

CHANNELS = 64
LR = 0.01
MAX_EPOCHS = 300
PATIENCE = 20
MIN_DELTA = 1e-5

GRADE_ORDER = ["D", "C", "B", "A"]
GRADE_TO_Y = {grade: i for i, grade in enumerate(GRADE_ORDER)} # D=0, C=1, B=2, A=3
Y_TO_GRADE = {i: grade for grade, i in GRADE_TO_Y.items()}     # 0=D, 1=C, 2=B, 3=A


class SkiCNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(CHANNELS, CHANNELS, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(CHANNELS, 4),
        )

    def forward(self, x):
        return self.net(x)


def make_split(df):
    """split data into train, val, test based on season, using the last seasons for test and val"""
    seasons = sorted(df["season"].unique(), key=lambda s: int(str(s).split("-")[0]))
    test_seasons = seasons[-TEST_SEASONS:]
    val_start = -(TEST_SEASONS + VAL_SEASONS)
    val_seasons = seasons[val_start:-TEST_SEASONS]
    train_seasons = seasons[:val_start]

    train_df = df[df["season"].isin(train_seasons)].reset_index(drop=True)
    val_df = df[df["season"].isin(val_seasons)].reset_index(drop=True)
    test_df = df[df["season"].isin(test_seasons)].reset_index(drop=True)

    print(f"train seasons:  {train_seasons[0]} through {train_seasons[-1]} ({len(train_df)} rows)")
    print(f"val seasons:  {val_seasons[0]} through {val_seasons[-1]} ({len(val_df)} rows)")
    print(f"test seasons: {test_seasons[0]} through {test_seasons[-1]} ({len(test_df)} rows)")
    return train_df, val_df, test_df


def weekly_features(df):
    """select numeric columns that start with 'avg_' or 'best_' and end with '_7d' """
    numeric_cols = [
        c for c in df.columns
        if (c.startswith("avg_") or c.startswith("best_"))
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return [c for c in numeric_cols if c.endswith("_7d")]


def make_tensors(train_df, val_df, test_df, feature_cols):
    """convert dataframes to tensors, normalizing features based on train__df statistics"""
    train_x = train_df[feature_cols].to_numpy(dtype="float32")
    val_x = val_df[feature_cols].to_numpy(dtype="float32")
    test_x = test_df[feature_cols].to_numpy(dtype="float32")

    mu = train_x.mean(axis=0, keepdims=True)
    sigma = train_x.std(axis=0, keepdims=True)
    sigma[sigma == 0] = 1

    train_x = (train_x - mu) / sigma
    val_x = (val_x - mu) / sigma
    test_x = (test_x - mu) / sigma

    # conv1d expects rows shaped like channels x feature sequence
    train_x = train_x[:, None, :]
    val_x = val_x[:, None, :]
    test_x = test_x[:, None, :]

    return (
        torch.tensor(train_x, dtype=torch.float32),
        torch.tensor(train_df["y"].to_numpy(), dtype=torch.long),
        torch.tensor(val_x, dtype=torch.float32),
        torch.tensor(val_df["y"].to_numpy(), dtype=torch.long),
        torch.tensor(test_x, dtype=torch.float32),
        torch.tensor(test_df["y"].to_numpy(), dtype=torch.long),
    )


def macro_f1(preds, y):
    scores = []
    for cls in range(4):
        tp = ((preds == cls) & (y == cls)).sum().item()
        fp = ((preds == cls) & (y != cls)).sum().item()
        fn = ((preds != cls) & (y == cls)).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))


def evaluate(model, x, y):
    model.eval()
    with torch.no_grad():
        preds = model(x).argmax(dim=1)
    return {
        "accuracy": (preds == y).float().mean().item(),
        # if the target is only 1 grade away from prediction (eg predicted B but it's actually A)
        "within_one": ((preds - y).abs() <= 1).float().mean().item(),
        "macro_f1": macro_f1(preds, y),
        "preds": preds,
    }


def train(model, x_train, y_train, x_val, y_val):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    bad_checks = 0 # number of consecutive epochs without improvement

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train() # need to set train mode for dropout to work
        opt.zero_grad()
        loss = F.cross_entropy(model(x_train), y_train)
        loss.backward()
        opt.step()

        model.eval() # back to eval mode for validation (no dropout, no grad)
        with torch.no_grad():
            val_loss = float(F.cross_entropy(model(x_val), y_val))

        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_checks = 0
        else:
            bad_checks += 1

        if bad_checks >= PATIENCE:
            break # stop early

    model.load_state_dict(best_state)
    return best_epoch, epoch, best_val_loss


def print_metrics(name, metrics):
    print(
        f"{name:>4}: "
        f"accuracy={metrics['accuracy']:.3f}, "
        f"macro_f1={metrics['macro_f1']:.3f}, "
        f"within_one={metrics['within_one']:.3f}"
    )


def save_checkpoint(model, feature_cols, best_epoch, stopped_epoch, best_val_loss):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "feature_cols": feature_cols,
        "grade_order": GRADE_ORDER,
        "config": {
            "channels": CHANNELS,
            "lr": LR,
            "seed": SEED,
            "test_seasons": TEST_SEASONS,
            "val_seasons": VAL_SEASONS,
            "best_epoch": best_epoch,
            "stopped_epoch": stopped_epoch,
            "best_val_loss": best_val_loss,
        },
    }
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, CHECKPOINT_PATH)
    print(f"saved checkpoint: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    df = pd.read_csv(DATA_PATH)
    df["y"] = df["grade"].map(GRADE_TO_Y)

    train_df, val_df, test_df = make_split(df)
    feature_cols = weekly_features(df)
    print(f"features: {len(feature_cols)}")

    x_train, y_train, x_val, y_val, x_test, y_test = make_tensors(
        train_df,
        val_df,
        test_df,
        feature_cols,
    )

    model = SkiCNN(x_train.shape[-1])

    from torchinfo import summary
    print(summary(model, input_size=(1, 1, x_train.shape[-1])))


    best_epoch, stopped_epoch, best_val_loss = train(model, x_train, y_train, x_val, y_val)

    print(f"best epoch: {best_epoch}")
    print(f"stopped epoch: {stopped_epoch}")
    print(f"best val loss: {best_val_loss:.4f}")
    save_checkpoint(model, feature_cols, best_epoch, stopped_epoch, best_val_loss)

    train_metrics = evaluate(model, x_train, y_train)
    val_metrics = evaluate(model, x_val, y_val)
    test_metrics = evaluate(model, x_test, y_test)

    print_metrics("train", train_metrics)
    print_metrics("val", val_metrics)
    print_metrics("test", test_metrics)

    confusion = pd.crosstab(
        pd.Series(test_df["y"].map(Y_TO_GRADE), name="actual"),
        pd.Series(pd.Series(test_metrics["preds"].numpy()).map(Y_TO_GRADE), name="predicted"),
    ).reindex(index=["A", "B", "C", "D"], columns=["A", "B", "C", "D"], fill_value=0)
    print("\ntest confusion matrix")
    print(confusion)
