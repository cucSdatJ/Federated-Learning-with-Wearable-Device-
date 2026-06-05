from pathlib import Path
import json
import random

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data.dataset import WearableDataset, FEATURE_COLS, TARGET_COL
from src.data.scaler import fit_scaler, transform_df
from src.models.mlp import MLPClassifier
from src.fl.common import compute_local_class_weights


SEED = 42
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIENT_IDS = [1, 2, 3, 4, 5,6,7,8]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(df, batch_size=512, shuffle=True):
    ds = WearableDataset(df)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
    }


def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    return metrics, all_labels, all_preds


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for X, y in loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def train_single_client(client_id, client_df, model_dir, exp_dir):
    print("\n" + "=" * 80)
    print(f"[CLIENT {client_id}] Start training")
    print("=" * 80)

    print(f"[CLIENT {client_id}] Full client shape: {client_df.shape}")
    print(f"[CLIENT {client_id}] Label distribution:")
    print(client_df[TARGET_COL].value_counts(normalize=True).sort_index())

    # split nội bộ theo stratify nếu đủ lớp
    train_df, test_df = train_test_split(
        client_df,
        test_size=0.2,
        random_state=SEED,
        stratify=client_df[TARGET_COL],
    )

    print(f"[CLIENT {client_id}] Train split: {train_df.shape}")
    print(f"[CLIENT {client_id}] Val split  : {test_df.shape}")

    # fit scaler riêng cho client
    scaler = fit_scaler(train_df)
    train_df = transform_df(train_df, scaler)
    test_df = transform_df(test_df, scaler)

    scaler_path = model_dir / f"scaler_client_{client_id}.pkl"
    joblib.dump(scaler, scaler_path)

    train_loader = make_loader(train_df, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = make_loader(test_df, batch_size=BATCH_SIZE, shuffle=False)

    model = MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)

    class_weights = compute_local_class_weights(train_df)

    print(f"[CLIENT {client_id}] Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_test_f1 = -1.0
    history = []

    model_path = model_dir / f"local_client_{client_id}.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        test_metrics, _, _ = evaluate(model, test_loader, DEVICE)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_accuracy": test_metrics["accuracy"],
            "test_precision_macro": test_metrics["precision_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
        }
        history.append(row)

        print(
            f"[CLIENT {client_id}] [Epoch {epoch:02d}] "
            f"loss={train_loss:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f} | "
            f"test_f1={test_metrics['f1_macro']:.4f}"
        )

        if test_metrics["f1_macro"] > best_test_f1:
            best_test_f1 = test_metrics["f1_macro"]
            torch.save(model.state_dict(), model_path)

    # load best model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    test_metrics, test_true, test_pred = evaluate(model, test_loader, DEVICE)
    test_report = classification_report(test_true, test_pred, digits=4, zero_division=0)

    # save reports
    with open(exp_dir / f"client_{client_id}_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(exp_dir / f"client_{client_id}_test_report.txt", "w", encoding="utf-8") as f:
        f.write(test_report)

    pd.DataFrame(history).to_csv(exp_dir / f"client_{client_id}_history.csv", index=False)

    summary_row = {
        "client_id": client_id,
        "train_samples": len(train_df),
        "test_samples": len(test_df),

        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
    }

    print(f"[CLIENT {client_id}] Test F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"[CLIENT {client_id}] Saved model -> {model_path}")
    print(f"[CLIENT {client_id}] Saved scaler -> {scaler_path}")

    return summary_row


def main():
    set_seed(SEED)

    data_dir = Path("data/processed")
    model_dir = Path("models")
    exp_dir = Path("experiments")

    model_dir.mkdir(exist_ok=True)
    exp_dir.mkdir(exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")

    all_results = []

    for client_id in CLIENT_IDS:
        client_path = data_dir / f"client_{client_id}.csv"
        client_df = pd.read_csv(client_path)

        result = train_single_client(
            client_id=client_id,
            client_df=client_df,
            model_dir=model_dir,
            exp_dir=exp_dir,
        )
        all_results.append(result)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(exp_dir / "local_results.csv", index=False)

    print("\n" + "=" * 80)
    print("[DONE] Local training finished")
    print("=" * 80)
    print(results_df)

    print("\nSaved:")
    print(" - models/local_client_1.pt ... local_client_8.pt")
    print(" - models/scaler_client_1.pkl ... scaler_client_8.pkl")
    print(" - experiments/client_*_test_metrics.json")
    print(" - experiments/client_*_test_report.txt")
    print(" - experiments/client_*_history.csv")
    print(" - experiments/local_results.csv")


if __name__ == "__main__":
    main()