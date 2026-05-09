from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.data.dataset import WearableDataset, FEATURE_COLS, TARGET_COL
from src.data.scaler import fit_scaler, transform_df
from src.models.mlp import MLPClassifier


SEED = 42
BATCH_SIZE = 512
EPOCHS = 10
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_train_test():
    data_dir = Path("data/processed")

    train_parts = []
    for i in range(1, 6):
        csv_path = data_dir / f"client_{i}.csv"
        train_parts.append(pd.read_csv(csv_path))

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.read_csv(data_dir / "test_set.csv")
    return train_df, test_df


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool):
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

    return compute_metrics(all_labels, all_preds), all_labels, all_preds


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


def main():
    set_seed(SEED)

    out_model_dir = Path("models")
    out_exp_dir = Path("experiments")
    out_model_dir.mkdir(exist_ok=True)
    out_exp_dir.mkdir(exist_ok=True)

    print(f"[INFO] Device: {DEVICE}")

    # 1. Load data
    train_df, test_df = load_train_test()

    print("[INFO] Full train shape:", train_df.shape)
    print("[INFO] Test shape:", test_df.shape)

    # 2. Train / validation split
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.2,
        random_state=SEED,
        stratify=train_df[TARGET_COL],
    )

    print("[INFO] Train split:", train_df.shape)
    print("[INFO] Val split:", val_df.shape)

    print("[INFO] Train label distribution:")
    print(train_df[TARGET_COL].value_counts(normalize=True).sort_index())

    print("[INFO] Val label distribution:")
    print(val_df[TARGET_COL].value_counts(normalize=True).sort_index())

    print("[INFO] Test label distribution:")
    print(test_df[TARGET_COL].value_counts(normalize=True).sort_index())

    # 3. Fit scaler on train only
    scaler = fit_scaler(train_df)
    train_df = transform_df(train_df, scaler)
    val_df = transform_df(val_df, scaler)
    test_df = transform_df(test_df, scaler)

    joblib.dump(scaler, out_model_dir / "scaler.pkl")
    print("[INFO] Saved scaler -> models/scaler.pkl")

    # 4. DataLoaders
    train_loader = make_loader(train_df, BATCH_SIZE, shuffle=True)
    val_loader = make_loader(val_df, BATCH_SIZE, shuffle=False)
    test_loader = make_loader(test_df, BATCH_SIZE, shuffle=False)

    # 5. Model
    model = MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)

    # 6. Class weights
    classes = np.array(sorted(train_df[TARGET_COL].unique()))
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df[TARGET_COL].values
    )
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    print("[INFO] Class weights:", class_weights.cpu().numpy())

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_f1 = -1.0
    history = []

    # 7. Training loop
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

        val_metrics, _, _ = evaluate(model, val_loader, DEVICE)

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision_macro": val_metrics["precision_macro"],
            "val_recall_macro": val_metrics["recall_macro"],
            "val_f1_macro": val_metrics["f1_macro"],
        })

        print(
            f"[Epoch {epoch:02d}] "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            torch.save(model.state_dict(), out_model_dir / "centralized_mlp.pt")
            print("[INFO] Best model updated -> models/centralized_mlp.pt")

    # 8. Load best model and evaluate on test
    model.load_state_dict(torch.load(out_model_dir / "centralized_mlp.pt", map_location=DEVICE))

    test_metrics, y_true, y_pred = evaluate(model, test_loader, DEVICE)

    print("\n[TEST RESULTS]")
    print("Accuracy      :", round(test_metrics["accuracy"], 4))
    print("PrecisionMacro:", round(test_metrics["precision_macro"], 4))
    print("RecallMacro   :", round(test_metrics["recall_macro"], 4))
    print("F1 Macro      :", round(test_metrics["f1_macro"], 4))
    print("ConfusionMatrix:")
    print(np.array(test_metrics["confusion_matrix"]))

    report = classification_report(y_true, y_pred, digits=4, zero_division=0)
    print("\nClassification Report:\n")
    print(report)

    # 9. Save outputs
    with open(out_exp_dir / "centralized_metrics.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    with open(out_exp_dir / "centralized_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    pd.DataFrame(history).to_csv(out_exp_dir / "centralized_history.csv", index=False)

    print("\n[DONE] Saved:")
    print(" - models/centralized_mlp.pt")
    print(" - models/scaler.pkl")
    print(" - experiments/centralized_metrics.json")
    print(" - experiments/centralized_classification_report.txt")
    print(" - experiments/centralized_history.csv")


if __name__ == "__main__":
    main()