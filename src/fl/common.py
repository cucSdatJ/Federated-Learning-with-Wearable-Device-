from pathlib import Path
import random
from typing import List, Tuple
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from src.data.dataset import FEATURE_COLS, TARGET_COL
from src.models.mlp import MLPClassifier


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def fit_global_scaler():
    data_dir = Path("data/processed")
    dfs = []
    for cid in range(1, 9):
        dfs.append(pd.read_csv(data_dir / f"client_{cid}.csv"))
    train_all = pd.concat(dfs, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(train_all[FEATURE_COLS])
    return scaler

def transform_df(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[FEATURE_COLS] = scaler.transform(out[FEATURE_COLS])
    return out
def get_or_create_fl_scaler():
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    scaler_path = model_dir / "flower_scaler.pkl"

    if scaler_path.exists():
        return joblib.load(scaler_path)

    scaler = fit_global_scaler()
    joblib.dump(scaler, scaler_path)
    return scaler

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_client_data(client_id: int, test_size: float = 0.2):
    data_dir = Path("data/processed")
    df = pd.read_csv(data_dir / f"client_{client_id}.csv")

    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        random_state=SEED,
        stratify=df[TARGET_COL],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def load_global_test():
    data_dir = Path("data/processed")
    return pd.read_csv(data_dir / "test_set.csv")


def df_to_loader(df: pd.DataFrame, batch_size: int = 512, shuffle: bool = True):
    X = torch.tensor(df[FEATURE_COLS].values, dtype=torch.float32)
    y = torch.tensor(df[TARGET_COL].values, dtype=torch.long)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def get_model() -> MLPClassifier:
    return MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)


def get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {
        k: torch.tensor(v, dtype=model.state_dict()[k].dtype)
        for k, v in params_dict
    }
    model.load_state_dict(state_dict, strict=True)


def compute_local_class_weights(df: pd.DataFrame) -> torch.Tensor:
    classes = np.array(sorted(df[TARGET_COL].unique()))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=df[TARGET_COL].values
    )

    # đảm bảo đủ 3 class weight theo thứ tự 0,1,2
    full_weights = np.ones(3, dtype=np.float32)
    for cls, w in zip(classes, weights):
        full_weights[int(cls)] = w

    return torch.tensor(full_weights, dtype=torch.float32).to(DEVICE)


def train_model(
    model: torch.nn.Module,
    train_loader,
    train_df: pd.DataFrame,
    epochs: int = 1,
    lr: float = 1e-3,
):
    class_weights = compute_local_class_weights(train_df)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    total_loss = 0.0
    total_batches = 0

    for _ in range(epochs):
        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


def evaluate_model(model: torch.nn.Module, loader):
    criterion = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(X)
            loss = criterion(logits, y)

            preds = torch.argmax(logits, dim=1)

            total_loss += loss.item()
            total_batches += 1

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(y.cpu().numpy().tolist())

    avg_loss = total_loss / max(total_batches, 1)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    }
    return avg_loss, metrics