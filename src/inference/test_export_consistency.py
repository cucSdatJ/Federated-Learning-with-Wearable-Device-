from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import onnxruntime as ort

from src.models.mlp import MLPClassifier
from src.inference.utils import FEATURE_COLS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


MODEL_PT = "models/flower_global_best.pt"
MODEL_TS = "models/exported/flower_global_best_torchscript.pt"
MODEL_ONNX = "models/exported/flower_global_best.onnx"
SCALER_PATH = "models/flower_scaler.pkl"
TEST_CSV = "data/processed/test_set.csv"


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def load_test_samples(n_samples: int = 10, random_state: int = 42):
    df = pd.read_csv(TEST_CSV)

    # lấy ngẫu nhiên một số mẫu từ test set
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=random_state).reset_index(drop=True)

    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(sample_df[FEATURE_COLS].astype(np.float32))
    y = sample_df["label"].values.astype(int)

    return sample_df, X.astype(np.float32), y


def load_pytorch_model():
    model = MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PT, map_location=DEVICE))
    model.eval()
    return model


def run_pytorch(model, X: np.ndarray):
    x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor).cpu().numpy()
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=1)
    return logits, probs, preds


def load_torchscript_model():
    model = torch.jit.load(MODEL_TS, map_location=DEVICE)
    model.eval()
    return model


def run_torchscript(model, X: np.ndarray):
    x_tensor = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        logits = model(x_tensor).cpu().numpy()
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=1)
    return logits, probs, preds


def load_onnx_session():
    providers = ["CPUExecutionProvider"]
    return ort.InferenceSession(MODEL_ONNX, providers=providers)


def run_onnx(session, X: np.ndarray):
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: X.astype(np.float32)})

    logits = outputs[0]
    probs = softmax_np(logits)
    preds = np.argmax(probs, axis=1)
    return logits, probs, preds


def compare_arrays(name_a: str, arr_a: np.ndarray, name_b: str, arr_b: np.ndarray):
    abs_diff = np.abs(arr_a - arr_b)
    max_diff = float(abs_diff.max())
    mean_diff = float(abs_diff.mean())

    print(f"[COMPARE] {name_a} vs {name_b}")
    print(f"  max abs diff : {max_diff:.10f}")
    print(f"  mean abs diff: {mean_diff:.10f}")
    print()


def main():
    print("=" * 80)
    print("TEST EXPORT CONSISTENCY")
    print("=" * 80)

    sample_df, X, y = load_test_samples(n_samples=10, random_state=42)
    print(f"[INFO] Loaded {len(X)} samples from test set")
    print(f"[INFO] Input shape: {X.shape}")
    print()

    # PyTorch
    pt_model = load_pytorch_model()
    pt_logits, pt_probs, pt_preds = run_pytorch(pt_model, X)

    # TorchScript
    ts_model = load_torchscript_model()
    ts_logits, ts_probs, ts_preds = run_torchscript(ts_model, X)

    # ONNX
    onnx_session = load_onnx_session()
    onnx_logits, onnx_probs, onnx_preds = run_onnx(onnx_session, X)

    # Compare logits
    compare_arrays("PyTorch logits", pt_logits, "TorchScript logits", ts_logits)
    compare_arrays("PyTorch logits", pt_logits, "ONNX logits", onnx_logits)
    compare_arrays("TorchScript logits", ts_logits, "ONNX logits", onnx_logits)

    # Compare probabilities
    compare_arrays("PyTorch probs", pt_probs, "TorchScript probs", ts_probs)
    compare_arrays("PyTorch probs", pt_probs, "ONNX probs", onnx_probs)
    compare_arrays("TorchScript probs", ts_probs, "ONNX probs", onnx_probs)

    # Compare predicted classes
    print("[PRED CLASS CHECK]")
    print("PyTorch preds    :", pt_preds.tolist())
    print("TorchScript preds:", ts_preds.tolist())
    print("ONNX preds       :", onnx_preds.tolist())
    print("Ground truth     :", y.tolist())
    print()

    print("[AGREEMENT]")
    print("PyTorch == TorchScript:", bool(np.array_equal(pt_preds, ts_preds)))
    print("PyTorch == ONNX       :", bool(np.array_equal(pt_preds, onnx_preds)))
    print("TorchScript == ONNX   :", bool(np.array_equal(ts_preds, onnx_preds)))
    print()

    # Detailed preview
    preview = pd.DataFrame({
        "true_label": y,
        "pt_pred": pt_preds,
        "ts_pred": ts_preds,
        "onnx_pred": onnx_preds,
        "pt_prob_ok": pt_probs[:, 0],
        "pt_prob_medium": pt_probs[:, 1],
        "pt_prob_high": pt_probs[:, 2],
    })
    print("[PREVIEW]")
    print(preview.head(10).to_string(index=False))

    print("\n[DONE] Export consistency test finished.")


if __name__ == "__main__":
    main()
