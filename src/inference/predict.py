import joblib
import numpy as np
import pandas as pd
import torch

from src.models.mlp import MLPClassifier
from src.inference.utils import FEATURE_COLS, LABEL_MAP
from src.inference.feature_builder import (
    build_feature_dict,
    build_feature_dict_from_activity_id,
    build_feature_dict_from_device_code,
    feature_dict_to_ordered_vector,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class WearableInferenceEngine:
    def __init__(
        self,
        model_path="models/flower_global_best.pt",
        scaler_path="models/flower_scaler.pkl",
    ):
        self.model = MLPClassifier(
            input_dim=len(FEATURE_COLS),
            num_classes=3,
        ).to(DEVICE)

        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.eval()

        self.scaler = joblib.load(scaler_path)
        # ===== DEBUG SCALER =====
        print("\n=== SCALER DEBUG ===")
        print("model_path =", model_path)
        print("scaler_path =", scaler_path)
        print("FEATURE_COLS =", FEATURE_COLS)

        if hasattr(self.scaler, "mean_"):
            print("scaler.mean_ =", self.scaler.mean_.tolist())
        else:
            print("scaler has no mean_")

        if hasattr(self.scaler, "scale_"):
            print("scaler.scale_ =", self.scaler.scale_.tolist())
        else:
            print("scaler has no scale_")

    def preprocess(self, feature_dict: dict):
        """
        Nhận feature_dict đúng schema 8 features,
        ép về đúng thứ tự FEATURE_COLS, scale, rồi convert sang tensor.
        """
        ordered_vector = feature_dict_to_ordered_vector(feature_dict)

        x_df = pd.DataFrame([ordered_vector], columns=FEATURE_COLS)
        x_scaled = self.scaler.transform(x_df).astype(np.float32)

        x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(DEVICE)
        return x_tensor, ordered_vector, x_scaled[0]

    def predict(self, feature_dict: dict, debug: bool = False):
        """
        Predict trực tiếp từ feature_dict đã được build sẵn.
        """
        x, ordered_vector, scaled_vector = self.preprocess(feature_dict)

        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_class = int(np.argmax(probs))

        if debug:
            print("\n=== DEBUG PREDICT ===")
            print("FEATURE_COLS =", FEATURE_COLS)
            print("feature_dict =", feature_dict)
            print("ordered_vector =", ordered_vector)
            print("scaled_x =", scaled_vector.tolist())
            print("logits =", logits.cpu().numpy()[0].tolist())
            print("probs =", probs.tolist())
            print("pred_class =", pred_class)
            print("pred_label =", LABEL_MAP[pred_class])

        return {
            "pred_class": pred_class,
            "pred_label": LABEL_MAP[pred_class],
            "probabilities": {
                "OK": float(probs[0]),
                "MEDIUM": float(probs[1]),
                "HIGH": float(probs[2]),
            },
            "features": feature_dict,
        }

    def predict_from_sensor(
        self,
        heart_rate: float,
        hr_window: list,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        hour_of_day: float,
        activity_group: str,
        debug: bool = False,
    ):
        """
        Build feature từ dữ liệu sensor-like với activity_group: rest/walk/brisk/run
        """
        feature_dict = build_feature_dict(
            heart_rate=heart_rate,
            hr_window=hr_window,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            activity_group=activity_group,
            hour_of_day=hour_of_day,
        )
        return self.predict(feature_dict, debug=debug)

    def predict_from_activity_id(
        self,
        heart_rate: float,
        hr_window: list,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        hour_of_day: float,
        activity_id: int,
        debug: bool = False,
    ):
        """
        Build feature từ PAMAP2 activity_id gốc
        """
        feature_dict = build_feature_dict_from_activity_id(
            heart_rate=heart_rate,
            hr_window=hr_window,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            activity_id=activity_id,
            hour_of_day=hour_of_day,
        )
        return self.predict(feature_dict, debug=debug)

    def predict_from_device_code(
        self,
        heart_rate: float,
        hr_window: list,
        acc_x: float,
        acc_y: float,
        acc_z: float,
        hour_of_day: float,
        device_activity_code: int,
        debug: bool = False,
    ):
        """
        Build feature từ code firmware CE:
        0 -> rest, 1 -> walk, 2 -> brisk, 4 -> run
        """
        feature_dict = build_feature_dict_from_device_code(
            heart_rate=heart_rate,
            hr_window=hr_window,
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            activity_code=device_activity_code,
            hour_of_day=hour_of_day,
        )
        return self.predict(feature_dict, debug=debug)


if __name__ == "__main__":
    engine = WearableInferenceEngine()

    # Test sample sensor-like theo protocol app -> API đã chốt [2]
    result = engine.predict_from_device_code(
        heart_rate=132.0,
        hr_window=[126, 128, 130, 131, 132],
        acc_x=0.8,
        acc_y=9.6,
        acc_z=1.5,
        hour_of_day=10.5,
        device_activity_code=2,  # brisk
        debug=True,
    )

    print("\n=== RESULT ===")
    print(result)
