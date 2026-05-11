from pathlib import Path
import torch

from src.models.mlp import MLPClassifier
from src.inference.utils import FEATURE_COLS, save_metadata


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def export_torchscript(
    model_path="models/flower_global_best.pt",
    output_path="models/exported/flower_global_best_torchscript.pt",
):
    model = MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    example_input = torch.randn(1, len(FEATURE_COLS)).to(DEVICE)
    traced_model = torch.jit.trace(model, example_input)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(output_path)
    print(f"[DONE] TorchScript saved to {output_path}")


def export_onnx(
    model_path="models/flower_global_best.pt",
    output_path="models/exported/flower_global_best.onnx",
):
    model = MLPClassifier(input_dim=len(FEATURE_COLS), num_classes=3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    dummy_input = torch.randn(1, len(FEATURE_COLS)).to(DEVICE)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=18,
    )
    print(f"[DONE] ONNX saved to {output_path}")


if __name__ == "__main__":
    save_metadata()
    export_torchscript()
    export_onnx()