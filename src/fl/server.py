from pathlib import Path
import json

import flwr as fl
import torch

from src.fl.common import (
    df_to_loader,
    evaluate_model,
    get_model,
    get_or_create_fl_scaler,
    get_parameters,
    load_global_test,
    set_parameters,
    set_seed,
    transform_df,
)


MODEL_DIR = Path("models")
EXP_DIR = Path("experiments")
MODEL_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)


def aggregate_fit_metrics(metrics):
    # metrics: List[Tuple[num_examples, metrics_dict]]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    if "train_loss" not in metrics[0][1]:
        return {}

    avg_train_loss = sum(
        num_examples * m["train_loss"] for num_examples, m in metrics
    ) / total_examples

    return {"train_loss": avg_train_loss}


def aggregate_eval_metrics(metrics):
    # metrics: List[Tuple[num_examples, metrics_dict]]
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    aggregated = {}

    for key in keys:
        aggregated[key] = sum(
            num_examples * m[key] for num_examples, m in metrics
        ) / total_examples

    return aggregated

def get_evaluate_fn():
    test_df = load_global_test()
    scaler = get_or_create_fl_scaler()
    test_df = transform_df(test_df, scaler)
    test_loader = df_to_loader(test_df, batch_size=512, shuffle=False)
    best = {"f1_macro": -1.0}
    history = []

    def evaluate(server_round, parameters, config):
        model = get_model()
        set_parameters(model, parameters)

        loss, metrics = evaluate_model(model, test_loader)

        row = {
            "round": int(server_round),
            "test_loss": float(loss),
            "test_accuracy": metrics["accuracy"],
            "test_precision_macro": metrics["precision_macro"],
            "test_recall_macro": metrics["recall_macro"],
            "test_f1_macro": metrics["f1_macro"],
        }
        history.append(row)

        print(
            f"[SERVER][ROUND {server_round}] "
            f"loss={loss:.4f} | "
            f"acc={metrics['accuracy']:.4f} | "
            f"f1={metrics['f1_macro']:.4f}"
        )

        # save round history every round
        import pandas as pd
        pd.DataFrame(history).to_csv(EXP_DIR / "flower_round_metrics.csv", index=False)

        # save best global model
        if metrics["f1_macro"] > best["f1_macro"]:
            best["f1_macro"] = metrics["f1_macro"]

            torch.save(model.state_dict(), MODEL_DIR / "flower_global_best.pt")

            with open(EXP_DIR / "flower_best_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

            print("[SERVER] Best model updated -> models/flower_global_best.pt")

        return float(loss), {
            "accuracy": metrics["accuracy"],
            "precision_macro": metrics["precision_macro"],
            "recall_macro": metrics["recall_macro"],
            "f1_macro": metrics["f1_macro"],
        }

    return evaluate


def main():
    set_seed()

    initial_model = get_model()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(initial_model))

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_fn=get_evaluate_fn(),
        fit_metrics_aggregation_fn=aggregate_fit_metrics,
        evaluate_metrics_aggregation_fn= aggregate_eval_metrics,
        initial_parameters=initial_parameters,
    )

    print("[SERVER] Starting Flower server on 0.0.0.0:8080")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=15),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()