from pathlib import Path
import pandas as pd
import json

import flwr as fl
import torch

from src.fl.common import (
    get_model,
    set_parameters,
    get_parameters,
    set_seed,
)


MODEL_DIR = Path("models")
EXP_DIR = Path("experiments")
MODEL_DIR.mkdir(exist_ok=True)
EXP_DIR.mkdir(exist_ok=True)

best = {"f1_macro": -1.0}
history = []
strategy_parameters = None

def aggregate_fit_metrics(metrics):
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
    global best_f1, best_params
    total_examples = sum(num_examples for num_examples, _ in metrics)
    if total_examples == 0:
        return {}

    keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    aggregated = {}

    for key in keys:
        aggregated[key] = sum(
            num_examples * m[key] for num_examples, m in metrics
        ) / total_examples

    # Lưu lịch sử từng round
    round_num = len(history) + 1
    row = {
        "round": round_num,
        **aggregated
    }
    history.append(row)
    pd.DataFrame(history).to_csv(EXP_DIR / "flower_round_metrics.csv", index=False)

    # Lưu best model
    if aggregated.get("f1_macro", -1.0) > best["f1_macro"]:
        best["f1_macro"] = aggregated["f1_macro"]
        print("[SERVER] Best model updated -> models/flower_global_best.pt")
        if strategy_parameters is not None:
            model = get_model()
            ndarrays = fl.common.parameters_to_ndarrays(strategy_parameters)
            set_parameters(model, ndarrays)
            torch.save(model.state_dict(), MODEL_DIR / "flower_global_best.pt")
        with open(EXP_DIR / "flower_best_metrics.json", "w", encoding="utf-8") as f:
            json.dump(aggregated, f, indent=2)
    return aggregated


class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        global strategy_parameters
        aggregated_result = super().aggregate_fit(server_round, results, failures)
        if aggregated_result is not None:
            strategy_parameters = aggregated_result[0]
        return aggregated_result
    def aggregate_evaluate(self, server_round, results, failures):
        return super().aggregate_evaluate(server_round, results, failures)


def main():
    set_seed()

    initial_model = get_model()
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(initial_model))


    strategy = CustomFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=8,
        min_evaluate_clients=8,
        min_available_clients=8,
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