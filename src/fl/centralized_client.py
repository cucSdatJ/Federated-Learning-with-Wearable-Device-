import argparse

import flwr as fl

from src.fl.common import (
    DEVICE,
    df_to_loader,
    evaluate_model,
    get_model,
    get_or_create_fl_scaler,
    get_parameters,
    load_client_data,
    set_parameters,
    set_seed,
    train_model,
    transform_df,
)


class WearableFlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, epochs: int = 1, batch_size: int = 512, lr: float = 1e-3):
        self.client_id = client_id
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        self.train_df, self.val_df = load_client_data(client_id)

        self.scaler = get_or_create_fl_scaler()
        self.train_df = transform_df(self.train_df, self.scaler)
        self.val_df = transform_df(self.val_df, self.scaler)

        self.train_loader = df_to_loader(self.train_df, batch_size=batch_size, shuffle=True)
        self.val_loader = df_to_loader(self.val_df, batch_size=batch_size, shuffle=False)

        self.model = get_model()

        print(f"[CLIENT {client_id}] device={DEVICE}")
        print(f"[CLIENT {client_id}] train={self.train_df.shape}, val={self.val_df.shape}")
        print(f"[CLIENT {client_id}] label dist train:")
        print(self.train_df["label"].value_counts(normalize=True).sort_index())

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)

        avg_loss = train_model(
            self.model,
            self.train_loader,
            self.train_df,
            epochs=self.epochs,
            lr=self.lr,
        )

        return (
            get_parameters(self.model),
            len(self.train_df),
            {
                "train_loss": float(avg_loss),
            },
        )

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)

        loss, metrics = evaluate_model(self.model, self.val_loader)

        return (
            float(loss),
            len(self.val_df),
            {
                "accuracy": metrics["accuracy"],
                "precision_macro": metrics["precision_macro"],
                "recall_macro": metrics["recall_macro"],
                "f1_macro": metrics["f1_macro"],
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cid", type=int, required=True, help="Client ID: 1..5")
    parser.add_argument("--server", type=str, default="127.0.0.1:8080")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    set_seed()

    client = WearableFlowerClient(
        client_id=args.cid,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )

    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )


if __name__ == "__main__":
    main()