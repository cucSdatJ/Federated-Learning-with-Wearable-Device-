from pathlib import Path
import json
import pandas as pd


EXP_DIR = Path("experiments")


def load_json(path: Path):
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(series):
    if len(series) == 0:
        return None
    return float(series.mean())


def df_to_simple_markdown(df: pd.DataFrame) -> str:
    cols = df.columns.tolist()

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append(f"{val:.6f}")
            else:
                vals.append(str(val))
        rows.append("| " + " | ".join(vals) + " |")

    return "\n".join([header, sep] + rows)


def metrics_row(setting_name: str, metrics: dict):
    return {
        "setting": setting_name,
        "accuracy": metrics.get("accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
    }


def main():
    centralized_path = EXP_DIR / "centralized_metrics.json"
    local_path = EXP_DIR / "local_results.csv"
    flower_path = EXP_DIR / "flower_best_metrics.json"

    centralized = load_json(centralized_path)
    flower = load_json(flower_path)

    if not local_path.exists():
        print(f"[WARN] Missing file: {local_path}")
        local_df = None
    else:
        local_df = pd.read_csv(local_path)

    rows = []

    # 1) Centralized
    if centralized is not None:
        rows.append(metrics_row("centralized", centralized))

    # 2) Local-only average + best local
    if local_df is not None and len(local_df) > 0:
        rows.append({
            "setting": "local_avg",
            "accuracy": safe_mean(local_df["test_accuracy"]),
            "precision_macro": safe_mean(local_df["test_precision_macro"]),
            "recall_macro": safe_mean(local_df["test_recall_macro"]),
            "f1_macro": safe_mean(local_df["test_f1_macro"]),
        })

        best_idx = local_df["test_f1_macro"].idxmax()
        best_row = local_df.loc[best_idx]

        rows.append({
            "setting": f"local_best_client_{int(best_row['client_id'])}",
            "accuracy": float(best_row["test_accuracy"]),
            "precision_macro": float(best_row["test_precision_macro"]),
            "recall_macro": float(best_row["test_recall_macro"]),
            "f1_macro": float(best_row["test_f1_macro"]),
        })

        # Optional: thêm từng local client riêng lẻ
        for _, row in local_df.iterrows():
            rows.append({
                "setting": f"local_client_{int(row['client_id'])}",
                "accuracy": float(row["test_accuracy"]),
                "precision_macro": float(row["test_precision_macro"]),
                "recall_macro": float(row["test_recall_macro"]),
                "f1_macro": float(row["test_f1_macro"]),
            })


    # 4) Flower
    if flower is not None:
        rows.append(metrics_row("flower_best", flower))

    if len(rows) == 0:
        print("[ERROR] No result files found.")
        return

    compare_df = pd.DataFrame(rows)
    compare_df = compare_df.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)

    out_csv = EXP_DIR / "compare_results.csv"
    compare_df.to_csv(out_csv, index=False)

    print("=" * 100)
    print("COMPARE RESULTS")
    print("=" * 100)
    print(compare_df.to_string(index=False))

    out_md = EXP_DIR / "compare_results.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Compare Results\n\n")
        f.write(df_to_simple_markdown(compare_df))
        f.write("\n")

    print("\n[DONE] Saved:")
    print(f" - {out_csv}")
    print(f" - {out_md}")

    best = compare_df.iloc[0]
    print("\nQuick summary:")
    print(
        f"Best setting: {best['setting']} | "
        f"Accuracy={best['accuracy']:.4f}, "
        f"F1={best['f1_macro']:.4f}"
    )

    # Tạo file gọn chỉ gồm các setting chính
    summary_rows = []
    for key in [
        "centralized",
        "local_avg",
        "federated_best_manual",
        "flower_best",
    ]:
        sub = compare_df[compare_df["setting"] == key]
        if len(sub) > 0:
            summary_rows.append(sub.iloc[0].to_dict())

    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_df = summary_df.sort_values(by="f1_macro", ascending=False).reset_index(drop=True)
        summary_csv = EXP_DIR / "compare_summary.csv"
        summary_df.to_csv(summary_csv, index=False)
        print(f" - {summary_csv}")


if __name__ == "__main__":
    main()