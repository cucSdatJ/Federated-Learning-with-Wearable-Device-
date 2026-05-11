from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import numpy as np

sns.set_theme(style="whitegrid", font_scale=1.0)

EXP_DIR = Path("experiments")
DATA_DIR = Path("data/processed")
PLOT_DIR = EXP_DIR / "plots"


def ensure_plot_dir():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def plot_compare_summary():
    """
    Vẽ compare cho các setting chính:
    - centralized
    - local_avg
    - federated_best_manual (nếu có)
    - flower_best
    """
    path = EXP_DIR / "compare_summary.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Empty file: {path}")
        return

    # sort để bar chart dễ nhìn
    df = df.sort_values("f1_macro", ascending=True)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="f1_macro",
        y="setting",
        hue="setting",
        palette="viridis",
        legend=False,
    )
    ax.set_title("Main Settings Comparison by Macro F1")
    ax.set_xlabel("F1 Macro")
    ax.set_ylabel("Setting")

    for i, v in enumerate(df["f1_macro"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center")

    plt.tight_layout()
    out = PLOT_DIR / "compare_summary_f1_macro.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df,
        x="accuracy",
        y="setting",
        hue="setting",
        palette="magma",
        legend=False,
    )
    ax.set_title("Main Settings Comparison by Accuracy")
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Setting")

    for i, v in enumerate(df["accuracy"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center")

    plt.tight_layout()
    out = PLOT_DIR / "compare_summary_accuracy.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")


def plot_compare_all():
    """
    Vẽ compare tất cả setting trong compare_results.csv
    (bao gồm local_best, local_client_i, flower_best, centralized, ...)
    """
    path = EXP_DIR / "compare_results.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Empty file: {path}")
        return

    # Chỉ lấy top 10 theo f1 để đỡ quá dài
    df = df.sort_values("f1_macro", ascending=False).head(10)
    df = df.sort_values("f1_macro", ascending=True)

    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=df,
        x="f1_macro",
        y="setting",
        hue="setting",
        palette="cubehelix",
        legend=False,
    )
    ax.set_title("Top 10 Model Settings by Macro F1")
    ax.set_xlabel("F1 Macro")
    ax.set_ylabel("Setting")

    for i, v in enumerate(df["f1_macro"]):
        ax.text(v + 0.001, i, f"{v:.4f}", va="center")

    plt.tight_layout()
    out = PLOT_DIR / "compare_all_top10_f1_macro.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")


def plot_fl_round_metrics_flower():
    """
    Plot Flower round metrics
    """
    path = EXP_DIR / "flower_round_metrics.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Empty file: {path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.lineplot(data=df, x="round", y="test_f1_macro", marker="o", ax=axes[0], color="tab:purple")
    axes[0].set_title("Flower FL: Test F1 Macro by Round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("F1 Macro")

    sns.lineplot(data=df, x="round", y="test_accuracy", marker="o", ax=axes[1], color="tab:orange")
    axes[1].set_title("Flower FL: Test Accuracy by Round")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("Accuracy")

    plt.tight_layout()
    out = PLOT_DIR / "flower_round_metrics.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="round", y="test_loss", marker="o", color="tab:red")
    plt.title("Flower FL: Test Loss by Round")
    plt.xlabel("Round")
    plt.ylabel("Test Loss")
    plt.tight_layout()
    out = PLOT_DIR / "flower_round_loss.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")


def plot_activity_counts():
    path = DATA_DIR / "activity_counts_all_subjects.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)

    # bỏ transient để nhìn activity thật rõ hơn
    df_no_transient = df[df["activity_id"] != 0].copy()
    df_no_transient = df_no_transient.sort_values("count", ascending=False)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_no_transient,
        x="activity_name",
        y="count",
        hue="activity_name",
        palette="crest",
        legend=False,
    )
    ax.set_title("Activity Counts Across All Subjects (Excluding Transient)")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f"{int(height):,}",
            (p.get_x() + p.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=8,
            xytext=(0, 3),
            textcoords="offset points",
        )

    plt.tight_layout()
    out = PLOT_DIR / "activity_counts.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")


def plot_activity_hr_means():
    path = DATA_DIR / "activity_hr_stats_all_subjects.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)

    df = df[df["activity_id"] != 0].copy()
    df = df.sort_values("mean", ascending=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df,
        x="activity_name",
        y="mean",
        hue="activity_name",
        palette="flare",
        legend=False,
    )
    ax.set_title("Mean Heart Rate by Activity")
    ax.set_xlabel("Activity")
    ax.set_ylabel("Mean Heart Rate")
    plt.xticks(rotation=45, ha="right")

    for i, v in enumerate(df["mean"]):
        ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out = PLOT_DIR / "activity_hr_means.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")

def plot_local_clients():
    path = EXP_DIR / "local_results.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Empty file: {path}")
        return

    df["client_name"] = df["client_id"].apply(lambda x: f"client_{int(x)}")

    # F1 macro
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df.sort_values("test_f1_macro", ascending=False),
        x="client_name",
        y="test_f1_macro",
        hue="client_name",
        palette="Set2",
        legend=False,
    )
    ax.set_title("Local Model Performance by Client (F1 Macro)")
    ax.set_xlabel("Client")
    ax.set_ylabel("F1 Macro")

    for i, v in enumerate(df.sort_values("test_f1_macro", ascending=False)["test_f1_macro"]):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = PLOT_DIR / "local_clients_f1_macro.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")

    # Accuracy
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=df.sort_values("test_accuracy", ascending=False),
        x="client_name",
        y="test_accuracy",
        hue="client_name",
        palette="Set3",
        legend=False,
    )
    ax.set_title("Local Model Performance by Client (Accuracy)")
    ax.set_xlabel("Client")
    ax.set_ylabel("Accuracy")

    for i, v in enumerate(df.sort_values("test_accuracy", ascending=False)["test_accuracy"]):
        ax.text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out = PLOT_DIR / "local_clients_accuracy.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")

import json
import numpy as np

def plot_confusion_matrix_from_json(json_path: Path, out_name: str, title: str):
    if not json_path.exists():
        print(f"[WARN] Missing file: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    cm = metrics.get("confusion_matrix")
    if cm is None:
        print(f"[WARN] confusion_matrix not found in {json_path}")
        return

    cm = np.array(cm)

    plt.figure(figsize=(6, 5))
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["OK", "MEDIUM", "HIGH"],
        yticklabels=["OK", "MEDIUM", "HIGH"],
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    plt.tight_layout()
    out = PLOT_DIR / out_name
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")


def plot_confusion_matrices():
    plot_confusion_matrix_from_json(
        EXP_DIR / "centralized_metrics.json",
        "confusion_matrix_centralized.png",
        "Centralized Confusion Matrix",
    )

    plot_confusion_matrix_from_json(
        EXP_DIR / "flower_best_metrics.json",
        "confusion_matrix_flower.png",
        "Flower FL Confusion Matrix",
    )

def plot_flower_vs_centralized():
    path = EXP_DIR / "compare_results.csv"
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return

    df = pd.read_csv(path)
    df = df[df["setting"].isin(["centralized", "flower_best"])].copy()

    if df.empty:
        print("[WARN] No centralized/flower_best rows found.")
        return

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=df,
        x="setting",
        y="f1_macro",
        hue="setting",
        palette=["#4C72B0", "#55A868"],
        legend=False,
    )
    ax.set_title("Flower FL vs Centralized (F1 Macro)")
    ax.set_xlabel("Setting")
    ax.set_ylabel("F1 Macro")

    for i, v in enumerate(df["f1_macro"]):
        ax.text(i, v + 0.001, f"{v:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    out = PLOT_DIR / "flower_vs_centralized_f1.png"
    plt.savefig(out, dpi=300)
    plt.close()
    print(f"[DONE] Saved {out}")

def main():
    ensure_plot_dir()

    # Compare plots
    plot_compare_summary()
    plot_compare_all()
    plot_flower_vs_centralized()
    plot_local_clients()

    # FL plots
    plot_fl_round_metrics_flower()

    # Confusion matrix
    plot_confusion_matrices()

    # Dataset analysis plots
    plot_activity_counts()
    plot_activity_hr_means()

    print("\nAll available plots saved to:")
    print(PLOT_DIR.resolve())


if __name__ == "__main__":
    main()