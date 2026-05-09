from pathlib import Path
import pandas as pd
from src.data.pamap2_loader import load_subject

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_dfs = []

    for subject_id in range(101, 110):
        file_path = RAW_DIR / f"subject{subject_id}.dat"
        if not file_path.exists():
            print(f"[WARN] Missing {file_path}")
            continue

        print(f"[INFO] Processing subject {subject_id}")
        df = load_subject(str(file_path), subject_id)
        print(df.shape)
        print(df["label"].value_counts(normalize=True).sort_index())
        all_dfs.append(df)

    if not all_dfs:
        print("[ERROR] No subject loaded.")
        return

    all_df = pd.concat(all_dfs, ignore_index=True)
    all_df.to_csv(OUT_DIR / "all_processed.csv", index=False)

    dist = (
        all_df.groupby(["subject_id", "label"])
        .size()
        .reset_index(name="count")
    )
    dist.to_csv(OUT_DIR / "class_distribution.csv", index=False)

    print("[DONE] Saved all_processed.csv and class_distribution.csv")

if __name__ == "__main__":
    main()