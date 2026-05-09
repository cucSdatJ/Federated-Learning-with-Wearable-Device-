from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/processed")

TRAIN_SUBJECTS = [101, 102, 103, 104, 105]
TEST_SUBJECTS = [106, 107, 108, 109]

def main():
    all_path = DATA_DIR / "all_processed.csv"
    df = pd.read_csv(all_path)

    for idx, sid in enumerate(TRAIN_SUBJECTS, start=1):
        client_df = df[df["subject_id"] == sid].copy()
        client_df.to_csv(DATA_DIR / f"client_{idx}.csv", index=False)
        print(f"[DONE] client_{idx}.csv -> {client_df.shape}")

    test_df = df[df["subject_id"].isin(TEST_SUBJECTS)].copy()
    test_df.to_csv(DATA_DIR / "test_set.csv", index=False)
    print(f"[DONE] test_set.csv -> {test_df.shape}")

if __name__ == "__main__":
    main()