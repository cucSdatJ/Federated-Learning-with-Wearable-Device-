import pandas as pd
from pathlib import Path

# data_dir = Path("data/processed")

# validate processed data
# for name in [f"client_{i}.csv" for i in range(1, 6)] + ["test_set.csv"]:
#     df = pd.read_csv(data_dir / name)
#     print(f"\n{name}")
#     print("shape:", df.shape)
#     print("NaN:\n", df.isna().sum())
#     print("label dist:\n", df["label"].value_counts(normalize=True).sort_index())

# verify activity 6

# df = pd.read_csv("data/raw/subject101.dat", sep=r"\s+", header=None, engine="python")
# df = df[[1, 2]].copy()
# df.columns = ["activity_id", "heart_rate"]
#
# df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
# df["heart_rate"] = df["heart_rate"].interpolate().bfill().ffill()
#
# df = df[df["activity_id"] != 0]
#
# print(df.groupby("activity_id")["heart_rate"].agg(["mean", "median", "std", "count"]).sort_index())

# RAW_DIR = Path("data/raw")
#
# ACTIVITY_MAP = {
#     0: "transient",
#     1: "lying",
#     2: "sitting",
#     3: "standing",
#     4: "walking",
#     5: "running",
#     6: "cycling",
#     7: "nordic_walking",
#     9: "watching_tv",
#     10: "computer_work",
#     11: "car_driving",
#     12: "ascending_stairs",
#     13: "descending_stairs",
#     16: "vacuum_cleaning",
#     17: "ironing",
#     18: "folding_laundry",
#     19: "house_cleaning",
#     20: "playing_soccer",
#     24: "rope_jumping",
# }
#
# EXPECTED_ACTIVITY_IDS = sorted(ACTIVITY_MAP.keys())
#
#
# def load_subject(file_path: Path, subject_id: int) -> pd.DataFrame:
#     df = pd.read_csv(file_path, sep=r"\s+", header=None, engine="python")
#
#     # Chỉ lấy cột cần cho kiểm tra activity + HR
#     df = df[[0, 1, 2]].copy()
#     df.columns = ["timestamp", "activity_id", "heart_rate"]
#     df["subject_id"] = subject_id
#
#     # numeric
#     df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
#     df["activity_id"] = pd.to_numeric(df["activity_id"], errors="coerce").astype("Int64")
#     df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce")
#
#     # Nội suy HR trong từng subject
#     df["heart_rate_interp"] = df["heart_rate"].interpolate(method="linear").bfill().ffill()
#
#     return df
#
#
# def main():
#     all_dfs = []
#
#     print("=" * 80)
#     print("PAMAP2 ACTIVITY CHECK ACROSS ALL SUBJECTS")
#     print("=" * 80)
#
#     for subject_id in range(101, 110):
#         file_path = RAW_DIR / f"subject{subject_id}.dat"
#         if not file_path.exists():
#             print(f"[WARN] Missing file: {file_path}")
#             continue
#
#         df = load_subject(file_path, subject_id)
#         all_dfs.append(df)
#
#         present_ids = sorted(df["activity_id"].dropna().unique().tolist())
#         missing_ids = sorted(set(EXPECTED_ACTIVITY_IDS) - set(present_ids))
#
#         print(f"\nSubject {subject_id}")
#         print(f"Shape: {df.shape}")
#         print(f"Present activity IDs: {present_ids}")
#         print(f"Missing expected IDs: {missing_ids}")
#
#     if not all_dfs:
#         print("[ERROR] No subject files loaded.")
#         return
#
#     all_df = pd.concat(all_dfs, ignore_index=True)
#
#     # map tên activity
#     all_df["activity_name"] = all_df["activity_id"].map(ACTIVITY_MAP)
#
#     print("\n" + "=" * 80)
#     print("GLOBAL ACTIVITY COUNTS")
#     print("=" * 80)
#     global_counts = all_df["activity_id"].value_counts().sort_index()
#     print(global_counts)
#
#     print("\n" + "=" * 80)
#     print("GLOBAL HR STATS BY ACTIVITY (INTERPOLATED HR)")
#     print("=" * 80)
#     global_stats = (
#         all_df.groupby(["activity_id", "activity_name"])["heart_rate_interp"]
#         .agg(["mean", "median", "std", "min", "max", "count"])
#         .reset_index()
#         .sort_values("activity_id")
#     )
#     print(global_stats.to_string(index=False))
#
#     print("\n" + "=" * 80)
#     print("PER-SUBJECT HR STATS BY ACTIVITY")
#     print("=" * 80)
#     per_subject_stats = (
#         all_df.groupby(["subject_id", "activity_id", "activity_name"])["heart_rate_interp"]
#         .agg(["mean", "median", "std", "min", "max", "count"])
#         .reset_index()
#         .sort_values(["subject_id", "activity_id"])
#     )
#     print(per_subject_stats.head(50).to_string(index=False))
#     print("\n[INFO] Full per-subject table will be saved to CSV.")
#
#     # Xuất file
#     out_dir = Path("data/processed")
#     out_dir.mkdir(parents=True, exist_ok=True)
#
#     global_counts_df = global_counts.reset_index()
#     global_counts_df.columns = ["activity_id", "count"]
#     global_counts_df["activity_name"] = global_counts_df["activity_id"].map(ACTIVITY_MAP)
#
#     global_counts_df.to_csv(out_dir / "activity_counts_all_subjects.csv", index=False)
#     global_stats.to_csv(out_dir / "activity_hr_stats_all_subjects.csv", index=False)
#     per_subject_stats.to_csv(out_dir / "activity_hr_stats_per_subject.csv", index=False)
#
#     # Check expected coverage
#     present_global = sorted(all_df["activity_id"].dropna().unique().tolist())
#     missing_global = sorted(set(EXPECTED_ACTIVITY_IDS) - set(present_global))
#
#     print("\n" + "=" * 80)
#     print("COVERAGE CHECK")
#     print("=" * 80)
#     print("Expected activity IDs:", EXPECTED_ACTIVITY_IDS)
#     print("Present activity IDs :", present_global)
#     print("Missing globally     :", missing_global)
#
#     print("\n[DONE] Saved files:")
#     print(" - data/processed/activity_counts_all_subjects.csv")
#     print(" - data/processed/activity_hr_stats_all_subjects.csv")
#     print(" - data/processed/activity_hr_stats_per_subject.csv")
#
#
# if __name__ == "__main__":
#     main()



# data_dir = Path("data/processed")
# train_df = pd.concat(
#     [pd.read_csv(data_dir / f"client_{i}.csv") for i in range(1, 6)],
#     ignore_index=True
# )
#
# print(train_df["label"].value_counts(normalize=True).sort_index())



# test_df = pd.read_csv("data/processed/test_set.csv")
# print(test_df["label"].value_counts(normalize=True).sort_index())




