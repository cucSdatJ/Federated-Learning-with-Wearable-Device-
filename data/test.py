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
# #
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

# import pandas as pd
#
# df_train = pd.read_csv("data/processed/client_1.csv")
# df_test = pd.read_csv("data/processed/test_set.csv")
#
# print("TRAIN hour stats:")
# print(df_train[["hour_sin", "hour_cos"]].describe())
#
# print("\nTEST hour stats:")
# print(df_test[["hour_sin", "hour_cos"]].describe())
#
# print("\nTRAIN min/max:")
# print("hour_sin:", df_train["hour_sin"].min(), df_train["hour_sin"].max())
# print("hour_cos:", df_train["hour_cos"].min(), df_train["hour_cos"].max())
#
# print("\nTEST min/max:")
# print("hour_sin:", df_test["hour_sin"].min(), df_test["hour_sin"].max())
# print("hour_cos:", df_test["hour_cos"].min(), df_test["hour_cos"].max())




#### threshold resoning


#
# import pandas as pd
# import numpy as np
# from pathlib import Path
#
# DATA_DIR = Path("data/processed")
#
# # ── 1. Load tất cả client CSVs ──────────────────────────────
# dfs = []
# for cid in range(1, 6):
#     path = DATA_DIR / f"client_{cid}.csv"
#     if path.exists():
#         dfs.append(pd.read_csv(path))
#
# df = pd.concat(dfs, ignore_index=True)
# print(f"Tổng rows: {len(df):,}")
# print(f"Columns: {list(df.columns)}\n")
#
# # ── 2. Reverse one-hot → activity_name ──────────────────────
# # act_rest=1 → "rest", act_walk=1 → "walk", v.v.
# def reverse_onehot(row):
#     if row["act_run"]   == 1: return "run"
#     if row["act_brisk"] == 1: return "brisk"
#     if row["act_walk"]  == 1: return "walk"
#     if row["act_rest"]  == 1: return "rest"
#     return "unknown"
#
# df["activity_name"] = df.apply(reverse_onehot, axis=1)
#
# # Cách nhanh hơn dùng np.select (không dùng apply):
# # conditions = [
# #     df["act_run"]   == 1,
# #     df["act_brisk"] == 1,
# #     df["act_walk"]  == 1,
# #     df["act_rest"]  == 1,
# # ]
# # choices = ["run", "brisk", "walk", "rest"]
# # df["activity_name"] = np.select(conditions, choices, default="unknown")
#
# # ── 3. Thống kê HR theo activity_name ───────────────────────
# print("=" * 65)
# print("HR Statistics by Activity Group")
# print("=" * 65)
#
# stats = (
#     df.groupby("activity_name")["heart_rate"]
#     .agg(
#         count="count",
#         mean="mean",
#         median="median",
#         std="std",
#         p10=lambda x: x.quantile(0.10),
#         p25=lambda x: x.quantile(0.25),
#         p75=lambda x: x.quantile(0.75),
#         p90=lambda x: x.quantile(0.90),
#     )
#     .sort_values("mean")
# )
#
# print(stats.round(2).to_string())
#
# # ── 4. Verify ngưỡng pseudo-label ───────────────────────────
# THRESHOLDS = {
#     "rest":  {"MEDIUM": 90,  "HIGH": 100},
#     "walk":  {"MEDIUM": 105, "HIGH": 120},
#     "brisk": {"MEDIUM": 120, "HIGH": 135},
#     "run":   {"MEDIUM": 150, "HIGH": 170},
# }
#
# print("\n" + "=" * 65)
# print("Label Distribution per Group (verify thresholds)")
# print("=" * 65)
#
# for group, thresh in THRESHOLDS.items():
#     sub = df[df["activity_name"] == group]["heart_rate"]
#     if len(sub) == 0:
#         continue
#     n_ok  = (sub <  thresh["MEDIUM"]).sum()
#     n_med = ((sub >= thresh["MEDIUM"]) & (sub < thresh["HIGH"])).sum()
#     n_hi  = (sub >= thresh["HIGH"]).sum()
#     total = len(sub)
#     print(
#         f"{group:6s}  n={total:,}  "
#         f"OK={n_ok/total*100:5.1f}%  "
#         f"MED={n_med/total*100:5.1f}%  "
#         f"HIGH={n_hi/total*100:5.1f}%  "
#         f"│ thresh MEDIUM≥{thresh['MEDIUM']}  HIGH≥{thresh['HIGH']}"
#     )
#
# # ── 5. Tại sao ngưỡng chọn vậy — in rõ reasoning ───────────
# print("\n" + "=" * 65)
# print("Threshold Reasoning (mean ± std)")
# print("=" * 65)
#
# for group in ["rest", "walk", "brisk", "run"]:
#     sub  = df[df["activity_name"] == group]["heart_rate"]
#     m, s = sub.mean(), sub.std()
#     t    = THRESHOLDS[group]
#     print(
#         f"{group:6s}  mean={m:6.2f}  std={s:5.2f}  "
#         f"│ MEDIUM={t['MEDIUM']} ({(t['MEDIUM']-m)/s:+.2f}σ)  "
#         f"HIGH={t['HIGH']} ({(t['HIGH']-m)/s:+.2f}σ)"
#     )

### print OK/MEDIUM/HIGH each client
# data/test.py

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/processed")

LABEL_MAP = {0: "OK", 1: "MEDIUM", 2: "HIGH"}

def reverse_onehot(df):
    conditions = [
        df["act_run"]   == 1,
        df["act_brisk"] == 1,
        df["act_walk"]  == 1,
        df["act_rest"]  == 1,
    ]
    choices = ["run", "brisk", "walk", "rest"]
    return np.select(conditions, choices, default="unknown")

# ── 1. Per-client label distribution ────────────────────────
print("=" * 70)
print("Label Distribution per Client")
print("=" * 70)
print(f"{'Client':<10} {'Subject':<10} {'N Rows':>10} {'OK %':>8} {'MEDIUM %':>10} {'HIGH %':>8}")
print("-" * 70)

all_dfs = []
for cid in range(1, 6):
    path = DATA_DIR / f"client_{cid}.csv"
    if not path.exists():
        print(f"  [WARN] {path} not found, skipping")
        continue

    df  = pd.read_csv(path)
    sid = df["subject_id"].iloc[0]
    n   = len(df)

    counts = df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    ok_pct  = counts[0] / n * 100
    med_pct = counts[1] / n * 100
    hi_pct  = counts[2] / n * 100

    print(
        f"  client_{cid:<4} subject_{sid:<4} {n:>10,} "
        f"{ok_pct:>7.1f}%  {med_pct:>8.1f}%  {hi_pct:>7.1f}%"
    )

    df["client_id"]     = cid
    df["activity_name"] = reverse_onehot(df)
    all_dfs.append(df)

# ── 2. Test set ──────────────────────────────────────────────
test_path = DATA_DIR / "test_set.csv"
if test_path.exists():
    test_df  = pd.read_csv(test_path)
    n        = len(test_df)
    counts   = test_df["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    ok_pct   = counts[0] / n * 100
    med_pct  = counts[1] / n * 100
    hi_pct   = counts[2] / n * 100
    subjects = sorted(test_df["subject_id"].unique().tolist())
    print("-" * 70)
    print(
        f"  test_set   subj {subjects}  {n:>10,} "
        f"{ok_pct:>7.1f}%  {med_pct:>8.1f}%  {hi_pct:>7.1f}%"
    )

# ── 3. Per-client × per-activity breakdown ───────────────────
if all_dfs:
    df_all = pd.concat(all_dfs, ignore_index=True)

    print("\n" + "=" * 70)
    print("Label Distribution per Client × Activity Group")
    print("=" * 70)

    for cid in range(1, 6):
        sub = df_all[df_all["client_id"] == cid]
        if len(sub) == 0:
            continue
        sid = sub["subject_id"].iloc[0]
        print(f"\n  client_{cid} (subject {sid}):")
        print(f"  {'Activity':<8} {'N':>8} {'OK':>8} {'MEDIUM':>8} {'HIGH':>8}")
        print(f"  {'-'*46}")

        for group in ["rest", "walk", "brisk", "run"]:
            g = sub[sub["activity_name"] == group]
            if len(g) == 0:
                continue
            n      = len(g)
            counts = g["label"].value_counts().reindex([0, 1, 2], fill_value=0)
            print(
                f"  {group:<8} {n:>8,} "
                f"{counts[0]/n*100:>7.1f}%"
                f"{counts[1]/n*100:>8.1f}%"
                f"{counts[2]/n*100:>8.1f}%"
            )

    # ── 4. Overall across all clients ───────────────────────
    print("\n" + "=" * 70)
    print("Overall Label Distribution (all 5 clients combined)")
    print("=" * 70)

    n      = len(df_all)
    counts = df_all["label"].value_counts().reindex([0, 1, 2], fill_value=0)
    for label_id, label_name in LABEL_MAP.items():
        bar_len = int(counts[label_id] / n * 40)
        bar     = "█" * bar_len
        print(
            f"  {label_name:<8} {counts[label_id]:>9,}  "
            f"({counts[label_id]/n*100:5.1f}%)  {bar}"
        )

    # ── 5. Non-IID score ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("Non-IID Quantification")
    print("=" * 70)
    print("(Std of per-client proportions — higher = more non-IID)")
    print()

    client_props = []
    for cid in range(1, 6):
        sub    = df_all[df_all["client_id"] == cid]
        n      = len(sub)
        counts = sub["label"].value_counts().reindex([0, 1, 2], fill_value=0)
        client_props.append([counts[i] / n for i in range(3)])

    props_arr = np.array(client_props)   # shape (5, 3)
    means     = props_arr.mean(axis=0)
    stds      = props_arr.std(axis=0)

    print(f"  {'Label':<10} {'Mean %':>8} {'Std %':>8}  {'Spread (max-min)':>18}")
    print(f"  {'-'*50}")
    for i, name in LABEL_MAP.items():
        col    = props_arr[:, i]
        spread = col.max() - col.min()
        print(
            f"  {name:<10} {means[i]*100:>7.1f}%  {stds[i]*100:>7.1f}%  "
            f"{spread*100:>17.1f}%"
        )

    print(f"\n  Interpretation:")
    print(f"  HIGH std → clients have very different label distributions")
    print(f"  → confirms genuine non-IID setup for FL experiment")
