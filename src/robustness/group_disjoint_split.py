# =========================================================
# GROUP-DISJOINT TRAIN / TEST SPLIT BY Tweet_ID
# =========================================================

import os
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit


# =========================================================
# 1. SETTINGS
# =========================================================

RANDOM_STATE = 42
TEST_SIZE = 0.10

INPUT_FILE = "data/processed/clean_train.csv"

OUTPUT_TRAIN = "data/group_disjoint/group_train.csv"
OUTPUT_TEST = "data/group_disjoint/group_test.csv"

os.makedirs("results", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("=" * 75)
print("GROUP-DISJOINT SPLIT BY Tweet_ID")
print("=" * 75)


# =========================================================
# 2. LOAD CLEAN TRAINING DATASET
# =========================================================

print("\nLoading cleaned training dataset...")

df = pd.read_csv(INPUT_FILE)

print("\nDataset shape:")
print(df.shape)


# =========================================================
# 3. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Tweet_ID",
    "Entity",
    "Sentiment",
    "Tweet"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing required column: {column}"
        )


# =========================================================
# 4. NORMALIZE Tweet_ID
# =========================================================

df["Tweet_ID"] = (
    df["Tweet_ID"]
    .astype(str)
    .str.strip()
)

print("\nUnique Tweet_IDs:")
print(df["Tweet_ID"].nunique())

print("\nRows:")
print(len(df))


# =========================================================
# 5. CHECK LABEL CONSISTENCY WITHIN Tweet_ID
# =========================================================

print("\n" + "=" * 75)
print("LABEL CONSISTENCY WITHIN Tweet_ID")
print("=" * 75)

labels_per_id = (
    df.groupby("Tweet_ID")["Sentiment"]
    .nunique()
)

inconsistent_ids = labels_per_id[
    labels_per_id > 1
]

print(
    "\nTweet_IDs with more than one sentiment label:",
    len(inconsistent_ids)
)

if len(inconsistent_ids) > 0:

    print(
        "\nWARNING: Some Tweet_ID groups contain "
        "multiple sentiment labels."
    )

else:

    print(
        "\nAll Tweet_ID groups have consistent labels."
    )


# =========================================================
# 6. CHECK ENTITY CONSISTENCY WITHIN Tweet_ID
# =========================================================

print("\n" + "=" * 75)
print("ENTITY CONSISTENCY WITHIN Tweet_ID")
print("=" * 75)

entities_per_id = (
    df.groupby("Tweet_ID")["Entity"]
    .nunique()
)

inconsistent_entity_ids = entities_per_id[
    entities_per_id > 1
]

print(
    "\nTweet_IDs with more than one entity:",
    len(inconsistent_entity_ids)
)

if len(inconsistent_entity_ids) == 0:

    print(
        "All Tweet_ID groups have consistent entities."
    )


# =========================================================
# 7. GROUP-DISJOINT SPLIT
# =========================================================

print("\n" + "=" * 75)
print("CREATING GROUP-DISJOINT SPLIT")
print("=" * 75)

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_indices, test_indices = next(
    splitter.split(
        X=df,
        y=df["Sentiment"],
        groups=df["Tweet_ID"]
    )
)

group_train = (
    df.iloc[train_indices]
    .copy()
    .reset_index(drop=True)
)

group_test = (
    df.iloc[test_indices]
    .copy()
    .reset_index(drop=True)
)

print("\nGroup training shape:")
print(group_train.shape)

print("\nGroup test shape:")
print(group_test.shape)


# =========================================================
# 8. Tweet_ID OVERLAP CHECK
# =========================================================

train_ids = set(
    group_train["Tweet_ID"].unique()
)

test_ids = set(
    group_test["Tweet_ID"].unique()
)

id_overlap = (
    train_ids
    .intersection(test_ids)
)

print("\n" + "=" * 75)
print("Tweet_ID LEAKAGE CHECK")
print("=" * 75)

print(
    "\nUnique training Tweet_IDs:",
    len(train_ids)
)

print(
    "Unique test Tweet_IDs:",
    len(test_ids)
)

print(
    "Tweet_ID overlap:",
    len(id_overlap)
)

if len(id_overlap) != 0:

    raise RuntimeError(
        "ERROR: Tweet_ID leakage detected!"
    )

print(
    "\nSUCCESS: Train and test are "
    "completely disjoint by Tweet_ID."
)


# =========================================================
# 9. EXACT TEXT OVERLAP CHECK
# =========================================================

train_texts = set(
    group_train["Tweet"]
    .astype(str)
    .str.strip()
)

test_texts = set(
    group_test["Tweet"]
    .astype(str)
    .str.strip()
)

text_overlap = (
    train_texts
    .intersection(test_texts)
)

print("\n" + "=" * 75)
print("EXACT TEXT LEAKAGE CHECK")
print("=" * 75)

print(
    "\nExact text overlap:",
    len(text_overlap)
)

if len(text_overlap) == 0:

    print(
        "SUCCESS: No exact tweet text overlap."
    )

else:

    print(
        "WARNING: Exact text overlap detected."
    )


# =========================================================
# 10. CLASS DISTRIBUTION
# =========================================================

print("\n" + "=" * 75)
print("CLASS DISTRIBUTION")
print("=" * 75)

train_distribution = (
    group_train["Sentiment"]
    .value_counts()
    .sort_index()
)

test_distribution = (
    group_test["Sentiment"]
    .value_counts()
    .sort_index()
)

print("\nTraining:")
print(train_distribution)

print("\nTest:")
print(test_distribution)


# =========================================================
# 11. CLASS PERCENTAGES
# =========================================================

train_percentages = (
    group_train["Sentiment"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

test_percentages = (
    group_test["Sentiment"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)

print("\nTraining percentages:")
print(
    train_percentages.round(2)
)

print("\nTest percentages:")
print(
    test_percentages.round(2)
)


# =========================================================
# 12. DISTRIBUTION DIFFERENCE
# =========================================================

all_classes = sorted(
    df["Sentiment"].unique()
)

distribution_rows = []

for label in all_classes:

    train_count = int(
        train_distribution.get(
            label,
            0
        )
    )

    test_count = int(
        test_distribution.get(
            label,
            0
        )
    )

    train_pct = float(
        train_percentages.get(
            label,
            0
        )
    )

    test_pct = float(
        test_percentages.get(
            label,
            0
        )
    )

    distribution_rows.append(
        {
            "Sentiment": label,

            "Train_Count":
                train_count,

            "Train_Percentage":
                train_pct,

            "Test_Count":
                test_count,

            "Test_Percentage":
                test_pct,

            "Absolute_Percentage_Difference":
                abs(
                    train_pct
                    - test_pct
                )
        }
    )


distribution_df = pd.DataFrame(
    distribution_rows
)

print("\nClass distribution comparison:")

print(
    distribution_df
    .round(2)
    .to_string(index=False)
)


# =========================================================
# 13. GROUP SIZE STATISTICS
# =========================================================

print("\n" + "=" * 75)
print("GROUP SIZE STATISTICS")
print("=" * 75)

train_group_sizes = (
    group_train
    .groupby("Tweet_ID")
    .size()
)

test_group_sizes = (
    group_test
    .groupby("Tweet_ID")
    .size()
)

print("\nTraining group size statistics:")

print(
    train_group_sizes
    .describe()
    .round(2)
)

print("\nTest group size statistics:")

print(
    test_group_sizes
    .describe()
    .round(2)
)


# =========================================================
# 14. SAVE DATASETS
# =========================================================

group_train.to_csv(
    OUTPUT_TRAIN,
    index=False
)

group_test.to_csv(
    OUTPUT_TEST,
    index=False
)

print("\n" + "=" * 75)
print("SAVING GROUP-DISJOINT DATASETS")
print("=" * 75)

print(
    "\nTraining dataset saved:"
)

print(
    OUTPUT_TRAIN
)

print(
    "\nTest dataset saved:"
)

print(
    OUTPUT_TEST
)


# =========================================================
# 15. SAVE DISTRIBUTION TABLE
# =========================================================

distribution_df.to_csv(
    "results/tables/"
    "group_disjoint_class_distribution.csv",
    index=False
)


# =========================================================
# 16. SAVE SPLIT SUMMARY
# =========================================================

summary_df = pd.DataFrame([
    {
        "Original_Rows":
            len(df),

        "Original_Unique_Tweet_IDs":
            df["Tweet_ID"].nunique(),

        "Training_Rows":
            len(group_train),

        "Test_Rows":
            len(group_test),

        "Training_Unique_Tweet_IDs":
            len(train_ids),

        "Test_Unique_Tweet_IDs":
            len(test_ids),

        "Tweet_ID_Overlap":
            len(id_overlap),

        "Exact_Text_Overlap":
            len(text_overlap),

        "Test_Size_Target":
            TEST_SIZE,

        "Random_State":
            RANDOM_STATE
    }
])

summary_df.to_csv(
    "results/tables/"
    "group_disjoint_split_summary.csv",
    index=False
)


# =========================================================
# 17. FINAL VALIDATION
# =========================================================

assert len(id_overlap) == 0

assert (
    len(group_train)
    + len(group_test)
    == len(df)
)

assert (
    set(group_train.index)
    is not set(group_test.index)
)


# =========================================================
# 18. FINISHED
# =========================================================

print("\n" + "=" * 75)
print("GROUP-DISJOINT SPLIT COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFinal summary:")

print(
    f"Original rows: {len(df)}"
)

print(
    f"Training rows: {len(group_train)}"
)

print(
    f"Test rows: {len(group_test)}"
)

print(
    f"Training unique Tweet_IDs: "
    f"{len(train_ids)}"
)

print(
    f"Test unique Tweet_IDs: "
    f"{len(test_ids)}"
)

print(
    f"Tweet_ID overlap: "
    f"{len(id_overlap)}"
)

print(
    f"Exact text overlap: "
    f"{len(text_overlap)}"
)

print("\nFiles created:")

print(
    "1. data/group_disjoint/group_train.csv"
)

print(
    "2. data/group_disjoint/group_test.csv"
)

print(
    "3. results/tables/"
    "group_disjoint_split_summary.csv"
)

print(
    "4. results/tables/"
    "group_disjoint_class_distribution.csv"
)