# =========================================================
# TWEET_ID LEAKAGE AUDIT
# =========================================================

import os
import pandas as pd
import numpy as np
from difflib import SequenceMatcher


# =========================================================
# 1. SETTINGS
# =========================================================

TRAIN_FILE = "data/processed/clean_train.csv"
VALIDATION_FILE = "data/processed/clean_validation.csv"

OUTPUT_DIR = "results/tables"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 75)
print("TWEET_ID LEAKAGE AUDIT")
print("=" * 75)


# =========================================================
# 2. LOAD DATASETS
# =========================================================

train_df = pd.read_csv(TRAIN_FILE)
val_df = pd.read_csv(VALIDATION_FILE)

print("\nDataset shapes:")
print("Training:", train_df.shape)
print("Validation:", val_df.shape)


# =========================================================
# 3. BASIC CHECKS
# =========================================================

required_columns = [
    "Tweet_ID",
    "Entity",
    "Sentiment",
    "Tweet"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing column in training dataset: {column}"
        )

    if column not in val_df.columns:
        raise ValueError(
            f"Missing column in validation dataset: {column}"
        )


# =========================================================
# 4. NORMALIZE Tweet_ID TYPE
# =========================================================

train_df["Tweet_ID"] = (
    train_df["Tweet_ID"]
    .astype(str)
    .str.strip()
)

val_df["Tweet_ID"] = (
    val_df["Tweet_ID"]
    .astype(str)
    .str.strip()
)


# =========================================================
# 5. UNIQUE Tweet_ID COUNTS
# =========================================================

train_unique_ids = set(
    train_df["Tweet_ID"].unique()
)

val_unique_ids = set(
    val_df["Tweet_ID"].unique()
)

common_ids = (
    train_unique_ids
    .intersection(val_unique_ids)
)


print("\n" + "=" * 75)
print("TWEET_ID OVERLAP")
print("=" * 75)

print(
    f"\nUnique Tweet_IDs in Training: "
    f"{len(train_unique_ids)}"
)

print(
    f"Unique Tweet_IDs in Validation: "
    f"{len(val_unique_ids)}"
)

print(
    f"Common Tweet_IDs: "
    f"{len(common_ids)}"
)

validation_overlap_percentage = (
    len(common_ids)
    / len(val_unique_ids)
    * 100
)

print(
    f"Validation Tweet_ID overlap: "
    f"{validation_overlap_percentage:.2f}%"
)


# =========================================================
# 6. ROWS ASSOCIATED WITH COMMON IDs
# =========================================================

train_overlap = train_df[
    train_df["Tweet_ID"].isin(common_ids)
].copy()

val_overlap = val_df[
    val_df["Tweet_ID"].isin(common_ids)
].copy()

print("\nRows associated with common Tweet_IDs:")

print(
    "Training rows:",
    len(train_overlap)
)

print(
    "Validation rows:",
    len(val_overlap)
)


# =========================================================
# 7. MERGE TRAIN AND VALIDATION BY Tweet_ID
# =========================================================

paired = train_overlap.merge(
    val_overlap,
    on="Tweet_ID",
    how="inner",
    suffixes=(
        "_train",
        "_validation"
    )
)

print(
    "\nTrain-validation pairs generated:",
    len(paired)
)


# =========================================================
# 8. EXACT TEXT MATCH CHECK
# =========================================================

paired["Exact_Text_Match"] = (
    paired["Tweet_train"]
    .astype(str)
    .str.strip()
    ==
    paired["Tweet_validation"]
    .astype(str)
    .str.strip()
)

exact_matches = int(
    paired["Exact_Text_Match"].sum()
)

print("\n" + "=" * 75)
print("EXACT TEXT CHECK")
print("=" * 75)

print(
    f"\nExact text matches among "
    f"same-ID pairs: {exact_matches}"
)


# =========================================================
# 9. ENTITY AGREEMENT
# =========================================================

paired["Same_Entity"] = (
    paired["Entity_train"]
    .astype(str)
    .str.strip()
    ==
    paired["Entity_validation"]
    .astype(str)
    .str.strip()
)

same_entity_count = int(
    paired["Same_Entity"].sum()
)

same_entity_percentage = (
    paired["Same_Entity"].mean()
    * 100
)

print("\n" + "=" * 75)
print("ENTITY AGREEMENT")
print("=" * 75)

print(
    f"\nSame entity pairs: "
    f"{same_entity_count}"
)

print(
    f"Same entity percentage: "
    f"{same_entity_percentage:.2f}%"
)


# =========================================================
# 10. LABEL AGREEMENT
# =========================================================

paired["Same_Label"] = (
    paired["Sentiment_train"]
    .astype(str)
    .str.strip()
    ==
    paired["Sentiment_validation"]
    .astype(str)
    .str.strip()
)

same_label_count = int(
    paired["Same_Label"].sum()
)

same_label_percentage = (
    paired["Same_Label"].mean()
    * 100
)

print("\n" + "=" * 75)
print("LABEL AGREEMENT")
print("=" * 75)

print(
    f"\nSame-label pairs: "
    f"{same_label_count}"
)

print(
    f"Same-label percentage: "
    f"{same_label_percentage:.2f}%"
)


# =========================================================
# 11. TEXT SIMILARITY
# =========================================================

print("\n" + "=" * 75)
print("TEXT SIMILARITY ANALYSIS")
print("=" * 75)

print(
    "\nCalculating similarity for same-ID pairs..."
)


def text_similarity(text1, text2):

    text1 = str(text1).lower().strip()
    text2 = str(text2).lower().strip()

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


paired["Text_Similarity"] = paired.apply(
    lambda row: text_similarity(
        row["Tweet_train"],
        row["Tweet_validation"]
    ),
    axis=1
)


print("\nText similarity statistics:")

print(
    paired["Text_Similarity"]
    .describe()
    .round(4)
)


# =========================================================
# 12. SIMILARITY THRESHOLDS
# =========================================================

thresholds = [
    0.50,
    0.70,
    0.80,
    0.90,
    0.95
]

print("\nSimilarity thresholds:")

threshold_results = []

for threshold in thresholds:

    count = int(
        (
            paired["Text_Similarity"]
            >= threshold
        ).sum()
    )

    percentage = (
        count
        / len(paired)
        * 100
    )

    print(
        f"Similarity >= {threshold:.2f}: "
        f"{count} pairs "
        f"({percentage:.2f}%)"
    )

    threshold_results.append(
        {
            "Threshold": threshold,
            "Pairs": count,
            "Percentage": percentage
        }
    )


# =========================================================
# 13. HIGH-SIMILARITY + SAME-LABEL PAIRS
# =========================================================

high_similarity = paired[
    paired["Text_Similarity"] >= 0.90
].copy()

high_similarity_same_label = high_similarity[
    high_similarity["Same_Label"]
].copy()

print("\n" + "=" * 75)
print("HIGH-SIMILARITY ANALYSIS")
print("=" * 75)

print(
    "\nPairs with similarity >= 0.90:",
    len(high_similarity)
)

print(
    "High-similarity pairs with same label:",
    len(high_similarity_same_label)
)

if len(high_similarity) > 0:

    high_similarity_label_agreement = (
        high_similarity["Same_Label"]
        .mean()
        * 100
    )

else:

    high_similarity_label_agreement = 0.0


print(
    "Label agreement among high-similarity pairs: "
    f"{high_similarity_label_agreement:.2f}%"
)


# =========================================================
# 14. DUPLICATE IDs WITHIN EACH DATASET
# =========================================================

train_id_counts = (
    train_df["Tweet_ID"]
    .value_counts()
)

val_id_counts = (
    val_df["Tweet_ID"]
    .value_counts()
)

train_duplicate_ids = train_id_counts[
    train_id_counts > 1
]

val_duplicate_ids = val_id_counts[
    val_id_counts > 1
]


print("\n" + "=" * 75)
print("WITHIN-DATASET Tweet_ID DUPLICATION")
print("=" * 75)

print(
    "\nTraining IDs occurring more than once:",
    len(train_duplicate_ids)
)

print(
    "Validation IDs occurring more than once:",
    len(val_duplicate_ids)
)

if len(train_duplicate_ids) > 0:

    print(
        "Maximum training rows for one Tweet_ID:",
        int(train_duplicate_ids.max())
    )

if len(val_duplicate_ids) > 0:

    print(
        "Maximum validation rows for one Tweet_ID:",
        int(val_duplicate_ids.max())
    )


# =========================================================
# 15. UNIQUE-ID LEVEL SUMMARY
# =========================================================

id_summary = []

for tweet_id in common_ids:

    train_group = train_overlap[
        train_overlap["Tweet_ID"] == tweet_id
    ]

    val_group = val_overlap[
        val_overlap["Tweet_ID"] == tweet_id
    ]

    train_labels = sorted(
        train_group["Sentiment"]
        .astype(str)
        .unique()
        .tolist()
    )

    val_labels = sorted(
        val_group["Sentiment"]
        .astype(str)
        .unique()
        .tolist()
    )

    train_entities = sorted(
        train_group["Entity"]
        .astype(str)
        .unique()
        .tolist()
    )

    val_entities = sorted(
        val_group["Entity"]
        .astype(str)
        .unique()
        .tolist()
    )

    id_summary.append(
        {
            "Tweet_ID": tweet_id,

            "Training_Rows":
                len(train_group),

            "Validation_Rows":
                len(val_group),

            "Training_Labels":
                " | ".join(train_labels),

            "Validation_Labels":
                " | ".join(val_labels),

            "Training_Entities":
                " | ".join(train_entities),

            "Validation_Entities":
                " | ".join(val_entities)
        }
    )


id_summary_df = pd.DataFrame(
    id_summary
)


# =========================================================
# 16. SAVE DETAILED RESULTS
# =========================================================

paired = paired.sort_values(
    by="Text_Similarity",
    ascending=False
)

paired.to_csv(
    "results/tables/"
    "tweet_id_overlap_pairs.csv",
    index=False
)

id_summary_df.to_csv(
    "results/tables/"
    "tweet_id_overlap_summary.csv",
    index=False
)

pd.DataFrame(
    threshold_results
).to_csv(
    "results/tables/"
    "tweet_id_similarity_thresholds.csv",
    index=False
)


# =========================================================
# 17. SAVE COMPACT AUDIT SUMMARY
# =========================================================

audit_summary = pd.DataFrame([
    {
        "Training_Rows":
            len(train_df),

        "Validation_Rows":
            len(val_df),

        "Training_Unique_Tweet_IDs":
            len(train_unique_ids),

        "Validation_Unique_Tweet_IDs":
            len(val_unique_ids),

        "Common_Tweet_IDs":
            len(common_ids),

        "Validation_ID_Overlap_Percentage":
            validation_overlap_percentage,

        "Same_ID_Pairs":
            len(paired),

        "Exact_Text_Matches":
            exact_matches,

        "Same_Entity_Percentage":
            same_entity_percentage,

        "Same_Label_Percentage":
            same_label_percentage,

        "Mean_Text_Similarity":
            paired["Text_Similarity"].mean(),

        "Median_Text_Similarity":
            paired["Text_Similarity"].median(),

        "Similarity_90_Pairs":
            len(high_similarity),

        "Similarity_90_Same_Label":
            len(high_similarity_same_label),

        "Similarity_90_Label_Agreement":
            high_similarity_label_agreement
    }
])

audit_summary.to_csv(
    "results/tables/"
    "tweet_id_leakage_audit_summary.csv",
    index=False
)


# =========================================================
# 18. AUTOMATIC RISK INDICATORS
# =========================================================

print("\n" + "=" * 75)
print("LEAKAGE RISK INDICATORS")
print("=" * 75)

high_sim_percentage = (
    len(high_similarity)
    / len(paired)
    * 100
)

print(
    f"\nValidation ID overlap: "
    f"{validation_overlap_percentage:.2f}%"
)

print(
    f"Mean same-ID text similarity: "
    f"{paired['Text_Similarity'].mean():.4f}"
)

print(
    f"Pairs with similarity >= 0.90: "
    f"{high_sim_percentage:.2f}%"
)

print(
    f"Same-ID label agreement: "
    f"{same_label_percentage:.2f}%"
)

print(
    f"High-similarity label agreement: "
    f"{high_similarity_label_agreement:.2f}%"
)


# =========================================================
# 19. FINISHED
# =========================================================

print("\n" + "=" * 75)
print("TWEET_ID LEAKAGE AUDIT COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(
    "1. results/tables/"
    "tweet_id_leakage_audit_summary.csv"
)

print(
    "2. results/tables/"
    "tweet_id_overlap_pairs.csv"
)

print(
    "3. results/tables/"
    "tweet_id_overlap_summary.csv"
)

print(
    "4. results/tables/"
    "tweet_id_similarity_thresholds.csv"
)