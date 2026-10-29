import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. PROJECT SETTINGS
# =========================================================

RANDOM_STATE = 42

os.makedirs("results", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)


# =========================================================
# 2. LOAD DATA
# =========================================================

columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]

train_df = pd.read_csv(
    "data/raw/twitter_training.csv",
    names=columns
)

val_df = pd.read_csv(
    "data/raw/twitter_validation.csv",
    names=columns
)


# =========================================================
# 3. BASIC INFORMATION
# =========================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("\nTraining shape:")
print(train_df.shape)

print("\nValidation shape:")
print(val_df.shape)

print("\nTraining dataset:")
print(train_df.head())

print("\nSentiment distribution - Training:")
print(train_df["Sentiment"].value_counts())

print("\nSentiment distribution - Validation:")
print(val_df["Sentiment"].value_counts())

print("\nMissing values - Training:")
print(train_df.isnull().sum())

print("\nMissing values - Validation:")
print(val_df.isnull().sum())

# =========================================================
# 4. DATA CLEANING
# =========================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# ---------------------------------------------------------
# 4.1 Remove missing tweets
# ---------------------------------------------------------

train_df = train_df.dropna(subset=["Tweet"]).copy()
val_df = val_df.dropna(subset=["Tweet"]).copy()

# ---------------------------------------------------------
# 4.2 Remove empty / whitespace-only tweets
# ---------------------------------------------------------

train_df["Tweet"] = train_df["Tweet"].astype(str).str.strip()
val_df["Tweet"] = val_df["Tweet"].astype(str).str.strip()

train_df = train_df[train_df["Tweet"] != ""].copy()
val_df = val_df[val_df["Tweet"] != ""].copy()

# ---------------------------------------------------------
# 4.3 Remove duplicate tweet texts from TRAINING
# ---------------------------------------------------------

train_df = train_df.drop_duplicates(
    subset=["Tweet"],
    keep="first"
).copy()

# ---------------------------------------------------------
# 4.4 Remove train-validation leakage
# ---------------------------------------------------------

validation_texts = set(val_df["Tweet"])

train_df = train_df[
    ~train_df["Tweet"].isin(validation_texts)
].copy()

# ---------------------------------------------------------
# 4.5 Reset indexes
# ---------------------------------------------------------

train_df.reset_index(drop=True, inplace=True)
val_df.reset_index(drop=True, inplace=True)

print("\nTraining shape after final cleaning:")
print(train_df.shape)

print("\nValidation shape:")
print(val_df.shape)

print("\nMissing values:")
print(train_df.isnull().sum())

print("\nClass distribution:")
print(train_df["Sentiment"].value_counts())

print("\nFinal cleaning completed successfully!")

# =========================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# ---------------------------------------------------------
# 5.1 Dataset Statistics Table
# ---------------------------------------------------------

class_order = ["Negative", "Neutral", "Positive", "Irrelevant"]

statistics = []

for sentiment in class_order:
    statistics.append({
        "Sentiment": sentiment,
        "Training": (train_df["Sentiment"] == sentiment).sum(),
        "Validation": (val_df["Sentiment"] == sentiment).sum()
    })

stats_df = pd.DataFrame(statistics)

stats_df.loc[len(stats_df)] = {
    "Sentiment": "TOTAL",
    "Training": len(train_df),
    "Validation": len(val_df)
}

print("\nDataset Statistics:")
print(stats_df)

stats_df.to_csv(
    "results/tables/dataset_statistics.csv",
    index=False
)


# ---------------------------------------------------------
# 5.2 Class Distribution
# ---------------------------------------------------------

train_counts = (
    train_df["Sentiment"]
    .value_counts()
    .reindex(class_order)
)

plt.figure(figsize=(8, 5))

bars = plt.bar(
    train_counts.index,
    train_counts.values
)

plt.title("Sentiment Class Distribution - Training Dataset")
plt.xlabel("Sentiment Class")
plt.ylabel("Number of Tweets")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height):,}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    "results/figures/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 5.3 Tweet Length
# ---------------------------------------------------------

train_df["tweet_length"] = train_df["Tweet"].astype(str).str.len()
train_df["word_count"] = train_df["Tweet"].astype(str).str.split().str.len()

plt.figure(figsize=(8, 5))

plt.hist(
    train_df["word_count"],
    bins=50
)

plt.title("Distribution of Tweet Lengths")
plt.xlabel("Number of Words")
plt.ylabel("Number of Tweets")

plt.tight_layout()

plt.savefig(
    "results/figures/tweet_length_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ---------------------------------------------------------
# 5.4 Descriptive Statistics
# ---------------------------------------------------------

print("\nTweet Length Statistics:")
print(train_df["word_count"].describe())

print("\nEDA completed successfully!")
print("Figures saved in: results/figures/")
print("Table saved in: results/tables/")

# =========================================================
# 6. DATA QUALITY / LEAKAGE CHECK
# =========================================================

print("\n" + "=" * 60)
print("DATA QUALITY & LEAKAGE CHECK")
print("=" * 60)

# Exact duplicate tweets inside the cleaned training set
duplicate_tweets_train = train_df.duplicated(
    subset=["Tweet"],
    keep=False
).sum()

print(f"\nDuplicate tweet texts in Training: {duplicate_tweets_train}")

# Exact duplicate tweets inside validation
duplicate_tweets_val = val_df.duplicated(
    subset=["Tweet"],
    keep=False
).sum()

print(f"Duplicate tweet texts in Validation: {duplicate_tweets_val}")

# Check whether the same tweet text occurs in both sets
train_tweets = set(train_df["Tweet"].astype(str))
val_tweets = set(val_df["Tweet"].astype(str))

overlap = train_tweets.intersection(val_tweets)

print(f"\nExact tweet overlap between Training and Validation: {len(overlap)}")

# Check Tweet_ID overlap
train_ids = set(train_df["Tweet_ID"])
val_ids = set(val_df["Tweet_ID"])

id_overlap = train_ids.intersection(val_ids)

print(f"Tweet_ID overlap between Training and Validation: {len(id_overlap)}")

print("\nData quality check completed!")

# =========================================================
# 7. SAVE FINAL CLEAN DATASETS
# =========================================================

print("\n" + "=" * 60)
print("SAVING FINAL CLEAN DATASETS")
print("=" * 60)

output_columns = ["Tweet_ID", "Entity", "Sentiment", "Tweet"]

train_df[output_columns].to_csv("data/processed/clean_train.csv", index=False)
val_df[output_columns].to_csv("data/processed/clean_validation.csv", index=False)

print("\nClean training dataset saved: data/processed/clean_train.csv")
print("Clean validation dataset saved: data/processed/clean_validation.csv")
print("\nFinal shapes:")
print("Training:", train_df[output_columns].shape)
print("Validation:", val_df[output_columns].shape)

if len(overlap) != 0:
    raise ValueError("Train-validation leakage detected.")

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 60)
