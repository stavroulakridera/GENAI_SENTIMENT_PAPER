import os
import pandas as pd

# =========================================================
# 1. LOAD CLEAN DATA
# =========================================================

print("=" * 60)
print("CREATING ENTITY + TWEET DATASET")
print("=" * 60)

train_df = pd.read_csv("results/clean_train.csv")
validation_df = pd.read_csv("results/clean_validation.csv")

print("\nOriginal training shape:", train_df.shape)
print("Original validation shape:", validation_df.shape)

# =========================================================
# 2. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = ["Entity", "Tweet", "Sentiment"]

for column in required_columns:
    if column not in train_df.columns:
        raise ValueError(f"Missing column in training data: {column}")

    if column not in validation_df.columns:
        raise ValueError(f"Missing column in validation data: {column}")

# =========================================================
# 3. CREATE ENTITY + TWEET INPUT
# =========================================================

train_df["Entity_Tweet"] = (
    train_df["Entity"].astype(str).str.strip()
    + " [SEP] "
    + train_df["Tweet"].astype(str).str.strip()
)

validation_df["Entity_Tweet"] = (
    validation_df["Entity"].astype(str).str.strip()
    + " [SEP] "
    + validation_df["Tweet"].astype(str).str.strip()
)

# =========================================================
# 4. BASIC CHECKS
# =========================================================

print("\nExample:")
print("Entity:", train_df.iloc[0]["Entity"])
print("Tweet:", train_df.iloc[0]["Tweet"])
print("Combined:")
print(train_df.iloc[0]["Entity_Tweet"])

print("\nMissing combined inputs:")
print("Training:", train_df["Entity_Tweet"].isna().sum())
print("Validation:", validation_df["Entity_Tweet"].isna().sum())

# =========================================================
# 5. SAVE NEW DATASETS
# =========================================================

os.makedirs("results", exist_ok=True)

train_path = "results/entity_tweet_train.csv"
validation_path = "results/entity_tweet_validation.csv"

train_df.to_csv(train_path, index=False)
validation_df.to_csv(validation_path, index=False)

print("\nSaved:")
print(train_path)
print(validation_path)

print("\nFinal training shape:", train_df.shape)
print("Final validation shape:", validation_df.shape)

print("\nEntity + Tweet datasets created successfully!")