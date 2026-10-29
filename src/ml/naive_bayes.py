# =========================================================
# NAIVE BAYES SENTIMENT ANALYSIS
# =========================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# =========================================================
# 1. SETTINGS
# =========================================================

RANDOM_STATE = 42

os.makedirs("results", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)


print("=" * 60)
print("NAIVE BAYES EXPERIMENT")
print("=" * 60)


# =========================================================
# 2. LOAD ORIGINAL DATASETS
# =========================================================

columns = [
    "Tweet_ID",
    "Entity",
    "Sentiment",
    "Tweet"
]

train_df = pd.read_csv(
    "twitter_training.csv",
    names=columns
)

val_df = pd.read_csv(
    "twitter_validation.csv",
    names=columns
)

print("\nOriginal Training shape:")
print(train_df.shape)

print("\nOriginal Validation shape:")
print(val_df.shape)


# =========================================================
# 3. DATA CLEANING
# =========================================================

print("\n" + "=" * 60)
print("DATA CLEANING")
print("=" * 60)

# Remove missing tweets
train_df = train_df.dropna(
    subset=["Tweet"]
).copy()

val_df = val_df.dropna(
    subset=["Tweet"]
).copy()


# Convert tweets to strings and remove surrounding whitespace
train_df["Tweet"] = (
    train_df["Tweet"]
    .astype(str)
    .str.strip()
)

val_df["Tweet"] = (
    val_df["Tweet"]
    .astype(str)
    .str.strip()
)


# Remove empty tweets
train_df = train_df[
    train_df["Tweet"] != ""
].copy()

val_df = val_df[
    val_df["Tweet"] != ""
].copy()


# Remove duplicate tweet texts from training
train_df = train_df.drop_duplicates(
    subset=["Tweet"],
    keep="first"
).copy()


# =========================================================
# 4. REMOVE TRAIN-VALIDATION TEXT LEAKAGE
# =========================================================

validation_texts = set(
    val_df["Tweet"]
)

train_df = train_df[
    ~train_df["Tweet"].isin(validation_texts)
].copy()


# Reset indexes
train_df.reset_index(
    drop=True,
    inplace=True
)

val_df.reset_index(
    drop=True,
    inplace=True
)


print("\nClean Training shape:")
print(train_df.shape)

print("\nValidation shape:")
print(val_df.shape)

print("\nTraining class distribution:")
print(train_df["Sentiment"].value_counts())


# =========================================================
# 5. FINAL DATA QUALITY CHECK
# =========================================================

train_texts = set(train_df["Tweet"])
validation_texts = set(val_df["Tweet"])

text_overlap = train_texts.intersection(
    validation_texts
)

print("\nExact tweet overlap between Training and Validation:")
print(len(text_overlap))

if len(text_overlap) != 0:
    raise ValueError(
        "ERROR: Train-validation leakage still exists!"
    )

print("\nData cleaning completed successfully!")


# =========================================================
# 6. SAVE CLEAN DATASETS
# =========================================================

train_df.to_csv(
    "data/processed/clean_train.csv",
    index=False
)

val_df.to_csv(
    "data/processed/clean_validation.csv",
    index=False
)

print("\nClean datasets saved:")
print("data/processed/clean_train.csv")
print("data/processed/clean_validation.csv")


# =========================================================
# 7. TF-IDF FEATURE EXTRACTION
# =========================================================

print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 60)

print("\nCreating TF-IDF features...")


tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)


start_tfidf = time.perf_counter()


X_train = tfidf.fit_transform(
    train_df["Tweet"]
)

X_val = tfidf.transform(
    val_df["Tweet"]
)


tfidf_time = time.perf_counter() - start_tfidf


y_train = train_df["Sentiment"]
y_val = val_df["Sentiment"]


print("\nTF-IDF completed!")

print("Training matrix:")
print(X_train.shape)

print("Validation matrix:")
print(X_val.shape)

print(
    f"TF-IDF Time: {tfidf_time:.4f} sec"
)


# =========================================================
# 8. MULTINOMIAL NAIVE BAYES
# =========================================================

print("\n" + "=" * 60)
print("MODEL: MULTINOMIAL NAIVE BAYES")
print("=" * 60)

model = MultinomialNB()


print("\nTraining Naive Bayes...")


start_training = time.perf_counter()

model.fit(
    X_train,
    y_train
)

training_time = (
    time.perf_counter()
    - start_training
)


print(
    f"Training completed in "
    f"{training_time:.4f} sec"
)


# =========================================================
# 9. PREDICTIONS
# =========================================================

print("\nMaking predictions...")


start_prediction = time.perf_counter()

predictions = model.predict(
    X_val
)

prediction_time = (
    time.perf_counter()
    - start_prediction
)


print(
    f"Predictions completed in "
    f"{prediction_time:.4f} sec"
)


# =========================================================
# 10. PERFORMANCE METRICS
# =========================================================

accuracy = accuracy_score(
    y_val,
    predictions
)


macro_precision = precision_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    y_val,
    predictions,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    y_val,
    predictions,
    average="weighted",
    zero_division=0
)


print("\n" + "=" * 60)
print("NAIVE BAYES RESULTS")
print("=" * 60)

print(
    f"Accuracy:        "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision: "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall:    "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1:        "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1:     "
    f"{weighted_f1:.4f}"
)

print(
    f"Training Time:   "
    f"{training_time:.4f} sec"
)

print(
    f"Prediction Time: "
    f"{prediction_time:.4f} sec"
)


# =========================================================
# 11. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_val,
        predictions,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 12. SAVE SUMMARY RESULTS
# =========================================================

results_df = pd.DataFrame(
    [
        {
            "Model": "Multinomial Naive Bayes",
            "Accuracy": accuracy,
            "Macro Precision": macro_precision,
            "Macro Recall": macro_recall,
            "Macro F1": macro_f1,
            "Weighted F1": weighted_f1,
            "Training Time (s)": training_time,
            "Prediction Time (s)": prediction_time
        }
    ]
)


results_df.to_csv(
    "results/tables/naive_bayes_results.csv",
    index=False
)


# =========================================================
# 13. SAVE PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame(
    {
        "Tweet_ID": val_df["Tweet_ID"].values,
        "Entity": val_df["Entity"].values,
        "Tweet": val_df["Tweet"].values,
        "True_Label": y_val.values,
        "Predicted_Label": predictions
    }
)


predictions_df.to_csv(
    "results/tables/naive_bayes_predictions.csv",
    index=False
)


# =========================================================
# 14. CONFUSION MATRIX
# =========================================================

labels = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]


cm = confusion_matrix(
    y_val,
    predictions,
    labels=labels
)


fig, ax = plt.subplots(
    figsize=(7, 6)
)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
)


disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)


plt.title(
    "Confusion Matrix - Multinomial Naive Bayes"
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    rotation=30
)

plt.tight_layout()


plt.savefig(
    "results/confusion_matrices/naive_bayes_cm.png",
    dpi=300,
    bbox_inches="tight"
)


# Do NOT use plt.show()
# It can block execution on macOS.
plt.close()


# =========================================================
# 15. FINISHED
# =========================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")

print(
    "1. data/processed/clean_train.csv"
)

print(
    "2. data/processed/clean_validation.csv"
)

print(
    "3. results/tables/naive_bayes_results.csv"
)

print(
    "4. results/tables/naive_bayes_predictions.csv"
)

print(
    "5. results/confusion_matrices/naive_bayes_cm.png"
)