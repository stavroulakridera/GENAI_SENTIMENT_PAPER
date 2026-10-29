# =========================================================
# SGD CLASSIFIER - SENTIMENT ANALYSIS
# =========================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
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
print("SGD CLASSIFIER EXPERIMENT")
print("=" * 60)


# =========================================================
# 2. LOAD CLEAN DATASETS
# =========================================================

print("\nLoading clean datasets...")

train_df = pd.read_csv(
    "data/processed/clean_train.csv"
)

val_df = pd.read_csv(
    "data/processed/clean_validation.csv"
)

print("\nTraining shape:")
print(train_df.shape)

print("\nValidation shape:")
print(val_df.shape)


# =========================================================
# 3. DATA CHECK
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


if train_df["Tweet"].isnull().any():
    raise ValueError(
        "Missing tweets detected in training dataset."
    )

if val_df["Tweet"].isnull().any():
    raise ValueError(
        "Missing tweets detected in validation dataset."
    )

print("\nDataset check completed successfully!")

print("\nTraining class distribution:")
print(
    train_df["Sentiment"].value_counts()
)


# =========================================================
# 4. TF-IDF FEATURE EXTRACTION
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
    train_df["Tweet"].astype(str)
)

X_val = tfidf.transform(
    val_df["Tweet"].astype(str)
)

tfidf_time = (
    time.perf_counter()
    - start_tfidf
)

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
# 5. SGD CLASSIFIER
# =========================================================

print("\n" + "=" * 60)
print("MODEL: SGD CLASSIFIER")
print("=" * 60)

model = SGDClassifier(
    loss="modified_huber",
    max_iter=1000,
    tol=1e-3,
    random_state=RANDOM_STATE
)

print("\nTraining SGD Classifier...")

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
# 6. PREDICTIONS
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
# 7. PERFORMANCE METRICS
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
print("SGD CLASSIFIER RESULTS")
print("=" * 60)

print(
    f"Accuracy:        {accuracy:.4f}"
)

print(
    f"Macro Precision: {macro_precision:.4f}"
)

print(
    f"Macro Recall:    {macro_recall:.4f}"
)

print(
    f"Macro F1:        {macro_f1:.4f}"
)

print(
    f"Weighted F1:     {weighted_f1:.4f}"
)

print(
    f"Training Time:   {training_time:.4f} sec"
)

print(
    f"Prediction Time: {prediction_time:.4f} sec"
)


# =========================================================
# 8. CLASSIFICATION REPORT
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
# 9. SAVE SUMMARY RESULTS
# =========================================================

results_df = pd.DataFrame(
    [
        {
            "Model": "SGD Classifier",
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
    "results/tables/sgd_classifier_results.csv",
    index=False
)


# =========================================================
# 10. SAVE PREDICTIONS
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
    "results/tables/sgd_classifier_predictions.csv",
    index=False
)


# =========================================================
# 11. CONFUSION MATRIX
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
    "Confusion Matrix - SGD Classifier"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "True Label"
)

plt.xticks(
    rotation=30
)

plt.tight_layout()

plt.savefig(
    "results/confusion_matrices/sgd_classifier_cm.png",
    dpi=300,
    bbox_inches="tight"
)

# Do not use plt.show() on macOS
plt.close()


# =========================================================
# 12. FINISHED
# =========================================================

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nFiles created:")

print(
    "1. results/tables/sgd_classifier_results.csv"
)

print(
    "2. results/tables/sgd_classifier_predictions.csv"
)

print(
    "3. results/confusion_matrices/sgd_classifier_cm.png"
)