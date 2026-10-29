# =========================================================
# GROUP-DISJOINT LINEAR SVM
# =========================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

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

TRAIN_FILE = "data/group_disjoint/group_train.csv"
TEST_FILE = "data/group_disjoint/group_test.csv"

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)

print("=" * 75)
print("GROUP-DISJOINT LINEAR SVM EXPERIMENT")
print("=" * 75)


# =========================================================
# 2. LOAD GROUP-DISJOINT DATASETS
# =========================================================

print("\nLoading group-disjoint datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print("\nTraining shape:")
print(train_df.shape)

print("\nTest shape:")
print(test_df.shape)


# =========================================================
# 3. DATA VALIDATION
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
            f"Missing training column: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing test column: {column}"
        )


# =========================================================
# 4. VERIFY GROUP DISJOINTNESS AGAIN
# =========================================================

train_ids = set(
    train_df["Tweet_ID"]
    .astype(str)
)

test_ids = set(
    test_df["Tweet_ID"]
    .astype(str)
)

id_overlap = train_ids.intersection(
    test_ids
)

train_texts_set = set(
    train_df["Tweet"]
    .astype(str)
    .str.strip()
)

test_texts_set = set(
    test_df["Tweet"]
    .astype(str)
    .str.strip()
)

text_overlap = train_texts_set.intersection(
    test_texts_set
)

print("\n" + "=" * 75)
print("LEAKAGE CHECK")
print("=" * 75)

print(
    f"\nTweet_ID overlap: {len(id_overlap)}"
)

print(
    f"Exact text overlap: {len(text_overlap)}"
)

if len(id_overlap) != 0:
    raise RuntimeError(
        "Tweet_ID leakage detected."
    )

if len(text_overlap) != 0:
    raise RuntimeError(
        "Exact text leakage detected."
    )

print(
    "\nSUCCESS: Group-disjoint evaluation confirmed."
)


# =========================================================
# 5. PREPARE DATA
# =========================================================

X_train_text = (
    train_df["Tweet"]
    .astype(str)
)

X_test_text = (
    test_df["Tweet"]
    .astype(str)
)

y_train = (
    train_df["Sentiment"]
    .astype(str)
)

y_test = (
    test_df["Sentiment"]
    .astype(str)
)


# =========================================================
# 6. TF-IDF
# SAME CONFIGURATION AS ORIGINAL ML EXPERIMENT
# =========================================================

print("\n" + "=" * 75)
print("TF-IDF VECTORIZATION")
print("=" * 75)

tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)

print(
    "\nFitting TF-IDF ONLY on group training data..."
)

X_train = tfidf.fit_transform(
    X_train_text
)

X_test = tfidf.transform(
    X_test_text
)

print(
    "\nTraining TF-IDF matrix:",
    X_train.shape
)

print(
    "Test TF-IDF matrix:",
    X_test.shape
)


# =========================================================
# 7. BUILD LINEAR SVM
# SAME CONFIGURATION AS ORIGINAL EXPERIMENT
# =========================================================

print("\n" + "=" * 75)
print("TRAINING LINEAR SVM")
print("=" * 75)

model = LinearSVC(
    C=1.0,
    random_state=RANDOM_STATE,
    max_iter=5000
)


# =========================================================
# 8. TRAINING
# =========================================================

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
    f"\nTraining Time: "
    f"{training_time:.4f} sec"
)


# =========================================================
# 9. PREDICTION
# =========================================================

print("\nMaking predictions...")

start_prediction = time.perf_counter()

predictions = model.predict(
    X_test
)

prediction_time = (
    time.perf_counter()
    - start_prediction
)

print(
    f"Prediction Time: "
    f"{prediction_time:.4f} sec"
)


# =========================================================
# 10. METRICS
# =========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

macro_precision = precision_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_test,
    predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)


# =========================================================
# 11. DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 75)
print("GROUP-DISJOINT LINEAR SVM RESULTS")
print("=" * 75)

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
# 12. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        predictions,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 13. SAVE SUMMARY RESULTS
# =========================================================

results_df = pd.DataFrame([
    {
        "Model": "Linear SVM",
        "Protocol": "Group-Disjoint",
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time,
        "Train Samples": len(train_df),
        "Test Samples": len(test_df),
        "Train Tweet_IDs": train_df["Tweet_ID"].nunique(),
        "Test Tweet_IDs": test_df["Tweet_ID"].nunique(),
        "Tweet_ID Overlap": len(id_overlap),
        "Exact Text Overlap": len(text_overlap)
    }
])

results_df.to_csv(
    "results/tables/"
    "group_linear_svm_results.csv",
    index=False
)


# =========================================================
# 14. SAVE PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame({
    "Tweet_ID":
        test_df["Tweet_ID"].values,

    "Entity":
        test_df["Entity"].values,

    "Tweet":
        test_df["Tweet"].values,

    "True_Label":
        y_test.values,

    "Predicted_Label":
        predictions,

    "Correct":
        (
            y_test.values
            == predictions
        )
})

predictions_df.to_csv(
    "results/tables/"
    "group_linear_svm_predictions.csv",
    index=False
)


# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

labels = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

cm = confusion_matrix(
    y_test,
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
    "Group-Disjoint Confusion Matrix - Linear SVM"
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
    "results/confusion_matrices/"
    "group_linear_svm_cm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 16. ORIGINAL VS GROUP-DISJOINT
# =========================================================

ORIGINAL_ACCURACY = 0.9010
ORIGINAL_MACRO_F1 = 0.8993

accuracy_change = (
    accuracy
    - ORIGINAL_ACCURACY
)

macro_f1_change = (
    macro_f1
    - ORIGINAL_MACRO_F1
)

accuracy_relative_change = (
    accuracy_change
    / ORIGINAL_ACCURACY
) * 100

macro_f1_relative_change = (
    macro_f1_change
    / ORIGINAL_MACRO_F1
) * 100


print("\n" + "=" * 75)
print("STANDARD VS GROUP-DISJOINT")
print("=" * 75)

print("\nSTANDARD SPLIT")

print(
    f"Accuracy: {ORIGINAL_ACCURACY:.4f}"
)

print(
    f"Macro F1: {ORIGINAL_MACRO_F1:.4f}"
)


print("\nGROUP-DISJOINT SPLIT")

print(
    f"Accuracy: {accuracy:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


print("\nCHANGE")

print(
    f"Accuracy change: "
    f"{accuracy_change:+.4f}"
)

print(
    f"Macro F1 change: "
    f"{macro_f1_change:+.4f}"
)

print(
    f"Relative Accuracy change: "
    f"{accuracy_relative_change:+.2f}%"
)

print(
    f"Relative Macro F1 change: "
    f"{macro_f1_relative_change:+.2f}%"
)


# =========================================================
# 17. SAVE COMPARISON
# =========================================================

comparison_df = pd.DataFrame([
    {
        "Model": "Linear SVM",
        "Protocol": "Standard",
        "Accuracy": ORIGINAL_ACCURACY,
        "Macro F1": ORIGINAL_MACRO_F1
    },
    {
        "Model": "Linear SVM",
        "Protocol": "Group-Disjoint",
        "Accuracy": accuracy,
        "Macro F1": macro_f1
    }
])

comparison_df.to_csv(
    "results/tables/"
    "linear_svm_protocol_comparison.csv",
    index=False
)


# =========================================================
# 18. FINISHED
# =========================================================

print("\n" + "=" * 75)
print("GROUP-DISJOINT LINEAR SVM COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(
    "1. results/tables/"
    "group_linear_svm_results.csv"
)

print(
    "2. results/tables/"
    "group_linear_svm_predictions.csv"
)

print(
    "3. results/tables/"
    "linear_svm_protocol_comparison.csv"
)

print(
    "4. results/confusion_matrices/"
    "group_linear_svm_cm.png"
)