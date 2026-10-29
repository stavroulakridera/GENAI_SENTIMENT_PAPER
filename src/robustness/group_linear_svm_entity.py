import os
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)


# =========================================================
# 1. SETTINGS
# =========================================================

RANDOM_STATE = 42

TRAIN_FILE = "data/group_disjoint/group_train.csv"
TEST_FILE = "data/group_disjoint/group_test.csv"

RESULTS_DIR = "results/tables"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "group_linear_svm_entity_results.csv"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "group_linear_svm_entity_predictions.csv"
)

CONFUSION_FILE = os.path.join(
    RESULTS_DIR,
    "group_linear_svm_entity_confusion_matrix.csv"
)


print("=" * 70)
print("GROUP-DISJOINT ROBUSTNESS ANALYSIS")
print("LINEAR SVM — ENTITY + TWEET")
print("=" * 70)


# =========================================================
# 2. LOAD GROUP-DISJOINT DATA
# =========================================================

print("\nLoading group-disjoint datasets...")

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

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


print(
    "Training shape:",
    train_df.shape
)

print(
    "Test shape:",
    test_df.shape
)


# =========================================================
# 3. VERIFY GROUP DISJOINTNESS
# =========================================================

train_ids = set(
    train_df["Tweet_ID"]
)

test_ids = set(
    test_df["Tweet_ID"]
)

common_ids = (
    train_ids.intersection(
        test_ids
    )
)


train_texts = set(
    train_df["Tweet"].astype(str)
)

test_texts = set(
    test_df["Tweet"].astype(str)
)

common_texts = (
    train_texts.intersection(
        test_texts
    )
)


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
    len(common_ids)
)

print(
    "Exact Tweet text overlap:",
    len(common_texts)
)


if len(common_ids) != 0:

    raise RuntimeError(
        "ERROR: Tweet_ID groups are not disjoint."
    )


if len(common_texts) != 0:

    raise RuntimeError(
        "ERROR: Exact tweet text overlap detected."
    )


print(
    "\nGroup-disjoint verification PASSED."
)


# =========================================================
# 4. CREATE ENTITY + TWEET INPUT
# =========================================================

train_df["Entity_Tweet"] = (
    train_df["Entity"]
    .astype(str)
    .str.strip()
    +
    " [SEP] "
    +
    train_df["Tweet"]
    .astype(str)
    .str.strip()
)

test_df["Entity_Tweet"] = (
    test_df["Entity"]
    .astype(str)
    .str.strip()
    +
    " [SEP] "
    +
    test_df["Tweet"]
    .astype(str)
    .str.strip()
)


print(
    "\nExample Entity + Tweet input:"
)

print(
    train_df[
        "Entity_Tweet"
    ].iloc[0]
)


# =========================================================
# 5. PREPARE LABELS
# =========================================================

X_train_text = (
    train_df[
        "Entity_Tweet"
    ].astype(str)
)

X_test_text = (
    test_df[
        "Entity_Tweet"
    ].astype(str)
)

y_train = (
    train_df[
        "Sentiment"
    ].astype(str)
)

y_test = (
    test_df[
        "Sentiment"
    ].astype(str)
)


print(
    "\nTraining class distribution:"
)

print(
    y_train.value_counts()
)


print(
    "\nTest class distribution:"
)

print(
    y_test.value_counts()
)


# =========================================================
# 6. TF-IDF
# =========================================================

print(
    "\nCreating TF-IDF representation..."
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)


X_train = vectorizer.fit_transform(
    X_train_text
)

X_test = vectorizer.transform(
    X_test_text
)


print(
    "Training TF-IDF shape:",
    X_train.shape
)

print(
    "Test TF-IDF shape:",
    X_test.shape
)


# =========================================================
# 7. LINEAR SVM
# =========================================================

print(
    "\nTraining Linear SVM..."
)

model = LinearSVC(
    C=1.0,
    random_state=RANDOM_STATE,
    max_iter=5000
)


training_start = time.perf_counter()

model.fit(
    X_train,
    y_train
)

training_time = (
    time.perf_counter()
    - training_start
)


print(
    "Training completed."
)


# =========================================================
# 8. PREDICTION
# =========================================================

prediction_start = time.perf_counter()

predictions = model.predict(
    X_test
)

prediction_time = (
    time.perf_counter()
    - prediction_start
)


# =========================================================
# 9. METRICS
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

balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        predictions
    )
)

mcc = matthews_corrcoef(
    y_test,
    predictions
)


# =========================================================
# 10. PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("GROUP-DISJOINT LINEAR SVM RESULTS")
print("=" * 70)

print(
    f"Accuracy:          "
    f"{accuracy:.4f}"
)

print(
    f"Macro Precision:   "
    f"{macro_precision:.4f}"
)

print(
    f"Macro Recall:      "
    f"{macro_recall:.4f}"
)

print(
    f"Macro F1:          "
    f"{macro_f1:.4f}"
)

print(
    f"Weighted F1:       "
    f"{weighted_f1:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy:.4f}"
)

print(
    f"MCC:               "
    f"{mcc:.4f}"
)

print(
    f"\nTraining Time:     "
    f"{training_time:.4f} sec"
)

print(
    f"Prediction Time:   "
    f"{prediction_time:.4f} sec"
)


# =========================================================
# 11. CLASSIFICATION REPORT
# =========================================================

labels = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        predictions,
        labels=labels,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 12. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_test,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"True_{label}"
        for label in labels
    ],
    columns=[
        f"Pred_{label}"
        for label in labels
    ]
)

cm_df.to_csv(
    CONFUSION_FILE
)


# =========================================================
# 13. SAVE PREDICTIONS
# =========================================================

prediction_df = pd.DataFrame({

    "Tweet_ID":
        test_df[
            "Tweet_ID"
        ].values,

    "Entity":
        test_df[
            "Entity"
        ].values,

    "Tweet":
        test_df[
            "Tweet"
        ].values,

    "True_Label":
        y_test.values,

    "Predicted_Label":
        predictions
})


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# =========================================================
# 14. SAVE SUMMARY
# =========================================================

result_df = pd.DataFrame([{

    "Model":
        "Linear SVM",

    "Evaluation":
        "Group-disjoint",

    "Input":
        "Entity + Tweet",

    "Train Samples":
        len(train_df),

    "Test Samples":
        len(test_df),

    "Train Tweet_ID Groups":
        len(train_ids),

    "Test Tweet_ID Groups":
        len(test_ids),

    "Tweet_ID Overlap":
        len(common_ids),

    "Exact Text Overlap":
        len(common_texts),

    "Accuracy":
        accuracy,

    "Macro Precision":
        macro_precision,

    "Macro Recall":
        macro_recall,

    "Macro F1":
        macro_f1,

    "Weighted F1":
        weighted_f1,

    "Balanced Accuracy":
        balanced_accuracy,

    "MCC":
        mcc,

    "Training Time (s)":
        training_time,

    "Prediction Time (s)":
        prediction_time
}])


result_df.to_csv(
    RESULT_FILE,
    index=False
)


# =========================================================
# 15. STANDARD BENCHMARK COMPARISON
# =========================================================

STANDARD_ACCURACY = 0.9080
STANDARD_MACRO_F1 = 0.9066
STANDARD_BALANCED_ACCURACY = 0.9056
STANDARD_MCC = 0.8766


print("\n" + "=" * 70)
print("STANDARD vs GROUP-DISJOINT")
print("=" * 70)

print(
    f"Accuracy: "
    f"{STANDARD_ACCURACY:.4f} "
    f"-> {accuracy:.4f} "
    f"({accuracy - STANDARD_ACCURACY:+.4f})"
)

print(
    f"Macro F1: "
    f"{STANDARD_MACRO_F1:.4f} "
    f"-> {macro_f1:.4f} "
    f"({macro_f1 - STANDARD_MACRO_F1:+.4f})"
)

print(
    f"Balanced Accuracy: "
    f"{STANDARD_BALANCED_ACCURACY:.4f} "
    f"-> {balanced_accuracy:.4f} "
    f"({balanced_accuracy - STANDARD_BALANCED_ACCURACY:+.4f})"
)

print(
    f"MCC: "
    f"{STANDARD_MCC:.4f} "
    f"-> {mcc:.4f} "
    f"({mcc - STANDARD_MCC:+.4f})"
)


# =========================================================
# 16. FINISHED
# =========================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    RESULT_FILE
)

print(
    PREDICTION_FILE
)

print(
    CONFUSION_FILE
)