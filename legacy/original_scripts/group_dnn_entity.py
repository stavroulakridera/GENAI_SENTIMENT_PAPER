import os
import time
import random
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

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

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. SETTINGS
# =========================================================

RANDOM_STATE = 42
SPLIT_SEED = 42

MAX_FEATURES = 10000

BATCH_SIZE = 128
EPOCHS = 15

TRAIN_FILE = "results/group_train.csv"
TEST_FILE = "results/group_test.csv"

RESULTS_DIR = "results/tables"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "group_dnn_entity_results.csv"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "group_dnn_entity_predictions.csv"
)

CONFUSION_FILE = os.path.join(
    RESULTS_DIR,
    "group_dnn_entity_confusion_matrix.csv"
)


print("=" * 70)
print("GROUP-DISJOINT ROBUSTNESS ANALYSIS")
print("DNN — ENTITY + TWEET")
print("=" * 70)


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# =========================================================
# 3. LOAD GROUP-DISJOINT DATA
# =========================================================

print(
    "\nLoading group-disjoint datasets..."
)

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
# 4. VERIFY GROUP DISJOINTNESS
# =========================================================

train_ids = set(
    train_df[
        "Tweet_ID"
    ]
)

test_ids = set(
    test_df[
        "Tweet_ID"
    ]
)


common_ids = (
    train_ids.intersection(
        test_ids
    )
)


train_texts = set(
    train_df[
        "Tweet"
    ].astype(str)
)

test_texts = set(
    test_df[
        "Tweet"
    ].astype(str)
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
# 5. CREATE ENTITY + TWEET INPUT
# =========================================================

train_df[
    "Entity_Tweet"
] = (
    train_df[
        "Entity"
    ]
    .astype(str)
    .str.strip()
    +
    " [SEP] "
    +
    train_df[
        "Tweet"
    ]
    .astype(str)
    .str.strip()
)


test_df[
    "Entity_Tweet"
] = (
    test_df[
        "Entity"
    ]
    .astype(str)
    .str.strip()
    +
    " [SEP] "
    +
    test_df[
        "Tweet"
    ]
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
# 6. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()


y_group_train = (
    label_encoder.fit_transform(
        train_df[
            "Sentiment"
        ].astype(str)
    )
)


y_test = (
    label_encoder.transform(
        test_df[
            "Sentiment"
        ].astype(str)
    )
)


NUM_CLASSES = len(
    label_encoder.classes_
)


print(
    "\nClasses:"
)

for index, label in enumerate(
    label_encoder.classes_
):

    print(
        f"{index}: {label}"
    )


# =========================================================
# 7. INTERNAL TRAIN / DEV SPLIT
# =========================================================
#
# IMPORTANT:
# The official group-disjoint TEST SET remains untouched.
#
# We create a fixed 90/10 internal split only from
# group_train.csv for early stopping.
#
# This mirrors the standard DNN protocol.
# =========================================================

indices = np.arange(
    len(train_df)
)


internal_train_indices, dev_indices = (
    train_test_split(
        indices,
        test_size=0.10,
        random_state=SPLIT_SEED,
        stratify=y_group_train
    )
)


internal_train_df = (
    train_df
    .iloc[
        internal_train_indices
    ]
    .reset_index(
        drop=True
    )
)


dev_df = (
    train_df
    .iloc[
        dev_indices
    ]
    .reset_index(
        drop=True
    )
)


y_internal_train = (
    y_group_train[
        internal_train_indices
    ]
)


y_dev = (
    y_group_train[
        dev_indices
    ]
)


print(
    "\nInternal training samples:",
    len(
        internal_train_df
    )
)

print(
    "Development samples:",
    len(
        dev_df
    )
)

print(
    "Group-disjoint test samples:",
    len(
        test_df
    )
)


# =========================================================
# 8. TF-IDF
# =========================================================

print(
    "\nCreating TF-IDF representation..."
)


vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)


X_internal_train = (
    vectorizer.fit_transform(
        internal_train_df[
            "Entity_Tweet"
        ].astype(str)
    )
)


X_dev = (
    vectorizer.transform(
        dev_df[
            "Entity_Tweet"
        ].astype(str)
    )
)


X_test = (
    vectorizer.transform(
        test_df[
            "Entity_Tweet"
        ].astype(str)
    )
)


print(
    "Internal training TF-IDF shape:",
    X_internal_train.shape
)

print(
    "Development TF-IDF shape:",
    X_dev.shape
)

print(
    "Test TF-IDF shape:",
    X_test.shape
)


# =========================================================
# 9. CONVERT TO DENSE FLOAT32
# =========================================================

print(
    "\nConverting TF-IDF matrices "
    "to dense float32..."
)


X_internal_train = (
    X_internal_train
    .toarray()
    .astype(
        np.float32
    )
)


X_dev = (
    X_dev
    .toarray()
    .astype(
        np.float32
    )
)


X_test = (
    X_test
    .toarray()
    .astype(
        np.float32
    )
)


print(
    "Dense conversion completed."
)


# =========================================================
# 10. BUILD DNN
# =========================================================

print(
    "\nBuilding DNN..."
)


model = Sequential([

    Input(
        shape=(
            MAX_FEATURES,
        )
    ),

    Dense(
        256,
        activation="relu"
    ),

    Dropout(
        0.5
    ),

    Dense(
        128,
        activation="relu"
    ),

    Dropout(
        0.3
    ),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(

    optimizer="adam",

    loss=(
        "sparse_categorical_crossentropy"
    ),

    metrics=[
        "accuracy"
    ]
)


model.summary()


# =========================================================
# 11. EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=2,

    restore_best_weights=True
)


# =========================================================
# 12. TRAIN DNN
# =========================================================

print(
    "\nTraining DNN..."
)


training_start = (
    time.perf_counter()
)


history = model.fit(

    X_internal_train,
    y_internal_train,

    validation_data=(
        X_dev,
        y_dev
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


training_time = (
    time.perf_counter()
    - training_start
)


epochs_completed = len(
    history.history[
        "loss"
    ]
)


print(
    "\nTraining completed."
)

print(
    "Epochs completed:",
    epochs_completed
)


# =========================================================
# 13. PREDICT GROUP-DISJOINT TEST SET
# =========================================================

print(
    "\nPredicting group-disjoint test set..."
)


prediction_start = (
    time.perf_counter()
)


probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)


prediction_time = (
    time.perf_counter()
    - prediction_start
)


predicted_indices = np.argmax(
    probabilities,
    axis=1
)


predictions = (
    label_encoder.inverse_transform(
        predicted_indices
    )
)


true_labels = (
    test_df[
        "Sentiment"
    ]
    .astype(str)
    .values
)


# =========================================================
# 14. METRICS
# =========================================================

accuracy = accuracy_score(
    true_labels,
    predictions
)


macro_precision = precision_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    true_labels,
    predictions,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    true_labels,
    predictions,
    average="weighted",
    zero_division=0
)


balanced_accuracy = (
    balanced_accuracy_score(
        true_labels,
        predictions
    )
)


mcc = matthews_corrcoef(
    true_labels,
    predictions
)


# =========================================================
# 15. PRINT RESULTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "GROUP-DISJOINT DNN RESULTS"
)

print(
    "=" * 70
)


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


print(
    f"Epochs completed:  "
    f"{epochs_completed}"
)


# =========================================================
# 16. CLASSIFICATION REPORT
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
        true_labels,
        predictions,
        labels=labels,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 17. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    true_labels,
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
# 18. SAVE PREDICTIONS
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
        true_labels,

    "Predicted_Label":
        predictions,

    "Confidence":
        np.max(
            probabilities,
            axis=1
        )
})


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# =========================================================
# 19. SAVE SUMMARY
# =========================================================

result_df = pd.DataFrame([{

    "Model":
        "DNN",

    "Seed":
        RANDOM_STATE,

    "Evaluation":
        "Group-disjoint",

    "Input":
        "Entity + Tweet",

    "Train Samples":
        len(
            train_df
        ),

    "Internal Train Samples":
        len(
            internal_train_df
        ),

    "Development Samples":
        len(
            dev_df
        ),

    "Test Samples":
        len(
            test_df
        ),

    "Train Tweet_ID Groups":
        len(
            train_ids
        ),

    "Test Tweet_ID Groups":
        len(
            test_ids
        ),

    "Tweet_ID Overlap":
        len(
            common_ids
        ),

    "Exact Text Overlap":
        len(
            common_texts
        ),

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
        prediction_time,

    "Epochs":
        epochs_completed
}])


result_df.to_csv(
    RESULT_FILE,
    index=False
)


# =========================================================
# 20. STANDARD BENCHMARK COMPARISON
# =========================================================
#
# For the DNN, the standard Entity + Tweet result
# below is the FIVE-SEED MEAN.
#
# The group-disjoint result in this script is seed 42.
# Therefore, this comparison is descriptive and should
# NOT be presented as a strict paired seed comparison.
# =========================================================

STANDARD_ACCURACY = 0.9750
STANDARD_MACRO_F1 = 0.9756
STANDARD_BALANCED_ACCURACY = 0.9755
STANDARD_MCC = 0.9664


print(
    "\n" + "=" * 70
)

print(
    "STANDARD vs GROUP-DISJOINT"
)

print(
    "=" * 70
)


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
# 21. CLEANUP
# =========================================================

del model
del probabilities

gc.collect()

tf.keras.backend.clear_session()


# =========================================================
# 22. FINISHED
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "EXPERIMENT COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)


print(
    "\nFiles created:"
)

print(
    RESULT_FILE
)

print(
    PREDICTION_FILE
)

print(
    CONFUSION_FILE
)