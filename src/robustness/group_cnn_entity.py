import os
import time
import random
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
    Embedding,
    Conv1D,
    GlobalMaxPooling1D,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 1. SETTINGS
# =========================================================

RANDOM_STATE = 42
SPLIT_SEED = 42

VOCAB_SIZE = 10000
MAX_LENGTH = 40
EMBEDDING_DIM = 128

BATCH_SIZE = 128
EPOCHS = 10

TRAIN_FILE = "results/group_train.csv"
TEST_FILE = "results/group_test.csv"

RESULTS_DIR = "results/tables"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "group_cnn_entity_results.csv"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "group_cnn_entity_predictions.csv"
)

CONFUSION_FILE = os.path.join(
    RESULTS_DIR,
    "group_cnn_entity_confusion_matrix.csv"
)


print("=" * 70)
print("GROUP-DISJOINT ROBUSTNESS ANALYSIS")
print("1D CNN — ENTITY + TWEET")
print("=" * 70)


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# =========================================================
# 3. LOAD GROUP-DISJOINT DATA
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
# 4. VERIFY GROUP DISJOINTNESS
# =========================================================

train_ids = set(
    train_df["Tweet_ID"]
)

test_ids = set(
    test_df["Tweet_ID"]
)

common_ids = train_ids.intersection(
    test_ids
)


train_texts = set(
    train_df["Tweet"].astype(str)
)

test_texts = set(
    test_df["Tweet"].astype(str)
)

common_texts = train_texts.intersection(
    test_texts
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
    train_df["Entity_Tweet"].iloc[0]
)


# =========================================================
# 6. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_group_train = label_encoder.fit_transform(
    train_df["Sentiment"].astype(str)
)

y_test = label_encoder.transform(
    test_df["Sentiment"].astype(str)
)

NUM_CLASSES = len(
    label_encoder.classes_
)


print("\nClasses:")

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
# group_test.csv remains completely untouched.
#
# The 90/10 split below is created ONLY from group_train
# and is used for validation / early stopping.
# =========================================================

indices = np.arange(
    len(train_df)
)

internal_train_indices, dev_indices = train_test_split(
    indices,
    test_size=0.10,
    random_state=SPLIT_SEED,
    stratify=y_group_train
)


internal_train_df = (
    train_df
    .iloc[internal_train_indices]
    .reset_index(drop=True)
)

dev_df = (
    train_df
    .iloc[dev_indices]
    .reset_index(drop=True)
)


y_internal_train = y_group_train[
    internal_train_indices
]

y_dev = y_group_train[
    dev_indices
]


print(
    "\nInternal training samples:",
    len(internal_train_df)
)

print(
    "Development samples:",
    len(dev_df)
)

print(
    "Group-disjoint test samples:",
    len(test_df)
)


# =========================================================
# 8. TOKENIZER
# =========================================================

print(
    "\nFitting tokenizer on internal training data..."
)

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    internal_train_df[
        "Entity_Tweet"
    ].astype(str)
)


# =========================================================
# 9. TEXT -> SEQUENCES
# =========================================================

train_sequences = tokenizer.texts_to_sequences(
    internal_train_df[
        "Entity_Tweet"
    ].astype(str)
)

dev_sequences = tokenizer.texts_to_sequences(
    dev_df[
        "Entity_Tweet"
    ].astype(str)
)

test_sequences = tokenizer.texts_to_sequences(
    test_df[
        "Entity_Tweet"
    ].astype(str)
)


# =========================================================
# 10. PADDING
# =========================================================

X_internal_train = pad_sequences(
    train_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_dev = pad_sequences(
    dev_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    test_sequences,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)


print(
    "\nInternal training shape:",
    X_internal_train.shape
)

print(
    "Development shape:",
    X_dev.shape
)

print(
    "Test shape:",
    X_test.shape
)


# =========================================================
# 11. BUILD CNN
# =========================================================

print(
    "\nBuilding 1D CNN..."
)


model = Sequential([

    Input(
        shape=(MAX_LENGTH,)
    ),

    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    Conv1D(
        filters=128,
        kernel_size=5,
        activation="relu"
    ),

    GlobalMaxPooling1D(),

    Dense(
        64,
        activation="relu"
    ),

    Dropout(
        0.5
    ),

    Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# =========================================================
# 12. EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# =========================================================
# 13. TRAIN CNN
# =========================================================

print(
    "\nTraining 1D CNN..."
)

training_start = time.perf_counter()


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
    history.history["loss"]
)


print(
    "\nTraining completed."
)

print(
    "Epochs completed:",
    epochs_completed
)


# =========================================================
# 14. PREDICT GROUP-DISJOINT TEST SET
# =========================================================

print(
    "\nPredicting group-disjoint test set..."
)

prediction_start = time.perf_counter()


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

predictions = label_encoder.inverse_transform(
    predicted_indices
)

true_labels = (
    test_df["Sentiment"]
    .astype(str)
    .values
)


# =========================================================
# 15. METRICS
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

balanced_accuracy = balanced_accuracy_score(
    true_labels,
    predictions
)

mcc = matthews_corrcoef(
    true_labels,
    predictions
)


# =========================================================
# 16. PRINT RESULTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "GROUP-DISJOINT CNN RESULTS"
)

print(
    "=" * 70
)


print(
    f"Accuracy:          {accuracy:.4f}"
)

print(
    f"Macro Precision:   {macro_precision:.4f}"
)

print(
    f"Macro Recall:      {macro_recall:.4f}"
)

print(
    f"Macro F1:          {macro_f1:.4f}"
)

print(
    f"Weighted F1:       {weighted_f1:.4f}"
)

print(
    f"Balanced Accuracy: {balanced_accuracy:.4f}"
)

print(
    f"MCC:               {mcc:.4f}"
)

print(
    f"\nTraining Time:     {training_time:.4f} sec"
)

print(
    f"Prediction Time:   {prediction_time:.4f} sec"
)

print(
    f"Epochs completed:  {epochs_completed}"
)


# =========================================================
# 17. CLASSIFICATION REPORT
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
# 18. CONFUSION MATRIX
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
# 19. SAVE PREDICTIONS
# =========================================================

prediction_df = pd.DataFrame({

    "Tweet_ID":
        test_df["Tweet_ID"].values,

    "Entity":
        test_df["Entity"].values,

    "Tweet":
        test_df["Tweet"].values,

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
# 20. SAVE SUMMARY
# =========================================================

result_df = pd.DataFrame([{

    "Model":
        "1D CNN",

    "Seed":
        RANDOM_STATE,

    "Evaluation":
        "Group-disjoint",

    "Input":
        "Entity + Tweet",

    "Train Samples":
        len(train_df),

    "Internal Train Samples":
        len(internal_train_df),

    "Development Samples":
        len(dev_df),

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
        prediction_time,

    "Epochs":
        epochs_completed
}])


result_df.to_csv(
    RESULT_FILE,
    index=False
)


# =========================================================
# 21. STANDARD BENCHMARK COMPARISON
# =========================================================
#
# Standard values below are the Entity + Tweet
# FIVE-SEED MEANS.
#
# The group-disjoint CNN in this script uses seed 42.
# Therefore the comparison is descriptive, rather than
# a strict paired seed comparison.
# =========================================================

STANDARD_ACCURACY = 0.9568
STANDARD_MACRO_F1 = 0.9562
STANDARD_BALANCED_ACCURACY = 0.9581
STANDARD_MCC = 0.9420


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
# 22. CLEANUP
# =========================================================

del model
del probabilities

gc.collect()

tf.keras.backend.clear_session()


# =========================================================
# 23. FINISHED
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