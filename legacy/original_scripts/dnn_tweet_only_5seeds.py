import os
import time
import random
import gc

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. SETTINGS
# =========================================================

SEEDS = [42, 52, 62, 72, 82]

SPLIT_SEED = 42

MAX_FEATURES = 10000
BATCH_SIZE = 128
EPOCHS = 15

TRAIN_FILE = "results/clean_train.csv"
TEST_FILE = "results/clean_validation.csv"

RESULTS_DIR = "results/tables"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "dnn_tweet_only_5seeds_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "dnn_tweet_only_5seeds_summary.csv"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "dnn_tweet_only_5seeds_predictions.csv"
)


print("=" * 75)
print("DNN — TWEET-ONLY — FIVE-SEED EXPERIMENT")
print("=" * 75)

print(
    "\nSeeds:",
    SEEDS
)

print(
    "Fixed split seed:",
    SPLIT_SEED
)


# =========================================================
# 2. LOAD CLEAN DATA
# =========================================================

print(
    "\nLoading datasets..."
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
            f"Missing validation column: {column}"
        )


print(
    "Training shape:",
    train_df.shape
)

print(
    "Validation shape:",
    test_df.shape
)


# =========================================================
# 3. PREPARE TWEET-ONLY INPUT
# =========================================================

train_df["Tweet_Input"] = (
    train_df["Tweet"]
    .astype(str)
    .str.strip()
)

test_df["Tweet_Input"] = (
    test_df["Tweet"]
    .astype(str)
    .str.strip()
)


print(
    "\nInput condition: TWEET ONLY"
)

print(
    "Example:"
)

print(
    train_df["Tweet_Input"].iloc[0]
)


# =========================================================
# 4. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_all = label_encoder.fit_transform(
    train_df["Sentiment"].astype(str)
)

y_test = label_encoder.transform(
    test_df["Sentiment"].astype(str)
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
# 5. FIXED INTERNAL TRAIN / DEV SPLIT
# =========================================================
#
# IMPORTANT:
# This split is created ONCE with seed 42.
#
# It stays identical for all five model seeds.
# Only model initialization/training randomness changes.
# =========================================================

all_indices = np.arange(
    len(train_df)
)


train_indices, dev_indices = train_test_split(
    all_indices,
    test_size=0.10,
    random_state=SPLIT_SEED,
    stratify=y_all
)


internal_train_df = (
    train_df
    .iloc[train_indices]
    .reset_index(drop=True)
)

dev_df = (
    train_df
    .iloc[dev_indices]
    .reset_index(drop=True)
)


y_train = y_all[
    train_indices
]

y_dev = y_all[
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
    "Final validation samples:",
    len(test_df)
)


# =========================================================
# 6. TF-IDF
# =========================================================
#
# Fit ONCE on the fixed internal training partition.
# Exactly the same representation is used for all seeds.
# =========================================================

print(
    "\nFitting Tweet-only TF-IDF..."
)


vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)


X_train_sparse = vectorizer.fit_transform(
    internal_train_df[
        "Tweet_Input"
    ]
)

X_dev_sparse = vectorizer.transform(
    dev_df[
        "Tweet_Input"
    ]
)

X_test_sparse = vectorizer.transform(
    test_df[
        "Tweet_Input"
    ]
)


print(
    "TF-IDF train shape:",
    X_train_sparse.shape
)

print(
    "TF-IDF dev shape:",
    X_dev_sparse.shape
)

print(
    "TF-IDF test shape:",
    X_test_sparse.shape
)


# =========================================================
# 7. CONVERT TO DENSE FLOAT32
# =========================================================

print(
    "\nConverting matrices to dense float32..."
)


X_train = (
    X_train_sparse
    .toarray()
    .astype(np.float32)
)

X_dev = (
    X_dev_sparse
    .toarray()
    .astype(np.float32)
)

X_test = (
    X_test_sparse
    .toarray()
    .astype(np.float32)
)


print(
    "Dense training shape:",
    X_train.shape
)


# Free sparse matrices
del X_train_sparse
del X_dev_sparse
del X_test_sparse

gc.collect()


# =========================================================
# 8. MODEL FUNCTION
# =========================================================

def build_model():

    model = Sequential([

        Input(
            shape=(MAX_FEATURES,)
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
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )


    return model


# =========================================================
# 9. RUN FIVE SEEDS
# =========================================================

all_results = []
all_predictions = []


for run_number, seed in enumerate(
    SEEDS,
    start=1
):

    print(
        "\n" + "=" * 75
    )

    print(
        f"RUN {run_number}/{len(SEEDS)} — SEED {seed}"
    )

    print(
        "=" * 75
    )


    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )


    # -----------------------------------------------------
    # Clear previous graph/session
    # -----------------------------------------------------

    tf.keras.backend.clear_session()

    gc.collect()


    # Reset seeds after clearing session
    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    tf.random.set_seed(
        seed
    )


    # -----------------------------------------------------
    # Build model
    # -----------------------------------------------------

    model = build_model()


    # -----------------------------------------------------
    # Early stopping
    # -----------------------------------------------------

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )


    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    training_start = time.perf_counter()


    history = model.fit(

        X_train,
        y_train,

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


    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------

    prediction_start = time.perf_counter()


    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
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


    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

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


    print(
        f"\nSeed {seed} RESULTS"
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
        f"Training Time:     {training_time:.4f} sec"
    )

    print(
        f"Prediction Time:   {prediction_time:.4f} sec"
    )

    print(
        f"Epochs completed:  {epochs_completed}"
    )


    # -----------------------------------------------------
    # Save run result
    # -----------------------------------------------------

    all_results.append({

        "Model":
            "DNN",

        "Input":
            "Tweet only",

        "Seed":
            seed,

        "Split Seed":
            SPLIT_SEED,

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
    })


    # -----------------------------------------------------
    # Save all test predictions
    # -----------------------------------------------------

    confidence = np.max(
        probabilities,
        axis=1
    )


    for i in range(
        len(test_df)
    ):

        all_predictions.append({

            "Tweet_ID":
                test_df.iloc[i]["Tweet_ID"],

            "Entity":
                test_df.iloc[i]["Entity"],

            "Tweet":
                test_df.iloc[i]["Tweet"],

            "True_Label":
                true_labels[i],

            "Predicted_Label":
                predictions[i],

            "Seed":
                seed,

            "Confidence":
                confidence[i]
        })


    # -----------------------------------------------------
    # Cleanup current model
    # -----------------------------------------------------

    del model
    del probabilities
    del predicted_indices
    del predictions

    tf.keras.backend.clear_session()

    gc.collect()


# =========================================================
# 10. SAVE PER-SEED RESULTS
# =========================================================

results_df = pd.DataFrame(
    all_results
)


results_df.to_csv(
    RESULT_FILE,
    index=False
)


predictions_df = pd.DataFrame(
    all_predictions
)


predictions_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# =========================================================
# 11. CALCULATE MEAN ± STD
# =========================================================

metrics = [
    "Accuracy",
    "Macro F1",
    "Balanced Accuracy",
    "MCC"
]


summary_data = []


for metric in metrics:

    mean_value = results_df[
        metric
    ].mean()

    std_value = results_df[
        metric
    ].std(
        ddof=1
    )


    summary_data.append({

        "Model":
            "DNN",

        "Input":
            "Tweet only",

        "Metric":
            metric,

        "Mean":
            mean_value,

        "Std":
            std_value
    })


summary_df = pd.DataFrame(
    summary_data
)


summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

print(
    "\n" + "=" * 75
)

print(
    "DNN TWEET-ONLY — FIVE-SEED SUMMARY"
)

print(
    "=" * 75
)


for metric in metrics:

    mean_value = results_df[
        metric
    ].mean()

    std_value = results_df[
        metric
    ].std(
        ddof=1
    )

    print(
        f"{metric:18s}: "
        f"{mean_value:.4f} ± {std_value:.4f}"
    )


# =========================================================
# 13. ENTITY + TWEET COMPARISON
# =========================================================
#
# Existing Entity+Tweet five-seed results:
#
# Accuracy          0.9750 ± 0.0012
# Macro F1          0.9756 ± 0.0012
# Balanced Accuracy 0.9755 ± 0.0007
# MCC               0.9664 ± 0.0016
#
# Both conditions now use:
# - same cleaned data
# - same fixed split seed
# - same five model seeds
# - same architecture
# - same hyperparameters
#
# =========================================================

ENTITY_MEANS = {

    "Accuracy":
        0.9750,

    "Macro F1":
        0.9756,

    "Balanced Accuracy":
        0.9755,

    "MCC":
        0.9664
}


print(
    "\n" + "=" * 75
)

print(
    "TWEET-ONLY vs ENTITY + TWEET"
)

print(
    "=" * 75
)


for metric in metrics:

    tweet_mean = results_df[
        metric
    ].mean()

    entity_mean = ENTITY_MEANS[
        metric
    ]

    difference = (
        entity_mean
        - tweet_mean
    )


    print(
        f"{metric:18s}: "
        f"{tweet_mean:.4f} "
        f"-> {entity_mean:.4f} "
        f"({difference:+.4f})"
    )


# =========================================================
# 14. BEST SEED
# =========================================================

best_index = results_df[
    "Macro F1"
].idxmax()


best_row = results_df.loc[
    best_index
]


print(
    "\n" + "=" * 75
)

print(
    "BEST TWEET-ONLY DNN SEED BY MACRO-F1"
)

print(
    "=" * 75
)


print(
    "Seed:",
    int(
        best_row["Seed"]
    )
)

print(
    f"Accuracy:          "
    f"{best_row['Accuracy']:.4f}"
)

print(
    f"Macro F1:          "
    f"{best_row['Macro F1']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_row['Balanced Accuracy']:.4f}"
)

print(
    f"MCC:               "
    f"{best_row['MCC']:.4f}"
)


# =========================================================
# 15. FINISHED
# =========================================================

print(
    "\n" + "=" * 75
)

print(
    "EXPERIMENT COMPLETED SUCCESSFULLY"
)

print(
    "=" * 75
)


print(
    "\nFiles created:"
)

print(
    RESULT_FILE
)

print(
    SUMMARY_FILE
)

print(
    PREDICTION_FILE
)