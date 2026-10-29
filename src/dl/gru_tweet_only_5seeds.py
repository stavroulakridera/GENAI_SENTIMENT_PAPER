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
    matthews_corrcoef
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Embedding,
    GRU,
    Dense,
    Dropout
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================================================
# 1. SETTINGS
# =========================================================

SEEDS = [42, 52, 62, 72, 82]
SPLIT_SEED = 42

VOCAB_SIZE = 10000
MAX_LENGTH = 40
EMBEDDING_DIM = 128
GRU_UNITS = 64

BATCH_SIZE = 128
EPOCHS = 10

TRAIN_FILE = "data/processed/clean_train.csv"
TEST_FILE = "data/processed/clean_validation.csv"

RESULTS_DIR = "results/tables"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "gru_tweet_only_5seeds_results.csv"
)

SUMMARY_FILE = os.path.join(
    RESULTS_DIR,
    "gru_tweet_only_5seeds_summary.csv"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "gru_tweet_only_5seeds_predictions.csv"
)


print("=" * 75)
print("GRU — TWEET-ONLY — FIVE-SEED EXPERIMENT")
print("=" * 75)

print("\nSeeds:", SEEDS)
print("Fixed split seed:", SPLIT_SEED)


# =========================================================
# 2. LOAD CLEAN DATA
# =========================================================

print("\nLoading datasets...")

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
# 3. TWEET-ONLY INPUT
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


print("\nInput condition: TWEET ONLY")

print("\nExample:")
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


print("\nClasses:")

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
# The split remains identical across all five runs.
# Only initialization/training seed changes.
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
# 6. TOKENIZER
# =========================================================
#
# Fit once on the fixed internal training partition.
# The tokenizer remains identical across all five seeds.
# =========================================================

print(
    "\nFitting Tweet-only tokenizer..."
)


tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)


tokenizer.fit_on_texts(
    internal_train_df[
        "Tweet_Input"
    ].astype(str)
)


# =========================================================
# 7. CONVERT TEXT TO SEQUENCES
# =========================================================

train_sequences = tokenizer.texts_to_sequences(
    internal_train_df[
        "Tweet_Input"
    ].astype(str)
)

dev_sequences = tokenizer.texts_to_sequences(
    dev_df[
        "Tweet_Input"
    ].astype(str)
)

test_sequences = tokenizer.texts_to_sequences(
    test_df[
        "Tweet_Input"
    ].astype(str)
)


# =========================================================
# 8. PADDING
# =========================================================

X_train = pad_sequences(
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
    "\nTraining sequence shape:",
    X_train.shape
)

print(
    "Development sequence shape:",
    X_dev.shape
)

print(
    "Validation sequence shape:",
    X_test.shape
)


# =========================================================
# 9. MODEL FUNCTION
# =========================================================

def build_model():

    model = Sequential([

        Input(
            shape=(MAX_LENGTH,)
        ),

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM
        ),

        GRU(
            GRU_UNITS
        ),

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


    return model


# =========================================================
# 10. FIVE-SEED EXPERIMENT
# =========================================================

all_results = []
all_predictions = []


true_labels = (
    test_df["Sentiment"]
    .astype(str)
    .values
)


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
    # Clear previous session
    # -----------------------------------------------------

    tf.keras.backend.clear_session()

    gc.collect()


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
    # Training
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
    # Prediction
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


    # -----------------------------------------------------
    # Print current seed results
    # -----------------------------------------------------

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
    # Store run metrics
    # -----------------------------------------------------

    all_results.append({

        "Model":
            "GRU",

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
    # Store test predictions
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
    # Cleanup
    # -----------------------------------------------------

    del model
    del probabilities
    del predicted_indices
    del predictions

    tf.keras.backend.clear_session()

    gc.collect()


# =========================================================
# 11. SAVE PER-SEED RESULTS
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
# 12. FIVE-SEED SUMMARY
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
            "GRU",

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
# 13. PRINT SUMMARY
# =========================================================

print(
    "\n" + "=" * 75
)

print(
    "GRU TWEET-ONLY — FIVE-SEED SUMMARY"
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
# 14. ENTITY + TWEET COMPARISON
# =========================================================

ENTITY_MEANS = {

    "Accuracy":
        0.9102,

    "Macro F1":
        0.9081,

    "Balanced Accuracy":
        0.9095,

    "MCC":
        0.8792
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
# 15. BEST SEED
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
    "BEST TWEET-ONLY GRU SEED BY MACRO-F1"
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
# 16. FINISHED
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