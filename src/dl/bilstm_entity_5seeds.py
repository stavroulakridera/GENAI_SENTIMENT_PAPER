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
    Embedding,
    LSTM,
    Bidirectional,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. SETTINGS
# =========================================================

SEEDS = [42, 52, 62, 72, 82]

# Fixed internal split for every run
SPLIT_SEED = 42

VOCAB_SIZE = 10000
MAX_LENGTH = 40
EMBEDDING_DIM = 128

LSTM_UNITS = 64

BATCH_SIZE = 128
EPOCHS = 10

os.makedirs("results/tables", exist_ok=True)

print("=" * 70)
print("BiLSTM — ENTITY + TWEET — 5 SEEDS")
print("=" * 70)


# =========================================================
# 2. LOAD ENTITY + TWEET DATA
# =========================================================

print("\nLoading Entity + Tweet datasets...")

train_df = pd.read_csv(
    "data/processed/entity_tweet_train.csv"
)

test_df = pd.read_csv(
    "data/processed/entity_tweet_validation.csv"
)

print("Training dataset:", train_df.shape)
print("Evaluation dataset:", test_df.shape)

required_columns = [
    "Tweet_ID",
    "Entity",
    "Sentiment",
    "Tweet",
    "Entity_Tweet"
]

for column in required_columns:

    if column not in train_df.columns:
        raise ValueError(
            f"Missing training column: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing evaluation column: {column}"
        )

if train_df["Entity_Tweet"].isnull().any():
    raise ValueError(
        "Missing Entity_Tweet values in training dataset."
    )

if test_df["Entity_Tweet"].isnull().any():
    raise ValueError(
        "Missing Entity_Tweet values in evaluation dataset."
    )

print("Dataset check completed successfully.")


# =========================================================
# 3. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_all = label_encoder.fit_transform(
    train_df["Sentiment"]
)

NUM_CLASSES = len(
    label_encoder.classes_
)

print("\nClasses:")

for i, label in enumerate(
    label_encoder.classes_
):
    print(f"{i}: {label}")

print(
    f"\nNumber of classes: {NUM_CLASSES}"
)


# =========================================================
# 4. FIXED INTERNAL TRAIN / DEV SPLIT
# =========================================================

indices = np.arange(
    len(train_df)
)

train_indices, dev_indices = train_test_split(
    indices,
    test_size=0.10,
    random_state=SPLIT_SEED,
    stratify=y_all
)

train_texts = train_df.iloc[
    train_indices
]["Entity_Tweet"].astype(str)

dev_texts = train_df.iloc[
    dev_indices
]["Entity_Tweet"].astype(str)

test_texts = test_df[
    "Entity_Tweet"
].astype(str)

y_train = y_all[
    train_indices
]

y_dev = y_all[
    dev_indices
]

true_labels = test_df[
    "Sentiment"
].values

print("\nFixed internal split:")

print(
    "Training samples:",
    len(train_texts)
)

print(
    "Development samples:",
    len(dev_texts)
)

print(
    "Evaluation samples:",
    len(test_texts)
)


# =========================================================
# 5. TOKENIZATION
# =========================================================

print(
    "\nFitting tokenizer ONLY on "
    "internal training data..."
)

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    train_texts
)

print(
    "Tokenizer fitted successfully."
)


# =========================================================
# 6. TEXT TO SEQUENCES
# =========================================================

print("\nCreating sequences...")

X_train = tokenizer.texts_to_sequences(
    train_texts
)

X_dev = tokenizer.texts_to_sequences(
    dev_texts
)

X_test = tokenizer.texts_to_sequences(
    test_texts
)

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_dev = pad_sequences(
    X_dev,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print(
    "Sequence preparation completed."
)

print(
    "Training matrix:",
    X_train.shape
)

print(
    "Development matrix:",
    X_dev.shape
)

print(
    "Evaluation matrix:",
    X_test.shape
)


# =========================================================
# 7. ORIGINAL BiLSTM ARCHITECTURE
# =========================================================

def build_model():

    model = Sequential([

        Embedding(
            input_dim=VOCAB_SIZE,
            output_dim=EMBEDDING_DIM
        ),

        Bidirectional(
            LSTM(
                LSTM_UNITS
            )
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

    model.build(
        input_shape=(
            None,
            MAX_LENGTH
        )
    )

    return model


# =========================================================
# 8. FIVE-SEED EXPERIMENT
# =========================================================

all_results = []
all_predictions = []

for run_number, seed in enumerate(
    SEEDS,
    start=1
):

    print(
        "\n" + "=" * 70
    )

    print(
        f"RUN {run_number}/{len(SEEDS)} "
        f"— SEED {seed}"
    )

    print(
        "=" * 70
    )

    # -----------------------------------------------------
    # Clear previous model
    # -----------------------------------------------------

    tf.keras.backend.clear_session()
    gc.collect()

    # -----------------------------------------------------
    # Set current training seed
    # -----------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # -----------------------------------------------------
    # Build fresh model
    # -----------------------------------------------------

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    start_training = time.perf_counter()

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
        - start_training
    )

    # -----------------------------------------------------
    # PREDICT
    # -----------------------------------------------------

    start_prediction = time.perf_counter()

    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0
    )

    prediction_time = (
        time.perf_counter()
        - start_prediction
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

    # -----------------------------------------------------
    # METRICS
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

    epochs_run = len(
        history.history["loss"]
    )

    # -----------------------------------------------------
    # PRINT CURRENT RUN
    # -----------------------------------------------------

    print(
        "\nRESULTS FOR THIS SEED"
    )

    print(
        "-" * 50
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
        f"Epochs:            "
        f"{epochs_run}"
    )

    print(
        f"Training Time:     "
        f"{training_time:.2f} sec"
    )

    print(
        f"Prediction Time:   "
        f"{prediction_time:.4f} sec"
    )

    # -----------------------------------------------------
    # STORE RESULTS
    # -----------------------------------------------------

    all_results.append({

        "Model":
            "BiLSTM",

        "Input":
            "Entity + Tweet",

        "Seed":
            seed,

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

        "Epochs":
            epochs_run,

        "Training Time (s)":
            training_time,

        "Prediction Time (s)":
            prediction_time
    })

    # -----------------------------------------------------
    # STORE PREDICTIONS
    # -----------------------------------------------------

    seed_predictions = pd.DataFrame({

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

        "Seed":
            seed,

        "Confidence":
            np.max(
                probabilities,
                axis=1
            )
    })

    all_predictions.append(
        seed_predictions
    )


# =========================================================
# 9. SAVE ALL FIVE RUNS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "results/tables/"
    "bilstm_entity_5seeds_results.csv",
    index=False
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    "results/tables/"
    "bilstm_entity_5seeds_predictions.csv",
    index=False
)


# =========================================================
# 10. MEAN ± STANDARD DEVIATION
# =========================================================

metrics = [
    "Accuracy",
    "Macro F1",
    "Balanced Accuracy",
    "MCC"
]

summary_rows = []

for metric in metrics:

    mean_value = results_df[
        metric
    ].mean()

    std_value = results_df[
        metric
    ].std(
        ddof=1
    )

    summary_rows.append({

        "Metric":
            metric,

        "Mean":
            mean_value,

        "Std":
            std_value
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    "results/tables/"
    "bilstm_entity_5seeds_summary.csv",
    index=False
)


# =========================================================
# 11. BEST RUN BY MACRO F1
# =========================================================

best_index = results_df[
    "Macro F1"
].idxmax()

best_run = results_df.loc[
    best_index
]


# =========================================================
# 12. FINAL SUMMARY
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL 5-SEED SUMMARY"
)

print(
    "=" * 70
)

for _, row in summary_df.iterrows():

    print(
        f"{row['Metric']}: "
        f"{row['Mean']:.4f} ± "
        f"{row['Std']:.4f}"
    )


print(
    "\nBest run by Macro F1:"
)

print(
    f"Seed: "
    f"{int(best_run['Seed'])}"
)

print(
    f"Accuracy: "
    f"{best_run['Accuracy']:.4f}"
)

print(
    f"Macro F1: "
    f"{best_run['Macro F1']:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_run['Balanced Accuracy']:.4f}"
)

print(
    f"MCC: "
    f"{best_run['MCC']:.4f}"
)


print(
    "\nFiles created:"
)

print(
    "results/tables/"
    "bilstm_entity_5seeds_results.csv"
)

print(
    "results/tables/"
    "bilstm_entity_5seeds_summary.csv"
)

print(
    "results/tables/"
    "bilstm_entity_5seeds_predictions.csv"
)

print(
    "\nBiLSTM 5-SEED EXPERIMENT "
    "COMPLETED SUCCESSFULLY"
)