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
    classification_report
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

os.makedirs("results", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)

print("=" * 70)
print("DNN + TF-IDF — ENTITY + TWEET — 5 SEEDS")
print("=" * 70)


# =========================================================
# 2. LOAD ENTITY + TWEET DATA
# =========================================================

print("\nLoading datasets...")

train_df = pd.read_csv(
    "data/processed/entity_tweet_train.csv"
)

test_df = pd.read_csv(
    "data/processed/entity_tweet_validation.csv"
)

print("Training dataset:", train_df.shape)
print("Final evaluation dataset:", test_df.shape)

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

print("Dataset check completed.")


# =========================================================
# 3. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y_all = label_encoder.fit_transform(
    train_df["Sentiment"]
)

y_test_encoded = label_encoder.transform(
    test_df["Sentiment"]
)

NUM_CLASSES = len(
    label_encoder.classes_
)

print("\nClasses:")

for i, label in enumerate(
    label_encoder.classes_
):
    print(i, label)


# =========================================================
# 4. FIXED INTERNAL TRAIN / DEV SPLIT
#
# IMPORTANT:
# The split remains identical across all 5 runs.
# Only the neural-network random seed changes.
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

print("\nFixed internal split:")
print("Training samples:", len(train_texts))
print("Development samples:", len(dev_texts))
print("Evaluation samples:", len(test_texts))


# =========================================================
# 5. TF-IDF
#
# Fitted once because the split and input data remain fixed.
# =========================================================

print("\nCreating TF-IDF features...")

tfidf = TfidfVectorizer(
    max_features=MAX_FEATURES,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)

start_tfidf = time.perf_counter()

X_train = tfidf.fit_transform(
    train_texts
)

X_dev = tfidf.transform(
    dev_texts
)

X_test = tfidf.transform(
    test_texts
)

tfidf_time = (
    time.perf_counter()
    - start_tfidf
)

print("TF-IDF completed.")
print("Training matrix:", X_train.shape)
print("Development matrix:", X_dev.shape)
print("Evaluation matrix:", X_test.shape)
print(f"TF-IDF time: {tfidf_time:.2f} sec")


# =========================================================
# 6. CONVERT TO DENSE FLOAT32
# =========================================================

print("\nConverting matrices to dense float32...")

X_train = X_train.toarray().astype(
    np.float32
)

X_dev = X_dev.toarray().astype(
    np.float32
)

X_test = X_test.toarray().astype(
    np.float32
)

print("Dense conversion completed.")


# =========================================================
# 7. FUNCTION TO BUILD SAME ORIGINAL DNN
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
# 8. RUN 5 SEEDS
# =========================================================

all_results = []
all_predictions = []

true_labels = test_df[
    "Sentiment"
].values

for run_number, seed in enumerate(
    SEEDS,
    start=1
):

    print("\n" + "=" * 70)
    print(
        f"RUN {run_number}/{len(SEEDS)} — SEED {seed}"
    )
    print("=" * 70)

    # -----------------------------------------------------
    # Clear previous TensorFlow model
    # -----------------------------------------------------

    tf.keras.backend.clear_session()
    gc.collect()

    # -----------------------------------------------------
    # Set random initialization/training seed
    # -----------------------------------------------------

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    # -----------------------------------------------------
    # Build same architecture
    # -----------------------------------------------------

    model = build_model()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    # -----------------------------------------------------
    # Train
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
    # Predict
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

    epochs_run = len(
        history.history["loss"]
    )

    print("\nRESULTS FOR THIS SEED")
    print("-" * 50)

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
        f"Epochs:            {epochs_run}"
    )

    print(
        f"Training Time:     {training_time:.2f} sec"
    )

    # -----------------------------------------------------
    # Store run results
    # -----------------------------------------------------

    all_results.append({
        "Model": "DNN",
        "Input": "Entity + Tweet",
        "Seed": seed,
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Balanced Accuracy": balanced_accuracy,
        "MCC": mcc,
        "Epochs": epochs_run,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time
    })

    # -----------------------------------------------------
    # Store predictions for statistical tests
    # -----------------------------------------------------

    seed_predictions = pd.DataFrame({
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
# 9. SAVE ALL 5 RUNS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "results/tables/dnn_entity_5seeds_results.csv",
    index=False
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    "results/tables/dnn_entity_5seeds_predictions.csv",
    index=False
)


# =========================================================
# 10. MEAN + STANDARD DEVIATION
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
    ].std(ddof=1)

    summary_rows.append({
        "Metric": metric,
        "Mean": mean_value,
        "Std": std_value
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    "results/tables/dnn_entity_5seeds_summary.csv",
    index=False
)


# =========================================================
# 11. FINAL SUMMARY
# =========================================================

print("\n" + "=" * 70)
print("FINAL 5-SEED SUMMARY")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"{row['Metric']}: "
        f"{row['Mean']:.4f} ± "
        f"{row['Std']:.4f}"
    )


best_index = results_df[
    "Macro F1"
].idxmax()

best_run = results_df.loc[
    best_index
]

print("\nBest run by Macro F1:")
print(
    f"Seed: {int(best_run['Seed'])}"
)
print(
    f"Accuracy: {best_run['Accuracy']:.4f}"
)
print(
    f"Macro F1: {best_run['Macro F1']:.4f}"
)
print(
    f"Balanced Accuracy: "
    f"{best_run['Balanced Accuracy']:.4f}"
)
print(
    f"MCC: {best_run['MCC']:.4f}"
)

print("\nFiles created:")
print(
    "results/tables/dnn_entity_5seeds_results.csv"
)
print(
    "results/tables/dnn_entity_5seeds_summary.csv"
)
print(
    "results/tables/dnn_entity_5seeds_predictions.csv"
)

print("\nDNN 5-SEED EXPERIMENT COMPLETED SUCCESSFULLY")