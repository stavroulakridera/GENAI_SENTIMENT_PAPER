# =========================================================
# LSTM - SENTIMENT ANALYSIS
# =========================================================

import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    LSTM,
    Dense,
    Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 1. SETTINGS
# =========================================================

RANDOM_STATE = 42

VOCAB_SIZE = 10000
MAX_LENGTH = 40
EMBEDDING_DIM = 128

LSTM_UNITS = 64

BATCH_SIZE = 128
EPOCHS = 10

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

os.makedirs("results", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/confusion_matrices", exist_ok=True)

print("=" * 65)
print("LSTM SENTIMENT ANALYSIS EXPERIMENT")
print("=" * 65)


# =========================================================
# 2. LOAD CLEAN DATASETS
# =========================================================

print("\nLoading clean datasets...")

train_df = pd.read_csv(
    "data/processed/clean_train.csv"
)

test_df = pd.read_csv(
    "data/processed/clean_validation.csv"
)

print("\nFull training dataset:")
print(train_df.shape)

print("\nFinal evaluation dataset:")
print(test_df.shape)


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
            f"Missing training column: {column}"
        )

    if column not in test_df.columns:
        raise ValueError(
            f"Missing validation column: {column}"
        )

if train_df["Tweet"].isnull().any():
    raise ValueError(
        "Missing tweets found in training dataset."
    )

if test_df["Tweet"].isnull().any():
    raise ValueError(
        "Missing tweets found in evaluation dataset."
    )

print("\nDataset check completed successfully!")


# =========================================================
# 4. LABEL ENCODING
# =========================================================

print("\n" + "=" * 65)
print("LABEL ENCODING")
print("=" * 65)

label_encoder = LabelEncoder()

y_all = label_encoder.fit_transform(
    train_df["Sentiment"]
)

y_test = label_encoder.transform(
    test_df["Sentiment"]
)

NUM_CLASSES = len(
    label_encoder.classes_
)

print("\nClasses:")

for i, label in enumerate(label_encoder.classes_):
    print(f"{i}: {label}")

print(
    f"\nNumber of classes: {NUM_CLASSES}"
)


# =========================================================
# 5. INTERNAL TRAIN / DEV SPLIT
# =========================================================

print("\n" + "=" * 65)
print("INTERNAL TRAIN / DEV SPLIT")
print("=" * 65)

train_texts, dev_texts, y_train, y_dev = train_test_split(
    train_df["Tweet"].astype(str),
    y_all,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=y_all
)

print(
    "\nTraining samples:",
    len(train_texts)
)

print(
    "Development samples:",
    len(dev_texts)
)

print(
    "Final evaluation samples:",
    len(test_df)
)


# =========================================================
# 6. TOKENIZATION
# =========================================================

print("\n" + "=" * 65)
print("TOKENIZATION")
print("=" * 65)

print("\nFitting tokenizer ONLY on internal training data...")

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(
    train_texts
)

print("Tokenizer fitted successfully.")


# =========================================================
# 7. TEXT TO SEQUENCES
# =========================================================

X_train = tokenizer.texts_to_sequences(
    train_texts
)

X_dev = tokenizer.texts_to_sequences(
    dev_texts
)

X_test = tokenizer.texts_to_sequences(
    test_df["Tweet"].astype(str)
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

print("\nSequence preparation completed.")

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
# 8. BUILD LSTM
# =========================================================

print("\n" + "=" * 65)
print("BUILDING LSTM")
print("=" * 65)

model = Sequential([
    Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=EMBEDDING_DIM
    ),

    LSTM(
        LSTM_UNITS
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
    input_shape=(None, MAX_LENGTH)
)

print("\nModel architecture:")
model.summary()


# =========================================================
# 9. EARLY STOPPING
# =========================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)


# =========================================================
# 10. TRAINING
# =========================================================

print("\n" + "=" * 65)
print("TRAINING LSTM")
print("=" * 65)

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

print(
    f"\nTraining completed in "
    f"{training_time:.2f} seconds."
)


# =========================================================
# 11. FINAL PREDICTIONS
# =========================================================

print("\n" + "=" * 65)
print("FINAL EVALUATION")
print("=" * 65)

start_prediction = time.perf_counter()

probabilities = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=0
)

predicted_indices = np.argmax(
    probabilities,
    axis=1
)

prediction_time = (
    time.perf_counter()
    - start_prediction
)

predictions = label_encoder.inverse_transform(
    predicted_indices
)

true_labels = test_df[
    "Sentiment"
].values


# =========================================================
# 12. PERFORMANCE METRICS
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


print("\n" + "=" * 65)
print("LSTM RESULTS")
print("=" * 65)

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
    f"Training Time:   {training_time:.2f} sec"
)

print(
    f"Prediction Time: {prediction_time:.4f} sec"
)


# =========================================================
# 13. CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:")

print(
    classification_report(
        true_labels,
        predictions,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 14. SAVE SUMMARY RESULTS
# =========================================================

results_df = pd.DataFrame([
    {
        "Model": "LSTM",
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "Training Time (s)": training_time,
        "Prediction Time (s)": prediction_time
    }
])

results_df.to_csv(
    "results/tables/lstm_results.csv",
    index=False
)


# =========================================================
# 15. SAVE PREDICTIONS + PROBABILITIES
# =========================================================

predictions_df = pd.DataFrame({
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


for class_index, class_name in enumerate(
    label_encoder.classes_
):

    predictions_df[
        f"Probability_{class_name}"
    ] = probabilities[:, class_index]


predictions_df.to_csv(
    "results/tables/lstm_predictions.csv",
    index=False
)


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================

labels = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

cm = confusion_matrix(
    true_labels,
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
    "Confusion Matrix - LSTM"
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
    "results/confusion_matrices/lstm_cm.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 17. TRAINING / VALIDATION LOSS
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    history.history["loss"],
    label="Training Loss"
)

ax.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Loss"
)

ax.set_title(
    "LSTM Training and Validation Loss"
)

ax.legend()

ax.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "results/figures/lstm_loss.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 18. TRAINING / VALIDATION ACCURACY
# =========================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

ax.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

ax.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Accuracy"
)

ax.set_title(
    "LSTM Training and Validation Accuracy"
)

ax.legend()

ax.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "results/figures/lstm_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =========================================================
# 19. FINISHED
# =========================================================

print("\n" + "=" * 65)
print("LSTM EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFiles created:")

print(
    "1. results/tables/lstm_results.csv"
)

print(
    "2. results/tables/lstm_predictions.csv"
)

print(
    "3. results/confusion_matrices/lstm_cm.png"
)

print(
    "4. results/figures/lstm_loss.png"
)

print(
    "5. results/figures/lstm_accuracy.png"
)