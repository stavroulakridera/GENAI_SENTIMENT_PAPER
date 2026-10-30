import os
import time
import random
import gc

import numpy as np
import pandas as pd
import torch

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

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)


# =========================================================
# 1. SETTINGS
# =========================================================

MODEL_NAME = "roberta-base"

SEEDS = [42, 52, 62, 72, 82]
SPLIT_SEED = 42

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

OUTPUT_DIR = "results/roberta_checkpoints"

os.makedirs("results/tables", exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("RoBERTa-BASE — ENTITY + TWEET — 5 SEEDS")
print("=" * 70)


# =========================================================
# 2. DEVICE
# =========================================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print("\nDevice:", DEVICE)


# =========================================================
# 3. LOAD DATA
# =========================================================

print("\nLoading Entity + Tweet datasets...")

train_df = pd.read_csv(
    "results/entity_tweet_train.csv"
)

test_df = pd.read_csv(
    "results/entity_tweet_validation.csv"
)

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

print("Training dataset:", train_df.shape)
print("Evaluation dataset:", test_df.shape)


# =========================================================
# 4. LABEL ENCODING
# =========================================================

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

for index, label in enumerate(
    label_encoder.classes_
):
    print(f"{index}: {label}")


# =========================================================
# 5. FIXED TRAIN / DEV SPLIT
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

internal_train_df["label"] = (
    y_all[train_indices]
)

dev_df["label"] = (
    y_all[dev_indices]
)

test_model_df = (
    test_df
    .copy()
    .reset_index(drop=True)
)

test_model_df["label"] = y_test

print(
    "\nInternal training samples:",
    len(internal_train_df)
)

print(
    "Development samples:",
    len(dev_df)
)

print(
    "Final evaluation samples:",
    len(test_model_df)
)


# =========================================================
# 6. TOKENIZER
# =========================================================

print(
    "\nLoading tokenizer:",
    MODEL_NAME
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =========================================================
# 7. CREATE DATASETS
# =========================================================

train_dataset = Dataset.from_dict({
    "text":
        internal_train_df[
            "Entity_Tweet"
        ].astype(str).tolist(),

    "label":
        internal_train_df[
            "label"
        ].tolist()
})

dev_dataset = Dataset.from_dict({
    "text":
        dev_df[
            "Entity_Tweet"
        ].astype(str).tolist(),

    "label":
        dev_df[
            "label"
        ].tolist()
})

test_dataset = Dataset.from_dict({
    "text":
        test_model_df[
            "Entity_Tweet"
        ].astype(str).tolist(),

    "label":
        test_model_df[
            "label"
        ].tolist()
})


# =========================================================
# 8. TOKENIZATION
# =========================================================

def tokenize_function(batch):

    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )


print("\nTokenizing datasets...")

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True
)

dev_dataset = dev_dataset.map(
    tokenize_function,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize_function,
    batched=True
)

train_dataset = train_dataset.remove_columns(
    ["text"]
)

dev_dataset = dev_dataset.remove_columns(
    ["text"]
)

test_dataset = test_dataset.remove_columns(
    ["text"]
)

train_dataset.set_format("torch")
dev_dataset.set_format("torch")
test_dataset.set_format("torch")

print("Tokenization completed.")


# =========================================================
# 9. TRAINER METRICS
# =========================================================

def compute_metrics(eval_pred):

    logits, labels = eval_pred

    predictions = np.argmax(
        logits,
        axis=-1
    )

    return {
        "accuracy":
            accuracy_score(
                labels,
                predictions
            ),

        "macro_f1":
            f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0
            )
    }


# =========================================================
# 10. FIVE-SEED EXPERIMENT
# =========================================================

all_results = []
all_predictions = []

for run_number, seed in enumerate(
    SEEDS,
    start=1
):

    print("\n" + "=" * 70)

    print(
        f"RoBERTa RUN "
        f"{run_number}/{len(SEEDS)} "
        f"— SEED {seed}"
    )

    print("=" * 70)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # -----------------------------------------------------
    # FRESH MODEL
    # -----------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES
        )
    )

    run_output_dir = os.path.join(
        OUTPUT_DIR,
        f"seed_{seed}"
    )

    # -----------------------------------------------------
    # TRAINING ARGUMENTS
    # -----------------------------------------------------

    training_args = TrainingArguments(

        output_dir=run_output_dir,

        num_train_epochs=EPOCHS,

        per_device_train_batch_size=
            BATCH_SIZE,

        per_device_eval_batch_size=
            BATCH_SIZE,

        learning_rate=
            LEARNING_RATE,

        weight_decay=
            0.01,

        eval_strategy=
            "epoch",

        save_strategy=
            "epoch",

        logging_strategy=
            "epoch",

        load_best_model_at_end=
            True,

        metric_for_best_model=
            "macro_f1",

        greater_is_better=
            True,

        save_total_limit=
            1,

        seed=
            seed,

        data_seed=
            seed,

        report_to=
            "none"
    )

    trainer = Trainer(

        model=model,

        args=training_args,

        train_dataset=
            train_dataset,

        eval_dataset=
            dev_dataset,

        processing_class=
            tokenizer,

        compute_metrics=
            compute_metrics
    )

    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    start_training = time.perf_counter()

    trainer.train()

    training_time = (
        time.perf_counter()
        - start_training
    )

    # -----------------------------------------------------
    # TEST PREDICTION
    # -----------------------------------------------------

    start_prediction = time.perf_counter()

    prediction_output = trainer.predict(
        test_dataset
    )

    prediction_time = (
        time.perf_counter()
        - start_prediction
    )

    logits = prediction_output.predictions

    predicted_indices = np.argmax(
        logits,
        axis=-1
    )

    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=-1
    ).numpy()

    predictions = (
        label_encoder.inverse_transform(
            predicted_indices
        )
    )

    true_labels = (
        test_df[
            "Sentiment"
        ].values
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

    # -----------------------------------------------------
    # PRINT CURRENT RESULT
    # -----------------------------------------------------

    print("\nRESULTS FOR THIS SEED")
    print("-" * 50)

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
            "RoBERTa-base",

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

    # -----------------------------------------------------
    # MEMORY CLEANUP
    # -----------------------------------------------------

    del trainer
    del model
    del logits
    del probabilities

    gc.collect()

    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# =========================================================
# 11. SAVE FIVE RUNS
# =========================================================

results_df = pd.DataFrame(
    all_results
)

results_df.to_csv(
    "results/tables/"
    "roberta_entity_5seeds_results.csv",
    index=False
)

predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)

predictions_df.to_csv(
    "results/tables/"
    "roberta_entity_5seeds_predictions.csv",
    index=False
)


# =========================================================
# 12. MEAN ± STD
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
    "roberta_entity_5seeds_summary.csv",
    index=False
)


# =========================================================
# 13. BEST RUN
# =========================================================

best_index = results_df[
    "Macro F1"
].idxmax()

best_run = results_df.loc[
    best_index
]


# =========================================================
# 14. FINAL SUMMARY
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


print("\nBest run by Macro F1:")

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

print("\nFiles created:")

print(
    "results/tables/"
    "roberta_entity_5seeds_results.csv"
)

print(
    "results/tables/"
    "roberta_entity_5seeds_summary.csv"
)

print(
    "results/tables/"
    "roberta_entity_5seeds_predictions.csv"
)

print(
    "\nRoBERTa 5-SEED EXPERIMENT "
    "COMPLETED SUCCESSFULLY"
)