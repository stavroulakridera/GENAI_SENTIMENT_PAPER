import os
import time
import re

import pandas as pd
import requests

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

MODEL_NAME = "qwen2.5:3b"

OLLAMA_URL = "http://localhost:11434/api/generate"

TEST_FILE = "data/group_disjoint/group_test.csv"

RESULTS_DIR = "results/tables"

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

RESULT_FILE = os.path.join(
    RESULTS_DIR,
    "group_qwen_few_shot_results.csv"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "group_qwen_few_shot_predictions.csv"
)

CHECKPOINT_FILE = os.path.join(
    RESULTS_DIR,
    "group_qwen_few_shot_checkpoint.csv"
)

CONFUSION_FILE = os.path.join(
    RESULTS_DIR,
    "group_qwen_few_shot_confusion_matrix.csv"
)


VALID_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

TEMPERATURE = 0
SEED = 42

CHECKPOINT_EVERY = 100


print("=" * 70)
print("GROUP-DISJOINT ROBUSTNESS TEST SUBSET")
print("QWEN2.5 3B — FEW-SHOT")
print("=" * 70)

print("\nModel:", MODEL_NAME)
print("Temperature:", TEMPERATURE)
print("Seed:", SEED)


# =========================================================
# 2. LOAD GROUP TEST DATA
# =========================================================

print(
    "\nLoading group-disjoint test dataset..."
)

df = pd.read_csv(
    TEST_FILE
)


required_columns = [
    "Tweet_ID",
    "Entity",
    "Sentiment",
    "Tweet"
]


for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing column: {column}"
        )


df = df.reset_index(
    drop=True
)


print(
    "Test samples:",
    len(df)
)

print(
    "Unique Tweet_ID groups:",
    df["Tweet_ID"].nunique()
)


print(
    "\nClass distribution:"
)

print(
    df["Sentiment"].value_counts()
)


# =========================================================
# 3. CREATE ENTITY + TWEET REPRESENTATION
# =========================================================

df["Entity_Tweet"] = (
    df["Entity"]
    .astype(str)
    .str.strip()
    +
    " [SEP] "
    +
    df["Tweet"]
    .astype(str)
    .str.strip()
)


print(
    "\nExample Entity + Tweet:"
)

print(
    df["Entity_Tweet"].iloc[0]
)


# =========================================================
# 4. FEW-SHOT PROMPT
# =========================================================

def build_prompt(entity, tweet):

    prompt = f"""
You are performing fine-grained sentiment classification.

Classify each social media post with respect to the
specified entity.

Use exactly one of these four labels:

Negative
Neutral
Positive
Irrelevant

Label definitions:

Negative:
The post expresses negative sentiment, criticism,
dissatisfaction, dislike, or an unfavorable opinion
toward the entity.

Neutral:
The post is relevant to the entity but does not express
clear positive or negative sentiment.

Positive:
The post expresses positive sentiment, approval,
satisfaction, praise, liking, or a favorable opinion
toward the entity.

Irrelevant:
The post is unrelated to the entity, mentions it only
incidentally, or does not express meaningful content
about the entity.

Here are four examples.

Example 1:
Entity: Microsoft
Post: I absolutely love the new Microsoft update.
Label: Positive

Example 2:
Entity: Google
Post: Google announced the release date of its next update.
Label: Neutral

Example 3:
Entity: Facebook
Post: Facebook keeps crashing and this is incredibly annoying.
Label: Negative

Example 4:
Entity: Amazon
Post: I was listening to music all morning and then went for a walk.
Label: Irrelevant

Now classify the following post.

Entity:
{entity}

Post:
{tweet}

Return ONLY one label:
Negative, Neutral, Positive, or Irrelevant.
"""

    return prompt.strip()


# =========================================================
# 5. CALL OLLAMA
# =========================================================

def query_ollama(prompt):

    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "seed": SEED
        }
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=300
    )

    response.raise_for_status()

    result = response.json()

    return result.get(
        "response",
        ""
    ).strip()


# =========================================================
# 6. NORMALIZE OUTPUT
# =========================================================

def normalize_prediction(raw_output):

    if raw_output is None:
        return None

    cleaned = raw_output.strip()

    cleaned = cleaned.replace(
        "**",
        ""
    )

    cleaned = cleaned.replace(
        "`",
        ""
    )

    cleaned = cleaned.strip()


    # Exact output
    for label in VALID_LABELS:

        if cleaned.lower() == label.lower():

            return label


    # Simple variants such as:
    # Positive.
    # Label: Positive
    # The label is Positive
    matches = re.findall(
        r"\b(Negative|Neutral|Positive|Irrelevant)\b",
        cleaned,
        flags=re.IGNORECASE
    )


    unique_matches = []

    for match in matches:

        normalized = match.capitalize()

        if normalized not in unique_matches:

            unique_matches.append(
                normalized
            )


    if len(unique_matches) == 1:

        return unique_matches[0]


    return None


# =========================================================
# 7. LOAD CHECKPOINT IF AVAILABLE
# =========================================================

records = []

start_index = 0


if os.path.exists(
    CHECKPOINT_FILE
):

    print(
        "\nExisting checkpoint found."
    )

    checkpoint_df = pd.read_csv(
        CHECKPOINT_FILE
    )

    if len(checkpoint_df) > 0:

        records = checkpoint_df.to_dict(
            orient="records"
        )

        start_index = len(
            checkpoint_df
        )

        print(
            "Resuming from sample:",
            start_index + 1
        )

        print(
            "Already completed:",
            start_index
        )

else:

    print(
        "\nNo checkpoint found. "
        "Starting from sample 1."
    )


# =========================================================
# 8. VALIDATE CHECKPOINT
# =========================================================

if start_index > len(df):

    raise RuntimeError(
        "Checkpoint contains more rows "
        "than the test dataset."
    )


if start_index == len(df):

    print(
        "\nAll samples already exist "
        "in the checkpoint."
    )


# =========================================================
# 9. RUN QWEN EVALUATION
# =========================================================

run_start = time.perf_counter()


for index in range(
    start_index,
    len(df)
):

    row = df.iloc[index]


    entity = str(
        row["Entity"]
    )

    tweet = str(
        row["Tweet"]
    )

    true_label = str(
        row["Sentiment"]
    )


    prompt = build_prompt(
        entity,
        tweet
    )


    sample_start = time.perf_counter()


    try:

        raw_output = query_ollama(
            prompt
        )

        prediction = normalize_prediction(
            raw_output
        )

        error = ""

    except Exception as exc:

        raw_output = ""

        prediction = None

        error = str(exc)


    latency = (
        time.perf_counter()
        - sample_start
    )


    valid = (
        prediction in VALID_LABELS
    )


    record = {

        "Dataset_Index":
            index,

        "Tweet_ID":
            row["Tweet_ID"],

        "Entity":
            entity,

        "Tweet":
            tweet,

        "True_Label":
            true_label,

        "Raw_Output":
            raw_output,

        "Predicted_Label":
            (
                prediction
                if prediction is not None
                else "INVALID"
            ),

        "Valid_Output":
            valid,

        "Latency_s":
            latency,

        "Error":
            error
    }


    records.append(
        record
    )


    completed = index + 1


    # -----------------------------------------------------
    # CHECKPOINT
    # -----------------------------------------------------

    if (
        completed % CHECKPOINT_EVERY == 0
        or completed == len(df)
    ):

        checkpoint_output = pd.DataFrame(
            records
        )

        checkpoint_output.to_csv(
            CHECKPOINT_FILE,
            index=False
        )


    # -----------------------------------------------------
    # PROGRESS
    # -----------------------------------------------------

    if (
        completed % 25 == 0
        or completed == len(df)
    ):

        print(
            f"Completed "
            f"{completed}/{len(df)}"
        )


run_time = (
    time.perf_counter()
    - run_start
)


# =========================================================
# 10. CREATE FINAL PREDICTION DATAFRAME
# =========================================================

predictions_df = pd.DataFrame(
    records
)


# Ensure correct order
predictions_df = (
    predictions_df
    .sort_values(
        "Dataset_Index"
    )
    .reset_index(
        drop=True
    )
)


if len(predictions_df) != len(df):

    raise RuntimeError(
        "Final prediction count does not "
        "match test dataset size."
    )


predictions_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# =========================================================
# 11. VALID / INVALID OUTPUTS
# =========================================================

valid_df = predictions_df[
    predictions_df[
        "Valid_Output"
    ] == True
].copy()


invalid_df = predictions_df[
    predictions_df[
        "Valid_Output"
    ] == False
].copy()


num_total = len(
    predictions_df
)

num_valid = len(
    valid_df
)

num_invalid = len(
    invalid_df
)


print(
    "\n" + "=" * 70
)

print(
    "OUTPUT VALIDITY"
)

print(
    "=" * 70
)


print(
    "Total samples:",
    num_total
)

print(
    "Valid outputs:",
    num_valid
)

print(
    "Invalid outputs:",
    num_invalid
)


# =========================================================
# 12. PERFORMANCE METRICS
# =========================================================

if num_valid == 0:

    raise RuntimeError(
        "No valid predictions were produced."
    )


true_labels = valid_df[
    "True_Label"
].astype(str).values


predicted_labels = valid_df[
    "Predicted_Label"
].astype(str).values


accuracy = accuracy_score(
    true_labels,
    predicted_labels
)


macro_precision = precision_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)


macro_recall = recall_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)


macro_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="macro",
    zero_division=0
)


weighted_f1 = f1_score(
    true_labels,
    predicted_labels,
    average="weighted",
    zero_division=0
)


balanced_accuracy = (
    balanced_accuracy_score(
        true_labels,
        predicted_labels
    )
)


mcc = matthews_corrcoef(
    true_labels,
    predicted_labels
)


# =========================================================
# 13. LATENCY
# =========================================================

total_recorded_latency = (
    predictions_df[
        "Latency_s"
    ].sum()
)


mean_latency = (
    predictions_df[
        "Latency_s"
    ].mean()
)


median_latency = (
    predictions_df[
        "Latency_s"
    ].median()
)


# =========================================================
# 14. PRINT RESULTS
# =========================================================

print(
    "\n" + "=" * 70
)

print(
    "GROUP-DISJOINT TEST SUBSET — "
    "QWEN2.5 3B FEW-SHOT RESULTS"
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
    f"\nRecorded Inference Time: "
    f"{total_recorded_latency:.2f} sec"
)

print(
    f"Current Run Time:         "
    f"{run_time:.2f} sec"
)

print(
    f"Mean Latency:             "
    f"{mean_latency:.4f} sec"
)

print(
    f"Median Latency:           "
    f"{median_latency:.4f} sec"
)


# =========================================================
# 15. CLASSIFICATION REPORT
# =========================================================

print(
    "\nClassification Report:"
)


print(
    classification_report(
        true_labels,
        predicted_labels,
        labels=VALID_LABELS,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# 16. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    true_labels,
    predicted_labels,
    labels=VALID_LABELS
)


cm_df = pd.DataFrame(

    cm,

    index=[
        f"True_{label}"
        for label in VALID_LABELS
    ],

    columns=[
        f"Pred_{label}"
        for label in VALID_LABELS
    ]
)


cm_df.to_csv(
    CONFUSION_FILE
)


# =========================================================
# 17. SAVE SUMMARY
# =========================================================

result_df = pd.DataFrame([{

    "Model":
        "Qwen2.5 3B",

    "Prompting":
        "Few-shot",

    "Evaluation":
        "Group-disjoint robustness test subset",

    "Input":
        "Entity + Tweet",

    "Total Samples":
        num_total,

    "Valid Outputs":
        num_valid,

    "Invalid Outputs":
        num_invalid,

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

    "Recorded Inference Time (s)":
        total_recorded_latency,

    "Mean Latency (s)":
        mean_latency,

    "Median Latency (s)":
        median_latency,

    "Temperature":
        TEMPERATURE,

    "Seed":
        SEED
}])


result_df.to_csv(
    RESULT_FILE,
    index=False
)


# =========================================================
# 18. STANDARD BENCHMARK COMPARISON
# =========================================================
#
# IMPORTANT:
#
# Qwen is NOT trained on group_train.csv.
#
# Therefore this does not establish that the LLM itself
# is training-data group-disjoint.
#
# We are comparing its performance on two different
# evaluation compositions:
#
# 1. Original 1,000-sample benchmark
# 2. 6,865-sample group-disjoint robustness test subset
#
# =========================================================

STANDARD_ACCURACY = 0.5390
STANDARD_MACRO_F1 = 0.4884
STANDARD_BALANCED_ACCURACY = 0.5084
STANDARD_MCC = 0.3792


print(
    "\n" + "=" * 70
)

print(
    "STANDARD BENCHMARK vs "
    "GROUP-DISJOINT TEST SUBSET"
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
# 19. REMOVE CHECKPOINT AFTER SUCCESS
# =========================================================

if os.path.exists(
    CHECKPOINT_FILE
):

    os.remove(
        CHECKPOINT_FILE
    )

    print(
        "\nTemporary checkpoint removed."
    )


# =========================================================
# 20. FINISHED
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