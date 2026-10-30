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
    classification_report
)


# =========================================================
# 1. SETTINGS
# =========================================================

MODEL_NAME = "gemma3:4b"

OLLAMA_URL = "http://localhost:11434/api/generate"

INPUT_FILE = "results/entity_tweet_validation.csv"

RESULTS_DIR = "results/tables"

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "gemma_few_shot_results.csv"
)

PREDICTIONS_FILE = os.path.join(
    RESULTS_DIR,
    "gemma_few_shot_predictions.csv"
)

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

VALID_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

TEMPERATURE = 0
SEED = 42


print("=" * 70)
print("GEMMA 3 4B — FEW-SHOT SENTIMENT CLASSIFICATION")
print("=" * 70)

print("\nModel:", MODEL_NAME)
print("Temperature:", TEMPERATURE)
print("Seed:", SEED)


# =========================================================
# 2. LOAD VALIDATION DATA
# =========================================================

print("\nLoading validation dataset...")

df = pd.read_csv(
    INPUT_FILE
)

required_columns = [
    "Tweet_ID",
    "Entity",
    "Sentiment",
    "Tweet",
    "Entity_Tweet"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Missing column: {column}"
        )


print(
    "Validation samples:",
    len(df)
)

print("\nClass distribution:")

print(
    df["Sentiment"].value_counts()
)


# =========================================================
# 3. FEW-SHOT PROMPT
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
# 4. CALL OLLAMA
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
# 5. NORMALIZE MODEL OUTPUT
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

    # Exact label
    for label in VALID_LABELS:

        if cleaned.lower() == label.lower():
            return label

    # Accept simple output variants such as:
    # "Positive."
    # "Label: Positive"
    # "The label is Positive"
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

    # Ambiguous or invalid output
    return None


# =========================================================
# 6. RUN FEW-SHOT EXPERIMENT
# =========================================================

records = []

total_start = time.perf_counter()

print(
    "\nStarting few-shot evaluation...\n"
)

for index, row in df.iterrows():

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

    records.append({

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
            prediction
            if prediction is not None
            else "INVALID",

        "Valid_Output":
            valid,

        "Latency_s":
            latency,

        "Error":
            error
    })

    completed = index + 1

    if (
        completed % 25 == 0
        or completed == len(df)
    ):

        print(
            f"Completed "
            f"{completed}/{len(df)}"
        )


total_time = (
    time.perf_counter()
    - total_start
)


# =========================================================
# 7. SAVE ALL PREDICTIONS
# =========================================================

predictions_df = pd.DataFrame(
    records
)

predictions_df.to_csv(
    PREDICTIONS_FILE,
    index=False
)


# =========================================================
# 8. VALID / INVALID OUTPUTS
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


print("\n" + "=" * 70)
print("OUTPUT VALIDITY")
print("=" * 70)

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
# 9. PERFORMANCE METRICS
# =========================================================

if num_valid == 0:

    raise RuntimeError(
        "No valid predictions were produced."
    )


true_labels = valid_df[
    "True_Label"
].values

predicted_labels = valid_df[
    "Predicted_Label"
].values


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
# 10. LATENCY
# =========================================================

mean_latency = predictions_df[
    "Latency_s"
].mean()

median_latency = predictions_df[
    "Latency_s"
].median()


# =========================================================
# 11. PRINT RESULTS
# =========================================================

print("\n" + "=" * 70)
print("GEMMA 3 4B — FEW-SHOT RESULTS")
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
    f"\nTotal Time:        "
    f"{total_time:.2f} sec"
)

print(
    f"Mean Latency:      "
    f"{mean_latency:.4f} sec"
)

print(
    f"Median Latency:    "
    f"{median_latency:.4f} sec"
)


# =========================================================
# 12. CLASSIFICATION REPORT
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
# 13. SAVE SUMMARY
# =========================================================

summary_df = pd.DataFrame([{

    "Model":
        "Gemma 3 4B",

    "Prompting":
        "Few-shot",

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

    "Total Time (s)":
        total_time,

    "Mean Latency (s)":
        mean_latency,

    "Median Latency (s)":
        median_latency,

    "Temperature":
        TEMPERATURE,

    "Seed":
        SEED
}])

summary_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 14. FINISHED
# =========================================================

print("\n" + "=" * 70)
print("EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    OUTPUT_FILE
)

print(
    PREDICTIONS_FILE
)