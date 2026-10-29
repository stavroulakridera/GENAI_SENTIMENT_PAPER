# =========================================================
# GENAI FEW-SHOT SENTIMENT ANALYSIS
# LLAMA 3.2 3B VIA OLLAMA
# 4-SHOT: ONE TRAINING EXAMPLE PER CLASS
# =========================================================

import os
import time
import pandas as pd
import ollama

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# SETTINGS
# =========================================================

MODEL = "llama3.2:3b"

TRAIN_FILE = "results/clean_train.csv"
TEST_FILE = "results/clean_validation.csv"

RANDOM_STATE = 42

VALID_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

OUTPUT_PREDICTIONS = (
    "results/tables/"
    "genai_few_shot_ollama_predictions.csv"
)

OUTPUT_RESULTS = (
    "results/tables/"
    "genai_few_shot_ollama_results.csv"
)

OUTPUT_CM = (
    "results/tables/"
    "genai_few_shot_ollama_confusion_matrix.csv"
)

OUTPUT_EXAMPLES = (
    "results/tables/"
    "genai_few_shot_examples.csv"
)

os.makedirs(
    "results/tables",
    exist_ok=True
)

print("=" * 75)
print("GENAI FEW-SHOT — LLAMA 3.2 3B")
print("=" * 75)

print(f"\nModel: {MODEL}")
print("Prompting strategy: 4-shot")
print("Examples: 1 per sentiment class")


# =========================================================
# LOAD DATA
# =========================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

test_df = pd.read_csv(
    TEST_FILE
)

print("\nTraining dataset:")
print(train_df.shape)

print("\nEvaluation dataset:")
print(test_df.shape)


# =========================================================
# SELECT ONE TRAINING EXAMPLE PER CLASS
# =========================================================

few_shot_examples = []

for label in VALID_LABELS:

    candidates = train_df[
        train_df["Sentiment"] == label
    ]

    if len(candidates) == 0:
        raise ValueError(
            f"No training examples found for {label}"
        )

    example = candidates.sample(
        n=1,
        random_state=RANDOM_STATE
    ).iloc[0]

    few_shot_examples.append(
        {
            "Entity": str(example["Entity"]),
            "Tweet": str(example["Tweet"]),
            "Sentiment": label
        }
    )

examples_df = pd.DataFrame(
    few_shot_examples
)

examples_df.to_csv(
    OUTPUT_EXAMPLES,
    index=False
)

print("\n" + "=" * 75)
print("FEW-SHOT EXAMPLES")
print("=" * 75)

for i, example in enumerate(
    few_shot_examples,
    start=1
):

    print(
        f"\nExample {i}"
    )

    print(
        f"Entity: {example['Entity']}"
    )

    print(
        f"Tweet: {example['Tweet']}"
    )

    print(
        f"Label: {example['Sentiment']}"
    )


# =========================================================
# BUILD FEW-SHOT PROMPT
# =========================================================

def build_prompt(entity, tweet):

    examples_text = ""

    for i, example in enumerate(
        few_shot_examples,
        start=1
    ):

        examples_text += f"""
Example {i}

Entity:
{example["Entity"]}

Post:
{example["Tweet"]}

Correct label:
{example["Sentiment"]}

"""

    prompt = f"""
You are a sentiment classification system.

Your task is to classify the sentiment of a social-media
post toward the specified entity.

You MUST select exactly one of these four labels:

Negative
Neutral
Positive
Irrelevant

Label definitions:

Negative:
The post expresses criticism, dissatisfaction, dislike,
hostility, disappointment, or another negative attitude
toward the specified entity.

Neutral:
The post is meaningfully related to the specified entity,
but does not express a clearly positive or negative
attitude toward it.

Positive:
The post expresses approval, satisfaction, praise,
enthusiasm, support, or another positive attitude toward
the specified entity.

Irrelevant:
The post does not meaningfully concern the specified
entity, or the content cannot reasonably be interpreted
as sentiment toward that entity.

Here are four labeled examples from the training data.

{examples_text}

Now classify the following new post.

Entity:
{entity}

Post:
{tweet}

Return ONLY one label:
Negative, Neutral, Positive, or Irrelevant.

Do not provide an explanation.
""".strip()

    return prompt


# =========================================================
# NORMALIZE OUTPUT
# =========================================================

def normalize_prediction(text):

    if text is None:
        return None

    cleaned = (
        str(text)
        .strip()
        .replace(".", "")
        .replace('"', "")
        .replace("'", "")
    )

    for label in VALID_LABELS:

        if cleaned.lower() == label.lower():
            return label

    lower_text = cleaned.lower()

    matches = []

    for label in VALID_LABELS:

        if lower_text.startswith(
            label.lower()
        ):
            matches.append(label)

    if len(matches) == 1:
        return matches[0]

    return None


# =========================================================
# CHECK OLLAMA
# =========================================================

print("\nChecking Ollama connection...")

try:

    response = ollama.chat(
        model=MODEL,

        messages=[
            {
                "role": "user",
                "content":
                    "Return only the word Positive."
            }
        ],

        options={
            "temperature": 0,
            "seed": RANDOM_STATE
        }
    )

    print(
        "Ollama test response:",
        response["message"]["content"].strip()
    )

except Exception as error:

    print("\nOLLAMA ERROR:")
    print(error)

    raise SystemExit


# =========================================================
# RUN FEW-SHOT EXPERIMENT
# =========================================================

predictions = []
raw_outputs = []
latencies = []
errors = []

print("\n" + "=" * 75)
print("STARTING FEW-SHOT CLASSIFICATION")
print("=" * 75)

experiment_start = time.perf_counter()

for index, row in test_df.iterrows():

    entity = str(
        row["Entity"]
    )

    tweet = str(
        row["Tweet"]
    )

    prompt = build_prompt(
        entity,
        tweet
    )

    request_start = time.perf_counter()

    try:

        response = ollama.chat(
            model=MODEL,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],

            options={
                "temperature": 0,
                "seed": RANDOM_STATE
            }
        )

        raw_output = (
            response["message"]["content"]
            .strip()
        )

        prediction = normalize_prediction(
            raw_output
        )

        error = ""

        if prediction is None:

            error = (
                "Invalid model output: "
                + raw_output
            )

    except Exception as e:

        raw_output = ""
        prediction = None
        error = str(e)

    latency = (
        time.perf_counter()
        - request_start
    )

    predictions.append(
        prediction
    )

    raw_outputs.append(
        raw_output
    )

    latencies.append(
        latency
    )

    errors.append(
        error
    )

    shown_prediction = (
        prediction
        if prediction is not None
        else "INVALID"
    )

    print(
        f"[{index + 1}/"
        f"{len(test_df)}] "
        f"True={row['Sentiment']} | "
        f"Pred={shown_prediction} | "
        f"{latency:.2f}s"
    )


total_time = (
    time.perf_counter()
    - experiment_start
)


# =========================================================
# CREATE PREDICTION DATAFRAME
# =========================================================

prediction_df = test_df.copy()

prediction_df[
    "Predicted_Label"
] = predictions

prediction_df[
    "Raw_Output"
] = raw_outputs

prediction_df[
    "Latency_s"
] = latencies

prediction_df[
    "Error"
] = errors

prediction_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# =========================================================
# VALID OUTPUTS
# =========================================================

valid_mask = (
    prediction_df[
        "Predicted_Label"
    ].isin(VALID_LABELS)
)

valid_df = (
    prediction_df[
        valid_mask
    ]
    .copy()
)

valid_outputs = len(
    valid_df
)

invalid_outputs = (
    len(prediction_df)
    - valid_outputs
)

if valid_outputs == 0:

    print(
        "\nERROR: No valid predictions."
    )

    raise SystemExit


# =========================================================
# METRICS
# =========================================================

y_true = valid_df[
    "Sentiment"
]

y_pred = valid_df[
    "Predicted_Label"
]

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_precision = precision_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    labels=VALID_LABELS,
    average="weighted",
    zero_division=0
)


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 75)
print("GENAI FEW-SHOT RESULTS")
print("=" * 75)

print(
    f"\nModel:             {MODEL}"
)

print(
    f"Evaluated Samples: {len(prediction_df)}"
)

print(
    f"Valid Outputs:     {valid_outputs}"
)

print(
    f"Invalid Outputs:   {invalid_outputs}"
)

print(
    f"\nAccuracy:          {accuracy:.4f}"
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
    f"\nTotal Time:        {total_time:.2f} sec"
)

print(
    f"Mean Latency:      "
    f"{prediction_df['Latency_s'].mean():.3f} sec"
)


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=VALID_LABELS,
        digits=4,
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    y_true,
    y_pred,
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
    OUTPUT_CM
)


# =========================================================
# SAVE SUMMARY
# =========================================================

summary_df = pd.DataFrame([
    {
        "Model":
            "Llama 3.2 3B",

        "Method":
            "Few-Shot (4-shot)",

        "Samples":
            len(prediction_df),

        "Valid Outputs":
            valid_outputs,

        "Invalid Outputs":
            invalid_outputs,

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

        "Total Time (s)":
            total_time,

        "Mean Latency (s)":
            prediction_df[
                "Latency_s"
            ].mean()
    }
])

summary_df.to_csv(
    OUTPUT_RESULTS,
    index=False
)


# =========================================================
# ZERO-SHOT VS FEW-SHOT
# =========================================================

ZERO_SHOT_ACCURACY = 0.4679
ZERO_SHOT_MACRO_F1 = 0.3581

print("\n" + "=" * 75)
print("ZERO-SHOT VS FEW-SHOT")
print("=" * 75)

print(
    f"\nZero-Shot Accuracy: "
    f"{ZERO_SHOT_ACCURACY:.4f}"
)

print(
    f"Few-Shot Accuracy:  "
    f"{accuracy:.4f}"
)

print(
    f"Accuracy Change:    "
    f"{accuracy - ZERO_SHOT_ACCURACY:+.4f}"
)

print(
    f"\nZero-Shot Macro F1: "
    f"{ZERO_SHOT_MACRO_F1:.4f}"
)

print(
    f"Few-Shot Macro F1:  "
    f"{macro_f1:.4f}"
)

print(
    f"Macro F1 Change:    "
    f"{macro_f1 - ZERO_SHOT_MACRO_F1:+.4f}"
)


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 75)
print("FEW-SHOT EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(OUTPUT_PREDICTIONS)
print(OUTPUT_RESULTS)
print(OUTPUT_CM)
print(OUTPUT_EXAMPLES)