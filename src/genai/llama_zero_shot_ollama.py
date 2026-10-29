# =========================================================
# GENAI ZERO-SHOT SENTIMENT ANALYSIS
# LOCAL LLAMA 3.2 3B VIA OLLAMA
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

INPUT_FILE = "data/processed/clean_validation.csv"

# First technical pilot.
# After successful execution change to None for all 1000.
SAMPLE_SIZE = None

RANDOM_STATE = 42

OUTPUT_PREDICTIONS = (
    "results/tables/"
    "genai_zero_shot_ollama_predictions.csv"
)

OUTPUT_RESULTS = (
    "results/tables/"
    "genai_zero_shot_ollama_results.csv"
)

OUTPUT_CM = (
    "results/tables/"
    "genai_zero_shot_ollama_confusion_matrix.csv"
)

os.makedirs(
    "results/tables",
    exist_ok=True
)

VALID_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

print("=" * 75)
print("GENAI ZERO-SHOT — LLAMA 3.2 3B")
print("=" * 75)

print(f"\nModel: {MODEL}")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

print(f"\nFull validation dataset: {df.shape}")

if SAMPLE_SIZE is None:

    eval_df = (
        df.copy()
        .reset_index(drop=True)
    )

else:

    eval_df = (
        df.sample(
            n=min(SAMPLE_SIZE, len(df)),
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

print(
    f"Samples to evaluate: {len(eval_df)}"
)


# =========================================================
# ZERO-SHOT PROMPT
# =========================================================

def build_prompt(entity, tweet):

    return f"""
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

Entity:
{entity}

Post:
{tweet}

Return ONLY one label:
Negative, Neutral, Positive, or Irrelevant.

Do not provide an explanation.
""".strip()


# =========================================================
# NORMALIZE MODEL OUTPUT
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

    # Conservative fallback:
    # accept a label only when the output clearly starts
    # with exactly one valid class name.

    matches = []

    lower_text = cleaned.lower()

    for label in VALID_LABELS:

        if lower_text.startswith(
            label.lower()
        ):
            matches.append(label)

    if len(matches) == 1:
        return matches[0]

    return None


# =========================================================
# TEST OLLAMA CONNECTION FIRST
# =========================================================

print("\nChecking Ollama connection...")

try:

    test_response = ollama.chat(
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

    test_text = (
        test_response["message"]["content"]
        .strip()
    )

    print(
        f"Ollama test response: {test_text}"
    )

except Exception as error:

    print("\nOLLAMA CONNECTION ERROR:")
    print(error)

    raise SystemExit(
        "\nExperiment stopped before processing tweets."
    )


# =========================================================
# CLASSIFICATION LOOP
# =========================================================

predictions = []
raw_outputs = []
latencies = []
errors = []

print("\n" + "=" * 75)
print("STARTING ZERO-SHOT CLASSIFICATION")
print("=" * 75)

experiment_start = time.perf_counter()

for index, row in eval_df.iterrows():

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
        f"[{index + 1}/{len(eval_df)}] "
        f"True={row['Sentiment']} | "
        f"Pred={shown_prediction} | "
        f"{latency:.2f}s"
    )


# =========================================================
# TOTAL TIME
# =========================================================

total_time = (
    time.perf_counter()
    - experiment_start
)


# =========================================================
# SAVE ALL PREDICTIONS
# =========================================================

prediction_df = eval_df.copy()

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


# =========================================================
# SAFETY CHECK
# =========================================================

if valid_outputs == 0:

    print("\n" + "=" * 75)
    print("NO VALID MODEL OUTPUTS")
    print("=" * 75)

    print(
        "\nNo valid predictions were produced."
    )

    print(
        "Check the Raw_Output and Error columns in:"
    )

    print(
        OUTPUT_PREDICTIONS
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
# RESULTS
# =========================================================

print("\n" + "=" * 75)
print("GENAI ZERO-SHOT RESULTS")
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
        f"True_{x}"
        for x in VALID_LABELS
    ],
    columns=[
        f"Pred_{x}"
        for x in VALID_LABELS
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
            "Zero-Shot",

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
# FINISHED
# =========================================================

print("\n" + "=" * 75)
print("ZERO-SHOT EXPERIMENT COMPLETED SUCCESSFULLY")
print("=" * 75)

print("\nFiles created:")

print(
    OUTPUT_PREDICTIONS
)

print(
    OUTPUT_RESULTS
)

print(
    OUTPUT_CM
)