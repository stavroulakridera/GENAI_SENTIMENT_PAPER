# =========================================================
# GENAI REASONING-ENHANCED SENTIMENT ANALYSIS
# LLAMA 3.2 3B VIA OLLAMA
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

TEST_FILE = "data/processed/clean_validation.csv"

RANDOM_STATE = 42

VALID_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]

OUTPUT_PREDICTIONS = (
    "results/tables/"
    "genai_reasoning_ollama_predictions.csv"
)

OUTPUT_RESULTS = (
    "results/tables/"
    "genai_reasoning_ollama_results.csv"
)

OUTPUT_CM = (
    "results/tables/"
    "genai_reasoning_ollama_confusion_matrix.csv"
)

os.makedirs(
    "results/tables",
    exist_ok=True
)


# =========================================================
# HEADER
# =========================================================

print("=" * 75)
print("GENAI REASONING-ENHANCED — LLAMA 3.2 3B")
print("=" * 75)

print(f"\nModel: {MODEL}")
print("Strategy: Reasoning-Enhanced Decision Procedure")
print("Temperature: 0")


# =========================================================
# LOAD DATA
# =========================================================

test_df = pd.read_csv(
    TEST_FILE
)

print("\nEvaluation dataset:")
print(test_df.shape)

print(
    f"\nSamples to evaluate: {len(test_df)}"
)


# =========================================================
# REASONING-ENHANCED PROMPT
# =========================================================

def build_prompt(entity, tweet):

    return f"""
You are a sentiment classification system for social-media
content.

Your task is to classify the sentiment of the post toward
the specified entity.

You MUST choose exactly one label:

Negative
Neutral
Positive
Irrelevant


Use the following decision procedure before selecting
the final label.


STEP 1 — RELEVANCE

Determine whether the post is meaningfully about the
specified entity.

If the post is unrelated to the entity, only mentions it
incidentally, or does not allow a meaningful judgment
about the entity:

Choose Irrelevant.

Do NOT classify an unrelated post as Positive or Negative
just because it contains emotional language.


STEP 2 — SENTIMENT PRESENCE

If the post is meaningfully about the entity, determine
whether it expresses a clear evaluative sentiment toward
that entity.

If it mainly states information, describes an event,
reports facts, asks a question, or discusses the entity
without a clearly favorable or unfavorable attitude:

Choose Neutral.

Do NOT force a Neutral post into Positive or Negative.


STEP 3 — SENTIMENT POLARITY

Only if the post is relevant AND contains a clear
evaluative sentiment:

Choose Positive when it expresses approval, satisfaction,
praise, enthusiasm, support, or another favorable attitude.

Choose Negative when it expresses criticism,
dissatisfaction, dislike, hostility, disappointment,
complaint, or another unfavorable attitude.


IMPORTANT DISTINCTION

Irrelevant =
the post is not meaningfully about the specified entity.

Neutral =
the post IS meaningfully about the specified entity,
but there is no clear positive or negative attitude.

Positive / Negative =
the post IS meaningfully about the specified entity
AND contains clear evaluative sentiment.


Now classify the following post.

Entity:
{entity}

Post:
{tweet}


Apply the decision procedure internally.

Return ONLY the final label.

Allowed output:
Negative
Neutral
Positive
Irrelevant

Do not provide an explanation.
""".strip()


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

    # Exact match first
    for label in VALID_LABELS:

        if cleaned.lower() == label.lower():
            return label

    # Conservative fallback
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
# CHECK OLLAMA CONNECTION
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
        test_response[
            "message"
        ][
            "content"
        ].strip()
    )

    print(
        f"Ollama test response: {test_text}"
    )

except Exception as error:

    print("\nOLLAMA CONNECTION ERROR:")
    print(error)

    raise SystemExit(
        "\nExperiment stopped."
    )


# =========================================================
# RUN EXPERIMENT
# =========================================================

predictions = []
raw_outputs = []
latencies = []
errors = []

print("\n" + "=" * 75)
print("STARTING REASONING-ENHANCED CLASSIFICATION")
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
            response[
                "message"
            ][
                "content"
            ]
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
        f"[{index + 1}/{len(test_df)}] "
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
# PREDICTION DATAFRAME
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
print("GENAI REASONING-ENHANCED RESULTS")
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
            "Reasoning-Enhanced",

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
# COMPARE ALL THREE GENAI STRATEGIES
# =========================================================

ZERO_SHOT_ACCURACY = 0.4679
ZERO_SHOT_MACRO_F1 = 0.3581

FEW_SHOT_ACCURACY = 0.4734
FEW_SHOT_MACRO_F1 = 0.3374


print("\n" + "=" * 75)
print("GENAI STRATEGY COMPARISON")
print("=" * 75)


comparison_df = pd.DataFrame([
    {
        "Strategy":
            "Zero-Shot",

        "Accuracy":
            ZERO_SHOT_ACCURACY,

        "Macro_F1":
            ZERO_SHOT_MACRO_F1
    },

    {
        "Strategy":
            "Few-Shot",

        "Accuracy":
            FEW_SHOT_ACCURACY,

        "Macro_F1":
            FEW_SHOT_MACRO_F1
    },

    {
        "Strategy":
            "Reasoning-Enhanced",

        "Accuracy":
            accuracy,

        "Macro_F1":
            macro_f1
    }
])


print(
    "\n"
    + comparison_df
    .round(4)
    .to_string(index=False)
)


comparison_df.to_csv(
    "results/tables/"
    "genai_strategy_comparison.csv",
    index=False
)


# =========================================================
# BEST GENAI STRATEGY
# =========================================================

best_row = (
    comparison_df
    .sort_values(
        "Macro_F1",
        ascending=False
    )
    .iloc[0]
)


print("\nBest GenAI strategy by Macro F1:")

print(
    f"{best_row['Strategy']} "
    f"| Accuracy = "
    f"{best_row['Accuracy']:.4f} "
    f"| Macro F1 = "
    f"{best_row['Macro_F1']:.4f}"
)


# =========================================================
# FINISHED
# =========================================================

print("\n" + "=" * 75)
print("REASONING-ENHANCED EXPERIMENT COMPLETED SUCCESSFULLY")
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

print(
    "results/tables/"
    "genai_strategy_comparison.csv"
)