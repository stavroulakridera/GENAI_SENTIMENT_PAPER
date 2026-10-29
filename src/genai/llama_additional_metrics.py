import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef
)


# =========================================================
# 1. SETTINGS
# =========================================================

RESULTS_DIR = "results/tables"

CONFIGURATIONS = {
    "Zero-shot": os.path.join(
        RESULTS_DIR,
        "genai_zero_shot_ollama_predictions.csv"
    ),
    "Few-shot": os.path.join(
        RESULTS_DIR,
        "genai_few_shot_ollama_predictions.csv"
    ),
    "Reasoning-enhanced": os.path.join(
        RESULTS_DIR,
        "genai_reasoning_ollama_predictions.csv"
    )
}

OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "llama32_3b_all_metrics.csv"
)

VALID_LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]


print("=" * 75)
print("LLAMA 3.2 3B — ADDITIONAL METRICS")
print("Balanced Accuracy + MCC")
print("=" * 75)


# =========================================================
# 2. HELPER: DETECT COLUMNS
# =========================================================

def find_column(df, possible_names):

    for name in possible_names:
        if name in df.columns:
            return name

    return None


# =========================================================
# 3. PROCESS EACH CONFIGURATION
# =========================================================

all_results = []


for configuration, filepath in CONFIGURATIONS.items():

    print("\n" + "=" * 75)
    print(configuration.upper())
    print("=" * 75)

    if not os.path.exists(filepath):

        raise FileNotFoundError(
            f"File not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    print(
        "File:",
        filepath
    )

    print(
        "Rows:",
        len(df)
    )

    print(
        "Columns:",
        list(df.columns)
    )


    # =====================================================
    # 4. DETECT TRUE LABEL COLUMN
    # =====================================================

    true_column = find_column(
        df,
        [
            "True_Label",
            "True Label",
            "True_Label_Name",
            "Actual",
            "Actual_Label",
            "Sentiment",
            "true_label",
            "label"
        ]
    )


    # =====================================================
    # 5. DETECT PREDICTION COLUMN
    # =====================================================

    prediction_column = find_column(
        df,
        [
            "Predicted_Label",
            "Predicted Label",
            "Prediction",
            "Predicted",
            "prediction",
            "predicted_label"
        ]
    )


    if true_column is None:

        raise ValueError(
            f"Could not identify true-label column "
            f"in {filepath}. "
            f"Available columns: {list(df.columns)}"
        )


    if prediction_column is None:

        raise ValueError(
            f"Could not identify prediction column "
            f"in {filepath}. "
            f"Available columns: {list(df.columns)}"
        )


    print(
        "True-label column:",
        true_column
    )

    print(
        "Prediction column:",
        prediction_column
    )


    # =====================================================
    # 6. NORMALIZE LABELS
    # =====================================================

    df[true_column] = (
        df[true_column]
        .astype(str)
        .str.strip()
    )

    df[prediction_column] = (
        df[prediction_column]
        .astype(str)
        .str.strip()
    )


    # Normalize capitalization
    label_mapping = {
        "negative": "Negative",
        "neutral": "Neutral",
        "positive": "Positive",
        "irrelevant": "Irrelevant"
    }


    df["_true_normalized"] = (
        df[true_column]
        .str.lower()
        .map(label_mapping)
    )

    df["_pred_normalized"] = (
        df[prediction_column]
        .str.lower()
        .map(label_mapping)
    )


    # =====================================================
    # 7. IDENTIFY VALID / INVALID OUTPUTS
    # =====================================================

    valid_mask = (
        df["_true_normalized"].isin(VALID_LABELS)
        &
        df["_pred_normalized"].isin(VALID_LABELS)
    )


    valid_df = df[
        valid_mask
    ].copy()


    total_samples = len(df)

    valid_outputs = len(valid_df)

    invalid_outputs = (
        total_samples
        - valid_outputs
    )

    invalid_rate = (
        invalid_outputs
        / total_samples
        if total_samples > 0
        else 0
    )


    print(
        "\nTotal samples:",
        total_samples
    )

    print(
        "Valid outputs:",
        valid_outputs
    )

    print(
        "Invalid outputs:",
        invalid_outputs
    )

    print(
        f"Invalid rate: {invalid_rate:.4%}"
    )


    if valid_outputs == 0:

        raise RuntimeError(
            f"No valid predictions found "
            f"for {configuration}."
        )


    # =====================================================
    # 8. METRICS
    # =====================================================

    y_true = valid_df[
        "_true_normalized"
    ].values

    y_pred = valid_df[
        "_pred_normalized"
    ].values


    accuracy = accuracy_score(
        y_true,
        y_pred
    )


    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )


    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )


    balanced_accuracy = balanced_accuracy_score(
        y_true,
        y_pred
    )


    mcc = matthews_corrcoef(
        y_true,
        y_pred
    )


    # =====================================================
    # 9. PRINT METRICS
    # =====================================================

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
        f"Balanced Accuracy: {balanced_accuracy:.4f}"
    )

    print(
        f"MCC:               {mcc:.4f}"
    )


    # =====================================================
    # 10. STORE RESULTS
    # =====================================================

    all_results.append({

        "Model":
            "Llama 3.2 3B",

        "Configuration":
            configuration,

        "Total Samples":
            total_samples,

        "Valid Outputs":
            valid_outputs,

        "Invalid Outputs":
            invalid_outputs,

        "Invalid Rate":
            invalid_rate,

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
            mcc
    })


# =========================================================
# 11. FINAL TABLE
# =========================================================

results_df = pd.DataFrame(
    all_results
)


print(
    "\n" + "=" * 75
)

print(
    "LLAMA 3.2 3B — FINAL COMPARISON"
)

print(
    "=" * 75
)


display_columns = [
    "Configuration",
    "Valid Outputs",
    "Invalid Outputs",
    "Accuracy",
    "Macro F1",
    "Balanced Accuracy",
    "MCC"
]


print(
    results_df[
        display_columns
    ].to_string(
        index=False
    )
)


# =========================================================
# 12. SAVE FINAL TABLE
# =========================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# 13. IDENTIFY BEST CONFIGURATION BY MACRO-F1
# =========================================================

best_index = (
    results_df["Macro F1"]
    .idxmax()
)


best_row = results_df.loc[
    best_index
]


print(
    "\n" + "=" * 75
)

print(
    "BEST LLAMA CONFIGURATION BY MACRO-F1"
)

print(
    "=" * 75
)


print(
    "Configuration:",
    best_row["Configuration"]
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
# 14. FINISHED
# =========================================================

print(
    "\n" + "=" * 75
)

print(
    "COMPLETED SUCCESSFULLY"
)

print(
    "=" * 75
)


print(
    "\nFile created:"
)

print(
    OUTPUT_FILE
)