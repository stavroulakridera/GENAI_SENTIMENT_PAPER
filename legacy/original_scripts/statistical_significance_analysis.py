import os
import itertools
import math

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score
)


# =========================================================
# 1. SETTINGS
# =========================================================

RESULTS_DIR = "results/tables"

SVM_FILE = os.path.join(
    RESULTS_DIR,
    "linear_svm_entity_predictions.csv"
)

DNN_FILE = os.path.join(
    RESULTS_DIR,
    "dnn_entity_5seeds_predictions.csv"
)

BERT_FILE = os.path.join(
    RESULTS_DIR,
    "bert_entity_5seeds_predictions.csv"
)

QWEN_FILE = os.path.join(
    RESULTS_DIR,
    "qwen_few_shot_predictions.csv"
)

CI_OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "macro_f1_bootstrap_95ci.csv"
)

MCNEMAR_OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "pairwise_mcnemar_tests.csv"
)

COMBINED_OUTPUT_FILE = os.path.join(
    RESULTS_DIR,
    "statistical_analysis_summary.csv"
)

BOOTSTRAP_ITERATIONS = 2000
BOOTSTRAP_SEED = 42

SELECTED_SEED = 42

LABELS = [
    "Negative",
    "Neutral",
    "Positive",
    "Irrelevant"
]


print("=" * 80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("95% BOOTSTRAP CI + PAIRWISE McNEMAR TESTS")
print("=" * 80)

print(
    "\nBootstrap iterations:",
    BOOTSTRAP_ITERATIONS
)

print(
    "Bootstrap seed:",
    BOOTSTRAP_SEED
)

print(
    "DNN/BERT selected seed:",
    SELECTED_SEED
)


# =========================================================
# 2. LOAD FILES
# =========================================================

print("\nLoading prediction files...")


svm_df = pd.read_csv(
    SVM_FILE
)

dnn_all_df = pd.read_csv(
    DNN_FILE
)

bert_all_df = pd.read_csv(
    BERT_FILE
)

qwen_df = pd.read_csv(
    QWEN_FILE
)


print(
    "SVM shape:",
    svm_df.shape
)

print(
    "DNN all-seed shape:",
    dnn_all_df.shape
)

print(
    "BERT all-seed shape:",
    bert_all_df.shape
)

print(
    "Qwen shape:",
    qwen_df.shape
)


# =========================================================
# 3. SELECT SEED 42 FOR DNN AND BERT
# =========================================================

dnn_df = (
    dnn_all_df[
        dnn_all_df["Seed"] == SELECTED_SEED
    ]
    .copy()
    .reset_index(drop=True)
)


bert_df = (
    bert_all_df[
        bert_all_df["Seed"] == SELECTED_SEED
    ]
    .copy()
    .reset_index(drop=True)
)


print(
    "\nDNN seed-42 rows:",
    len(dnn_df)
)

print(
    "BERT seed-42 rows:",
    len(bert_df)
)


if len(dnn_df) != 1000:

    raise RuntimeError(
        "Expected exactly 1000 DNN predictions "
        f"for seed {SELECTED_SEED}, "
        f"but found {len(dnn_df)}."
    )


if len(bert_df) != 1000:

    raise RuntimeError(
        "Expected exactly 1000 BERT predictions "
        f"for seed {SELECTED_SEED}, "
        f"but found {len(bert_df)}."
    )


# =========================================================
# 4. STANDARDIZE DATAFRAMES
# =========================================================

svm = pd.DataFrame({

    "Tweet_ID":
        svm_df["Tweet_ID"],

    "Entity":
        svm_df["Entity"],

    "Tweet":
        svm_df["Tweet"],

    "True_Label":
        svm_df["Sentiment"],

    "Prediction":
        svm_df["Prediction"]
})


dnn = pd.DataFrame({

    "Tweet_ID":
        dnn_df["Tweet_ID"],

    "Entity":
        dnn_df["Entity"],

    "Tweet":
        dnn_df["Tweet"],

    "True_Label":
        dnn_df["True_Label"],

    "Prediction":
        dnn_df["Predicted_Label"]
})


bert = pd.DataFrame({

    "Tweet_ID":
        bert_df["Tweet_ID"],

    "Entity":
        bert_df["Entity"],

    "Tweet":
        bert_df["Tweet"],

    "True_Label":
        bert_df["True_Label"],

    "Prediction":
        bert_df["Predicted_Label"]
})


qwen = pd.DataFrame({

    "Tweet_ID":
        qwen_df["Tweet_ID"],

    "Entity":
        qwen_df["Entity"],

    "Tweet":
        qwen_df["Tweet"],

    "True_Label":
        qwen_df["True_Label"],

    "Prediction":
        qwen_df["Predicted_Label"]
})


# =========================================================
# 5. NORMALIZE STRINGS
# =========================================================

datasets = {
    "Linear SVM": svm,
    "DNN": dnn,
    "BERT": bert,
    "Qwen2.5 3B Few-shot": qwen
}


for name, df in datasets.items():

    df["Entity"] = (
        df["Entity"]
        .astype(str)
        .str.strip()
    )

    df["Tweet"] = (
        df["Tweet"]
        .astype(str)
        .str.strip()
    )

    df["True_Label"] = (
        df["True_Label"]
        .astype(str)
        .str.strip()
    )

    df["Prediction"] = (
        df["Prediction"]
        .astype(str)
        .str.strip()
    )

    print(
        f"\n{name}: {len(df)} samples"
    )


# =========================================================
# 6. VERIFY COMMON TEST SET
# =========================================================
#
# Tweet_ID alone is NOT guaranteed to be unique in this
# dataset. Therefore we verify alignment row-by-row using:
#
# Tweet_ID + Entity + Tweet + True_Label
#
# =========================================================

reference_name = "Linear SVM"
reference_df = datasets[
    reference_name
].reset_index(drop=True)


if len(reference_df) != 1000:

    raise RuntimeError(
        "Reference test set does not contain "
        "exactly 1000 samples."
    )


for name, df in datasets.items():

    df = df.reset_index(
        drop=True
    )

    if len(df) != len(reference_df):

        raise RuntimeError(
            f"{name} has {len(df)} samples, "
            f"expected {len(reference_df)}."
        )


    id_match = np.array_equal(
        reference_df["Tweet_ID"].to_numpy(),
        df["Tweet_ID"].to_numpy()
    )


    entity_match = np.array_equal(
        reference_df["Entity"].to_numpy(),
        df["Entity"].to_numpy()
    )


    tweet_match = np.array_equal(
        reference_df["Tweet"].to_numpy(),
        df["Tweet"].to_numpy()
    )


    label_match = np.array_equal(
        reference_df["True_Label"].to_numpy(),
        df["True_Label"].to_numpy()
    )


    print(
        f"\nAlignment check — {name}"
    )

    print(
        "Tweet_ID match:",
        id_match
    )

    print(
        "Entity match:",
        entity_match
    )

    print(
        "Tweet match:",
        tweet_match
    )

    print(
        "True-label match:",
        label_match
    )


    if not (
        id_match
        and entity_match
        and tweet_match
        and label_match
    ):

        raise RuntimeError(
            f"Test-set alignment failed for {name}. "
            "Do NOT perform paired statistical tests."
        )


print(
    "\nCOMMON TEST-SET ALIGNMENT PASSED."
)

print(
    "All four models use the same 1,000 samples "
    "in the same order."
)


# =========================================================
# 7. VERIFY LABEL VALIDITY
# =========================================================

for name, df in datasets.items():

    invalid_true = (
        ~df["True_Label"].isin(LABELS)
    ).sum()

    invalid_predictions = (
        ~df["Prediction"].isin(LABELS)
    ).sum()


    print(
        f"\n{name}"
    )

    print(
        "Invalid true labels:",
        invalid_true
    )

    print(
        "Invalid predictions:",
        invalid_predictions
    )


    if invalid_true != 0:

        raise RuntimeError(
            f"{name} contains invalid true labels."
        )


    if invalid_predictions != 0:

        raise RuntimeError(
            f"{name} contains invalid predictions. "
            "Paired analysis requires a prediction "
            "for every common test sample."
        )


# =========================================================
# 8. POINT ESTIMATES
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "POINT ESTIMATES"
)

print(
    "=" * 80
)


for name, df in datasets.items():

    accuracy = accuracy_score(
        df["True_Label"],
        df["Prediction"]
    )

    macro_f1 = f1_score(
        df["True_Label"],
        df["Prediction"],
        labels=LABELS,
        average="macro",
        zero_division=0
    )

    print(
        f"{name:24s} "
        f"Accuracy={accuracy:.4f}  "
        f"Macro-F1={macro_f1:.4f}"
    )


# =========================================================
# 9. BOOTSTRAP FUNCTION
# =========================================================

def bootstrap_macro_f1(
    y_true,
    y_pred,
    iterations=2000,
    seed=42
):

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    n = len(
        y_true
    )

    rng = np.random.default_rng(
        seed
    )

    scores = np.empty(
        iterations,
        dtype=np.float64
    )


    for i in range(
        iterations
    ):

        indices = rng.integers(
            0,
            n,
            size=n
        )

        bootstrap_true = y_true[
            indices
        ]

        bootstrap_pred = y_pred[
            indices
        ]


        scores[i] = f1_score(
            bootstrap_true,
            bootstrap_pred,
            labels=LABELS,
            average="macro",
            zero_division=0
        )


    lower = np.percentile(
        scores,
        2.5
    )

    upper = np.percentile(
        scores,
        97.5
    )


    return (
        lower,
        upper,
        scores
    )


# =========================================================
# 10. 95% BOOTSTRAP CI FOR MACRO-F1
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "95% BOOTSTRAP CONFIDENCE INTERVALS — MACRO-F1"
)

print(
    "=" * 80
)


ci_results = []


for model_index, (name, df) in enumerate(
    datasets.items()
):

    y_true = df[
        "True_Label"
    ].to_numpy()

    y_pred = df[
        "Prediction"
    ].to_numpy()


    point_macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=LABELS,
        average="macro",
        zero_division=0
    )


    # Use a deterministic but distinct bootstrap stream
    # for each model.
    model_bootstrap_seed = (
        BOOTSTRAP_SEED
        + model_index
    )


    ci_lower, ci_upper, bootstrap_scores = (
        bootstrap_macro_f1(
            y_true,
            y_pred,
            iterations=BOOTSTRAP_ITERATIONS,
            seed=model_bootstrap_seed
        )
    )


    ci_results.append({

        "Model":
            name,

        "N":
            len(df),

        "Macro F1":
            point_macro_f1,

        "CI Lower 95%":
            ci_lower,

        "CI Upper 95%":
            ci_upper,

        "Bootstrap Iterations":
            BOOTSTRAP_ITERATIONS
    })


    print(
        f"{name:24s} "
        f"{point_macro_f1:.4f} "
        f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    )


ci_df = pd.DataFrame(
    ci_results
)


ci_df.to_csv(
    CI_OUTPUT_FILE,
    index=False
)


# =========================================================
# 11. EXACT BINOMIAL TWO-SIDED P-VALUE
# =========================================================
#
# McNemar's exact test depends only on the discordant
# pairs b and c.
#
# Under H0:
# b ~ Binomial(b+c, 0.5)
#
# We implement the exact two-sided binomial test directly
# to avoid requiring statsmodels/scipy-specific APIs.
# =========================================================

def exact_binomial_two_sided(
    b,
    c
):

    n = b + c

    if n == 0:
        return 1.0


    k = min(
        b,
        c
    )


    cumulative_probability = 0.0


    for i in range(
        0,
        k + 1
    ):

        cumulative_probability += (
            math.comb(n, i)
            *
            (0.5 ** n)
        )


    p_value = min(
        1.0,
        2.0 * cumulative_probability
    )


    return p_value


# =========================================================
# 12. McNEMAR PAIRWISE TEST
# =========================================================

def mcnemar_test(
    y_true,
    pred_a,
    pred_b
):

    y_true = np.asarray(
        y_true
    )

    pred_a = np.asarray(
        pred_a
    )

    pred_b = np.asarray(
        pred_b
    )


    correct_a = (
        pred_a == y_true
    )

    correct_b = (
        pred_b == y_true
    )


    both_correct = np.sum(
        correct_a
        &
        correct_b
    )


    a_correct_b_wrong = np.sum(
        correct_a
        &
        ~correct_b
    )


    a_wrong_b_correct = np.sum(
        ~correct_a
        &
        correct_b
    )


    both_wrong = np.sum(
        ~correct_a
        &
        ~correct_b
    )


    discordant = (
        a_correct_b_wrong
        +
        a_wrong_b_correct
    )


    p_value = exact_binomial_two_sided(
        int(a_correct_b_wrong),
        int(a_wrong_b_correct)
    )


    return {
        "Both Correct":
            int(both_correct),

        "A Correct B Wrong":
            int(a_correct_b_wrong),

        "A Wrong B Correct":
            int(a_wrong_b_correct),

        "Both Wrong":
            int(both_wrong),

        "Discordant Pairs":
            int(discordant),

        "Exact p-value":
            p_value
    }


# =========================================================
# 13. RUN ALL SIX PAIRWISE McNEMAR TESTS
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "PAIRWISE McNEMAR TESTS"
)

print(
    "=" * 80
)


model_names = list(
    datasets.keys()
)


mcnemar_results = []


for model_a, model_b in itertools.combinations(
    model_names,
    2
):

    df_a = datasets[
        model_a
    ]

    df_b = datasets[
        model_b
    ]


    y_true = df_a[
        "True_Label"
    ].to_numpy()

    pred_a = df_a[
        "Prediction"
    ].to_numpy()

    pred_b = df_b[
        "Prediction"
    ].to_numpy()


    result = mcnemar_test(
        y_true,
        pred_a,
        pred_b
    )


    accuracy_a = accuracy_score(
        y_true,
        pred_a
    )

    accuracy_b = accuracy_score(
        y_true,
        pred_b
    )


    row = {

        "Model A":
            model_a,

        "Model B":
            model_b,

        "Accuracy A":
            accuracy_a,

        "Accuracy B":
            accuracy_b,

        **result
    }


    mcnemar_results.append(
        row
    )


    print(
        f"\n{model_a} vs {model_b}"
    )

    print(
        f"Accuracy: "
        f"{accuracy_a:.4f} vs {accuracy_b:.4f}"
    )

    print(
        "A correct / B wrong:",
        result["A Correct B Wrong"]
    )

    print(
        "A wrong / B correct:",
        result["A Wrong B Correct"]
    )

    print(
        "Discordant pairs:",
        result["Discordant Pairs"]
    )

    print(
        f"Exact p-value: "
        f"{result['Exact p-value']:.8g}"
    )


mcnemar_df = pd.DataFrame(
    mcnemar_results
)


# =========================================================
# 14. HOLM MULTIPLE-COMPARISON CORRECTION
# =========================================================

def holm_adjust(
    p_values
):

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    m = len(
        p_values
    )

    order = np.argsort(
        p_values
    )

    adjusted = np.empty(
        m,
        dtype=float
    )


    running_max = 0.0


    for rank, index in enumerate(
        order
    ):

        multiplier = (
            m - rank
        )

        candidate = (
            multiplier
            *
            p_values[index]
        )

        candidate = min(
            candidate,
            1.0
        )


        running_max = max(
            running_max,
            candidate
        )


        adjusted[index] = min(
            running_max,
            1.0
        )


    return adjusted


mcnemar_df[
    "Holm Adjusted p-value"
] = holm_adjust(

    mcnemar_df[
        "Exact p-value"
    ].to_numpy()
)


mcnemar_df[
    "Significant Raw p<0.05"
] = (

    mcnemar_df[
        "Exact p-value"
    ]
    < 0.05
)


mcnemar_df[
    "Significant Holm p<0.05"
] = (

    mcnemar_df[
        "Holm Adjusted p-value"
    ]
    < 0.05
)


# =========================================================
# 15. PRINT CORRECTED McNEMAR RESULTS
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "McNEMAR RESULTS WITH HOLM CORRECTION"
)

print(
    "=" * 80
)


for _, row in mcnemar_df.iterrows():

    print(
        f"\n{row['Model A']} "
        f"vs "
        f"{row['Model B']}"
    )

    print(
        f"Raw p-value:  "
        f"{row['Exact p-value']:.8g}"
    )

    print(
        f"Holm p-value: "
        f"{row['Holm Adjusted p-value']:.8g}"
    )

    print(
        "Significant after Holm correction:",
        row[
            "Significant Holm p<0.05"
        ]
    )


mcnemar_df.to_csv(
    MCNEMAR_OUTPUT_FILE,
    index=False
)


# =========================================================
# 16. COMBINED SUMMARY
# =========================================================

summary_rows = []


for _, row in ci_df.iterrows():

    summary_rows.append({

        "Analysis":
            "95% Bootstrap CI",

        "Model / Comparison":
            row["Model"],

        "Statistic":
            "Macro F1",

        "Value":
            row["Macro F1"],

        "Lower":
            row["CI Lower 95%"],

        "Upper":
            row["CI Upper 95%"],

        "p-value":
            np.nan,

        "Adjusted p-value":
            np.nan
    })


for _, row in mcnemar_df.iterrows():

    summary_rows.append({

        "Analysis":
            "McNemar",

        "Model / Comparison":
            (
                f"{row['Model A']} "
                f"vs {row['Model B']}"
            ),

        "Statistic":
            "Exact paired correctness test",

        "Value":
            np.nan,

        "Lower":
            np.nan,

        "Upper":
            np.nan,

        "p-value":
            row["Exact p-value"],

        "Adjusted p-value":
            row[
                "Holm Adjusted p-value"
            ]
    })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    COMBINED_OUTPUT_FILE,
    index=False
)


# =========================================================
# 17. FINAL OUTPUT
# =========================================================

print(
    "\n" + "=" * 80
)

print(
    "95% CI SUMMARY"
)

print(
    "=" * 80
)


print(
    ci_df[
        [
            "Model",
            "Macro F1",
            "CI Lower 95%",
            "CI Upper 95%"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "FINAL McNEMAR SUMMARY"
)

print(
    "=" * 80
)


print(
    mcnemar_df[
        [
            "Model A",
            "Model B",
            "A Correct B Wrong",
            "A Wrong B Correct",
            "Exact p-value",
            "Holm Adjusted p-value",
            "Significant Holm p<0.05"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\n" + "=" * 80
)

print(
    "ANALYSIS COMPLETED SUCCESSFULLY"
)

print(
    "=" * 80
)


print(
    "\nFiles created:"
)

print(
    CI_OUTPUT_FILE
)

print(
    MCNEMAR_OUTPUT_FILE
)

print(
    COMBINED_OUTPUT_FILE
)