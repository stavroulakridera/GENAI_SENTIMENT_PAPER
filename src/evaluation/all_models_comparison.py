# =========================================================
# ALL MODELS COMPARISON
# MACHINE LEARNING + DEEP LEARNING
# =========================================================

import os
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. SETTINGS
# =========================================================

os.makedirs("results", exist_ok=True)
os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

print("=" * 80)
print("ALL MODELS COMPARISON - ML + DL")
print("=" * 80)


# =========================================================
# 2. FILE PATHS
# =========================================================

ml_file = "results/tables/ml_results.csv"
dl_file = "results/tables/dl_results.csv"

if not os.path.exists(ml_file):
    raise FileNotFoundError(
        f"Missing file: {ml_file}"
    )

if not os.path.exists(dl_file):
    raise FileNotFoundError(
        f"Missing file: {dl_file}"
    )

print("\nFiles found successfully:")
print(ml_file)
print(dl_file)


# =========================================================
# 3. LOAD RESULTS
# =========================================================

ml_results = pd.read_csv(ml_file)
dl_results = pd.read_csv(dl_file)

print("\nML models loaded:", len(ml_results))
print("DL models loaded:", len(dl_results))


# =========================================================
# 4. REMOVE OLD RANK COLUMNS
# =========================================================

if "Rank" in ml_results.columns:
    ml_results = ml_results.drop(
        columns=["Rank"]
    )

if "Rank" in dl_results.columns:
    dl_results = dl_results.drop(
        columns=["Rank"]
    )


# =========================================================
# 5. ADD MODEL FAMILY
# =========================================================

ml_results.insert(
    1,
    "Family",
    "Machine Learning"
)

dl_results.insert(
    1,
    "Family",
    "Deep Learning"
)


# =========================================================
# 6. COMBINE RESULTS
# =========================================================

all_results = pd.concat(
    [
        ml_results,
        dl_results
    ],
    ignore_index=True
)


# =========================================================
# 7. CHECK REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Model",
    "Family",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Training Time (s)",
    "Prediction Time (s)"
]

missing_columns = [
    column
    for column in required_columns
    if column not in all_results.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )


# =========================================================
# 8. SORT BY MACRO F1
# =========================================================

all_results = all_results.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)

all_results.insert(
    0,
    "Rank",
    range(1, len(all_results) + 1)
)


# =========================================================
# 9. DISPLAY FINAL RANKING
# =========================================================

display_columns = [
    "Rank",
    "Model",
    "Family",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Training Time (s)",
    "Prediction Time (s)"
]

print("\n" + "=" * 80)
print("FINAL ML + DL MODEL RANKING")
print("=" * 80)

print(
    all_results[display_columns]
    .round(4)
    .to_string(index=False)
)


# =========================================================
# 10. SAVE COMPLETE TABLE
# =========================================================

all_results.to_csv(
    "results/tables/all_models_results.csv",
    index=False
)

print(
    "\nComplete results saved:"
    "\nresults/tables/all_models_results.csv"
)


# =========================================================
# 11. PUBLICATION TABLE
# =========================================================
# Compact table containing the most useful metrics
# for the final paper.

publication_table = all_results[
    [
        "Rank",
        "Model",
        "Family",
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ]
].copy()

publication_table[
    [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ]
] = publication_table[
    [
        "Accuracy",
        "Macro Precision",
        "Macro Recall",
        "Macro F1",
        "Weighted F1"
    ]
].round(4)

publication_table.to_csv(
    "results/tables/all_models_publication_table.csv",
    index=False
)

print(
    "\nPublication table saved:"
    "\nresults/tables/all_models_publication_table.csv"
)


# =========================================================
# 12. MACRO F1 - ALL MODELS
# =========================================================

plot_df = all_results.sort_values(
    by="Macro F1",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.barh(
    plot_df["Model"],
    plot_df["Macro F1"]
)

ax.set_xlabel(
    "Macro F1 Score"
)

ax.set_ylabel(
    "Model"
)

ax.set_title(
    "Macro F1 Comparison Across ML and DL Models"
)

ax.set_xlim(
    0,
    1
)

ax.grid(
    axis="x",
    alpha=0.25
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.006,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}",
        va="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "results/figures/all_models_macro_f1.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nMacro F1 figure saved:"
    "\nresults/figures/all_models_macro_f1.png"
)


# =========================================================
# 13. ACCURACY VS MACRO F1
# =========================================================

plot_df = all_results.sort_values(
    by="Macro F1",
    ascending=True
)

y_positions = list(
    range(len(plot_df))
)

bar_height = 0.35

fig, ax = plt.subplots(
    figsize=(11, 7)
)

ax.barh(
    [
        y - bar_height / 2
        for y in y_positions
    ],
    plot_df["Accuracy"],
    height=bar_height,
    label="Accuracy"
)

ax.barh(
    [
        y + bar_height / 2
        for y in y_positions
    ],
    plot_df["Macro F1"],
    height=bar_height,
    label="Macro F1"
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    plot_df["Model"]
)

ax.set_xlabel(
    "Score"
)

ax.set_ylabel(
    "Model"
)

ax.set_title(
    "Accuracy and Macro F1 Across ML and DL Models"
)

ax.set_xlim(
    0,
    1
)

ax.legend()

ax.grid(
    axis="x",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    "results/figures/all_models_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPerformance figure saved:"
    "\nresults/figures/all_models_performance.png"
)


# =========================================================
# 14. TRAINING TIME COMPARISON
# =========================================================

time_df = all_results.sort_values(
    by="Training Time (s)",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.barh(
    time_df["Model"],
    time_df["Training Time (s)"]
)

ax.set_xlabel(
    "Training Time (seconds)"
)

ax.set_ylabel(
    "Model"
)

ax.set_title(
    "Training Time Across ML and DL Models"
)

ax.grid(
    axis="x",
    alpha=0.25
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 1,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.2f}s",
        va="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    "results/figures/all_models_training_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nTraining time figure saved:"
    "\nresults/figures/all_models_training_time.png"
)


# =========================================================
# 15. PREDICTION TIME COMPARISON
# =========================================================

prediction_df = all_results.sort_values(
    by="Prediction Time (s)",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 7)
)

bars = ax.barh(
    prediction_df["Model"],
    prediction_df["Prediction Time (s)"]
)

ax.set_xlabel(
    "Prediction Time (seconds)"
)

ax.set_ylabel(
    "Model"
)

ax.set_title(
    "Prediction Time Across ML and DL Models"
)

ax.grid(
    axis="x",
    alpha=0.25
)

for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.005,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}s",
        va="center",
        fontsize=8
    )

plt.tight_layout()

plt.savefig(
    "results/figures/all_models_prediction_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPrediction time figure saved:"
    "\nresults/figures/all_models_prediction_time.png"
)


# =========================================================
# 16. BEST MODEL BY FAMILY
# =========================================================

best_ml = (
    all_results[
        all_results["Family"]
        == "Machine Learning"
    ]
    .sort_values(
        "Macro F1",
        ascending=False
    )
    .iloc[0]
)

best_dl = (
    all_results[
        all_results["Family"]
        == "Deep Learning"
    ]
    .sort_values(
        "Macro F1",
        ascending=False
    )
    .iloc[0]
)


print("\n" + "=" * 80)
print("BEST MODEL BY FAMILY")
print("=" * 80)

print("\nBest Machine Learning Model:")
print(
    f"{best_ml['Model']} | "
    f"Accuracy = {best_ml['Accuracy']:.4f} | "
    f"Macro F1 = {best_ml['Macro F1']:.4f}"
)

print("\nBest Deep Learning Model:")
print(
    f"{best_dl['Model']} | "
    f"Accuracy = {best_dl['Accuracy']:.4f} | "
    f"Macro F1 = {best_dl['Macro F1']:.4f}"
)


# =========================================================
# 17. BEST OVERALL MODEL
# =========================================================

best_overall = all_results.iloc[0]

print("\n" + "=" * 80)
print("BEST OVERALL MODEL")
print("=" * 80)

print(
    f"\nModel: {best_overall['Model']}"
)

print(
    f"Family: {best_overall['Family']}"
)

print(
    f"Accuracy: "
    f"{best_overall['Accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{best_overall['Macro Precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{best_overall['Macro Recall']:.4f}"
)

print(
    f"Macro F1: "
    f"{best_overall['Macro F1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{best_overall['Weighted F1']:.4f}"
)


# =========================================================
# 18. DL IMPROVEMENT OVER BEST ML
# =========================================================

macro_f1_difference = (
    best_dl["Macro F1"]
    - best_ml["Macro F1"]
)

accuracy_difference = (
    best_dl["Accuracy"]
    - best_ml["Accuracy"]
)

relative_macro_f1_improvement = (
    macro_f1_difference
    / best_ml["Macro F1"]
) * 100


print("\n" + "=" * 80)
print("BEST DL VS BEST ML")
print("=" * 80)

print(
    f"\nBest DL: {best_dl['Model']}"
)

print(
    f"Best ML: {best_ml['Model']}"
)

print(
    f"\nAbsolute Accuracy Difference: "
    f"{accuracy_difference:.4f}"
)

print(
    f"Absolute Macro F1 Difference: "
    f"{macro_f1_difference:.4f}"
)

print(
    f"Relative Macro F1 Improvement: "
    f"{relative_macro_f1_improvement:.2f}%"
)


# =========================================================
# 19. FINISHED
# =========================================================

print("\n" + "=" * 80)
print("ALL MODELS COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nFiles created:")

print(
    "1. results/tables/"
    "all_models_results.csv"
)

print(
    "2. results/tables/"
    "all_models_publication_table.csv"
)

print(
    "3. results/figures/"
    "all_models_macro_f1.png"
)

print(
    "4. results/figures/"
    "all_models_performance.png"
)

print(
    "5. results/figures/"
    "all_models_training_time.png"
)

print(
    "6. results/figures/"
    "all_models_prediction_time.png"
)