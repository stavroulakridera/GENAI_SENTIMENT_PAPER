# =========================================================
# DEEP LEARNING RESULTS COMPARISON
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

print("=" * 70)
print("DEEP LEARNING RESULTS COMPARISON")
print("=" * 70)


# =========================================================
# 2. RESULT FILES
# =========================================================

result_files = [
    "results/tables/dnn_results.csv",
    "results/tables/cnn_results.csv",
    "results/tables/lstm_results.csv",
    "results/tables/gru_results.csv",
    "results/tables/bilstm_results.csv"
]


# =========================================================
# 3. CHECK FILES
# =========================================================

print("\nChecking result files...")

for file in result_files:

    if not os.path.exists(file):
        raise FileNotFoundError(
            f"Missing result file: {file}"
        )

    print(f"Found: {file}")


# =========================================================
# 4. LOAD RESULTS
# =========================================================

results = []

for file in result_files:

    df = pd.read_csv(file)

    if len(df) != 1:
        raise ValueError(
            f"Expected one result row in {file}, "
            f"but found {len(df)}."
        )

    results.append(df)


dl_results = pd.concat(
    results,
    ignore_index=True
)


# =========================================================
# 5. SORT BY MACRO F1
# =========================================================

dl_results = dl_results.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)


# Add ranking
dl_results.insert(
    0,
    "Rank",
    range(1, len(dl_results) + 1)
)


# =========================================================
# 6. DISPLAY FINAL RESULTS
# =========================================================

print("\n" + "=" * 70)
print("FINAL DEEP LEARNING RESULTS")
print("=" * 70)

display_columns = [
    "Rank",
    "Model",
    "Accuracy",
    "Macro Precision",
    "Macro Recall",
    "Macro F1",
    "Weighted F1",
    "Training Time (s)",
    "Prediction Time (s)"
]

print(
    dl_results[display_columns]
    .round(4)
    .to_string(index=False)
)


# =========================================================
# 7. SAVE FINAL TABLE
# =========================================================

dl_results.to_csv(
    "results/tables/dl_results.csv",
    index=False
)

print(
    "\nFinal DL table saved:"
    "\nresults/tables/dl_results.csv"
)


# =========================================================
# 8. ACCURACY + MACRO F1 COMPARISON
# =========================================================

plot_df = dl_results.sort_values(
    by="Macro F1",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

y_positions = list(
    range(len(plot_df))
)

bar_height = 0.35

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
    "Deep Learning Model"
)

ax.set_title(
    "Deep Learning Model Performance"
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
    "results/figures/dl_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPerformance figure saved:"
    "\nresults/figures/dl_performance_comparison.png"
)


# =========================================================
# 9. MACRO F1 COMPARISON
# =========================================================

plot_df = dl_results.sort_values(
    by="Macro F1",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.barh(
    plot_df["Model"],
    plot_df["Macro F1"]
)

ax.set_xlabel(
    "Macro F1 Score"
)

ax.set_ylabel(
    "Deep Learning Model"
)

ax.set_title(
    "Macro F1 Comparison of Deep Learning Models"
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
        width + 0.008,
        bar.get_y()
        + bar.get_height() / 2,
        f"{width:.4f}",
        va="center"
    )


plt.tight_layout()

plt.savefig(
    "results/figures/dl_macro_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nMacro F1 figure saved:"
    "\nresults/figures/dl_macro_f1_comparison.png"
)


# =========================================================
# 10. TRAINING TIME COMPARISON
# =========================================================

time_df = dl_results.sort_values(
    by="Training Time (s)",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.barh(
    time_df["Model"],
    time_df["Training Time (s)"]
)

ax.set_xlabel(
    "Training Time (seconds)"
)

ax.set_ylabel(
    "Deep Learning Model"
)

ax.set_title(
    "Deep Learning Training Time Comparison"
)

ax.grid(
    axis="x",
    alpha=0.25
)


for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 1,
        bar.get_y()
        + bar.get_height() / 2,
        f"{width:.2f}s",
        va="center"
    )


plt.tight_layout()

plt.savefig(
    "results/figures/dl_training_time_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nTraining time figure saved:"
    "\nresults/figures/dl_training_time_comparison.png"
)


# =========================================================
# 11. PREDICTION TIME COMPARISON
# =========================================================

prediction_df = dl_results.sort_values(
    by="Prediction Time (s)",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.barh(
    prediction_df["Model"],
    prediction_df["Prediction Time (s)"]
)

ax.set_xlabel(
    "Prediction Time (seconds)"
)

ax.set_ylabel(
    "Deep Learning Model"
)

ax.set_title(
    "Deep Learning Prediction Time Comparison"
)

ax.grid(
    axis="x",
    alpha=0.25
)


for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.005,
        bar.get_y()
        + bar.get_height() / 2,
        f"{width:.4f}s",
        va="center"
    )


plt.tight_layout()

plt.savefig(
    "results/figures/dl_prediction_time_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPrediction time figure saved:"
    "\nresults/figures/dl_prediction_time_comparison.png"
)


# =========================================================
# 12. BEST DL MODEL
# =========================================================

best_model = dl_results.iloc[0]

print("\n" + "=" * 70)
print("BEST DEEP LEARNING MODEL")
print("=" * 70)

print(
    f"\nModel: {best_model['Model']}"
)

print(
    f"Accuracy: "
    f"{best_model['Accuracy']:.4f}"
)

print(
    f"Macro Precision: "
    f"{best_model['Macro Precision']:.4f}"
)

print(
    f"Macro Recall: "
    f"{best_model['Macro Recall']:.4f}"
)

print(
    f"Macro F1: "
    f"{best_model['Macro F1']:.4f}"
)

print(
    f"Weighted F1: "
    f"{best_model['Weighted F1']:.4f}"
)

print(
    f"Training Time: "
    f"{best_model['Training Time (s)']:.2f} sec"
)


# =========================================================
# 13. FINISHED
# =========================================================

print("\n" + "=" * 70)
print("DL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nFiles created:")

print(
    "1. results/tables/dl_results.csv"
)

print(
    "2. results/figures/"
    "dl_performance_comparison.png"
)

print(
    "3. results/figures/"
    "dl_macro_f1_comparison.png"
)

print(
    "4. results/figures/"
    "dl_training_time_comparison.png"
)

print(
    "5. results/figures/"
    "dl_prediction_time_comparison.png"
)