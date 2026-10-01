# =========================================================
# MACHINE LEARNING RESULTS COMPARISON
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

print("=" * 65)
print("MACHINE LEARNING RESULTS COMPARISON")
print("=" * 65)


# =========================================================
# 2. RESULT FILES
# =========================================================

result_files = [
    "results/tables/naive_bayes_results.csv",
    "results/tables/logistic_regression_results.csv",
    "results/tables/linear_svm_results.csv",
    "results/tables/sgd_classifier_results.csv",
    "results/tables/random_forest_results.csv"
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
    results.append(df)


ml_results = pd.concat(
    results,
    ignore_index=True
)


# =========================================================
# 5. SORT BY MACRO F1
# =========================================================

ml_results = ml_results.sort_values(
    by="Macro F1",
    ascending=False
).reset_index(drop=True)


# Add ranking
ml_results.insert(
    0,
    "Rank",
    range(1, len(ml_results) + 1)
)


# =========================================================
# 6. DISPLAY FINAL RESULTS
# =========================================================

print("\n" + "=" * 65)
print("FINAL MACHINE LEARNING RESULTS")
print("=" * 65)

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
    ml_results[display_columns]
    .round(4)
    .to_string(index=False)
)


# =========================================================
# 7. SAVE FINAL TABLE
# =========================================================

ml_results.to_csv(
    "results/tables/ml_results.csv",
    index=False
)

print(
    "\nFinal ML table saved:"
    "\nresults/tables/ml_results.csv"
)


# =========================================================
# 8. PERFORMANCE FIGURE
# =========================================================

plot_df = ml_results.sort_values(
    by="Macro F1",
    ascending=True
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

y_positions = range(
    len(plot_df)
)

bar_height = 0.35


ax.barh(
    [y - bar_height / 2 for y in y_positions],
    plot_df["Accuracy"],
    height=bar_height,
    label="Accuracy"
)

ax.barh(
    [y + bar_height / 2 for y in y_positions],
    plot_df["Macro F1"],
    height=bar_height,
    label="Macro F1"
)


ax.set_yticks(
    list(y_positions)
)

ax.set_yticklabels(
    plot_df["Model"]
)

ax.set_xlabel(
    "Score"
)

ax.set_ylabel(
    "Machine Learning Model"
)

ax.set_title(
    "Machine Learning Model Performance"
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
    "results/figures/ml_performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nPerformance figure saved:"
    "\nresults/figures/ml_performance_comparison.png"
)


# =========================================================
# 9. MACRO F1 FIGURE
# =========================================================

plot_df = ml_results.sort_values(
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
    "Machine Learning Model"
)

ax.set_title(
    "Macro F1 Comparison of Machine Learning Models"
)

ax.set_xlim(
    0,
    1
)

ax.grid(
    axis="x",
    alpha=0.25
)


# Add values to bars
for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.4f}",
        va="center"
    )


plt.tight_layout()

plt.savefig(
    "results/figures/ml_macro_f1_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nMacro F1 figure saved:"
    "\nresults/figures/ml_macro_f1_comparison.png"
)


# =========================================================
# 10. TRAINING TIME FIGURE
# =========================================================

time_df = ml_results.sort_values(
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
    "Machine Learning Model"
)

ax.set_title(
    "Training Time Comparison"
)

ax.grid(
    axis="x",
    alpha=0.25
)


for bar in bars:

    width = bar.get_width()

    ax.text(
        width + 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{width:.3f}s",
        va="center"
    )


plt.tight_layout()

plt.savefig(
    "results/figures/ml_training_time_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print(
    "\nTraining time figure saved:"
    "\nresults/figures/ml_training_time_comparison.png"
)


# =========================================================
# 11. BEST MODEL
# =========================================================

best_model = ml_results.iloc[0]

print("\n" + "=" * 65)
print("BEST MACHINE LEARNING MODEL")
print("=" * 65)

print(
    f"\nModel: {best_model['Model']}"
)

print(
    f"Accuracy: {best_model['Accuracy']:.4f}"
)

print(
    f"Macro F1: {best_model['Macro F1']:.4f}"
)

print(
    f"Weighted F1: {best_model['Weighted F1']:.4f}"
)


# =========================================================
# 12. FINISHED
# =========================================================

print("\n" + "=" * 65)
print("ML COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 65)

print("\nFiles created:")

print(
    "1. results/tables/ml_results.csv"
)

print(
    "2. results/figures/ml_performance_comparison.png"
)

print(
    "3. results/figures/ml_macro_f1_comparison.png"
)

print(
    "4. results/figures/ml_training_time_comparison.png"
)
