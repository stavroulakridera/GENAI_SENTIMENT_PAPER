"""Recompute all manuscript class metrics from saved row-level predictions.

No model training and no changes to historical results/tables are performed.
Invalid LLM outputs count as incorrect on the full evaluation denominator,
while format-validity rates are reported separately.
Run: python src/evaluation/recompute_class_metrics.py
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "derived"
LABELS = ("Negative", "Neutral", "Positive", "Irrelevant")
INVALID = "__INVALID__"
SEEDS = {42, 52, 62, 72, 82}

# Each tuple: human-readable model, protocol, condition, results filename,
# required rows per individual seed. Include original and corrected runs,
# but never mix their metrics under the same experimental name.
EXPERIMENTS = []
for model, tweet_file, entity_file in [
    ("Naive Bayes", "naive_bayes", "naive_bayes_entity"),
    ("Logistic Regression", "logistic_regression", "logistic_regression_entity"),
    ("Linear SVM", "linear_svm", "linear_svm_entity"),
    ("SGD", "sgd_classifier", "sgd_entity"),
    ("Random Forest", "random_forest", "random_forest_entity"),
]:
    EXPERIMENTS.extend([
        (model, "standard", "Tweet-only", tweet_file + "_predictions.csv", 1000),
        (model, "standard", "Entity+Tweet", entity_file + "_predictions.csv", 1000),
    ])
for model, stem in [
    ("DNN", "dnn"), ("CNN", "cnn"), ("LSTM", "lstm"),
    ("GRU", "gru"), ("BiLSTM", "bilstm"), ("BERT", "bert"),
    ("RoBERTa", "roberta")
]:
    for condition, suffix in [("Tweet-only", "tweet_only"), ("Entity+Tweet", "entity")]:
        EXPERIMENTS.append(
            (model, "standard", condition,
             f"{stem}_{suffix}_5seeds_predictions.csv", 1000)
        )
for model, stem in [
    ("Llama 3.2 3B", "genai"), ("Qwen2.5 3B", "qwen"),
    ("Gemma 3 4B", "gemma")
]:
    for strategy, suffix in [
        ("Zero-shot", "zero_shot"),
        ("Few-shot", "few_shot"),
        ("Reasoning", "reasoning"),
    ]:
        filename = (
            f"genai_{suffix}_ollama_predictions.csv"
            if stem == "genai" else f"{stem}_{suffix}_predictions.csv"
        )
        EXPERIMENTS.append(
            (f"{model} / {strategy}", "standard", "Entity+Tweet", filename, 1000)
        )
for model, filename in [
    ("Linear SVM", "group_linear_svm_entity_predictions.csv"),
    ("DNN / historical row-level dev split", "group_dnn_entity_predictions.csv"),
    ("CNN / historical row-level dev split", "group_cnn_entity_predictions.csv"),
    ("Qwen2.5 3B / group-test only", "group_qwen_few_shot_predictions.csv"),
    ("DNN / corrected group-aware dev", "group_dnn_entity_groupaware_predictions.csv"),
    ("CNN / corrected group-aware dev", "group_cnn_entity_groupaware_predictions.csv"),
    ("BERT / corrected group-aware dev", "group_bert_entity_5seeds_predictions.csv"),
]:
    EXPERIMENTS.append((model, "group-disjoint", "Entity+Tweet", filename, 6865))


def normalize_schema(frame, filename):
    """Support original ML/Llama and later standardized prediction CSVs."""
    rename = {}
    if "True_Label" not in frame.columns and "Sentiment" in frame.columns:
        rename["Sentiment"] = "True_Label"
    if "Predicted_Label" not in frame.columns and "Prediction" in frame.columns:
        rename["Prediction"] = "Predicted_Label"
    frame = frame.rename(columns=rename)
    for key in ("Tweet_ID", "Entity", "Tweet", "True_Label", "Predicted_Label"):
        if key not in frame.columns:
            raise ValueError(f"{filename}: missing expected column {key}")
    # Make invalid/missing output explicit; never impute a sentiment label.
    frame["True_Label"] = frame["True_Label"].astype(str).str.strip()
    frame["Predicted_Label"] = (
        frame["Predicted_Label"].fillna(INVALID).astype(str).str.strip()
    )
    frame.loc[~frame["Predicted_Label"].isin(LABELS), "Predicted_Label"] = INVALID
    if not frame["True_Label"].isin(LABELS).all():
        raise ValueError(f"{filename}: unexpected true sentiment label")
    if frame["Tweet_ID"].isna().any():
        raise ValueError(f"{filename}: missing Tweet_ID")
    return frame


def evaluate_rows(sample, protocol, model, condition, seed):
    """Return aggregate/per-class/confusion rows for one fixed test set."""
    truth, prediction = sample["True_Label"], sample["Predicted_Label"]
    invalid = int((prediction == INVALID).sum())
    common = dict(Protocol=protocol, Model=model, Condition=condition, Seed=seed)
    aggregate = dict(
        **common, N=len(sample), Invalid_Outputs=invalid,
        Valid_Output_Rate=1 - invalid / len(sample),
        Accuracy=accuracy_score(truth, prediction),
        Macro_F1=f1_score(truth, prediction, labels=LABELS,
                          average="macro", zero_division=0),
    )
    p, r, f1, support = precision_recall_fscore_support(
        truth, prediction, labels=LABELS, zero_division=0
    )
    by_class = [
        dict(**common, Label=label, Precision=float(p[i]), Recall=float(r[i]),
             F1=float(f1[i]), Support=int(support[i]))
        for i, label in enumerate(LABELS)
    ]
    labels = list(LABELS) + [INVALID]
    matrix = confusion_matrix(truth, prediction, labels=labels)
    confusion = [
        dict(**common, Actual=actual, Predicted=predicted,
             Count=int(matrix[i, j]))
        for i, actual in enumerate(LABELS)
        for j, predicted in enumerate(labels)
    ]
    return aggregate, by_class, confusion


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overall, by_class, confusion, missing = [], [], [], []
    for model, protocol, condition, filename, expected_n in EXPERIMENTS:
        filepath = INPUT / filename
        if not filepath.is_file():
            missing.append(filename)
            continue
        frame = normalize_schema(pd.read_csv(filepath, dtype={"Tweet_ID": "string"}), filename)
        if "Seed" in frame.columns:
            if "5seeds" in filename:
                observed_seeds = set(frame["Seed"].astype(int))
                if observed_seeds != SEEDS:
                    raise ValueError(
                        f"{filename}: expected seeds {sorted(SEEDS)}, "
                        f"observed {sorted(observed_seeds)}"
                    )
            grouped = frame.groupby("Seed", sort=True)
        else:
            grouped = [("single", frame)]
        for seed, sample in grouped:
            if len(sample) != expected_n:
                raise ValueError(
                    f"{filename} seed {seed}: expected {expected_n} rows, got {len(sample)}"
                )
            record, classes, counts = evaluate_rows(
                sample, protocol, model, condition, seed
            )
            overall.append(record)
            by_class.extend(classes)
            confusion.extend(counts)
    if not overall:
        raise RuntimeError("No saved prediction files found")
    pd.DataFrame(overall).to_csv(OUTPUT / "overall_metrics.csv", index=False)
    pd.DataFrame(by_class).to_csv(OUTPUT / "class_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(OUTPUT / "confusion_counts.csv", index=False)
    (OUTPUT / "missing_optional_inputs.txt").write_text(
        "\n".join(missing) + ("\n" if missing else ""), encoding="utf-8"
    )
    print(f"Audited {len(overall)} model/seed configurations across "
          f"{len(EXPERIMENTS) - len(missing)} experiment files")
    print(f"Saved outputs under: {OUTPUT}")
    if missing:
        print("Not yet generated:", ", ".join(missing))


if __name__ == "__main__":
    main()
