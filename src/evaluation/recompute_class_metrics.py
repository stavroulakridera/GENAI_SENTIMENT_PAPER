"""Recompute class-wise metrics from SAVED predictions, without training.

Outputs go only to results/derived. Historical results/tables files are never
modified. In strict LLM scoring invalid/unparseable outputs count as errors,
while the valid-output rate is retained as a separate diagnostic.
"""
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "results" / "tables"
OUTPUT = ROOT / "results" / "derived"
LABELS = ("Negative", "Neutral", "Positive", "Irrelevant")
INVALID = "__INVALID__"
FILES = {
    "Linear SVM": ("standard", "linear_svm_entity_predictions.csv", 1000),
    "DNN": ("standard", "dnn_entity_5seeds_predictions.csv", 1000),
    "BERT": ("standard", "bert_entity_5seeds_predictions.csv", 1000),
    "Qwen few-shot": ("standard", "qwen_few_shot_predictions.csv", 1000),
    "Llama zero-shot": ("standard", "genai_zero_shot_ollama_predictions.csv", 1000),
    "Llama few-shot": ("standard", "genai_few_shot_ollama_predictions.csv", 1000),
    "Llama reasoning": ("standard", "genai_reasoning_ollama_predictions.csv", 1000),
    "Group SVM": ("group-disjoint", "group_linear_svm_entity_predictions.csv", 6865),
    "Group DNN": ("group-disjoint", "group_dnn_entity_predictions.csv", 6865),
    "Group-aware DNN": ("group-disjoint", "group_dnn_entity_groupaware_predictions.csv", 6865),
    "Group CNN": ("group-disjoint", "group_cnn_entity_predictions.csv", 6865),
    "Group-aware CNN": ("group-disjoint", "group_cnn_entity_groupaware_predictions.csv", 6865),
    "Group Qwen": ("group-disjoint", "group_qwen_few_shot_predictions.csv", 6865),
    "Group BERT": ("group-disjoint", "group_bert_entity_5seeds_predictions.csv", 6865),
}

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overall, by_class, confusion, skipped = [], [], [], []
    for model, (protocol, filename, expected_n) in FILES.items():
        filepath = INPUT / filename
        if not filepath.exists():
            skipped.append(filename)
            continue
        df = pd.read_csv(filepath)
        for column in ("True_Label", "Predicted_Label"):
            if column not in df.columns:
                raise ValueError(f"{filename} lacks {column}")
        group_by_seed = df.groupby("Seed", sort=True) if "Seed" in df.columns else [("single", df)]
        for seed, sample in group_by_seed:
            if len(sample) != expected_n:
                raise ValueError(f"{filename} seed {seed}: expected {expected_n} rows, got {len(sample)}")
            y_true = sample["True_Label"].astype(str).str.strip()
            if not y_true.isin(LABELS).all():
                raise ValueError(f"Unknown true labels in {filename}")
            y_pred = sample["Predicted_Label"].fillna(INVALID).astype(str).str.strip()
            y_pred = y_pred.where(y_pred.isin(LABELS), INVALID)
            count_invalid = int((y_pred == INVALID).sum())
            record = dict(Protocol=protocol, Model=model, Seed=seed,
                          N=len(sample), Invalid=count_invalid,
                          Valid_Rate=1 - count_invalid / len(sample),
                          Accuracy=accuracy_score(y_true, y_pred),
                          Macro_F1=f1_score(y_true, y_pred, labels=LABELS,
                                            average="macro", zero_division=0))
            overall.append(record)
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true, y_pred, labels=LABELS, zero_division=0
            )
            for idx, label in enumerate(LABELS):
                by_class.append(dict(Protocol=protocol, Model=model, Seed=seed,
                                     Label=label, Precision=precision[idx],
                                     Recall=recall[idx], F1=f1[idx],
                                     Support=int(support[idx])))
            labels_with_invalid = list(LABELS) + [INVALID]
            cm = confusion_matrix(y_true, y_pred, labels=labels_with_invalid)
            for i, actual in enumerate(LABELS):
                for j, predicted in enumerate(labels_with_invalid):
                    confusion.append(dict(Protocol=protocol, Model=model, Seed=seed,
                                          Actual=actual, Predicted=predicted,
                                          Count=int(cm[i, j])))
    if not overall:
        raise RuntimeError("No prediction files available to evaluate")
    pd.DataFrame(overall).to_csv(OUTPUT / "overall_metrics.csv", index=False)
    pd.DataFrame(by_class).to_csv(OUTPUT / "class_metrics.csv", index=False)
    pd.DataFrame(confusion).to_csv(OUTPUT / "confusion_counts.csv", index=False)
    (OUTPUT / "missing_optional_inputs.txt").write_text(
        "\n".join(skipped) + "\n", encoding="utf-8"
    )
    print(f"Audited {len(overall)} model/seed configurations; "
          f"saved results in {OUTPUT}")
    if skipped:
        print("Unavailable experiment outputs:", ", ".join(skipped))

if __name__ == "__main__":
    main()
