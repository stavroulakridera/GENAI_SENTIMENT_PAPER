"""Extract representative misclassifications from saved model predictions.

Requires a locally generated results CSV; does not require model training.

Example:
    python src/evaluation/export_class_errors.py \
      --predictions results/tables/group_bert_entity_5seeds_predictions.csv --seed 42

Outputs are written under ignored results/derived/.
"""
import argparse
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-per-confusion", type=int, default=10)
    args = parser.parse_args()
    path = args.predictions
    if not path.is_absolute():
        path = ROOT / path
    df = pd.read_csv(path)
    df = df.rename(columns={
        "Sentiment": "True_Label", "Prediction": "Predicted_Label"
    })
    required = ("Tweet_ID", "Entity", "Tweet", "True_Label", "Predicted_Label")
    if any(key not in df.columns for key in required):
        raise ValueError("Prediction file lacks the required labels or text")
    if "Seed" in df.columns:
        if args.seed is None:
            parser.error("Specify --seed for multi-seed predictions")
        df = df[df["Seed"] == args.seed]
    if df.empty:
        raise ValueError("No rows found for requested configuration")
    wrong = df[df["True_Label"] != df["Predicted_Label"]].copy()
    wrong["Confusion"] = (
        wrong["True_Label"].astype(str) + " -> "
        + wrong["Predicted_Label"].fillna("INVALID").astype(str)
    )
    wrong = (
        wrong.sort_values(["Confusion", "Tweet_ID"])
        .groupby("Confusion", group_keys=False)
        .head(args.max_per_confusion)
    )
    output = ROOT / "results/derived"
    output.mkdir(parents=True, exist_ok=True)
    name = path.stem + (f"_seed{args.seed}" if args.seed else "") + "_errors.csv"
    wrong.to_csv(output / name, index=False)
    print(f"Saved {len(wrong)} error examples: {output/name}")


if __name__ == "__main__":
    main()
