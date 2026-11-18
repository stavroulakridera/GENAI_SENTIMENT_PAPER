"""Run corrected group-aware DNN/CNN using all five fixed random seeds.

From repository root:
    python src/robustness/run_group_neural_5seeds.py --model all

Individual model scripts write per-run temporary outputs under ignored
results/tables. The runner aggregates all seeds before saving final outputs.
Historical standard-benchmark CSVs are never touched.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/tables"
SEEDS = (42, 52, 62, 72, 82)
CONFIG = {
    "dnn": ("group_dnn_entity.py", "group_dnn_entity_groupaware"),
    "cnn": ("group_cnn_entity.py", "group_cnn_entity_groupaware"),
}


def run(name):
    script, stem = CONFIG[name]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs, predictions = [], []
    for seed in SEEDS:
        env = dict(os.environ, SENTIMENT_SEED=str(seed))
        subprocess.run(
            [sys.executable, str(Path(__file__).parent / script)],
            cwd=ROOT, env=env, check=True
        )
        result = pd.read_csv(OUTPUT / f"{stem}_results.csv")
        preds = pd.read_csv(OUTPUT / f"{stem}_predictions.csv")
        if len(result) != 1 or len(preds) != 6865:
            raise ValueError(f"Unexpected row count for {name}, seed={seed}")
        if int(result["Seed"].iloc[0]) != seed:
            raise ValueError(f"Output seed mismatch for {name}: {seed}")
        preds["Seed"] = seed
        runs.append(result)
        predictions.append(preds)
    df = pd.concat(runs, ignore_index=True)
    all_preds = pd.concat(predictions, ignore_index=True)
    df.to_csv(OUTPUT / f"{stem}_5seeds_results.csv", index=False)
    all_preds.to_csv(OUTPUT / f"{stem}_5seeds_predictions.csv", index=False)
    summary = pd.DataFrame([
        {"Metric": metric, "Mean": float(df[metric].mean()),
         "Std": float(df[metric].std(ddof=1))}
        for metric in ("Accuracy", "Macro F1", "Balanced Accuracy", "MCC")
    ])
    summary.to_csv(OUTPUT / f"{stem}_5seeds_summary.csv", index=False)
    print(f"{name}: completed five-seed group-aware evaluation")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("all", "dnn", "cnn"), default="all")
    chosen = parser.parse_args().model
    for name in CONFIG if chosen == "all" else (chosen,):
        run(name)


if __name__ == "__main__":
    main()
