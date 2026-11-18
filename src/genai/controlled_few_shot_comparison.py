"""Controlled four-shot evaluation across the three locally hosted LLMs.

All models receive the identical prompt template and identical held-out rows.
Results are saved locally only; historical configurations are unchanged.

    python src/genai/controlled_few_shot_comparison.py --model all
    python src/genai/controlled_few_shot_comparison.py --model qwen --limit 20

--limit produces pilot-only files and must not be treated as a full evaluation.
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "data/processed/clean_validation.csv"
PROMPT = ROOT / "prompts/qwen/few_shot.txt"
OUTPUT = ROOT / "results/tables"
MODELS = {"llama": "llama3.2:3b", "qwen": "qwen2.5:3b", "gemma": "gemma3:4b"}
LABELS = ("Negative", "Neutral", "Positive", "Irrelevant")


def build_prompt(entity, tweet):
    return PROMPT.read_text(encoding="utf-8").format(
        entity=str(entity), tweet=str(tweet)
    )


def normalize_label(response):
    if response is None:
        return None
    value = str(response).strip().strip('"').strip("'").strip()
    if value.endswith("."):
        value = value[:-1].strip()
    return next((x for x in LABELS if value.lower() == x.lower()), None)


def inference(model, prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False,
              "options": {"temperature": 0, "seed": 42}},
        timeout=300,
    )
    response.raise_for_status()
    return response.json().get("response", "")


def evaluate(alias, data, pilot=False):
    rows = []
    for _, item in data.iterrows():
        start = time.perf_counter()
        raw, error = "", ""
        try:
            raw = inference(MODELS[alias], build_prompt(item["Entity"], item["Tweet"]))
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
        label = normalize_label(raw)
        rows.append({
            "Tweet_ID": item["Tweet_ID"], "Entity": item["Entity"],
            "Tweet": item["Tweet"], "True_Label": item["Sentiment"],
            "Raw_Output": raw, "Predicted_Label": label,
            "Valid_Output": label is not None,
            "Latency_s": time.perf_counter() - start, "Error": error,
        })
    df = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    name = f"controlled_few_shot_{alias}" + ("_pilot" if pilot else "")
    df.to_csv(OUTPUT / f"{name}_predictions.csv", index=False)
    predicted = df["Predicted_Label"].fillna("__INVALID__")
    true = df["True_Label"]
    report = pd.DataFrame([{
        "Model": MODELS[alias], "N": len(df),
        "Valid_Outputs": int(df["Valid_Output"].sum()),
        "Invalid_Outputs": int((~df["Valid_Output"]).sum()),
        "Accuracy": accuracy_score(true, predicted),
        "Macro F1": f1_score(true, predicted, labels=LABELS,
                              average="macro", zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(true, predicted),
        "MCC": matthews_corrcoef(true, predicted),
        "Mean_Latency_s": df["Latency_s"].mean(),
        "Temperature": 0, "Seed": 42, "Pilot": pilot,
    }])
    report.to_csv(OUTPUT / f"{name}_results.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("all", *MODELS), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    data = pd.read_csv(INPUT)
    if len(data) != 1000:
        raise ValueError("Expected the original 1000-instance held-out dataset")
    if args.limit is not None:
        if not 1 <= args.limit <= 1000:
            parser.error("--limit must be between 1 and 1000")
        data = data.head(args.limit)
    for alias in MODELS if args.model == "all" else (args.model,):
        evaluate(alias, data, pilot=args.limit is not None)


if __name__ == "__main__":
    main()
