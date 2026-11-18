"""Regression tests for standard and historical prediction schemas."""
import unittest
import pandas as pd
from src.evaluation.recompute_class_metrics import (
    normalize_schema, evaluate_rows, INVALID, LABELS, EXPERIMENTS
)

class TestSavedPredictionMetrics(unittest.TestCase):
    def test_legacy_ml_columns(self):
        rows = pd.DataFrame({
            "Tweet_ID": [1,2,3,4],
            "Entity": ["X"]*4, "Tweet": ["a","b","c","d"],
            "Sentiment": list(LABELS), "Prediction": list(LABELS)
        })
        normalized = normalize_schema(rows, "test_legacy.csv")
        agg, cls, cm = evaluate_rows(normalized, "standard", "SVM", "Entity+Tweet", "single")
        self.assertEqual(agg["Accuracy"], 1)
        self.assertEqual(agg["Macro_F1"], 1)
        self.assertEqual(len(cls), 4)
        self.assertEqual(sum(x["Count"] for x in cm), 4)

    def test_invalid_counted_in_full_denominator(self):
        rows = pd.DataFrame({
            "Tweet_ID": [1,2,3,4], "Entity": ["X"]*4,
            "Tweet": ["a","b","c","d"], "Sentiment": list(LABELS),
            "Predicted_Label": ["Negative", "Neutral", "bad text", None]
        })
        normalized = normalize_schema(rows, "test_llama.csv")
        agg, cls, cm = evaluate_rows(
            normalized, "standard", "Llama", "Entity+Tweet", "single"
        )
        self.assertEqual(agg["N"], 4)
        self.assertEqual(agg["Invalid_Outputs"], 2)
        self.assertEqual(agg["Valid_Output_Rate"], 0.5)
        self.assertEqual(agg["Accuracy"], 0.5)
        self.assertEqual(sum(x["Count"] for x in cm if x["Predicted"] == INVALID), 2)

    def test_manuscript_experiment_coverage(self):
        self.assertEqual(len(EXPERIMENTS), 40)
        self.assertTrue(any("group_bert_entity_5seeds" in row[3] for row in EXPERIMENTS))
        self.assertTrue(any("genai_few_shot_ollama" in row[3] for row in EXPERIMENTS))

if __name__ == "__main__":
    unittest.main()
