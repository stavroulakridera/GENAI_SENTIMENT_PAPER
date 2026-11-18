"""Tests for group-aware development separation (no training required)."""
import unittest
import pandas as pd
from src.robustness.group_split_utils import group_aware_train_dev_split

class TestGroupSplit(unittest.TestCase):
    def setUp(self):
        # Four labels; each group contributes two textual variants.
        self.frame = pd.DataFrame([
            {"Tweet_ID": g, "Sentiment": ("Negative", "Neutral", "Positive", "Irrelevant")[g % 4],
             "Tweet": f"Text {g} variant {v}"}
            for g in range(200) for v in range(2)
        ])

    def test_disjoint_complete_and_deterministic(self):
        a, b = group_aware_train_dev_split(self.frame, random_state=42)
        c, d = group_aware_train_dev_split(self.frame, random_state=42)
        self.assertEqual(list(a), list(c))
        self.assertEqual(list(b), list(d))
        self.assertEqual(len(a) + len(b), len(self.frame))
        self.assertFalse(set(a) & set(b))
        self.assertFalse(set(self.frame.iloc[a].Tweet_ID) &
                         set(self.frame.iloc[b].Tweet_ID))
        self.assertEqual(set(self.frame.iloc[a].Sentiment),
                         set(self.frame.Sentiment.unique()))
        self.assertEqual(set(self.frame.iloc[b].Sentiment),
                         set(self.frame.Sentiment.unique()))

if __name__ == "__main__":
    unittest.main()
