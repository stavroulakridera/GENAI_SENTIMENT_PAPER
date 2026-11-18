"""Test shared four-shot prompts and strict output parsing without Ollama."""
import unittest
from src.genai.controlled_few_shot_comparison import (
    MODELS, LABELS, build_prompt, normalize_label,
)


class TestControlledPrompt(unittest.TestCase):
    def test_same_prompt_template_is_model_independent(self):
        prompt = build_prompt("Microsoft", "The update is great.")
        self.assertIn("Entity:\nMicrosoft", prompt)
        self.assertIn("Post:\nThe update is great.", prompt)
        self.assertIn("Example 1:", prompt)
        self.assertIn("Example 4:", prompt)
        self.assertEqual(len(MODELS), 3)
        self.assertEqual(len(LABELS), 4)

    def test_invalid_outputs_are_not_guessed(self):
        self.assertEqual(normalize_label("  Positive. "), "Positive")
        self.assertEqual(normalize_label("negative"), "Negative")
        self.assertIsNone(normalize_label("The answer is Positive"))
        self.assertIsNone(normalize_label(None))
        self.assertIsNone(normalize_label(""))


if __name__ == "__main__":
    unittest.main()
