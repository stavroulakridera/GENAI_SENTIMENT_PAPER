# Prompt templates

The templates in `llama/`, `qwen/` and `gemma/` preserve the original
zero-shot, four-shot and reasoning-guided prompts from the experimental
scripts. Llama originally selects a training demonstration per class,
while Qwen and Gemma use fixed illustrative examples.

The additional `src/genai/controlled_few_shot_comparison.py` compares
all three LLMs using the same four-shot template from
`prompts/qwen/few_shot.txt`. It does not change the original experiments.
