# Reproducibility and smoke-test scope

The public repository contains experiment code, prompt templates, dataset
inputs, software snapshots and tests. Generated results, predictions, model
weights and figures are **excluded from Git**.

Automated checks validate Python syntax, test group-aware split and
evaluation functions, and verify the frozen original/group-disjoint input
partitions. They do **not** train the machine/deep-learning/Transformer
models or run local Ollama models.

See `README.md` for setup and `docs/EXPERIMENTS.md` for the complete
script-to-experiment mapping. The pre-cleanup code and result files remain
recoverable through historical Git commits.
