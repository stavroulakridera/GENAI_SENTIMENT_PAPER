# Reproducibility Check

Repository-level reproducibility audit completed after the paper repository reorganization.

## Verified

- All original experiment scripts are preserved.
- The organized `src/` hierarchy contains 53 Python scripts covering preprocessing, ML, DL, Transformers, GenAI, robustness, statistical evaluation, and cross-model evaluation.
- Raw datasets are stored under `data/raw/`.
- Processed datasets are stored under `data/processed/`.
- Group-disjoint datasets are stored under `data/group_disjoint/`.
- Organized scripts use relative dataset paths compatible with the new folder structure.
- No organized script contains local machine paths such as `/Users/...` or `C:\\Users\\...`.
- Existing experiment outputs remain under `results/`, including 131 CSV result files.
- Prompt templates for Llama, Qwen, and Gemma are preserved under `prompts/`.
- Large model checkpoints are excluded from the repository.
- Historical root-level scripts are retained for traceability; `src/` is the reproducibility entry point.
- Existing numerical results, random seeds, model settings, and experimental logic were not changed during restructuring.

## Execution convention

Run organized scripts from the repository root, for example:

```bash
python src/ml/linear_svm_entity.py
python src/dl/dnn_entity_5seeds.py
python src/transformers/bert_entity_5seeds.py
python src/genai/qwen_few_shot_ollama.py
python src/robustness/group_linear_svm_entity.py
python src/statistical_evaluation/statistical_significance_analysis.py
```

## Dependency provenance

The active project virtual environment was captured after the repository reorganization.

- Operating system: macOS 26.5.2 (Build 25F84)
- Architecture: arm64
- Hardware: Apple M2, 8 GB RAM
- Python: 3.13.1
- Full package snapshot: `environment/pip-freeze.txt`
- Machine/OS snapshot: `environment/system-info.txt`
- Install-oriented pinned dependencies: `requirements.txt`
- Ollama model snapshot: `environment/ollama-models.txt`

Key recorded versions include scikit-learn 1.9.0, TensorFlow 2.21.0, PyTorch 2.14.0, Transformers 5.17.0, pandas 3.0.5, NumPy 2.5.2, and Ollama 0.6.2.

The local GenAI models are `llama3.2:3b`, `qwen2.5:3b`, and `gemma3:4b`.

## Scope of this check

This was a static repository and artifact consistency audit. It verifies organization, preserved artifacts, relative paths, input/output locations, and absence of machine-specific paths. It does not claim a fresh full retraining of every computationally expensive model.
