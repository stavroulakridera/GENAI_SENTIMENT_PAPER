# Reliability and Generalization of Supervised Models and Large Language Models in Entity-Level Sentiment Classification

Research code for the manuscript by **Stavroula Kridera and Andreas Kanavos** (Department of Informatics, Ionian University, Greece).

**Journal:** *Frontiers in Artificial Intelligence*  
**Research Topic:** [Societal Implications of Artificial Intelligence and Natural Language Processing: Trust, Ethics, and Human-Centered AI](https://www.frontiersin.org/research-topics/81031/societal-implications-of-artificial-intelligence-and-natural-language-processing-trust-ethics-and-human-centered-ai)

This is a **code-first research repository**. It provides the experimental scripts, original and processed input datasets, prompt templates, software environment and basic validation tests. **Generated predictions, metrics, figures, logs and checkpoints are not version-controlled**. Scripts create their results locally under `results/`.

## Research scope

Four-class, target-aware Twitter sentiment classification (**Negative, Neutral, Positive, Irrelevant**), including:

- Five traditional ML classifiers: Multinomial Naive Bayes, Logistic Regression, Linear SVM, SGD and Random Forest.
- Five supervised deep learning models: TF–IDF DNN, CNN, LSTM, GRU and BiLSTM.
- Two fine-tuned Transformers: BERT-base and RoBERTa-base.
- Three locally run generative LLMs: Llama 3.2 3B, Qwen2.5 3B and Gemma 3 4B, with zero-shot, four-shot and reasoning-guided prompting.

Supervised models have **Tweet-only** and **Entity+Tweet** variants. The main stochastic experiments use **five seeds: 42, 52, 62, 72, 82**.

## Repository structure

```text
data/
  raw/                   Original Twitter CSV files
  processed/             Cleaned CSVs and Entity+Tweet representations
  group_disjoint/        Frozen Tweet-ID-separated training/test input files
src/
  preprocessing/         Cleaning and input preparation
  ml/                    Five traditional ML classifier families
  dl/                    DNN, CNN, LSTM, GRU, BiLSTM
  transformers/          BERT and RoBERTa
  genai/                 Llama/Qwen/Gemma scripts, plus controlled four-shot test
  robustness/            Overlap audit, group splits, SVM, DNN, CNN, BERT, Qwen
  evaluation/            Metric calculation and comparison
  statistical_evaluation/ Bootstrap confidence intervals and McNemar tests
prompts/                 Original prompt templates
environment/             Software and local LLM version snapshots
tests/                   Small reproducibility tests
docs/EXPERIMENTS.md      Full experiment-to-script reference
requirements.txt
results/                  GENERATED LOCALLY; ignored by Git
```

Historical duplicate source files have been removed from the working tree; previous versions are recoverable through Git history.

## Setup

Execute from the **repository root** in a Python 3.13 environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

For the local GenAI scripts install and start [Ollama](https://ollama.com/) and download the specified models:

```bash
ollama pull llama3.2:3b
ollama pull qwen2.5:3b
ollama pull gemma3:4b
```

Exact Python-package and Ollama-version snapshots are documented in `environment/`.

## Input preparation

```bash
python src/preprocessing/sentiment_analysis.py
python src/preprocessing/create_entity_tweet_data.py
```

The fixed group-disjoint datasets are included in `data/group_disjoint/`. The split-generation script is also retained for auditability; **do not silently overwrite the frozen split** with a new experimental cohort.

## Main supervised experiments

```bash
python src/ml/linear_svm_entity.py
python src/dl/dnn_entity_5seeds.py
python src/dl/cnn_entity_5seeds.py
python src/transformers/bert_entity_5seeds.py
python src/transformers/roberta_entity_5seeds.py
```

The full Tweet-only/Entity+Tweet mapping of all 12 supervised models is in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md). The original single-run DL scripts are retained to reproduce baseline configurations; the named five-seed scripts are used for multi-seed comparisons.

## Original nine LLM configurations

```bash
python src/genai/llama_zero_shot_ollama.py
python src/genai/llama_few_shot_ollama.py
python src/genai/llama_reasoning_ollama.py
python src/genai/qwen_zero_shot_ollama.py
python src/genai/qwen_few_shot_ollama.py
python src/genai/qwen_reasoning_ollama.py
python src/genai/gemma_zero_shot_ollama.py
python src/genai/gemma_few_shot_ollama.py
python src/genai/gemma_reasoning_ollama.py
```

The original few-shot protocols differ: Llama draws one example per sentiment class from the training data; Qwen and Gemma use fixed illustrative examples. The nine original experiments are therefore descriptive across model/prompt configurations, **not a single identical-prompt model comparison**.

## Generalization, overlap and robustness experiments

```bash
python src/robustness/tweet_id_leakage_audit.py
python src/robustness/group_linear_svm_entity.py
python src/robustness/run_group_neural_5seeds.py --model all
python src/robustness/group_bert_entity_5seeds.py
python src/robustness/group_qwen_few_shot_ollama.py
```

The revised DNN/CNN experiments use a Tweet-ID-disjoint internal train/development separation and the same fixed external group-disjoint test set. The five-seed runner collects DNN and CNN runs for seeds 42, 52, 62, 72, 82. The new BERT group-disjoint script uses the same external test and five-seed design.

**Important:** Standard held-out validation and group-disjoint testing use different test cohorts. Differences between their scores are *descriptive*, not causal estimates of leakage. A local group-disjoint test does not establish anything about the external pretraining data of an LLM.

## Additional controlled four-shot comparison

```bash
python src/genai/controlled_few_shot_comparison.py --model all
```

This optional script uses **one identical four-shot prompt**, entity/text input, inference temperature and test set for Llama, Qwen and Gemma. The original LLM scripts and prompts are not changed. For a quick pilot, use `--model qwen --limit 20`; pilot outputs use separate `_pilot` filenames.

## Post-experiment analysis

```bash
python src/evaluation/recompute_class_metrics.py
python src/statistical_evaluation/statistical_significance_analysis.py
```

These scripts work from **locally generated predictions**. The metrics script supports original and modern CSV layouts, separates LLM output validity from classification correctness, and counts invalid LLM responses as errors over the full test denominator.

## Code and output policy

Only **source code, data inputs, prompt templates, tests and documentation** are committed. Outputs under `results/` are ignored by Git and are optional for users of the repository. GitHub Actions runs lightweight syntax, split and schema checks; it does **not** run computationally expensive training or Ollama inference.

The original experiment code is preserved in Git history. No numerical manuscript results are asserted from the existence of a script.
