# A Systematic Multi-Paradigm Evaluation of Machine Learning, Deep Learning, and Generative AI for Fine-Grained Entity-Level Sentiment Classification

This repository contains the experimental code, dataset organization, prompt templates, predictions, and statistical evaluation artifacts for the paper.

## Experimental scope

The study compares four model families for four-class entity-level sentiment classification (Negative, Neutral, Positive, Irrelevant):

- **Machine Learning:** Multinomial Naive Bayes, Logistic Regression, Linear SVM, SGD, Random Forest.
- **Deep Learning:** DNN, 1D CNN, LSTM, GRU, BiLSTM.
- **Transformers:** BERT-base and RoBERTa-base.
- **Generative AI:** Llama 3.2 3B, Qwen2.5 3B, and Gemma 3 4B under zero-shot, four-shot/few-shot, and reasoning-enhanced prompting.

Supervised models are evaluated using both Tweet-only and Entity+Tweet representations. Stochastic supervised experiments use five seeds: **42, 52, 62, 72, 82**.

## Repository structure

```text
.
├── data/
│   ├── raw/
│   ├── processed/
│   └── group_disjoint/
├── src/
│   ├── preprocessing/
│   ├── ml/
│   ├── dl/
│   ├── transformers/
│   ├── genai/
│   ├── robustness/
│   ├── statistical_evaluation/
│   └── evaluation/
├── prompts/
│   ├── llama/
│   ├── qwen/
│   └── gemma/
├── results/
├── requirements.txt
└── README.md
```

## Data

The original Twitter sentiment files are stored under `data/raw/`.

Processed files used by the experiments are organized as:

```text
data/processed/clean_train.csv
data/processed/clean_validation.csv
data/processed/entity_tweet_train.csv
data/processed/entity_tweet_validation.csv
```

The group-disjoint robustness split is:

```text
data/group_disjoint/group_train.csv
data/group_disjoint/group_test.csv
```

The group-disjoint split is constructed by Tweet_ID with random state 42 and is checked for zero Tweet_ID overlap and zero exact-text overlap between train and test.

## Running the experiments

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Run scripts from the repository root. The organized source folders correspond directly to experiment families:

```bash
python src/ml/linear_svm_entity.py
python src/dl/dnn_entity_5seeds.py
python src/transformers/bert_entity_5seeds.py
python src/genai/qwen_few_shot_ollama.py
python src/robustness/group_linear_svm_entity.py
python src/statistical_evaluation/statistical_significance_analysis.py
```

Before rerunning the organized scripts, verify that their data/output paths point to the new `data/` and `results/` locations. The original root-level scripts are temporarily retained during repository migration to preserve the exact historical experiment code.

## GenAI experiments

Generative AI experiments run locally through Ollama. The evaluated model families are:

- Llama 3.2 3B
- Qwen2.5 3B
- Gemma 3 4B

Exact prompt templates are stored under `prompts/`. The prompts require the model to output exactly one of the four labels. Reasoning-enhanced prompts instruct the model to apply relevance, sentiment-presence, and polarity decisions internally while returning only the final label.

For Llama few-shot evaluation, one training example per class is selected by the experiment script and the selected examples are preserved in the experiment outputs. Qwen and Gemma few-shot experiments use the four fixed examples recorded in their prompt templates.

## Evaluation metrics

The principal metrics are:

- Accuracy
- Macro-averaged Precision
- Macro-averaged Recall
- Macro F1
- Weighted F1 where applicable
- Balanced Accuracy
- Matthews Correlation Coefficient (MCC)

Prediction CSV files preserve true and predicted labels so that additional metrics, confusion matrices, and paired statistical tests can be computed without retraining.

## Robustness evaluation

The `src/robustness/` directory contains:

- Tweet-ID leakage audit
- Group-disjoint split generation
- Group-disjoint Linear SVM experiments
- Group-disjoint DNN experiment
- Group-disjoint CNN experiment
- Group-disjoint Qwen few-shot experiment

The preserved group-disjoint split contains 61,760 training instances and 6,865 test instances, with no Tweet_ID or exact-text overlap.

## Statistical evaluation

`src/statistical_evaluation/statistical_significance_analysis.py` performs:

- Macro-F1 bootstrap confidence intervals
- 2,000 bootstrap iterations
- Bootstrap seed 42
- Pairwise McNemar tests
- Holm correction for multiple comparisons

For paired statistical tests involving five-seed DNN/BERT outputs, seed 42 is used as the selected deterministic comparison run.

## Reproducibility parameters

Key fixed parameters include:

- Data/split random state: 42
- Five-seed experiments: 42, 52, 62, 72, 82
- TF-IDF vocabulary limit in core supervised experiments: 10,000 features
- Linear SVM: C=1.0
- Statistical bootstrap iterations: 2,000
- Statistical bootstrap seed: 42

Model-specific hyperparameters remain explicitly defined in each experiment script.

## Results

The repository's result artifacts are intended to preserve both aggregate metrics and row-level predictions. This allows evaluation and statistical analyses to be reproduced without repeating expensive model training.

Large BERT/RoBERTa checkpoint directories are intentionally excluded from version control. They are not required for reproducing analyses from the saved predictions and metrics.

## Script-to-experiment mapping

| Experiment family | Main scripts |
| --- | --- |
| Preprocessing | `src/preprocessing/` |
| Traditional ML | `src/ml/` |
| DNN/CNN/LSTM/GRU/BiLSTM | `src/dl/` |
| BERT/RoBERTa | `src/transformers/` |
| Llama/Qwen/Gemma | `src/genai/` |
| Group-disjoint & leakage | `src/robustness/` |
| Bootstrap/McNemar | `src/statistical_evaluation/` |
| Cross-model summaries | `src/evaluation/` |

## Computational environment

The experiments were developed and executed in Python-based environments, with Transformer experiments using PyTorch/Transformers, neural baselines using TensorFlow/Keras, and local GenAI inference using Ollama.

For archival reproducibility, exact package versions should be captured from the original experiment environment where available. The repository intentionally does not claim versions that were not recorded by the experiment scripts.

## Reproducibility policy

No published numerical result should be changed during repository reorganization. Repository restructuring is limited to organization, documentation, relative-path adaptation, and preservation of existing experiment outputs. New experiments, if any, should be introduced in separate commits after the reproducibility baseline has been verified.
