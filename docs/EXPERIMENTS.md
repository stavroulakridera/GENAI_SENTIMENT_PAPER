# Experiment inventory and commands

All scripts are run from the repository root. Outputs are generated locally
under `results/` and are ignored by Git.

## Data preparation

| Experiment or check | Script |
| --- | --- |
| Clean raw data | `src/preprocessing/sentiment_analysis.py` |
| Create entity-aware inputs | `src/preprocessing/create_entity_tweet_data.py` |
| Audit train/test related-ID overlaps | `src/robustness/tweet_id_leakage_audit.py` |
| Rebuild a group-disjoint split (only if intentionally needed) | `src/robustness/group_disjoint_split.py` |

Frozen inputs are already saved in `data/group_disjoint/`. Do not
silently regenerate them when replicating existing comparisons.

## Standard supervised experiments

| Family | Tweet-only | Entity+Tweet |
| --- | --- | --- |
| Naive Bayes | `src/ml/naive_bayes.py` | `src/ml/naive_bayes_entity.py` |
| Logistic Regression | `src/ml/logistic_regression.py` | `src/ml/logistic_regression_entity.py` |
| Linear SVM | `src/ml/linear_svm.py` | `src/ml/linear_svm_entity.py` |
| SGD | `src/ml/sgd_classifier.py` | `src/ml/sgd_entity.py` |
| Random Forest | `src/ml/random_forest.py` | `src/ml/random_forest_entity.py` |
| DNN (five seeds) | `src/dl/dnn_tweet_only_5seeds.py` | `src/dl/dnn_entity_5seeds.py` |
| 1D CNN (five seeds) | `src/dl/cnn_tweet_only_5seeds.py` | `src/dl/cnn_entity_5seeds.py` |
| LSTM (five seeds) | `src/dl/lstm_tweet_only_5seeds.py` | `src/dl/lstm_entity_5seeds.py` |
| GRU (five seeds) | `src/dl/gru_tweet_only_5seeds.py` | `src/dl/gru_entity_5seeds.py` |
| BiLSTM (five seeds) | `src/dl/bilstm_tweet_only_5seeds.py` | `src/dl/bilstm_entity_5seeds.py` |
| BERT (five seeds) | `src/transformers/bert_tweet_only_5seeds.py` | `src/transformers/bert_entity_5seeds.py` |
| RoBERTa (five seeds) | `src/transformers/roberta_tweet_only_5seeds.py` | `src/transformers/roberta_entity_5seeds.py` |

Original single-run DL scripts are retained for historical baselines. The
five-seed versions are preferred for manuscript-comparable stochastic runs.

## Original LLM experiments

All scripts are in `src/genai/`.

| Model | Zero-shot | Four-shot | Reasoning-guided |
| --- | --- | --- | --- |
| Llama 3.2 3B | `llama_zero_shot_ollama.py` | `llama_few_shot_ollama.py` | `llama_reasoning_ollama.py` |
| Qwen2.5 3B | `qwen_zero_shot_ollama.py` | `qwen_few_shot_ollama.py` | `qwen_reasoning_ollama.py` |
| Gemma 3 4B | `gemma_zero_shot_ollama.py` | `gemma_few_shot_ollama.py` | `gemma_reasoning_ollama.py` |

Exact original prompt templates are preserved under `prompts/`.
The original Llama four-shot demonstrations come from the training set;
Qwen and Gemma use fixed illustrative examples.

## Group-disjoint robustness and targeted extensions

| Experiment | Script |
| --- | --- |
| Linear SVM Tweet-only | `src/robustness/group_linear_svm.py` |
| Linear SVM Entity+Tweet | `src/robustness/group_linear_svm_entity.py` |
| DNN with Tweet-ID-disjoint internal development | `src/robustness/group_dnn_entity.py` |
| CNN with Tweet-ID-disjoint internal development | `src/robustness/group_cnn_entity.py` |
| Five-seed corrected group DNN and CNN | `src/robustness/run_group_neural_5seeds.py --model all` |
| Five-seed group-disjoint BERT | `src/robustness/group_bert_entity_5seeds.py` |
| Qwen four-shot on group-disjoint cohort | `src/robustness/group_qwen_few_shot_ollama.py` |
| Controlled identical four-shot prompt across Llama/Qwen/Gemma | `src/genai/controlled_few_shot_comparison.py --model all` |

The five-seed DNN/CNN runner passes `SENTIMENT_SEED` to the existing
group-aware DNN/CNN scripts. Outputs from different seeds are aggregated
locally; original result files are not required in the public repository.

## Statistical and error analyses

| Analysis | Script |
| --- | --- |
| Consolidated model comparison | `src/evaluation/all_models_comparison.py` |
| Class precision, recall, F1 and invalid-output-aware metrics | `src/evaluation/recompute_class_metrics.py` |
| Bootstrap intervals and Holm-adjusted McNemar comparisons | `src/statistical_evaluation/statistical_significance_analysis.py` |

Statistics require locally generated row-level predictions as inputs.

## Interpretation

Original standard validation shares Tweet-ID groups with cleaned training.
Group-disjoint evaluation uses an alternative set of test samples. Hence
differences between the two protocols must not be interpreted as a causal
measure of leakage. Results need not be published in GitHub for the
experiment scripts to be complete or useful.
