# Frontiers methodological correction plan

This branch makes targeted changes to the *code*, not to the historical
manuscript results. The reference manuscript is titled
"Reliability and Generalization of Supervised Models and Large Language
Models in Entity-Level Sentiment Classification" (30-page Frontiers draft).

## Changes in this branch

1. Restore the **original 10.3-MB training CSV** to
   `data/raw/twitter_training.csv` using the pre-existing Git blob from
   the identical previously versioned `legacy/original_data/twitter_training.csv` blob (previous raw file was empty). The duplicate `legacy/` folder has since been removed from the working tree, but the pre-cleanup Git commit preserves it.
2. Make the internal DNN/CNN development splits **Tweet-ID-disjoint**.
   The fixed external group-disjoint test dataset is not touched.
3. Add `src/robustness/group_bert_entity_5seeds.py`, which uses the
   original BERT architecture and hyperparameters, the same fixed
   group-disjoint data, the new group-aware internal split, and seeds
   42, 52, 62, 72 and 82.
4. Add a **read-only** metric audit to recompute class-specific Precision,
   Recall, F1, confusion counts and strict-denominator LLM scores from
   preserved predictions; it does not overwrite historical CSV results.
5. Repair the Llama few-shot prompt template so that it contains
   exactly the template constructed by the executable code.
6. Add a deterministic unit test for group-aware split integrity.

## Run order (repository root)

```bash
python -m unittest discover -s tests -v
python src/preprocessing/sentiment_analysis.py
python src/evaluation/recompute_class_metrics.py
python src/robustness/group_dnn_entity.py
python src/robustness/group_cnn_entity.py
python src/robustness/group_bert_entity_5seeds.py
python src/evaluation/recompute_class_metrics.py
```

The preprocessing script regenerates cleaned datasets and may replace the
original processed CSV files. **Before executing it**, archive or checksum
the current processed data, and compare regenerated row counts and labels
to the preserved baseline. The group-disjoint files are already frozen:
do not rerun `group_disjoint_split.py` unless intentionally creating a
different experiment with a separately named output path.

## Expected file outputs from newly executed experiments

- `results/tables/group_dnn_entity_groupaware_predictions.csv`
- `results/tables/group_cnn_entity_groupaware_predictions.csv`
- `results/tables/group_bert_entity_5seeds_predictions.csv`
- `results/tables/group_bert_entity_5seeds_results.csv`
- `results/tables/group_bert_entity_5seeds_summary.csv`
- `results/derived/overall_metrics.csv`
- `results/derived/class_metrics.csv`
- `results/derived/confusion_counts.csv`

The corrected DNN/CNN scripts intentionally use new \`*_groupaware_*\` filenames, so the historical single-seed output CSVs are retained. Commit newly generated outputs only after verification, and never replace paper scores automatically.

## Methodological decisions not yet implemented

- Cross-family few-shot prompt comparison: current Llama demonstrations
  are sampled from training; Qwen and Gemma use fixed illustrative
  demonstrations. Original results remain descriptive, not a fully
  prompt-controlled between-model comparison.
- Class-wise result tables and error examples must be checked and
  selected from the newly generated `results/derived` CSVs.
- The original test partition shares 985 Tweet IDs with cleaned training
  data; differences between standard and group-disjoint tests are
  descriptive and not causal leakage estimates.
- No new scores are claimed in this branch. GPU/Apple MPS training has
  not been executed as part of this code-only review.
