# Generative AI and Entity-Aware Sentiment Analysis

## Overview

This repository contains the source code, experimental pipeline, evaluation procedures, and results developed as part of a research study on advanced sentiment analysis of social media content.

The study explores and compares traditional Machine Learning (ML), Deep Learning (DL), transformer-based architectures, and Large Language Models (LLMs) within a common experimental framework.

A particular focus is placed on investigating whether entity-related information can improve sentiment classification compared with approaches based exclusively on textual content.

## Purpose of the Study

The primary objective of this study is to systematically investigate the effectiveness of different computational approaches for sentiment classification and to examine the contribution of entity-aware information to predictive performance.

The experimental framework compares conventional machine learning algorithms, neural network architectures, transformer-based models, and Large Language Models.

For selected models, two main configurations are investigated:

- **Tweet-only:** sentiment classification based exclusively on the textual content of the tweet.
- **Entity-aware:** sentiment classification incorporating information related to the entity associated with the text.

Through this comparison, the study aims to determine whether the inclusion of entity-related contextual information provides measurable benefits over text-only sentiment classification.

## Models and Approaches

The repository includes experiments involving several modelling families, including:

- Logistic Regression
- Random Forest
- Support Vector Machines (SVM)
- XGBoost
- Deep Neural Networks
- BERT
- RoBERTa
- Large Language Models (LLMs)

Both conventional and modern NLP approaches are evaluated under a common experimental methodology.

## Experimental Framework

The experimental pipeline includes:

1. Data preprocessing and preparation
2. Text and entity-aware feature configurations
3. Machine Learning experiments
4. Deep Learning experiments
5. Transformer-based experiments
6. Large Language Model experiments
7. Performance evaluation
8. Statistical comparison of model performance
9. Generation of tables and figures for analysis

The use of a unified experimental framework enables a systematic comparison between different modelling paradigms.

## Evaluation

Model performance is evaluated using standard classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score

Additional statistical analyses are performed to investigate whether observed differences between modelling approaches are statistically meaningful.

## Repository Structure

The repository contains Python scripts for:

- Data preprocessing
- Machine Learning experiments
- Deep Learning experiments
- BERT and RoBERTa experiments
- Tweet-only and entity-aware configurations
- LLM-based sentiment classification
- Statistical analysis
- Generation of figures and result tables

Experimental outputs and supporting result files are organised within the corresponding results directories.

## Reproducibility

The source code is provided to support transparency and reproducibility of the experimental methodology presented in the associated research study.

Large model checkpoints and local virtual-environment files are not included in the repository.

## Research Context

This repository accompanies research conducted within the context of doctoral research in the areas of Artificial Intelligence, Natural Language Processing, and online social behaviour analysis.

The work investigates how different generations of computational models—from traditional machine learning to transformer architectures and Generative AI—can be used to analyse sentiment expressed in social media content.

## Citation

If you use this repository or its associated methodology in academic work, please cite the corresponding research paper.

Full citation information will be added upon publication.

## License

This repository is intended for academic and research purposes.
