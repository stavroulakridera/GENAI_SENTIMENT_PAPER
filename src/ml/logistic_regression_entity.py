import os
import time
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    classification_report,
    confusion_matrix
)

# =========================================================
# 1. LOAD ENTITY + TWEET DATA
# =========================================================

print("=" * 65)
print("LOGISTIC REGRESSION — ENTITY + TWEET")
print("=" * 65)

train_df = pd.read_csv("results/entity_tweet_train.csv")
test_df = pd.read_csv("results/entity_tweet_validation.csv")

X_train_text = train_df["Entity_Tweet"].astype(str)
X_test_text = test_df["Entity_Tweet"].astype(str)

y_train = train_df["Sentiment"]
y_test = test_df["Sentiment"]

print("Training samples:", len(train_df))
print("Validation samples:", len(test_df))

# =========================================================
# 2. TF-IDF
# Same configuration as Tweet-only baseline
# =========================================================

print("\nCreating TF-IDF features...")

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 1),
    min_df=2,
    sublinear_tf=True,
    dtype=np.float32
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

print("Training matrix:", X_train.shape)
print("Validation matrix:", X_test.shape)

# =========================================================
# 3. TRAIN MODEL
# =========================================================

model = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=42
)

start_train = time.perf_counter()
model.fit(X_train, y_train)
training_time = time.perf_counter() - start_train

# =========================================================
# 4. PREDICT
# =========================================================

start_pred = time.perf_counter()
y_pred = model.predict(X_test)
prediction_time = time.perf_counter() - start_pred

# =========================================================
# 5. METRICS
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

macro_precision = precision_score(
    y_test, y_pred, average="macro", zero_division=0
)

macro_recall = recall_score(
    y_test, y_pred, average="macro", zero_division=0
)

macro_f1 = f1_score(
    y_test, y_pred, average="macro", zero_division=0
)

weighted_f1 = f1_score(
    y_test, y_pred, average="weighted", zero_division=0
)

balanced_accuracy = balanced_accuracy_score(y_test, y_pred)

mcc = matthews_corrcoef(y_test, y_pred)

print("\n" + "=" * 65)
print("RESULTS")
print("=" * 65)

print(f"Accuracy:          {accuracy:.4f}")
print(f"Macro Precision:   {macro_precision:.4f}")
print(f"Macro Recall:      {macro_recall:.4f}")
print(f"Macro F1:          {macro_f1:.4f}")
print(f"Weighted F1:       {weighted_f1:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"MCC:               {mcc:.4f}")
print(f"Training Time:     {training_time:.4f} sec")
print(f"Prediction Time:   {prediction_time:.4f} sec")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# =========================================================
# 6. SAVE RESULTS
# =========================================================

os.makedirs("results/tables", exist_ok=True)

results = pd.DataFrame({
    "Model": ["Logistic Regression"],
    "Input": ["Entity + Tweet"],
    "Accuracy": [accuracy],
    "Macro Precision": [macro_precision],
    "Macro Recall": [macro_recall],
    "Macro F1": [macro_f1],
    "Weighted F1": [weighted_f1],
    "Balanced Accuracy": [balanced_accuracy],
    "MCC": [mcc],
    "Training Time": [training_time],
    "Prediction Time": [prediction_time]
})

results.to_csv(
   "results/tables/logistic_regression_entity_results.csv",
    index=False
)

predictions = test_df[
    ["Tweet_ID", "Entity", "Tweet", "Sentiment"]
].copy()

predictions["Prediction"] = y_pred

predictions.to_csv(
    "results/tables/logistic_regression_entity_predictions.csv",
    index=False
)

cm = confusion_matrix(y_test, y_pred)

pd.DataFrame(
    cm,
    index=model.classes_,
    columns=model.classes_
).to_csv(
    "results/tables/logistic_regression_entity_confusion_matrix.csv"
)

print("\nResults saved successfully.")