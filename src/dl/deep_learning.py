import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# =========================================================
# 1. REPRODUCIBILITY
# =========================================================

RANDOM_STATE = 42

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

# =========================================================
# 2. LOAD CLEAN DATA
# =========================================================

print("=" * 60)
print("DEEP LEARNING EXPERIMENTS")
print("=" * 60)

train_df = pd.read_csv("data/processed/clean_train.csv")
test_df = pd.read_csv("data/processed/clean_validation.csv")

print("\nTraining:", train_df.shape)
print("Final evaluation:", test_df.shape)

# =========================================================
# 3. LABEL ENCODING
# =========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(train_df["Sentiment"])
y_test = label_encoder.transform(test_df["Sentiment"])

num_classes = len(label_encoder.classes_)

print("\nClasses:")
print(label_encoder.classes_)

y = to_categorical(y, num_classes=num_classes)
y_test = to_categorical(y_test, num_classes=num_classes)

# =========================================================
# 4. INTERNAL TRAIN / DEVELOPMENT SPLIT
# =========================================================

train_texts, dev_texts, y_train, y_dev = train_test_split(
    train_df["Tweet"].astype(str),
    y,
    test_size=0.10,
    random_state=RANDOM_STATE,
    stratify=train_df["Sentiment"]
)

print("\nInternal training:", len(train_texts))
print("Internal development:", len(dev_texts))
print("Final held-out evaluation:", len(test_df))

# =========================================================
# 5. TOKENIZATION
# =========================================================

VOCAB_SIZE = 20000
MAX_LENGTH = 60

tokenizer = Tokenizer(
    num_words=VOCAB_SIZE,
    oov_token="<OOV>"
)

tokenizer.fit_on_texts(train_texts)

X_train = tokenizer.texts_to_sequences(train_texts)
X_dev = tokenizer.texts_to_sequences(dev_texts)
X_test = tokenizer.texts_to_sequences(
    test_df["Tweet"].astype(str)
)

X_train = pad_sequences(
    X_train,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_dev = pad_sequences(
    X_dev,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

X_test = pad_sequences(
    X_test,
    maxlen=MAX_LENGTH,
    padding="post",
    truncating="post"
)

print("\nSequence shapes:")
print("Train:", X_train.shape)
print("Development:", X_dev.shape)
print("Final evaluation:", X_test.shape)

print("\nDeep Learning data preparation completed!")