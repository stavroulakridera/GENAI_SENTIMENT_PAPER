"""Leakage-safe internal train/development separation by Tweet_ID.

The external group-disjoint test set is NOT used here. The same deterministic
candidate search and selection must be applied across DNN, CNN and BERT.
"""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

LABELS = ("Negative", "Neutral", "Positive", "Irrelevant")


def group_aware_train_dev_split(frame, *, test_size=0.10,
                                random_state=42, n_candidates=50):
    """Return disjoint row indices for training and development.

    GroupShuffleSplit controls group overlap; among deterministic candidate
    splits we choose one with class proportions closest to the input frame.
    Neither a trained model nor the external test-set results influence choice.
    """
    for column in ("Tweet_ID", "Sentiment"):
        if column not in frame.columns:
            raise ValueError(f"Missing required column: {column}")
    if frame["Tweet_ID"].isna().any() or frame["Sentiment"].isna().any():
        raise ValueError("Tweet_ID and Sentiment must not contain missing values")
    groups = frame["Tweet_ID"].astype(str).to_numpy()
    labels = frame["Sentiment"].astype(str).to_numpy()
    if not set(labels).issubset(LABELS):
        raise ValueError(f"Unexpected sentiment labels: {set(labels)-set(LABELS)}")
    reference = np.array([(labels == x).mean() for x in LABELS])
    splitter = GroupShuffleSplit(
        n_splits=n_candidates, test_size=test_size, random_state=random_state
    )
    candidates = []
    for train_idx, dev_idx in splitter.split(np.zeros(len(frame)), labels, groups):
        train_groups = set(groups[train_idx])
        dev_groups = set(groups[dev_idx])
        assert train_groups.isdisjoint(dev_groups)
        # Every class must occur on BOTH sides, otherwise reject the candidate.
        train_dist = np.array([(labels[train_idx] == x).mean() for x in LABELS])
        dev_dist = np.array([(labels[dev_idx] == x).mean() for x in LABELS])
        if any((labels[train_idx] == x).sum() == 0 for x in LABELS):
            continue
        if any((labels[dev_idx] == x).sum() == 0 for x in LABELS):
            continue
        score = (np.abs(dev_dist - reference).sum()
                 + np.abs(train_dist - reference).sum()
                 + abs(len(dev_idx) / len(frame) - test_size))
        candidates.append((score, train_idx, dev_idx))
    if not candidates:
        raise RuntimeError("Could not obtain group-disjoint train/dev split with all classes")
    _, train_idx, dev_idx = min(candidates, key=lambda item: item[0])
    assert not set(groups[train_idx]) & set(groups[dev_idx])
    assert len(train_idx) + len(dev_idx) == len(frame)
    print("Group-aware train/dev split:",
          len(train_idx), "train,", len(dev_idx), "development;",
          "shared Tweet_ID groups: 0")
    return train_idx, dev_idx
