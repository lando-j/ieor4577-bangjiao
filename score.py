#!/usr/bin/env python3
"""IEOR 4577 official scorer.

Usage:
    python3 score.py                  -> scores the popularity baseline
    python3 score.py predictions.csv [labels.csv]

predictions.csv: columns user_id,item_id — your ranked recommendations,
best first, at least 20 rows per user. Extra columns are ignored.
labels.csv defaults to dev_labels.csv in this folder.

The metric (identical for everyone, all term): rank the FULL catalog,
exclude items already in the user's given history, then
Recall@20 = fraction of the user's held-out items found in your top 20,
Precision@20 = hits / 20. Reported as means over users.
"""
import sys

import pandas as pd

K = 20


def main():
    labels_path = sys.argv[2] if len(sys.argv) > 2 else "dev_labels.csv"
    inter_path = sys.argv[3] if len(sys.argv) > 3 else "interactions.csv"

    labels = pd.read_csv(labels_path)
    inter = pd.read_csv(inter_path, usecols=["user_id", "item_id"])

    if len(sys.argv) > 1:
        preds = pd.read_csv(sys.argv[1])
    else:
        top = inter["item_id"].value_counts().head(100).index.tolist()
        users = labels["user_id"].unique()
        preds = pd.DataFrame(
            [(u, it) for u in users for it in top],
            columns=["user_id", "item_id"],
        )
        print("no predictions file given — scoring the popularity baseline")

    hist = inter.groupby("user_id")["item_id"].apply(set)
    lab = labels.groupby("user_id")["item_id"].apply(set)
    pred = preds.groupby("user_id")["item_id"].apply(list)

    recalls, precisions, missing = [], [], 0
    for u, truth in lab.items():
        ranked = pred.get(u)
        if ranked is None:
            missing += 1
            ranked = []
        seen = hist.get(u, set())
        top = [it for it in ranked if it not in seen][:K]
        hits = len(set(top) & truth)
        recalls.append(hits / len(truth))
        precisions.append(hits / K)

    print(f"users scored: {len(recalls)}  (missing predictions: {missing})")
    print(f"Recall@{K}:    {sum(recalls) / len(recalls):.4f}")
    print(f"Precision@{K}: {sum(precisions) / len(precisions):.4f}")


if __name__ == "__main__":
    main()
