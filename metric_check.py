"""Recall@20 under two denominators: course score.py (|held-out|) vs Liang et al. 2018 / RecVAE (min(20, |held-out|)).
Usage: python metric_check.py preds1.csv [preds2.csv ...]"""
import sys, numpy as np, pandas as pd
D = "/Users/landojiao/Downloads/ieor4577-student-dataset/"
inter = pd.read_csv(D + "interactions.csv", usecols=["user_id", "item_id"])
labels = pd.read_csv(D + "dev_labels.csv")
hist = inter.groupby("user_id").item_id.apply(set); lab = labels.groupby("user_id").item_id.apply(set)
for f in sys.argv[1:]:
    pred = pd.read_csv(f).groupby("user_id").item_id.apply(list)
    r_full, r_min = [], []
    for u, truth in lab.items():
        top = [i for i in pred[u] if i not in hist.get(u, set())][:20]
        h = len(set(top) & truth)
        r_full.append(h / len(truth)); r_min.append(h / min(20, len(truth)))
    print(f"{f}: course Recall@20 = {np.mean(r_full):.4f} | paper-style (denominator min(20,n)) = {np.mean(r_min):.4f}")
