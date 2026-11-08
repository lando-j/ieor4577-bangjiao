import os
import numpy as np, pandas as pd, scipy.sparse as sp
# folder holding interactions.csv / items.csv / dev_labels.csv (not in the repo); override with IEOR_DATA
D = os.environ.get("IEOR_DATA", "/Users/landojiao/Downloads/ieor4577-student-dataset/").rstrip("/") + "/"
K = 20

def load():
    inter = pd.read_csv(D + "interactions.csv", usecols=["user_id", "item_id"])
    labels = pd.read_csv(D + "dev_labels.csv")
    uids, urow = np.unique(inter.user_id.values, return_inverse=True)
    iids, icol = np.unique(inter.item_id.values, return_inverse=True)
    X = sp.csr_matrix((np.ones(len(inter), np.float32), (urow, icol)), shape=(len(uids), len(iids)))
    X.data[:] = 1.0
    dev = np.unique(labels.user_id.values)
    dev_rows = np.searchsorted(uids, dev)
    imap = pd.Series(np.arange(len(iids)), index=iids)
    lab = labels.groupby("user_id").item_id.apply(lambda s: set(imap.reindex(s.values).dropna().astype(int)))
    n_lab = labels.groupby("user_id").size()  # denominator includes labels for unseen items
    return X, uids, iids, dev, dev_rows, lab, n_lab

def topk(scores, Xd, k=K):
    scores = scores.copy()
    r, c = Xd.nonzero()
    scores[r, c] = -np.inf
    idx = np.argpartition(-scores, k, axis=1)[:, :k]
    o = np.argsort(-np.take_along_axis(scores, idx, 1), axis=1)
    return np.take_along_axis(idx, o, 1)

def evaluate(top, dev, lab, n_lab):
    rec = [len(set(top[i]) & lab[u]) / n_lab[u] for i, u in enumerate(dev)]
    return float(np.mean(rec))

def write_preds(top, dev, iids, path):
    pd.DataFrame({"user_id": np.repeat(dev, top.shape[1]), "item_id": iids[top].ravel()}).to_csv(path, index=False)
