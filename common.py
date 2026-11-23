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


def val_split(X, dev_rows, seed=0, n_val=5000, frac=0.2):
    """5,000 held-out *training* users for model selection (identical to the split vae.py used up to 11-23:
    same seed, same order of rng calls). Each val user's items are split at random: 80% fold-in / 20% target.
    Returns val_rows, Xv_in (fold-in histories, csr), val_lab (list of target item sets), Xfit (X with the
    val users' 20% targets removed -- fit models on this so nothing about val targets leaks)."""
    rng = np.random.default_rng(seed)
    is_dev = np.zeros(X.shape[0], bool); is_dev[dev_rows] = True
    val_rows = rng.choice(np.where(~is_dev)[0], n_val, replace=False)
    Xv = X[val_rows].tocsr(); r, c = Xv.nonzero()
    hide = rng.random(len(r)) < frac
    Xv_in = sp.csr_matrix((np.ones((~hide).sum(), np.float32), (r[~hide], c[~hide])), shape=Xv.shape)
    val_lab = [set() for _ in range(n_val)]
    for i, j in zip(r[hide], c[hide]): val_lab[i].add(j)
    keep = np.ones(X.shape[0], bool); keep[val_rows] = False
    Xfit = sp.vstack([X[keep], Xv_in]).tocsr()         # row order does not matter for item-item models
    return val_rows, Xv_in, val_lab, Xfit


def per_user_recall(top, labs, n):
    """course-style recall per user: hits / n (n = number of held-out items)."""
    return np.array([len(set(top[i]) & labs[i]) for i in range(len(labs))]) / np.asarray(n, float)


def paired_bootstrap(a, b, n_boot=2000, seed=0):
    """mean of per-user (a - b) with a 95% percentile bootstrap CI over users."""
    d = np.asarray(a) - np.asarray(b); rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
