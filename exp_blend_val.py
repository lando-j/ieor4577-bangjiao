"""Blend a neural model with a closed-form one, with the weight AND the score normalization chosen on the
5,000 validation users (proposal protocol); dev evaluated once. Also paired bootstrap of single models vs EASE-300.
Needs runs/{val,dev}scores_<a>.npy and runs/{val,dev}scores_<b>.npy.
Usage: python exp_blend_val.py recvae edlae_l300_p0.2
"""
import sys, numpy as np, pandas as pd, scipy.linalg as la
from common import *
from log import log, bins

na, nb_ = sys.argv[1], sys.argv[2]
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows].tocsr()
val_rows, Xv_in, val_lab, _ = val_split(X, dev_rows)
val_n = np.array([max(len(s), 1) for s in val_lab])
dev_lab = [lab[u] for u in dev]; dev_n = np.array([n_lab[u] for u in dev])

def norm(S, Xin, how):
    S = S.astype(np.float32); r, c = Xin.nonzero(); S[r, c] = np.nan
    if how == "z":
        S = (S - np.nanmean(S, 1, keepdims=True)) / np.nanstd(S, 1, keepdims=True)
    elif how == "top-z":   # standardize using each user's top-200 scores only (the part that decides the ranking)
        T = -np.sort(-np.nan_to_num(S, nan=-np.inf), 1)[:, :200]
        S = (S - T.mean(1, keepdims=True)) / T.std(1, keepdims=True)
    elif how == "rank":    # reciprocal rank
        o = np.argsort(-np.nan_to_num(S, nan=-np.inf), 1); R = np.empty_like(o); R[np.arange(len(S))[:, None], o] = np.arange(S.shape[1])
        S = 1.0 / (60 + R)
    S[r, c] = -np.inf
    return S

def recall_of(S, Xin, labs, n): return per_user_recall(topk(S, Xin), labs, n)

Va, Vb = np.load(f"runs/valscores_{na}.npy"), np.load(f"runs/valscores_{nb_}.npy")
ws = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
best = (-1, None, None)
for how in ["z", "top-z", "rank"]:
    A, B = norm(Va, Xv_in, how), norm(Vb, Xv_in, how)
    for w in ws:
        S = np.where(np.isinf(A), -np.inf, w * np.where(np.isinf(A), 0, A) + (1 - w) * np.where(np.isinf(B), 0, B))
        v = recall_of(S, Xv_in, val_lab, val_n).mean()
        print(f"val  {how:5s} w_{na}={w:<4}  {v:.4f}", flush=True)
        if v > best[0]: best = (v, how, w)
del Va, Vb
print(f"\nchosen on val: norm={best[1]} w={best[2]} (val {best[0]:.4f})")

Da, Db = np.load(f"runs/devscores_{na}.npy"), np.load(f"runs/devscores_{nb_}.npy")
A, B = norm(Da, Xd, best[1]), norm(Db, Xd, best[1]); del Da, Db
S = np.where(np.isinf(A), -np.inf, best[2] * np.where(np.isinf(A), 0, A) + (1 - best[2]) * np.where(np.isinf(B), 0, B))
top = topk(S, Xd); rb = per_user_recall(top, dev_lab, dev_n)
ra = recall_of(A, Xd, dev_lab, dev_n)

G = (X.T @ X).toarray().astype(np.float32); G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False); Bm = P / (-np.diag(P))[None, :]; np.fill_diagonal(Bm, 0); del G, P
r0 = per_user_recall(topk(np.asarray(Xd @ Bm, dtype=np.float32), Xd), dev_lab, dev_n); del Bm
pv = pd.read_csv("runs/pred_paper.csv")       # Mult-VAE (paper schedule) predictions, for the comparison table
pt = pv.groupby("user_id").item_id.apply(list); imap = pd.Series(np.arange(len(iids)), index=iids)
rv = per_user_recall(np.array([imap[pt[u]].values for u in dev]), dev_lab, dev_n)

hits = np.array([len(set(top[i]) & dev_lab[i]) for i in range(len(dev))])
print(f"\ndev: EASE-300 {r0.mean():.4f} | Mult-VAE {rv.mean():.4f} | {na} {ra.mean():.4f} | blend {rb.mean():.4f} "
      f"(paper-style {(hits/np.minimum(20, dev_n)).mean():.4f})")
for name, x, y in [(f"{na} vs EASE", ra, r0), (f"{na} vs Mult-VAE", ra, rv), (f"blend vs EASE", rb, r0), (f"blend vs {na}", rb, ra)]:
    m, lo, hi = paired_bootstrap(x, y); print(f"  {name:22s} {m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
pop = np.asarray(X.sum(0)).ravel(); rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
m, lo, hi = paired_bootstrap(rb, r0)
log(f"{na}+{nb_} 融合", f"norm={best[1]} w_{na}={best[2]} (chosen on 5k val users, val={best[0]:.4f})", rb.mean(), hits.mean() / K,
    bins(top, dev, lab, rank), len(np.unique(top)), f"论文口径 {(hits/np.minimum(20, dev_n)).mean():.4f}; vs EASE-300 {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
write_preds(top, dev, iids, f"runs/pred_blend_{na}_{nb_}.csv")
