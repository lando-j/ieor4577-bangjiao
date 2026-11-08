"""Exp 1: reduce popularity bias on top of EASE.
 (a) post-hoc: score_j / pop_j^alpha
 (b) popularity-weighted ridge (EDLAE-style): reg_j = lam + a * pop_j  (closed form is still B = I - P diag(1/P_jj))
Usage: python exp_debias.py
"""
import sys, time, numpy as np, scipy.linalg as la
from common import *
from log import log, bins

t0 = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows]
G0 = (X.T @ X).toarray().astype(np.float32)
pop = np.diag(G0).copy()
rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
OUT = "/Users/landojiao/Downloads/ieor4577-student-dataset/experiments/"


def ease_B(reg):
    G = G0.copy(); G[np.diag_indices_from(G)] += reg
    P = la.inv(G, overwrite_a=True, check_finite=False)
    B = P / (np.diag(P))[None, :] * -1
    np.fill_diagonal(B, 0)
    return B


def report(method, params, S, note=""):
    top = topk(S, Xd)
    r = evaluate(top, dev, lab, n_lab)
    hits = np.mean([len(set(top[i]) & lab[u]) for i, u in enumerate(dev)]) / K
    log(method, params, r, hits, bins(top, dev, lab, rank), len(np.unique(top)), note)
    return r, top


best = (0, None, "")
# earlier runs (no band stats), recorded for completeness
log("Popularity", "top100", 0.1472, 0.0796, note="官方基线")
log("Item-KNN", "cosine, all neighbors, pow=1", 0.2566, 0.1272, note="首轮")

B = ease_B(300.0)
S0 = Xd @ B
r, top = report("EASE", "lam=300", S0, "基线,带分档统计")
best = (r, top, "EASE lam=300")

# (a) post-hoc popularity discount
for alpha in [0.1, 0.2, 0.3, 0.5]:
    # scores can be negative, so shift via sign-preserving division
    S = S0 / (pop[None, :] ** alpha)
    r, top = report("EASE + 热门折扣", f"lam=300, score/pop^{alpha}", S, "后处理")
    if r > best[0]: best = (r, top, f"EASE lam=300 alpha={alpha}")
del B, S0

# (b) popularity-weighted ridge
for lam, a in [(100, 0.05), (100, 0.2), (100, 0.5), (300, 0.2), (300, 1.0), (50, 1.0)]:
    B = ease_B(lam + a * pop)
    r, top = report("EASE 热门加权正则", f"reg_j=lam+a*pop_j, lam={lam}, a={a}", Xd @ B, f"{time.time()-t0:.0f}s")
    if r > best[0]: best = (r, top, f"lam={lam} a={a}")

write_preds(best[1], dev, iids, OUT + "pred_best_exp1.csv")
print("best", best[2], f"{best[0]:.4f}", "-> pred_best_exp1.csv")
