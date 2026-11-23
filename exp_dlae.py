"""Closed-form linear autoencoders, all hyperparameters chosen on the 5,000 validation users (common.val_split);
dev is touched once per family at the end.

  EASE  (Steck 2019):  P = (G + lam*I)^-1,             B = I - P / diag(P)   (zero diagonal)
  EDLAE (Steck 2020):  P = (G + Lam)^-1,               B = I - P / diag(P)   (zero diagonal, "emphasized" denoising)
  DLAE  (Steck 2020):  P = (G + Lam)^-1,               B = I - P * Lam       (full rank, dropout-denoising)
  with Lam = lam*I + diag(p/(1-p) * diag(G)),  G = X^T X,  p = dropout probability.

Models for validation are fit on Xfit (val users' 20% targets removed); the chosen config is refit on all of X for dev.
Outputs: rows in RESULTS.md, runs/dlae_grid.json, runs/{val,dev}scores_<name>.npy (float16) for later blending.
Usage: python exp_dlae.py
"""
import json, time, numpy as np, scipy.linalg as la
from common import *
from log import log, bins

t0 = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows].tocsr()
val_rows, Xv_in, val_lab, Xfit = val_split(X, dev_rows)
val_n = np.array([max(len(s), 1) for s in val_lab])
dev_lab = [lab[u] for u in dev]; dev_n = np.array([n_lab[u] for u in dev])
pop = np.asarray(X.sum(0)).ravel(); rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
print(f"data + val split {time.time()-t0:.0f}s", flush=True)

def fit(Xtr, kind, lam, p=0.0):
    G = (Xtr.T @ Xtr).toarray().astype(np.float32)
    Lam = lam + (p / (1 - p)) * np.diag(G) if p > 0 else np.full(G.shape[0], lam, np.float32)
    G[np.diag_indices_from(G)] += Lam
    P = la.inv(G, overwrite_a=True, check_finite=False); del G
    if kind == "DLAE":
        B = -P * Lam[None, :].astype(np.float32); B[np.diag_indices_from(B)] += 1
    else:
        B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0)
    return B

def run(Xin, B, labs, n):
    S = np.asarray(Xin @ B, dtype=np.float32)
    top = topk(S, Xin)
    return S, top, per_user_recall(top, labs, n)

grid = [("EASE", l, 0.0) for l in [100, 200, 300, 500, 1000, 2000]] + \
       [("EDLAE", l, p) for p in [0.1, 0.2, 0.33, 0.5] for l in [10, 100, 300]] + \
       [("DLAE", l, p) for p in [0.2, 0.33, 0.5, 0.67] for l in [10, 100]]
res = []
for kind, lam, p in grid:
    t = time.time()
    _, _, r = run(Xv_in, fit(Xfit, kind, lam, p), val_lab, val_n)
    res.append(dict(kind=kind, lam=lam, p=p, val=float(r.mean())))
    print(f"{kind:5s} lam={lam:<5} p={p:<4}  val Recall@20 {r.mean():.4f}  ({time.time()-t:.0f}s)", flush=True)
json.dump(res, open("runs/dlae_grid.json", "w"), indent=1)

# baseline for every comparison: the proposal's EASE (lam=300), refit on all of X, on dev
B0 = fit(X, "EASE", 300)
_, top0, r0 = run(Xd, B0, dev_lab, dev_n); del B0
print(f"\nbaseline EASE lam=300 on dev: {r0.mean():.4f}")
for kind in ["EASE", "EDLAE", "DLAE"]:
    best = max((x for x in res if x["kind"] == kind), key=lambda x: x["val"])
    name = f"{kind.lower()}_l{best['lam']}_p{best['p']}"
    # val scores of the val-fit model (for choosing blend weights later), then refit on all X for dev
    Sv, _, _ = run(Xv_in, fit(Xfit, kind, best["lam"], best["p"]), val_lab, val_n)
    np.save(f"runs/valscores_{name}.npy", Sv.astype(np.float16)); del Sv
    S, top, r = run(Xd, fit(X, kind, best["lam"], best["p"]), dev_lab, dev_n)
    np.save(f"runs/devscores_{name}.npy", S.astype(np.float16)); del S
    m, lo, hi = paired_bootstrap(r, r0)
    hits = np.array([len(set(top[i]) & dev_lab[i]) for i in range(len(dev))])
    paper = (hits / np.minimum(20, dev_n)).mean()
    print(f"{kind:5s} best on val: lam={best['lam']} p={best['p']} val {best['val']:.4f} | dev {r.mean():.4f} "
          f"(paper-style {paper:.4f}) | vs EASE-300 {m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]", flush=True)
    log(kind, f"lam={best['lam']} p={best['p']} (chosen on 5k val users, val={best['val']:.4f})", r.mean(), hits.mean() / K,
        bins(top, dev, lab, rank), len(np.unique(top)), f"论文口径 {paper:.4f}; vs EASE-300 {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
print(f"done {time.time()-t0:.0f}s")
