"""Task 0 follow-up: calibrated re-ranking (Steck 2018) on popularity bands.
The held-out 20% is a random split, so each user's held-out band mix ~= their history band mix (checked in exp_popularity.py).
Greedy re-rank of each user's top-100 candidates: maximize (1-lam) * sum z-score - lam * KL(p_user || q_rec),
p_user = band mix of history (or, as an oracle upper bound, of the held-out set).
lam is chosen with 2-fold cross-fitting on dev users; paired SE reported.
Usage: python exp_calib.py [ease|blend]
"""
import sys, time, numpy as np, scipy.linalg as la
from common import *

which = sys.argv[1] if len(sys.argv) > 1 else "blend"
t0 = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows].tocsr()
pop = np.asarray(X.sum(0)).ravel()
rank = np.empty(len(pop), int); rank[np.argsort(-pop, kind="stable")] = np.arange(len(pop))
edges = [100, 500, 2000]; band_of = np.searchsorted(edges, rank, side="right"); nb = 4

G = (X.T @ X).toarray().astype(np.float32); G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False)
B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0); del G, P
Se = np.asarray(Xd @ B, dtype=np.float32); del B
hr, hc = Xd.nonzero()
def znorm(S):
    S = S.copy(); S[hr, hc] = np.nan
    S = (S - np.nanmean(S, 1, keepdims=True)) / np.nanstd(S, 1, keepdims=True)
    S[hr, hc] = -np.inf; return S
Z = znorm(Se)
if which == "blend":
    Z = 0.2 * Z + 0.8 * znorm(np.load("runs/devscores_vae_base.npy").astype(np.float32))
C = topk(Z, Xd, 100)                                   # candidates, sorted
Cs = np.take_along_axis(Z, C, 1); del Z, Se
Cb = band_of[C]
print(f"candidates ready {time.time()-t0:.0f}s", flush=True)

nU = len(dev)
def mix(item_lists):
    M = np.full((nU, nb), 1e-3)
    for i, it in enumerate(item_lists): M[i] += np.bincount(band_of[it], minlength=nb)
    return M / M.sum(1, keepdims=True)
p_hist = mix([Xd.indices[Xd.indptr[i]:Xd.indptr[i+1]] for i in range(nU)])
p_true = mix([np.fromiter(lab[u], int) for u in dev])
labsets = [lab[u] for u in dev]; nl = np.array([n_lab[u] for u in dev])
onehot = np.eye(nb)[Cb]                                 # (nU, 100, nb)

def rerank(p, lam, k=K, a=0.01):
    chosen = np.zeros(C.shape, bool); cnt = np.zeros((nU, nb)); rel = np.zeros(nU); out = np.zeros((nU, k), int)
    for step in range(k):
        q = (cnt[:, None, :] + onehot) / (step + 1)                         # mix if candidate added
        q = (1 - a) * q + a * p[:, None, :]
        kl = (p[:, None, :] * np.log(p[:, None, :] / q)).sum(2)
        obj = (1 - lam) * (rel[:, None] + Cs) / (step + 1) - lam * kl
        obj[chosen] = -np.inf
        j = obj.argmax(1); r = np.arange(nU)
        chosen[r, j] = True; cnt += onehot[r, j]; rel += Cs[r, j]; out[:, step] = C[r, j]
    return out

def recall(top): return np.array([len(set(top[i]) & labsets[i]) for i in range(nU)]) / nl

base = recall(C[:, :K])
print(f"{which} no re-rank: {base.mean():.4f}")
lams = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
for name, p in [("history mix (usable)", p_hist), ("held-out mix (ORACLE, not usable)", p_true)]:
    R = np.stack([recall(rerank(p, l)) for l in lams])
    print(f"\n{name}:")
    for l, r in zip(lams, R): print(f"  lam={l:<5} recall {r.mean():.4f}  diff vs none {(r-base).mean():+.4f} ± {(r-base).std()/np.sqrt(nU):.4f}")
    fold = np.random.default_rng(0).integers(0, 2, nU); ch = np.zeros(nU, int)
    for f in [0, 1]: ch[fold == f] = int(np.argmax(R[:, fold != f].mean(1)))
    cf = R[ch, np.arange(nU)]; d = cf - base
    print(f"  cross-fitted lam per fold {[lams[ch[fold==f][0]] for f in [0,1]]}: {cf.mean():.4f}, {d.mean():+.4f} ± {d.std()/np.sqrt(nU):.4f}")
print(f"done {time.time()-t0:.0f}s")
