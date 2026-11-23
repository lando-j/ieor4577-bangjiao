"""Exp 3: blend EASE (lam=300) and Mult-VAE (runs/devscores_vae_base.npy) scores.
Per-user normalization of each model's scores over non-history items, then  w*z_vae + (1-w)*z_ease.
The weight w is a single parameter; it is picked with 2-fold cross-fitting over dev users (and the full curve is printed).
Reports both the course Recall@20 (/ n) and the paper-style one (/ min(20, n)).
Usage: python exp_fusion.py [vae_tag]
"""
import sys, time, numpy as np, scipy.linalg as la
from common import *
from log import log, bins

tag = sys.argv[1] if len(sys.argv) > 1 else "vae_base"
t0 = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows]
G = (X.T @ X).toarray().astype(np.float32)
pop = np.diag(G).copy(); rank_pop = np.empty(len(pop), int); rank_pop[np.argsort(-pop)] = np.arange(len(pop))
G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False)
B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0)
Se = Xd @ B; del G, P, B
Sv = np.load(f"runs/devscores_{tag}.npy").astype(np.float32)
print(f"loaded EASE + {tag} scores {time.time()-t0:.0f}s", flush=True)

hr, hc = Xd.nonzero()
def znorm(S):
    S = S.copy(); S[hr, hc] = np.nan
    m = np.nanmean(S, 1, keepdims=True); s = np.nanstd(S, 1, keepdims=True)
    S = (S - m) / s
    S[hr, hc] = -np.inf
    return S
Ze, Zv = znorm(Se), znorm(Sv); del Se, Sv

nl = np.array([n_lab[u] for u in dev])
def per_user(top):
    h = np.array([len(set(top[i]) & lab[u]) for i, u in enumerate(dev)])
    return h / nl, h / np.minimum(20, nl), h.mean() / K

ws = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
res = {}
for w in ws:
    S = np.where(np.isinf(Ze), -np.inf, w * Zv + (1 - w) * Ze)
    top = topk(S, Xd)
    res[w] = (top,) + per_user(top)
    print(f"w_vae={w:.1f}: course {res[w][1].mean():.4f} | paper-style {res[w][2].mean():.4f}", flush=True)

R = np.stack([res[w][1] for w in ws])
rng = np.random.default_rng(0); fold = rng.integers(0, 2, len(dev)); chosen = np.zeros(len(dev), int)
for f in [0, 1]:
    chosen[fold == f] = int(np.argmax(R[:, fold != f].mean(1)))
print("cross-fitted w choices per fold:", [ws[int(np.argmax(R[:, fold != f].mean(1)))] for f in [0, 1]])
cf = R[chosen, np.arange(len(dev))]
d = cf - R[ws.index(0.0)]
print(f"cross-fitted blend {cf.mean():.4f} vs EASE alone {R[ws.index(0.0)].mean():.4f}: {d.mean():+.4f} ± {d.std()/np.sqrt(len(d)):.4f}")
d2 = R[ws.index(0.5)] - R[ws.index(0.0)]; print(f"w=0.5 vs EASE: {d2.mean():+.4f} ± {d2.std()/np.sqrt(len(d2)):.4f}")
d3 = R[ws.index(0.5)] - R[ws.index(1.0)]; print(f"w=0.5 vs VAE:  {d3.mean():+.4f} ± {d3.std()/np.sqrt(len(d3)):.4f}")
for name, a_ in [("VAE alone", R[ws.index(1.0)]), ("cross-fitted blend", cf)]:
    m, lo, hi = paired_bootstrap(a_, R[ws.index(0.0)]); print(f"bootstrap {name} vs EASE: {m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")
m, lo, hi = paired_bootstrap(cf, R[ws.index(1.0)]); print(f"bootstrap blend vs VAE alone: {m:+.4f} [95% CI {lo:+.4f}, {hi:+.4f}]")

wb = ws[int(np.argmax(R.mean(1)))]
for w in sorted({0.0, 1.0, 0.5, wb}):
    top, rc, rp, pr = res[w]
    log("EASE+VAE 融合" if 0 < w < 1 else ("EASE(融合表基准)" if w == 0 else "VAE(融合表基准)"),
        f"z-score blend, w_vae={w}, vae={tag}", rc.mean(), pr, bins(top, dev, lab, rank_pop), len(np.unique(top)), f"论文口径 {rp.mean():.4f}")
top = res[wb][0]
write_preds(top, dev, iids, f"runs/pred_fusion_{tag}.csv")
print("best w", wb, "-> runs/pred_fusion_%s.csv" % tag)
