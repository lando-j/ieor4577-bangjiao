"""Exp 2: (1) release-year vs rating-date constraint, (4) per-user (history-length-dependent) popularity discount.
Base model: EASE lam=300. Per-bin alpha is chosen with 2-fold cross-fitting over dev users.
"""
import time, numpy as np, pandas as pd, scipy.linalg as la
from common import *
from log import log, bins

t0 = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows]
G = (X.T @ X).toarray().astype(np.float32)
pop = np.diag(G).copy()
rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False)
B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0)
S0 = Xd @ B
del G, P, B
print(f"EASE ready {time.time()-t0:.0f}s", flush=True)

# ---- dates / years ----
inter = pd.read_csv(D + "interactions.csv", usecols=["user_id", "date"])
inter = inter[inter.user_id.isin(dev)]
inter["y"] = pd.to_datetime(inter.date).dt.year
uy_s = inter.groupby("user_id").y.max()
uy = uy_s.reindex(dev).values.astype(float)
items = pd.read_csv(D + "items.csv").set_index("item_id")
iy = items.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float).reindex(iids).values
print(f"items with parsed year: {np.isfinite(iy).mean():.1%}; user years {np.nanmin(uy):.0f}-{np.nanmax(uy):.0f}")

# sanity: how often does the *given history* / labels violate "release year <= rating year"?
r, c = Xd.nonzero()
d_hist = iy[c] - uy[r]
print("history: 上映年份 - 评分年份  >0 的比例", f"{np.nanmean(d_hist > 0):.2%}", " >1:", f"{np.nanmean(d_hist > 1):.2%}", " >2:", f"{np.nanmean(d_hist > 2):.2%}")
lr = [(i, j) for i, u in enumerate(dev) for j in lab[u]]
li = np.array([a for a, _ in lr]); lj = np.array([b for _, b in lr])
d_lab = iy[lj] - uy[li]
print("labels:  上映年份 - 评分年份  >0 的比例", f"{np.nanmean(d_lab > 0):.2%}", " >1:", f"{np.nanmean(d_lab > 1):.2%}", " >2:", f"{np.nanmean(d_lab > 2):.2%}")

top0 = topk(S0, Xd)
for m in [0, 1, 2]:
    viol = (iy[top0] > uy[:, None] + m)
    print(f"基线 top-20 中违反约束(margin={m})的推荐比例: {np.nanmean(viol):.2%}; 至少有 1 部违反的用户占 {viol.any(1).mean():.1%}")

rec_of = lambda top: np.array([len(set(top[i]) & lab[u]) / n_lab[u] for i, u in enumerate(dev)])
prec_of = lambda top: np.mean([len(set(top[i]) & lab[u]) for i, u in enumerate(dev)]) / K
def rep(method, params, top, note=""):
    r = rec_of(top).mean()
    log(method, params, r, prec_of(top), bins(top, dev, lab, rank), len(np.unique(top)), note)
    return r

rep("EASE", "lam=300 (重算)", top0, "实验2基线")

# ---- (1) time constraint ----
best_m, best_r, best_mask = None, rec_of(top0).mean(), None
for m in [0, 1, 2]:
    mask = iy[None, :] > (uy[:, None] + m)
    S = S0.copy(); S[mask] = -np.inf
    top = topk(S, Xd)
    r = rep("EASE + 时间约束", f"lam=300, 过滤 上映年>评分年+{m}", top, f"过滤了 {np.nanmean(mask):.2%} 的(用户,电影)对")
    if r > best_r: best_m, best_r, best_mask = m, r, mask
print("best margin:", best_m)
mask = best_mask  # None if no help

# ---- (4) per-bin alpha with cross-fitting ----
alphas = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]
hl = np.asarray(Xd.sum(1)).ravel()
edges = np.quantile(hl, np.linspace(0, 1, 7)[1:-1])   # 6 bins
ub = np.searchsorted(edges, hl, side="right")
tops, recs = {}, {}
for a in alphas:
    S = S0 / (pop[None, :] ** a)
    if mask is not None: S[mask] = -np.inf
    tops[a] = topk(S, Xd); recs[a] = rec_of(tops[a])
R = np.stack([recs[a] for a in alphas])      # (alphas, users)
print("\n全局 α 扫描 (用户平均 Recall):", {a: round(recs[a].mean(), 4) for a in alphas})
print("按历史长度分档的最优 α(整个 dev,仅供观察):")
for b in range(6):
    sel = ub == b
    print(f"  bin{b} 历史 {int(hl[sel].min())}-{int(hl[sel].max())} ({sel.sum()}人): " + "  ".join(f"{a:+.1f}:{recs[a][sel].mean():.4f}" for a in alphas))

rng = np.random.default_rng(0)
fold = rng.integers(0, 2, len(dev))
chosen = np.zeros(len(dev), int)
for f in [0, 1]:
    tune, ev = fold != f, fold == f
    for b in range(6):
        s = tune & (ub == b)
        chosen[ev & (ub == b)] = int(np.argmax(R[:, s].mean(1)))
cf = R[chosen, np.arange(len(dev))]
base = recs[0.0]
g_best = max(alphas, key=lambda a: recs[a].mean())
print(f"\n交叉拟合后的每档 α: Recall = {cf.mean():.4f}  | 全局 α=0: {base.mean():.4f} | 最佳全局 α={g_best}: {recs[g_best].mean():.4f}")
diff = cf - base
print(f"相对 α=0 的提升 {diff.mean():+.4f} ± {diff.std()/np.sqrt(len(diff)):.4f} (标准误)")
comp = np.stack([tops[alphas[chosen[i]]][i] for i in range(len(dev))])
rep("EASE + 分档热门折扣", f"lam=300, 6 档 α∈{alphas}, 2 折交叉拟合" + ("" if mask is None else f", 含时间约束 m={best_m}"), comp, "每档 α 由另一半用户选出")
np.save("/Users/landojiao/Downloads/ieor4577-student-dataset/experiments/exp2_chosen_alpha_idx.npy", chosen)
