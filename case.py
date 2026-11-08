"""Case study: show history / top-20 / held-out labels for specific dev users (EASE lam=300)."""
import numpy as np, pandas as pd, scipy.linalg as la
from common import *
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows]
G = (X.T @ X).toarray().astype(np.float32)
pop = np.diag(G).copy(); rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False)
B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0)
S = Xd @ B
top = topk(S, Xd)
items = pd.read_csv(D + "items.csv").set_index("item_id")
title = lambda j: f"{items.title.get(iids[j], '?')}  [{items.genres.get(iids[j], '')}]"
hits = np.array([len(set(top[i]) & lab[u]) for i, u in enumerate(dev)])
hl = np.asarray(Xd.sum(1)).ravel(); nl = np.array([n_lab[u] for u in dev])
rng = np.random.default_rng(0)
good = np.where((hl >= 20) & (hl <= 40) & (nl >= 6) & (nl <= 10) & (hits >= 3))[0]
bad = np.where((hl >= 20) & (hl <= 40) & (nl >= 6) & (nl <= 10) & (hits == 0))[0]
out = []
for tag, pool in [("有命中的用户", good), ("完全没命中的用户", bad)]:
    i = rng.choice(pool); u = dev[i]
    hist = Xd[i].indices; labset = lab[u]
    out.append(f"\n{'='*20} {tag}: user_id={u} {'='*20}")
    out.append(f"历史 {len(hist)} 部,被隐藏标签 {n_lab[u]} 部,top-20 命中 {hits[i]} 部 -> Recall@20 = {hits[i]/n_lab[u]:.2f}")
    out.append("\n[已知历史(模型的输入)]")
    for j in hist: out.append(f"  - {title(j)}   (热门排名 {rank[j]+1})")
    out.append("\n[模型推荐的前 20 部]   ✓ = 命中被隐藏的标签")
    for k, j in enumerate(top[i], 1):
        out.append(f"  {k:2d}. {'✓' if j in labset else ' '} {title(j)}   (热门排名 {rank[j]+1}, 得分 {S[i, j]:.2f})")
    out.append("\n[被隐藏的真实标签]   模型把它排在第几名(全目录 20205 部,已排除历史)")
    sc = S[i].copy(); sc[hist] = -np.inf
    order = np.argsort(-sc); pos = np.empty(len(sc), int); pos[order] = np.arange(len(sc))
    for j in sorted(labset, key=lambda j: pos[j]):
        out.append(f"  - 排第 {pos[j]+1:5d} 名  {title(j)}   (热门排名 {rank[j]+1})")
txt = "\n".join(out); print(txt)
open("CASE.md", "w").write("# 案例:EASE(λ=300)对两个 dev 用户的推荐\n\n```" + txt + "\n```\n")
