"""Task 0: popularity diagnosis. How popular are the movies each model puts in its top-20,
compared with what users actually held out (dev labels) and with their own histories?

Models: EASE (lam=300), Mult-VAE (runs/devscores_vae_base.npy), z-score blend w_vae=0.8.
Popularity = number of users in interactions.csv who have the movie; bands by global rank.
Output: printed tables + runs/popularity_report.md (incl. hand-inspection of sample users).
Usage: python exp_popularity.py
"""
import time, numpy as np, pandas as pd, scipy.linalg as la
from common import *

t0 = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows].tocsr()
pop = np.asarray(X.sum(0)).ravel()
rank = np.empty(len(pop), int); rank[np.argsort(-pop, kind="stable")] = np.arange(len(pop))
BANDS = [("top 1-100", 0, 100), ("101-500", 100, 500), ("501-2000", 500, 2000), ("2001+", 2000, 10**9)]
def band(r): return next(i for i, (_, lo, hi) in enumerate(BANDS) if lo <= r < hi)
band_of = np.array([band(r) for r in rank])
nb = len(BANDS)

# ---- scores
G = (X.T @ X).toarray().astype(np.float32); G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False)
B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0); del G, P
Se = np.asarray(Xd @ B, dtype=np.float32); del B
Sv = np.load("runs/devscores_vae_base.npy").astype(np.float32)
hr, hc = Xd.nonzero()
def znorm(S):
    S = S.copy(); S[hr, hc] = np.nan
    S = (S - np.nanmean(S, 1, keepdims=True)) / np.nanstd(S, 1, keepdims=True)
    S[hr, hc] = -np.inf; return S
Zf = 0.8 * znorm(Sv) + 0.2 * znorm(Se)
tops = {"EASE": topk(Se, Xd), "Mult-VAE": topk(Sv, Xd), "Blend 0.8": topk(Zf, Xd)}
del Se, Sv, Zf
print(f"scores ready {time.time()-t0:.0f}s", flush=True)

# ---- per-user band shares
hlen = np.diff(Xd.indptr)
def share(rows_items):  # list of item arrays -> (n_users, nb) share matrix
    M = np.zeros((len(rows_items), nb))
    for i, it in enumerate(rows_items):
        if len(it): M[i] = np.bincount(band_of[it], minlength=nb) / len(it)
    return M
H = share([Xd.indices[Xd.indptr[i]:Xd.indptr[i+1]] for i in range(len(dev))])
L = share([np.fromiter(lab[u], int) for u in dev])
R = {m: share(list(t)) for m, t in tops.items()}
labsets = [lab[u] for u in dev]
hits = {m: np.array([len(set(t[i]) & labsets[i]) for i in range(len(dev))]) for m, t in tops.items()}
nl = np.array([n_lab[u] for u in dev])

out = []
def p(s=""): print(s, flush=True); out.append(s)
p("# Popularity diagnosis (dev, 10,000 users)\n")
p(f"Bands by global popularity rank. Item counts per band: " + ", ".join(f"{n} = {(band_of==i).sum()}" for i, (n, _, _) in enumerate(BANDS)))
p("\n## 1. Where do top-20 slots go vs where held-out movies are (mean per-user share)\n")
p("| | " + " | ".join(b[0] for b in BANDS) + " | avg # of top-100 movies in top-20 |")
p("|---" * (nb + 2) + "|")
p("| user history | " + " | ".join(f"{x:.1%}" for x in H.mean(0)) + " | — |")
p("| **held-out (truth)** | " + " | ".join(f"**{x:.1%}**" for x in L.mean(0)) + f" | {20*L[:,0].mean():.1f} (if top-20 matched truth mix) |")
for m in tops: p(f"| {m} top-20 | " + " | ".join(f"{x:.1%}" for x in R[m].mean(0)) + f" | {20*R[m][:,0].mean():.1f} |")

p("\n## 2. Precision of each band: of the slots a model spends in a band, how many hit?\n")
p("| model | " + " | ".join(b[0] for b in BANDS) + " | Recall@20 |"); p("|---" * (nb + 2) + "|")
for m, t in tops.items():
    slot = np.zeros(nb); hit = np.zeros(nb)
    for i in range(len(dev)):
        for j in t[i]:
            slot[band_of[j]] += 1; hit[band_of[j]] += j in labsets[i]
    p(f"| {m} | " + " | ".join(f"{h/s:.1%} ({int(s/len(dev)*10)/10} slots)" for h, s in zip(hit, slot)) + f" | {(hits[m]/nl).mean():.4f} |")
p("\n## 3. Recall of each band: of held-out movies in a band, how many are found?\n")
p("| model | " + " | ".join(b[0] for b in BANDS) + " |"); p("|---" * (nb + 1) + "|")
for m, t in tops.items():
    found = np.zeros(nb); tot = np.zeros(nb)
    for i in range(len(dev)):
        ts = set(t[i])
        for j in labsets[i]: tot[band_of[j]] += 1; found[band_of[j]] += j in ts
    p(f"| {m} | " + " | ".join(f"{f/n:.1%}" for f, n in zip(found, tot)) + " |")
    if m == "EASE": p("| (share of all held-out movies) | " + " | ".join(f"{n/tot.sum():.1%}" for n in tot) + " |")

p("\n## 4. By user history length: share of top-100 movies (band 1)\n")
qs = [0, 20, 50, 100, 200, 10**9]
p("| history length | users | history | truth | EASE | VAE | Blend | EASE recall | Blend recall |"); p("|---" * 9 + "|")
for lo, hi in zip(qs[:-1], qs[1:]):
    s = (hlen >= lo) & (hlen < hi)
    p(f"| {lo}-{hi if hi<10**9 else '∞'} | {s.sum()} | {H[s,0].mean():.1%} | {L[s,0].mean():.1%} | {R['EASE'][s,0].mean():.1%} | "
      f"{R['Mult-VAE'][s,0].mean():.1%} | {R['Blend 0.8'][s,0].mean():.1%} | {(hits['EASE'][s]/nl[s]).mean():.4f} | {(hits['Blend 0.8'][s]/nl[s]).mean():.4f} |")

p("\n## 5. Do models follow each user's own taste for popular vs niche?\n")
hp = H[:, 0] + H[:, 1]; lp = L[:, 0] + L[:, 1]
for m in tops:
    rp = R[m][:, 0] + R[m][:, 1]
    p(f"- {m}: corr(user's history share of top-500, model's top-20 share of top-500) = {np.corrcoef(hp, rp)[0,1]:.3f}; "
      f"corr with held-out share = {np.corrcoef(lp, rp)[0,1]:.3f}")
p(f"- reference: corr(history share, held-out share) = {np.corrcoef(hp, lp)[0,1]:.3f}")
q = np.quantile(hp, [0.2, 0.8])
for name, s in [("niche users (bottom 20% history share of top-500)", hp <= q[0]), ("mainstream users (top 20%)", hp >= q[1])]:
    p(f"- {name}: history {hp[s].mean():.1%}, truth {lp[s].mean():.1%}, " +
      ", ".join(f"{m} {(R[m][s,0]+R[m][s,1]).mean():.1%}" for m in tops) +
      f"; EASE recall {(hits['EASE'][s]/nl[s]).mean():.4f}, Blend recall {(hits['Blend 0.8'][s]/nl[s]).mean():.4f}")

# ---- hand inspection
items = pd.read_csv(D + "items.csv").set_index("item_id").title
T = lambda j: f"{items.get(iids[j], iids[j])} [#{rank[j]+1}]"
rng = np.random.default_rng(1)
p("\n## 6. Hand inspection (blend top-20; ✓ = held-out hit; [#k] = global popularity rank)\n")
picks = []
for lo, hi in [(0, 20), (20, 50), (50, 200), (200, 10**9)]:
    c = np.where((hlen >= lo) & (hlen < hi))[0]; picks += list(rng.choice(c, 2, replace=False))
c = np.where(hp <= q[0])[0]; picks += list(rng.choice(c, 2, replace=False))
for i in picks:
    h = Xd.indices[Xd.indptr[i]:Xd.indptr[i+1]]
    hs = h[np.argsort(rank[h])]
    p(f"### user {dev[i]} — history {len(h)}, held-out {nl[i]}, blend hits {hits['Blend 0.8'][i]}, EASE hits {hits['EASE'][i]}")
    p(f"- history share top-100 {H[i,0]:.0%}, top-500 {hp[i]:.0%}; held-out top-500 {lp[i]:.0%}")
    p("- history (sample of up to 12, most popular → least): " + "; ".join(T(j) for j in (hs if len(hs) <= 12 else hs[np.linspace(0, len(hs)-1, 12).astype(int)])))
    p("- blend top-20: " + "; ".join(("✓ " if j in labsets[i] else "") + T(j) for j in tops["Blend 0.8"][i]))
    miss = sorted(labsets[i] - set(tops["Blend 0.8"][i]), key=lambda j: rank[j])
    p(f"- missed held-out ({len(miss)}): " + "; ".join(T(j) for j in miss[:15]) + (" …" if len(miss) > 15 else ""))
    p()
open("runs/popularity_report.md", "w").write("\n".join(out) + "\n")
print(f"done {time.time()-t0:.0f}s -> runs/popularity_report.md")
