"""Where do the gains come from? Per-user Recall@20 on dev, split by user history length and by the popularity
band of held-out movies, for EASE-300, Mult-VAE (paper schedule), RecVAE, and the RecVAE+EDLAE blend.
Paired bootstrap CIs vs EASE within each history bucket. Output: printed + runs/breakdown_report.md
Usage: python exp_breakdown.py
"""
import numpy as np, pandas as pd, scipy.linalg as la
from common import *

X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows].tocsr(); hlen = np.diff(Xd.indptr)
dev_lab = [lab[u] for u in dev]; dev_n = np.array([n_lab[u] for u in dev])
pop = np.asarray(X.sum(0)).ravel(); rank = np.empty(len(pop), int); rank[np.argsort(-pop, kind="stable")] = np.arange(len(pop))
band_of = np.searchsorted([100, 500, 2000], rank, side="right"); BANDS = ["top 1-100", "101-500", "501-2000", "2001+"]
imap = pd.Series(np.arange(len(iids)), index=iids)
def read(path):
    p = pd.read_csv(path).groupby("user_id").item_id.apply(list)
    return np.array([imap[p[u]].values for u in dev])
G = (X.T @ X).toarray().astype(np.float32); G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False); B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0); del G, P
tops = {"EASE": topk(np.asarray(Xd @ B, dtype=np.float32), Xd), "Mult-VAE": read("runs/pred_paper.csv"),
        "RecVAE": read("runs/pred_recvae.csv"), "RecVAE+EDLAE": read("runs/pred_blend_recvae_edlae_l300_p0.2.csv")}
del B
R = {m: per_user_recall(t, dev_lab, dev_n) for m, t in tops.items()}

out = []
def p(s=""): print(s, flush=True); out.append(s)
p("# Where the gains come from (dev, 10,000 users, course Recall@20)\n")
p("## 1. By user history length\n")
p("| history | users | EASE | Mult-VAE | RecVAE | RecVAE+EDLAE | RecVAE − EASE [95% CI] | blend − EASE [95% CI] |")
p("|---|---|---|---|---|---|---|---|")
for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10**9)]:
    s = (hlen >= lo) & (hlen < hi)
    d1 = paired_bootstrap(R["RecVAE"][s], R["EASE"][s]); d2 = paired_bootstrap(R["RecVAE+EDLAE"][s], R["EASE"][s])
    p(f"| {lo}-{hi if hi < 10**9 else '∞'} | {s.sum()} | " + " | ".join(f"{R[m][s].mean():.4f}" for m in tops) +
      f" | {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] | {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}] |")
d1 = paired_bootstrap(R["RecVAE"], R["EASE"]); d2 = paired_bootstrap(R["RecVAE+EDLAE"], R["EASE"])
p(f"| all | {len(dev)} | " + " | ".join(f"{R[m].mean():.4f}" for m in tops) +
  f" | {d1[0]:+.4f} [{d1[1]:+.4f}, {d1[2]:+.4f}] | {d2[0]:+.4f} [{d2[1]:+.4f}, {d2[2]:+.4f}] |")
# share of the total gain by bucket
gain = (R["RecVAE"] - R["EASE"]); p("\nShare of RecVAE's total gain over EASE contributed by each bucket (gain summed over users):")
for lo, hi in [(0, 20), (20, 50), (50, 100), (100, 200), (200, 10**9)]:
    s = (hlen >= lo) & (hlen < hi); p(f"- {lo}-{hi if hi < 10**9 else '∞'}: {gain[s].sum() / gain.sum():.0%} of the gain, {s.mean():.0%} of users")

p("\n## 2. By popularity band of the held-out movie: % of held-out movies found in the top-20\n")
p("| model | " + " | ".join(BANDS) + " |"); p("|---" * 5 + "|")
tot = np.zeros(4)
for i in range(len(dev)):
    for j in dev_lab[i]: tot[band_of[j]] += 1
for m, t in tops.items():
    f = np.zeros(4)
    for i in range(len(dev)):
        ts = set(t[i])
        for j in dev_lab[i]: f[band_of[j]] += j in ts
    p(f"| {m} | " + " | ".join(f"{x/n:.1%}" for x, n in zip(f, tot)) + " |")
p("| (share of held-out movies) | " + " | ".join(f"{n/tot.sum():.1%}" for n in tot) + " |")

p("\n## 3. Popularity of what each model recommends: share of top-20 slots per band\n")
p("| model | " + " | ".join(BANDS) + " | distinct movies recommended |"); p("|---" * 6 + "|")
for m, t in tops.items():
    s = np.bincount(band_of[t.ravel()], minlength=4) / t.size
    p(f"| {m} | " + " | ".join(f"{x:.1%}" for x in s) + f" | {len(np.unique(t))} |")
open("runs/breakdown_report.md", "w").write("\n".join(out) + "\n")
