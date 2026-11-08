"""Mult-VAE (Liang et al. 2018) on the course data.

Model selection uses 5,000 held-out *training* users (80% fold-in / 20% target, random split);
dev is evaluated once at the end with the best-validation epoch.
Training set = remaining training users + the given 80% histories of dev users (no dev labels).

Usage: python vae.py --tag base [--epochs 100 --hidden 600 --latent 200 --dropout 0.5 --beta 0.2 --lr 1e-3 --wd 0]
"""
import argparse, json, time, copy, numpy as np, scipy.sparse as sp, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from log import log, bins

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="base")
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--hidden", type=int, default=600)
ap.add_argument("--latent", type=int, default=200)
ap.add_argument("--dropout", type=float, default=0.5)
ap.add_argument("--beta", type=float, default=0.2)
ap.add_argument("--anneal_epochs", type=int, default=40)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=0.0)
ap.add_argument("--batch", type=int, default=500)
ap.add_argument("--dae", action="store_true", help="no KL / no sampling (Mult-DAE)")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--save_scores", action="store_true", help="save dev score matrix (float16) for ensembling")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev_ = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
OUT = "/Users/landojiao/Downloads/ieor4577-student-dataset/experiments/runs/"
import os; os.makedirs(OUT, exist_ok=True)

X, uids, iids, dev, dev_rows, lab, n_lab = load()
n_users, n_items = X.shape
is_dev = np.zeros(n_users, bool); is_dev[dev_rows] = True
train_pool = np.where(~is_dev)[0]
val_rows = rng.choice(train_pool, 5000, replace=False)
is_val = np.zeros(n_users, bool); is_val[val_rows] = True
train_rows = np.where(~is_val)[0]                      # includes dev users' given histories
print(f"train users {len(train_rows)}  val users {len(val_rows)}  dev users {len(dev_rows)}  items {n_items}", flush=True)

# validation split: random 80/20 of each val user's items
Xv = X[val_rows].tocsr(); r, c = Xv.nonzero()
hide = rng.random(len(r)) < 0.2
Xv_in = sp.csr_matrix((np.ones((~hide).sum(), np.float32), (r[~hide], c[~hide])), shape=Xv.shape)
val_lab = [set() for _ in range(len(val_rows))]
for i, j in zip(r[hide], c[hide]): val_lab[i].add(j)
val_n = np.array([max(len(s), 1) for s in val_lab])
Xtr = X[train_rows].tocsr(); Xdev = X[dev_rows].tocsr()

class VAE(nn.Module):
    def __init__(s):
        super().__init__()
        s.enc = nn.Linear(n_items, a.hidden); s.mu = nn.Linear(a.hidden, a.latent); s.lv = nn.Linear(a.hidden, a.latent)
        s.dec1 = nn.Linear(a.latent, a.hidden); s.dec2 = nn.Linear(a.hidden, n_items)
        for m in [s.enc, s.mu, s.lv, s.dec1, s.dec2]: nn.init.xavier_normal_(m.weight); nn.init.normal_(m.bias, std=0.001)
    def forward(s, x):
        h = F.normalize(x, dim=1)
        h = F.dropout(h, a.dropout, s.training)
        h = torch.tanh(s.enc(h)); mu, lv = s.mu(h), s.lv(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if (s.training and not a.dae) else mu
        return s.dec2(torch.tanh(s.dec1(z))), mu, lv

def dense(m): return torch.from_numpy(m.toarray()).to(dev_)

@torch.no_grad()
def score(model, Xin, bs=1000):
    model.eval(); out = []
    for i in range(0, Xin.shape[0], bs):
        out.append(model(dense(Xin[i:i+bs]))[0].float().cpu().numpy())
    return np.concatenate(out)

model = VAE().to(dev_)
opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
steps_per_epoch = int(np.ceil(len(train_rows) / a.batch)); anneal_steps = a.anneal_epochs * steps_per_epoch
step = 0; best = (-1, None, 0); hist = []; t0 = time.time()
for ep in range(1, a.epochs + 1):
    model.train(); perm = rng.permutation(len(train_rows)); tot = 0
    for i in range(0, len(perm), a.batch):
        x = dense(Xtr[perm[i:i+a.batch]])
        beta = a.beta * min(1.0, step / anneal_steps); step += 1
        logits, mu, lv = model(x)
        nll = -(F.log_softmax(logits, 1) * x).sum(1).mean()
        kl = (-0.5 * (1 + lv - mu**2 - lv.exp()).sum(1)).mean()
        loss = nll if a.dae else nll + beta * kl
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    if ep % 5 == 0 or ep == a.epochs:
        top = topk(score(model, Xv_in), Xv_in)
        vr = float(np.mean([len(set(top[i]) & val_lab[i]) / val_n[i] for i in range(len(val_rows))]))
        hist.append((ep, tot / steps_per_epoch, vr))
        print(f"ep {ep:3d} loss {tot/steps_per_epoch:8.2f} val Recall@20 {vr:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        if vr > best[0]: best = (vr, copy.deepcopy(model.state_dict()), ep)

model.load_state_dict(best[1]); print(f"best val {best[0]:.4f} at epoch {best[2]}")
Sd = score(model, Xdev); top = topk(Sd, Xdev)
r_dev = evaluate(top, dev, lab, n_lab)
hits = np.mean([len(set(top[i]) & lab[u]) for i, u in enumerate(dev)]) / K
pop = np.asarray(X.sum(0)).ravel(); rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
params = f"{a.tag}: hid={a.hidden} lat={a.latent} drop={a.dropout} beta={a.beta} anneal={a.anneal_epochs}ep lr={a.lr} wd={a.wd} bs={a.batch}, best ep {best[2]}/{a.epochs}, val={best[0]:.4f}"
log("Mult-DAE" if a.dae else "Mult-VAE", params, r_dev, hits, bins(top, dev, lab, rank), len(np.unique(top)), f"{time.time()-t0:.0f}s, seed {a.seed}")
json.dump(dict(args=vars(a), val_curve=hist, best_epoch=best[2], best_val=best[0], dev_recall=r_dev), open(OUT + a.tag + ".json", "w"))
write_preds(top, dev, iids, OUT + f"pred_{a.tag}.csv")
if a.save_scores: np.save(OUT + f"devscores_{a.tag}.npy", Sd.astype(np.float16))
