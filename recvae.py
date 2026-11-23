"""RecVAE (Shenbin et al., WSDM 2020) on the course data, following the official code
(github.com/ilya-shenbin/RecVAE, master: model.py / run.py; its README notes master differs slightly from the paper).

What differs from our Mult-VAE (vae.py):
  * encoder: 5 dense-residual layers (h_k = LN(swish(W_k h_{k-1} + h_1 + ... + h_{k-1}))), LayerNorm eps 0.1, swish
  * decoder: a single linear layer latent -> items
  * composite prior: mixture [3/20, 3/4, 1/10] of N(0, I), the *previous* encoder's posterior q_old(z|x), and N(0, e^10 I)
  * KL weight per user = gamma * |x_u| (number of items in the user's input), not a global annealed beta
  * alternating training: per outer epoch, 3 encoder epochs (input dropout 0.5), then copy encoder -> prior's old encoder,
    then 1 decoder epoch (no dropout; z still sampled). Separate Adam optimizers, lr 5e-4.
Official ML-20M setting: 50 outer epochs, gamma 0.005, hidden 600, latent 200, batch 500.

Our protocol (same as vae.py): model selection on the 5,000 validation users of common.val_split by course-style
Recall@20 every outer epoch (the official code selects by NDCG@100); dev evaluated once with the best model.
A checkpoint is written after every outer epoch, so a dropped Colab session can continue with --resume.

Usage: python recvae.py --tag recvae [--epochs 50 --gamma 0.005 --resume]
       python recvae.py --tag smoke --epochs 1 --max_train 3000      (quick local check)
"""
import argparse, json, os, time, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import *
from log import log, bins

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="recvae")
ap.add_argument("--epochs", type=int, default=50, help="outer epochs (each = n_enc encoder + n_dec decoder passes)")
ap.add_argument("--n_enc", type=int, default=3)
ap.add_argument("--n_dec", type=int, default=1)
ap.add_argument("--hidden", type=int, default=600)
ap.add_argument("--latent", type=int, default=200)
ap.add_argument("--gamma", type=float, default=0.005)
ap.add_argument("--dropout", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--batch", type=int, default=500)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max_train", type=int, default=0, help="use only this many training users (smoke tests)")
ap.add_argument("--resume", action="store_true")
a = ap.parse_args()
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
dev_ = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs") + "/"; os.makedirs(OUT, exist_ok=True)
CKPT = OUT + f"ckpt_{a.tag}.pt"
print("device:", dev_, flush=True)

X, uids, iids, dev, dev_rows, lab, n_lab = load()
n_users, n_items = X.shape
val_rows, Xv_in, val_lab, _ = val_split(X, dev_rows, seed=0)        # fixed split, shared by every model
val_n = np.array([max(len(s), 1) for s in val_lab])
is_val = np.zeros(n_users, bool); is_val[val_rows] = True
train_rows = np.where(~is_val)[0]                                   # includes dev users' given histories
if a.max_train: train_rows = rng.choice(train_rows, a.max_train, replace=False)
Xtr = X[train_rows].tocsr(); Xdev = X[dev_rows].tocsr()
print(f"train users {len(train_rows)}  val users {len(val_rows)}  dev users {len(dev_rows)}  items {n_items}", flush=True)

LOG2PI = float(np.log(2 * np.pi))
def log_norm_pdf(z, mu, logvar): return -0.5 * (logvar + LOG2PI + (z - mu).pow(2) / logvar.exp())

class Encoder(nn.Module):
    def __init__(s, eps=1e-1):
        super().__init__()
        s.fc = nn.ModuleList([nn.Linear(n_items, a.hidden)] + [nn.Linear(a.hidden, a.hidden) for _ in range(4)])
        s.ln = nn.ModuleList([nn.LayerNorm(a.hidden, eps=eps) for _ in range(5)])
        s.mu = nn.Linear(a.hidden, a.latent); s.logvar = nn.Linear(a.hidden, a.latent)
    def forward(s, x, dropout):
        x = F.normalize(x, dim=1)                                      # official: x / ||x||_2
        x = F.dropout(x, p=dropout, training=s.training)
        hs = [s.ln[0](F.silu(s.fc[0](x)))]                             # silu == swish
        for k in range(1, 5):
            hs.append(s.ln[k](F.silu(s.fc[k](hs[-1]) + sum(hs))))
        return s.mu(hs[-1]), s.logvar(hs[-1])

class RecVAE(nn.Module):
    W = [3 / 20, 3 / 4, 1 / 10]
    def __init__(s):
        super().__init__()
        s.encoder = Encoder(); s.encoder_old = Encoder(); s.encoder_old.requires_grad_(False)
        s.decoder = nn.Linear(a.latent, n_items)
        s.update_prior()
    def update_prior(s): s.encoder_old.load_state_dict(copy.deepcopy(s.encoder.state_dict()))
    def log_prior(s, x, z):
        pmu, plv = s.encoder_old(x, 0)
        zero = torch.zeros_like(z)
        comps = [log_norm_pdf(z, zero, zero), log_norm_pdf(z, pmu, plv), log_norm_pdf(z, zero, zero + 10)]
        return torch.logsumexp(torch.stack([c + np.log(w) for c, w in zip(comps, s.W)], -1), -1)
    def forward(s, x, dropout=0.0, loss=True):
        mu, logvar = s.encoder(x, dropout)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if s.training else mu
        logits = s.decoder(z)
        if not loss: return logits
        mll = (F.log_softmax(logits, -1) * x).sum(-1).mean()
        kl_w = a.gamma * x.sum(-1)                                     # per-user KL weight
        kld = ((log_norm_pdf(z, mu, logvar) - s.log_prior(x, z)).sum(-1) * kl_w).mean()
        return -(mll - kld), mll.item(), kld.item()

def dense(m): return torch.from_numpy(m.toarray()).to(dev_)

@torch.no_grad()
def score(model, Xin, bs=1000):
    model.eval(); out = []
    for i in range(0, Xin.shape[0], bs):
        out.append(model(dense(Xin[i:i+bs]), loss=False).float().cpu().numpy())
    return np.concatenate(out)

def val_recall(model):
    top = topk(score(model, Xv_in), Xv_in)
    return float(per_user_recall(top, val_lab, val_n).mean())

def run_epochs(model, opt, n, dropout):
    model.train(); tot = np.zeros(3); nb = 0
    for _ in range(n):
        perm = rng.permutation(len(train_rows))
        for i in range(0, len(perm), a.batch):
            x = dense(Xtr[perm[i:i+a.batch]])
            opt.zero_grad()
            loss, mll, kld = model(x, dropout)
            loss.backward(); opt.step()
            tot += (loss.item(), mll, kld); nb += 1
    return tot / nb

model = RecVAE().to(dev_)
opt_enc = torch.optim.Adam(model.encoder.parameters(), lr=a.lr)
opt_dec = torch.optim.Adam(model.decoder.parameters(), lr=a.lr)
start, best, hist = 1, (-1.0, None, 0), []
if a.resume and os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=dev_, weights_only=False)
    model.load_state_dict(ck["model"]); opt_enc.load_state_dict(ck["opt_enc"]); opt_dec.load_state_dict(ck["opt_dec"])
    start, best, hist = ck["epoch"] + 1, (ck["best_val"], ck["best_state"], ck["best_epoch"]), ck["hist"]
    rng = np.random.default_rng(a.seed + ck["epoch"])
    print(f"resumed from outer epoch {ck['epoch']} (best val {best[0]:.4f} at {best[2]})", flush=True)

t0 = time.time()
for ep in range(start, a.epochs + 1):
    le = run_epochs(model, opt_enc, a.n_enc, a.dropout)                # encoder passes, dropout 0.5
    model.update_prior()                                               # old posterior <- current encoder
    ld = run_epochs(model, opt_dec, a.n_dec, 0.0)                      # decoder pass, no dropout
    vr = val_recall(model)
    hist.append((ep, float(le[0]), float(ld[0]), vr))
    if vr > best[0]: best = (vr, copy.deepcopy(model.state_dict()), ep)
    print(f"ep {ep:3d} enc loss {le[0]:8.2f} (mll {le[1]:7.2f} kl {le[2]:6.2f}) dec loss {ld[0]:8.2f} "
          f"val Recall@20 {vr:.4f}  best {best[0]:.4f}@{best[2]}  ({time.time()-t0:.0f}s)", flush=True)
    torch.save(dict(model=model.state_dict(), opt_enc=opt_enc.state_dict(), opt_dec=opt_dec.state_dict(), epoch=ep,
                    best_val=best[0], best_state=best[1], best_epoch=best[2], hist=hist, args=vars(a)), CKPT)

model.load_state_dict(best[1]); print(f"best val {best[0]:.4f} at outer epoch {best[2]}", flush=True)
Sd = score(model, Xdev); top = topk(Sd, Xdev)
dev_lab = [lab[u] for u in dev]; dev_n = np.array([n_lab[u] for u in dev])
r = per_user_recall(top, dev_lab, dev_n)
hits = np.array([len(set(top[i]) & dev_lab[i]) for i in range(len(dev))])
paper = (hits / np.minimum(20, dev_n)).mean()
print(f"dev Recall@20 {r.mean():.4f} (paper-style {paper:.4f})", flush=True)
if a.max_train: raise SystemExit("smoke run (--max_train): nothing logged or saved")
pop = np.asarray(X.sum(0)).ravel(); rank = np.empty(len(pop), int); rank[np.argsort(-pop)] = np.arange(len(pop))
params = (f"{a.tag}: hid={a.hidden} lat={a.latent} gamma={a.gamma} drop={a.dropout} lr={a.lr} enc/dec={a.n_enc}/{a.n_dec} "
          f"bs={a.batch}, best outer ep {best[2]}/{a.epochs}, val={best[0]:.4f}" + (f", max_train={a.max_train}" if a.max_train else ""))
log("RecVAE", params, r.mean(), hits.mean() / K, bins(top, dev, lab, rank), len(np.unique(top)),
    f"论文口径 {paper:.4f}; {time.time()-t0:.0f}s, seed {a.seed}")
json.dump(dict(args=vars(a), hist=hist, best_epoch=best[2], best_val=best[0], dev_recall=float(r.mean()), dev_paper=float(paper)),
          open(OUT + a.tag + ".json", "w"), indent=1)
write_preds(top, dev, iids, OUT + f"pred_{a.tag}.csv")
np.save(OUT + f"devscores_{a.tag}.npy", Sd.astype(np.float16))
np.save(OUT + f"valscores_{a.tag}.npy", score(model, Xv_in).astype(np.float16))
torch.save(model.state_dict(), OUT + f"model_{a.tag}.pt")
print("saved scores / preds / model to runs/", flush=True)
