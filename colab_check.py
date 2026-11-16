"""Colab readiness check for the Mult-VAE / RecVAE runs. Run once per new Colab session:
    !IEOR_DATA=/content/drive/MyDrive/ieor4577 python colab_check.py
Checks: GPU + memory, library versions, data files, data shape vs expected, dense GPU matmul speed,
and the time for one full Mult-VAE-sized training epoch (to estimate a 200-epoch run).
"""
import os, sys, time, shutil
ok = True
def check(name, cond, info=""):
    global ok; ok &= bool(cond); print(f"[{'OK ' if cond else 'FAIL'}] {name} {info}", flush=True)

import numpy as np, scipy, pandas as pd, torch
print(f"python {sys.version.split()[0]} | numpy {np.__version__} | scipy {scipy.__version__} | pandas {pd.__version__} | torch {torch.__version__}")
check("CUDA GPU available", torch.cuda.is_available(), "(Runtime > Change runtime type > GPU)")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    check("GPU memory >= 12 GB", p.total_memory / 2**30 >= 12, f"{p.name}, {p.total_memory/2**30:.1f} GB")
try:
    import psutil; ram = psutil.virtual_memory().total / 2**30
    check("system RAM >= 12 GB", ram >= 12, f"{ram:.1f} GB (EASE's 20k x 20k inverse needs ~5 GB)")
except ImportError:
    print("psutil missing; skip RAM check")
check("disk free >= 5 GB", shutil.disk_usage(".").free / 2**30 >= 5, f"{shutil.disk_usage('.').free/2**30:.0f} GB free")

# Colab ships an unrelated pip package called "common"; make sure our experiments/common.py wins.
here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
check("running from experiments/ folder", os.path.exists(os.path.join(here, "common.py")),
      f"(cwd = {here}; %cd into the experiments folder on Drive first)")
if not ok: sys.exit("fix the FAIL lines first")
sys.path.insert(0, here); sys.modules.pop("common", None)
from common import D, load
for f in ["interactions.csv", "dev_labels.csv", "items.csv"]:
    check(f"data file {f}", os.path.exists(D + f), D + f)
if not ok: sys.exit("fix the FAIL lines first")

t = time.time(); X, uids, iids, dev, dev_rows, lab, n_lab = load()
# 20,720 movies in items.csv, but only 20,205 appear in interactions.csv (the matrix columns)
check("data shape", X.shape == (126677, 20205) and X.nnz == 8_691_549,
      f"{X.shape}, {X.nnz:,} interactions, {len(dev)} dev users, load {time.time()-t:.0f}s")
check("dev users = 10,000", len(dev) == 10000)

dev_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a = torch.randn(4096, 4096, device=dev_); torch.cuda.synchronize() if dev_.type == "cuda" else None
t = time.time()
for _ in range(20): a @ a
torch.cuda.synchronize() if dev_.type == "cuda" else None
print(f"matmul: {20*2*4096**3/(time.time()-t)/1e12:.1f} TFLOPS fp32")

# one Mult-VAE-shaped epoch: 600-200-600, full softmax over 20,720 items, batch 500
import torch.nn as nn, torch.nn.functional as F
n_items = X.shape[1]
net = nn.Sequential(nn.Linear(n_items, 600), nn.Tanh(), nn.Linear(600, 400), nn.Tanh(),
                    nn.Linear(400, 600), nn.Tanh(), nn.Linear(600, n_items)).to(dev_)
opt = torch.optim.Adam(net.parameters(), 1e-3)
rows = np.random.default_rng(0).permutation(X.shape[0]); t = time.time()
for i in range(0, len(rows), 500):
    x = torch.from_numpy(X[rows[i:i+500]].toarray()).to(dev_)
    loss = -(F.log_softmax(net(F.normalize(x, dim=1)), 1) * x).sum(1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
torch.cuda.synchronize() if dev_.type == "cuda" else None
ep = time.time() - t
print(f"one epoch {ep:.0f}s -> 200 epochs ~{200*ep/60:.0f} min (RecVAE alternating updates: roughly 2-3x that)")
check("200-epoch Mult-VAE fits in a 3h session", 200 * ep < 3 * 3600)
print("\nALL OK" if ok else "\nsome checks FAILED")
