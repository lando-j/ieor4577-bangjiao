import sys, time, numpy as np
from common import *
S = sys.argv[1]
t = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
print("loaded", X.shape, X.nnz, f"{time.time()-t:.0f}s", flush=True)
G = (X.T @ X).toarray().astype(np.float32)
d = np.diag(G).copy(); np.fill_diagonal(G, 0)
Xd = X[dev_rows]
norm = np.sqrt(np.maximum(d, 1))
C = G / norm[:, None] / norm[None, :]          # cosine
print("gram", f"{time.time()-t:.0f}s", flush=True)
best = (0, None)
for k in [50, 200, 1000, None]:
    if k:
        Ck = np.zeros_like(C)
        idx = np.argpartition(-C, k, axis=1)[:, :k]
        np.put_along_axis(Ck, idx, np.take_along_axis(C, idx, 1), 1)
    else:
        Ck = C
    for p in [1.0, 2.0]:
        W = Ck ** p if p != 1 else Ck
        top = topk(Xd @ W, Xd)
        r = evaluate(top, dev, lab, n_lab)
        print(f"knn k={k} pow={p} Recall@20={r:.4f}", flush=True)
        if r > best[0]: best = (r, top)
write_preds(best[1], dev, iids, S + "/pred_knn.csv")
print("best", best[0])
