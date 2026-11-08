import sys, time, numpy as np, scipy.linalg as la
from common import *
S = sys.argv[1]; lams = [float(x) for x in sys.argv[2:]]
t = time.time()
X, uids, iids, dev, dev_rows, lab, n_lab = load()
G0 = (X.T @ X).toarray().astype(np.float32)
Xd = X[dev_rows]
best = (0, None)
for lam in lams:
    G = G0.copy(); G[np.diag_indices_from(G)] += lam
    P = la.inv(G, overwrite_a=True, check_finite=False)
    B = P / (-np.diag(P))[None, :]
    np.fill_diagonal(B, 0)
    top = topk(Xd @ B, Xd)
    r = evaluate(top, dev, lab, n_lab)
    print(f"ease lambda={lam} Recall@20={r:.4f} ({time.time()-t:.0f}s)", flush=True)
    if r > best[0]: best = (r, top, lam)
write_preds(best[1], dev, iids, S + "/pred_ease.csv")
print("best", best[0], "lambda", best[2])
