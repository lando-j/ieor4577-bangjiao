"""Append-only experiment log: every call adds one row to RESULTS.md."""
import datetime, os
import numpy as np

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.md")
HEADER = (
    "# 实验结果(dev 集,Recall@20 / Precision@20,全目录排序)\n\n"
    "脚本每跑完一个实验自动追加一行。分档命中率 = 标签电影按全局热门排名分档后的命中率。\n\n"
    "| 时间 | 方法 | 参数 | Recall@20 | Prec@20 | 命中率 101-500 | 501-2000 | 2001+ | 推荐覆盖电影数 | 备注 |\n"
    "|---|---|---|---|---|---|---|---|---|---|\n"
)


def bins(top, dev, lab, rank):
    """hit rate of label items by global popularity rank band."""
    out = {}
    h = {"101-500": [0, 0], "501-2000": [0, 0], "2001+": [0, 0]}
    for i, u in enumerate(dev):
        t = set(top[i])
        for j in lab[u]:
            r = rank[j]
            k = "101-500" if 100 <= r < 500 else "501-2000" if 500 <= r < 2000 else "2001+" if r >= 2000 else None
            if k:
                h[k][0] += j in t
                h[k][1] += 1
    return {k: v[0] / max(v[1], 1) for k, v in h.items()}


def log(method, params, recall, prec, b=None, cover=None, note=""):
    if not os.path.exists(PATH):
        with open(PATH, "w") as f:
            f.write(HEADER)
    b = b or {}
    f4 = lambda x: "" if x is None else f"{x:.4f}"
    row = (f"| {datetime.datetime.now():%m-%d %H:%M} | {method} | {params} | {recall:.4f} | {prec:.4f} | "
           f"{f4(b.get('101-500'))} | {f4(b.get('501-2000'))} | {f4(b.get('2001+'))} | {cover or ''} | {note} |\n")
    with open(PATH, "a") as f:
        f.write(row)
    print(row.strip(), flush=True)
