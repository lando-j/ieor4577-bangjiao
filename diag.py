import re, numpy as np, pandas as pd, scipy.linalg as la
from common import *
X, uids, iids, dev, dev_rows, lab, n_lab = load()
Xd = X[dev_rows]
G = (X.T @ X).toarray().astype(np.float32)
pop = np.diag(G).copy()
G[np.diag_indices_from(G)] += 300
P = la.inv(G, overwrite_a=True, check_finite=False)
B = P / (-np.diag(P))[None, :]; np.fill_diagonal(B, 0)
top = topk(Xd @ B, Xd)
hits = np.array([len(set(top[i]) & lab[u]) for i, u in enumerate(dev)])
nl = np.array([n_lab[u] for u in dev])
hl = np.asarray(Xd.sum(1)).ravel()
rec = hits / nl
print(f"overall recall {rec.mean():.4f}")

print("\n[1] 按用户历史长度分组")
df = pd.DataFrame(dict(hl=hl, nl=nl, rec=rec, hits=hits))
df["bin"] = pd.qcut(df.hl, 8, duplicates="drop")
print(df.groupby("bin", observed=True).agg(users=("rec","size"), mean_labels=("nl","mean"), recall=("rec","mean")).round(4))

print("\n[2] 上限与标签数")
orc = np.minimum(nl, 20) / nl
print(f"oracle Recall@20 (理论上限) {orc.mean():.4f}; 平均标签数 {nl.mean():.1f}, 中位数 {np.median(nl):.0f}; >20个标签的用户占 {(nl>20).mean():.1%}")
df["nlbin"] = pd.cut(df.nl, [0,5,10,20,40,80,10000])
print(df.groupby("nlbin", observed=True).agg(users=("rec","size"), recall=("rec","mean"), oracle=("nl", lambda s: (np.minimum(s,20)/s).mean())).round(4))

print("\n[3] 标签电影热门度 (按全局点赞数十分位)")
iid_pos = pd.Series(np.arange(len(iids)), index=iids)
labels = pd.read_csv(D + "dev_labels.csv")
labels["j"] = iid_pos.reindex(labels.item_id.values).values
print(f"标签电影不在训练里的比例: {labels.j.isna().mean():.4%}")
labels = labels.dropna(subset=["j"]); labels["j"] = labels.j.astype(int)
topset = {u: set(top[i]) for i, u in enumerate(dev)}
labels["hit"] = [j in topset[u] for u, j in zip(labels.user_id, labels.j)]
labels["pop"] = pop[labels.j]
order = np.argsort(-pop); rank = np.empty(len(pop), int); rank[order] = np.arange(len(pop))
labels["rank"] = rank[labels.j]
labels["rbin"] = pd.cut(labels["rank"], [-1,19,99,499,1999,5999,25000], labels=["top20","21-100","101-500","501-2000","2001-6000","6000+"])
t = labels.groupby("rbin", observed=True).agg(labels=("hit","size"), hit_rate=("hit","mean"))
t["share_of_labels"] = t.labels / t.labels.sum(); print(t.round(4))

print("\n[4] 推荐列表的热门度")
rr = rank[top.ravel()]
print("推荐电影热门排名: 中位数", int(np.median(rr)), " 落在热门前100的比例", f"{(rr<100).mean():.1%}", " 前500", f"{(rr<500).mean():.1%}", " 前2000", f"{(rr<2000).mean():.1%}")
print("不同的推荐电影数(覆盖率):", len(np.unique(top)), "/", len(iids))

print("\n[5] 日期结构")
inter = pd.read_csv(D + "interactions.csv")
inter["date"] = pd.to_datetime(inter.date)
print("日期范围", inter.date.min().date(), "~", inter.date.max().date())
g = inter.groupby("user_id")
nuniq = g.date.nunique(); cnt = g.size()
print(f"每用户:交互数中位数 {cnt.median():.0f};不同日期数中位数 {nuniq.median():.0f}")
print(f"用户中 >50% 交互发生在同一天的比例: {(g.date.agg(lambda s: s.value_counts().iloc[0]) / cnt > 0.5).mean():.1%}")
span = (g.date.max() - g.date.min()).dt.days
print(f"每用户时间跨度(天)中位数 {span.median():.0f}, 跨度为0的用户占 {(span==0).mean():.1%}")

print("\n[6] 是随机划分还是按时间划分? 用电影年份对比")
items = pd.read_csv(D + "items.csv")
items["year"] = items.title.str.extract(r"\((\d{4})\)\s*$")[0].astype(float)
yr = items.set_index("item_id").year
hist = inter[inter.user_id.isin(dev)]
lab_all = pd.read_csv(D + "dev_labels.csv")
print(f"dev 历史电影年份均值 {yr.reindex(hist.item_id.values).mean():.2f}, 标签电影年份均值 {yr.reindex(lab_all.item_id.values).mean():.2f}")
tr = inter[~inter.user_id.isin(dev)]
print(f"训练用户电影年份均值 {yr.reindex(tr.item_id.values).mean():.2f}")
# 逐用户: 标签年份均值 - 历史年份均值
hy = hist.assign(y=yr.reindex(hist.item_id.values).values).groupby("user_id").y.mean()
ly = lab_all.assign(y=yr.reindex(lab_all.item_id.values).values).groupby("user_id").y.mean()
d = (ly - hy).dropna(); print(f"逐用户 (标签年份-历史年份) 均值 {d.mean():+.2f} 年, 标准差 {d.std():.2f}")
print("标准误上界参考 (随机划分时均值应≈0):", f"{d.std()/np.sqrt(len(d)):.3f}")
