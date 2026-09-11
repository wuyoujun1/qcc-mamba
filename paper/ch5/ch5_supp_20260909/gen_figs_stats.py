# -*- coding: utf-8 -*-
"""
QCCK-M 第五章补充实验 · 可解释性(原始耦合核 f) 分析与绘图
============================================================
口径(与定稿第五章一致)：
  - 耦合核 f[i,j] = |<psi_i|psi_j>|^2  对称、对角≈1、0..1、逐样本
  - 此处展示/统计的均为“测试集前若干 batch 的逐样本 f 求平均”
    的原始对称核（与 paper/figs/f_*.npz 完全同一来源），
    不 softmax / 不 temperature / 不使用消息权重 K_n。
输入：/home/youjun/paper/figs/f_{tag}.npz   (tag=ETTh1_96,ETTm2_96,ECL_96,Traffic_96,ETTh1_720,ETTh2_96,ETTm1_96)
输出：本目录 figs/、results/（任务单命名）
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from scipy import signal as sg
from scipy.stats import spearmanr
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
F   = os.path.join(HERE, "figs"); R = os.path.join(HERE, "results")
os.makedirs(F, exist_ok=True); os.makedirs(R, exist_ok=True)
SRC = "/home/youjun/paper/figs"
ETT_CSV = "/home/youjun/dataops_ws/dataset/ETT-small"
def loadF(tag): return np.load(f"{SRC}/f_{tag}.npz")["F"]

# ---------------------------------------------------------------- 变量名
def varnames(ds):
    cols = list(pd.read_csv(f"{ETT_CSV}/{ds}.csv", nrows=0).columns)
    return [c for c in cols if c != "date"]

# ---------------------------------------------------------------- 1.x 热力图
def heatmap(Farr, names, outpath, title):
    import matplotlib.cm as cm
    from matplotlib.colors import LinearSegmentedColormap
    V = Farr.shape[0]
    vmax = max(0.01, float(np.max(Farr)))
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)   # 平方根展色，低值区可区分
    # 连续浅底→蓝绿→黄：低值不近黑/不深紫（审稿口径），单调于 f
    cmap = LinearSegmentedColormap.from_list("couple_light", [
        "#EDF4FC", "#D3E5F5", "#AFCDEA", "#79B6DA",
        "#4FA8C4", "#2FA395", "#7CC24F", "#E6C93E", "#FFD93F"])
    cmap.set_bad("#BFBFBF")   # 隐藏对角显示为中灰带，区别于低值浅底
    masked = np.ma.masked_invalid(Farr.astype(float)).copy()
    for i in range(V): masked[i, i] = np.ma.masked
    fig, ax = plt.subplots(figsize=(4.2, 3.7))
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal",
                   interpolation="nearest")
    for i in range(V):
        for j in range(V):
            if i == j: continue
            c = cmap(norm(Farr[i, j]))[:3]
            lum = 0.299*c[0] + 0.587*c[1] + 0.114*c[2]
            ax.text(j, i, f"{Farr[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if lum < 0.5 else "black")
    ax.set_xticks(range(V)); ax.set_yticks(range(V))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title(title, fontsize=10)
    cb = plt.colorbar(im, ax=ax, fraction=0.046); cb.ax.tick_params(labelsize=8)
    fig.tight_layout(); fig.savefig(outpath, dpi=200); plt.close(fig)

varn = {d: varnames(d) for d in ["ETTh1", "ETTm2"]}
for ds, tag in [("ETTh1", "ETTh1_96"), ("ETTm2", "ETTm2_96")]:
    Fm = loadF(tag)
    np.save(f"{R}/K_{tag}.npy", Fm)
    heatmap(Fm, varn[ds], f"{F}/K_{tag}.png", f"Learned coupling kernel f · {ds}")
F72 = loadF("ETTh1_720")
np.save(f"{R}/K_ETTh1_720.npy", F72)
heatmap(F72, varn["ETTh1"], f"{F}/K_ETTh1_720.png",
        "Learned coupling kernel f · ETTh1 (720)")

# ---------------------------------------------------------------- 1.3 ECDF
dsets = ["ETTh1_96", "ETTm2_96", "ECL_96", "Traffic_96"]
off = {}
for tag in dsets:
    FF = loadF(tag); V = FF.shape[0]
    off[tag] = FF[np.triu_indices(V, 1)]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
for tag in dsets:
    x = np.sort(off[tag]); y = np.arange(1, len(x)+1)/len(x)
    ax.plot(x, y, lw=1.8, label=tag.replace("_96", ""))
ax.set_xlabel("off-diagonal coupling f"); ax.set_ylabel("CDF")
ax.set_title("ECDF of off-diagonal f across four datasets", fontsize=10)
ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{F}/K_distribution_ECDF.png", dpi=200)
plt.close(fig)
# values：npz 存各盘 1D 非对角；.npy 存拼接（任务单命名）
np.savez(f"{R}/K_distribution_values.npz", **off)
np.save(f"{R}/K_distribution_values.npy",
        np.concatenate([off[t] for t in dsets]))

# ---------------------------------------------------------------- 三 Spearman
def data_coupling(csv):
    X = pd.read_csv(csv).select_dtypes(include=[np.number]).values.astype(np.float64).T
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    n, T = X.shape; L = max(8, int(T*0.05)); C = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            cc = sg.correlate(X[i], X[j], mode="same")/T
            C[i, j] = C[j, i] = np.abs(cc[T//2-L:T//2+L+1]).max()
    return C
PROV_K = "raw symmetric fidelity kernel f=|<psi_i|psi_j>|^2, test-set mean (== paper figs f_*.npz)"
PROV_S = "data-driven: per-variable z-normalized full series, lag-aligned (<=5% len) max|cross-correlation|"
rows = []
allk, allc = [], []
for ds, tag in [("ETTh1", "ETTh1_96"), ("ETTh2", "ETTh2_96"),
                ("ETTm1", "ETTm1_96"), ("ETTm2", "ETTm2_96")]:
    FF = loadF(tag); V = FF.shape[0]
    C = data_coupling(f"{ETT_CSV}/{ds}.csv"); ii = np.triu_indices(V, 1)
    k = FF[ii]; c = C[ii]
    rho, p = spearmanr(k, c)
    allk += list(k); allc += list(c)
    rows.append([ds, round(rho, 4), round(float(p), 4), len(k)])
rk, rp = spearmanr(allk, allc)
rows.append(["merged_4ETT", round(rk, 4), round(float(rp), 6), len(allk)])
df = pd.DataFrame(rows, columns=["Dataset", "Spearman_rho", "p_value", "n_pairs"])
df["K_source"] = PROV_K; df["stat_source"] = PROV_S
df["diag_removed"] = "yes"
df.to_csv(f"{R}/spearman_results.csv", index=False)

# ---------------------------------------------------------------- 四 Top-K
def topk(ds, tag, top=5):
    FF = loadF(tag); V = FF.shape[0]; names = varnames(ds)
    ii = np.triu_indices(V, 1); vals = FF[ii]; order = np.argsort(-vals)
    out = []
    for r in range(top):
        idx = order[r]; i, j = ii[0][idx], ii[1][idx]
        out.append([ds, names[i], names[j], round(float(vals[idx]), 6), r+1])
    return out
tk = topk("ETTh1", "ETTh1_96") + topk("ETTm2", "ETTm2_96")
(pd.DataFrame(tk, columns=["Dataset", "Variable_i", "Variable_j",
                           "K_value", "Rank"])
   .to_csv(f"{R}/topK_pairs.csv", index=False))

print("figs:", sorted(os.listdir(F)))
print("results:", sorted(os.listdir(R)))
print(df.to_string(index=False))
