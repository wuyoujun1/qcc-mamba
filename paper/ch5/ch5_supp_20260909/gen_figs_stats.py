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
import sys; sys.path.insert(0, os.path.dirname(HERE))
import palette_qcc as P
F   = os.path.join(HERE, "figs"); R = os.path.join(HERE, "results")
os.makedirs(F, exist_ok=True); os.makedirs(R, exist_ok=True)
SRC = os.environ.get("QCC_F_DIR", os.path.join(HERE, "..", "figs"))   # f_*.npz 所在（仓库 paper/ch5/figs）
PDFF= os.path.join(HERE, "..", "figs")                                # 第五章 PDF 引用的图
ETT_CSV = os.environ.get("QCC_ETT_CSV", "/home/youjun/dataops_ws/dataset/ETT-small")
def loadF(tag): return np.load(f"{SRC}/f_{tag}.npz")["F"]

# ---------------------------------------------------------------- 变量名
def varnames(ds):
    cols = list(pd.read_csv(f"{ETT_CSV}/{ds}.csv", nrows=0).columns)
    return [c for c in cols if c != "date"]

# ---------------------------------------------------------------- 1.x 热力图
def heatmap(Farr, names, hue, title, out_supp, out_pdf):
    """C3 风格：细白格线 + 对角纯浅灰 + 数字按 WCAG 选色；色带=单色相 OKLCH。"""
    V = Farr.shape[0]
    # 定标用「非对角最大值」：对角恒为 1 且被隐藏，若把它算进 vmax，
    # 可见数据只用到色带下半截、colorbar 顶端还会标出永不出现的值。
    _off = Farr[~np.eye(V, dtype=bool)]
    vmax = max(0.01, float(np.max(_off)))
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    cmap = P.cmap(hue)
    masked = np.ma.masked_invalid(Farr.astype(float)).copy()
    for i in range(V): masked[i, i] = np.ma.masked
    fig, ax = plt.subplots(figsize=(4.3, 3.9))
    ax.imshow(masked, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    for i in range(V):
        for j in range(V):
            if i == j: continue
            ax.text(j, i, f"{Farr[i, j]:.2f}", ha="center", va="center",
                    fontsize=P.NUM_FS, color=P.text_on(cmap(norm(Farr[i, j]))[:3]))
    for i in range(V):   # 对角：纯浅灰，无纹理
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, facecolor=P.DIAG_BG,
                                   edgecolor="none", zorder=3))
    for i in range(1, V):   # 细白格线
        ax.axhline(i - .5, color="white", lw=P.GRID_LW, zorder=4)
        ax.axvline(i - .5, color="white", lw=P.GRID_LW, zorder=4)
    ax.set_xticks(range(V)); ax.set_yticks(range(V))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.tick_params(length=2.5, width=0.8, pad=2)
    for s in ax.spines.values(): s.set_linewidth(0.9)
    ax.set_title(title, fontsize=10)
    from matplotlib.cm import ScalarMappable
    cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("coupling f", fontsize=8); cb.ax.tick_params(labelsize=7.5)
    cb.outline.set_linewidth(0.6)
    # 刻度必须落在 0–vmax 内，否则 colorbar 会标到 1.0 而数据最大只有 vmax（误导）
    tk = np.linspace(0.0, vmax, 4)
    cb.set_ticks(tk); cb.set_ticklabels([f"{x:.2f}" for x in tk])
    fig.tight_layout()
    for op in (out_supp, out_pdf):
        fig.savefig(op, dpi=300)
    plt.close(fig)


varn = {d: varnames(d) for d in ["ETTh1", "ETTm2"]}
# 2026-09-13 定：图2a 赭褐 / 图2b 绛红 / 图2d 墨紫
for ds, tag, hue, pdfname, title in [
        ("ETTh1", "ETTh1_96", "赭褐", "ch5_k_ETTh1.png",     "Learned coupling kernel f · ETTh1"),
        ("ETTm2", "ETTm2_96", "绛红", "ch5_k_ETTm2.png",     "Learned coupling kernel f · ETTm2")]:
    Fm = loadF(tag)
    np.save(f"{R}/K_{tag}.npy", Fm)
    heatmap(Fm, varn[ds], hue, title, f"{F}/K_{tag}.png", f"{PDFF}/{pdfname}")
F72 = loadF("ETTh1_720")
np.save(f"{R}/K_ETTh1_720.npy", F72)
heatmap(F72, varn["ETTh1"], "墨紫", "Learned coupling kernel f · ETTh1 (720)",
        f"{F}/K_ETTh1_720.png", f"{PDFF}/ch5_k_ETTh1_720.png")

# ---------------------------------------------------------------- 1.3 ECDF
dsets = ["ETTh1_96", "ETTm2_96", "ECL_96", "Traffic_96"]
off = {}
for tag in dsets:
    FF = loadF(tag); V = FF.shape[0]
    off[tag] = FF[np.triu_indices(V, 1)]
fig, ax = plt.subplots(figsize=(5.4, 3.7))
# 赭褐与绛红的深端太接近，四条线用「同族色 + 线型冗余编码」区分
ECDF_STYLE = {"ETTh1_96":   (P.ramp("赭褐")[8], "-",  2.2),   # 深褐   实线
              "ETTm2_96":   (P.ramp("绛红")[7], "--", 2.2),   # 绛红   虚线
              "ECL_96":     (P.ramp("墨紫")[5], "-.", 2.2),   # 中紫   点划线（提亮+换型，避免被压住）
              "Traffic_96": (P.ramp("深青")[8], "-",  2.2)}   # 深青   实线
# ECL 与 Traffic 的 CDF 几乎重合，ECL 必须最后画否则被压住
for tag in [d for d in dsets if d != "ECL_96"] + ["ECL_96"]:
    x = np.sort(off[tag]); y = np.arange(1, len(x)+1)/len(x)
    col, ls, lw = ECDF_STYLE[tag]
    ax.plot(x, y, lw=lw, color=col, ls=ls, label=tag.replace("_96", ""))
ax.set_xlabel("off-diagonal coupling f", fontsize=10); ax.set_ylabel("CDF", fontsize=10)
ax.set_title("ECDF of off-diagonal f across four datasets", fontsize=10)
ax.tick_params(labelsize=9); ax.legend(fontsize=9, frameon=False)
for s in ax.spines.values(): s.set_linewidth(0.9)
fig.tight_layout()
fig.savefig(f"{F}/K_distribution_ECDF.png", dpi=300)
fig.savefig(f"{PDFF}/ch5_k_dist.png", dpi=300)
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
