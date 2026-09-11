# -*- coding: utf-8 -*-
"""图2：敏感性（2×2 版面）。
列 = 数据集（ETTh1 / ETTm2），行 = 超参数（量子比特数 / 门初值）；
每条折线 = 一个预测长度，线尾直接标步长。纵轴=相对该设置默认取值(base)的 MSE 变化%。
"""
import os, re, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
font_manager.fontManager.addfont("/home/youjun/cjkfont/wqy-zenhei.ttf")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"
plt.rcParams["axes.unicode_minus"] = False

L = "/home/youjun/dataops_ws/logs"
def mse_of(log):
    try:
        txt = open(log, encoding="utf-8", errors="ignore").read()
    except FileNotFoundError:
        return None
    m = re.findall(r"mse:([\d.]+)", txt)
    return float(m[-1]) if m else None

CELLS = {"ETTh1": ["96", "192", "336", "720"], "ETTm2": ["96", "192"]}
data = {}
for ds, hs in CELLS.items():
    for h in hs:
        for v in ["nq3", "nq7", "g0", "g1", "base"]:
            m = mse_of(f"{L}/sens_{ds}_{h}_{v}.log")
            if m is not None:
                data[(ds, h, v)] = m
json.dump({f"{a}|{b}|{c}": v for (a, b, c), v in data.items()}, open("/tmp/sens_full.json", "w"))

def dev(ds, h, v):
    b = data[(ds, h, "base")]
    return (data[(ds, h, v)] - b) / b * 100.0

HCO = {"96": "#1f77b4", "192": "#ff7f0e", "336": "#2ca02c", "720": "#d62728"}
def maptag(is_nq, x):
    if is_nq: return "base" if x == 5 else ("nq3" if x == 3 else "nq7")
    return "base" if x == 0.1 else ("g0" if x == 0.0 else "g1")

fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.2), dpi=300)
PANELS = [(0, 0, "ETTh1", True), (0, 1, "ETTm2", True),
          (1, 0, "ETTh1", False), (1, 1, "ETTm2", False)]
for r, c, ds, is_nq in PANELS:
    ax = axes[r][c]
    pvals = [3, 5, 7] if is_nq else [0.0, 0.1, 1.0]
    for h in CELLS[ds]:
        xs = pvals; yy = [dev(ds, h, maptag(is_nq, x)) for x in pvals]
        ax.plot(xs, yy, "-o", color=HCO[h], ms=4, lw=1.5)
        ax.annotate(f" H={h}", (xs[-1], yy[-1]), color=HCO[h], fontsize=8,
                    va="center", ha="left")
    ax.axhline(0, color="gray", lw=0.9, ls="--")
    ax.set_xlabel("量子比特数 n_qubits" if is_nq else "门初值 gate_init", fontsize=10)
    ax.set_ylabel("相对默认 MSE 变化 (%)", fontsize=10)
    ax.set_title(f"{ds} · {'量子比特数' if is_nq else '门初值'}", fontsize=11)
    ax.set_xlim(min(pvals) - 0.3, max(pvals) + 0.6)
    ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig("/home/youjun/paper/figs/ch5_sens_full.png", dpi=300, bbox_inches="tight")
print("saved ch5_sens_full.png (2x2)")
print(f"{'cell':13s} {'default':>9s} {'nq3':>9s} {'nq7':>9s} {'gate0':>9s} {'gate1':>9s}")
for ds, hs in CELLS.items():
    for h in hs:
        if (ds, h, "base") not in data: continue
        print(f"{ds+' '+h:13s} " + " ".join(f"{data[(ds,h,v)]:.4f}" if (ds, h, v) in data else "  --    "
                                            for v in ["base", "nq3", "nq7", "g0", "g1"]))
