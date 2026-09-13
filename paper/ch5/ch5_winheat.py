# -*- coding: utf-8 -*-
"""图1：QCCK-M 相对 S-Mamba 的 MSE 降低百分比（9 盘 × 4 预测长），高分辨率。
数值与主表同源：S-Mamba 取官方复现（含 Traffic/Exchange 个别覆盖），QCC 取 md 主表。
"""
import os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import palette_qcc as P
# 中文字体（仓库自带 wqy）
import matplotlib.font_manager as _fm
_fm.fontManager.addfont(os.path.join(BASE, "cjkfont", "wqy-zenhei.ttf"))
plt.rcParams["font.sans-serif"] = ["WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

MD = open(os.environ.get("QCC_DATA_MD", os.path.join(BASE, "data", "第五章_主表消融_MSEMAE_20260904.md")), encoding="utf-8").read()

def parse_md_main(part):
    lines = [l for l in part.splitlines() if l.startswith("|")]
    hdr = [x.strip() for x in lines[0].split("|")[1:-1]]
    rows = [[x.strip() for x in l.split("|")[1:-1]] for l in lines[2:]]
    return hdr, rows

def splitmm(s):
    if s in ("—/—", "—", "--", "") or "/" not in s:
        return (None, None)
    a, b = s.split("/", 1)
    return (a.strip(), b.strip())

hdr, rows = parse_md_main(MD.split("## 主表")[1].split("## 主表 · Beijing")[0])
mi = {m: i + 2 for i, m in enumerate(hdr[2:])}
SMOV = {("Traffic","96"):("0.3785","0.2585"),("Traffic","336"):("0.4165","0.2795"),
        ("Traffic","720"):("0.4637","0.2983"),("Exchange","720"):("0.8595","0.7003")}
def cell(ds, h, meth):
    for r in rows:
        if r[0] == ds and r[1] == h:
            return splitmm(r[mi[meth]])
    return (None, None)

DS = ["ETTh1","ETTh2","ETTm1","ETTm2","Weather","ECL","Traffic","Exchange","Beijing"]
HS = ["96","192","336","720"]
bj_q = {"96":("0.4520","0.3167"),"192":("0.4759","0.3365"),"336":("0.4909","0.3499"),"720":("0.4733","0.3593")}
bj_sm= {"96":("0.4788","0.3394"),"192":("0.4924","0.3487"),"336":("0.5094","0.3618"),"720":("0.5346","0.3961")}

G = np.zeros((len(DS), len(HS)))
for i, ds in enumerate(DS):
    for j, h in enumerate(HS):
        if ds == "Beijing":
            q = float(bj_q[h][0]); s = float(bj_sm[h][0])
        else:
            qv = cell(ds, h, "QCC"); q = float(qv[0]) if qv and qv[0] else float("nan")
            sv = SMOV.get((ds, h)) or cell(ds, h, "S-Mamba")
            s = float(sv[0]) if sv and sv[0] else float("nan")
        G[i, j] = (s - q) / s * 100.0 if s and not np.isnan(q) and not np.isnan(s) else 0.0

vmax = max(1.0, float(np.nanmax(G)))
# 行=预测长度(720..96 上到下)，列=数据集；色带 magma（论文风）
D = G.T  # (4, 9) rows=horizon, cols=dataset
HROWS = ["720", "336", "192", "96"]
D = np.array([D[HS.index(h)] for h in HROWS])  # top=720
cmap = plt.get_cmap("Blues")   # 保持原配色（用户 2026-09-13 定：图1 不改色）
fig, ax = plt.subplots(figsize=(10.0, 4.6), dpi=300)
X, Y = np.meshgrid(np.arange(D.shape[1] + 1), np.arange(D.shape[0] + 1))
pcm = ax.pcolormesh(X, Y, D, cmap=cmap, vmin=0, vmax=vmax, edgecolor="white",
                    linewidth=1.6, shading="flat")
ax.set_xticks(np.arange(len(DS)) + 0.5); ax.set_xticklabels(DS, fontsize=13)
ax.set_yticks(np.arange(len(HROWS)) + 0.5); ax.set_yticklabels(HROWS, fontsize=13)
ax.set_xlabel("数据集", fontsize=13); ax.set_ylabel("预测长度", fontsize=13)
ax.set_xlim(0, len(DS)); ax.set_ylim(0, len(HROWS))
for i in range(len(HROWS)):
    for j in range(len(DS)):
        v = D[i, j]
        if np.isnan(v): continue
        txt = f"{v:.1f}" if abs(v) < 9.5 else f"{v:.0f}"
        c = cmap(min(1.0, v / vmax))[:3]
        lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
        ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=12,
                fontweight="bold", color="black" if lum > 0.5 else "white")
cb = fig.colorbar(pcm, ax=ax, fraction=0.028, pad=0.02)
cb.set_label("相对 S-Mamba 的 MSE 降低 (%)", fontsize=11)
cb.ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "figs", "ch5_winheat.png"), dpi=300, bbox_inches="tight")
print("saved ch5_winheat.png ; max imp =", round(float(np.nanmax(G)), 2),
      "Beijing:", [f"{G[8,j]:.1f}" for j in range(4)])
