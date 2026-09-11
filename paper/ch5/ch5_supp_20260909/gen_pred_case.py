# -*- coding: utf-8 -*-
"""
QCCK-M 第五章补充实验 · 预测案例(prediction_case)
==================================================
任务单 §五：找 QCCK-M 比 S-Mamba 明显改善的局部测试窗口（定性案例，不入主表）。
选窗规则（防 cherry-pick，可复现）：
  1) 整段 ETTh1 测试集逐窗(96)预测，算 OT 通道窗口 MSE；
  2) 保留 Δ = MSE_SM - MSE_QCC > 0 的窗口；
  3) 在这些窗口中，选 QCC 自身误差落在全窗 QCC 误差中位数附近(±~半IQR)
     、且 Δ 最大的窗口 —— 典型 QCC 表现、但 S-Mamba 明显更差的代表窗。
  若差异不明显(见 print)，程序会提示用 720 再试（任务单兜底）。
输出：figs/prediction_case.png   results/case_info.txt
"""
import sys, re, torch, numpy as np, argparse, glob, importlib, os
_WS = os.environ.get("QCC_WS", "/home/youjun/dataops_ws")  # 工作副本路径（含 run.py/logs/checkpoints）
os.chdir(_WS)
sys.path.insert(0, "/home/youjun/dataops_ws")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
F = os.path.join(HERE, "figs"); R = os.path.join(HERE, "results")
os.makedirs(F, exist_ok=True); os.makedirs(R, exist_ok=True)

def cfg_from_log(log):
    txt = open(log, encoding="utf-8", errors="ignore").read()
    ns = re.search(r"Namespace\((.+?)\)\n", txt, re.S).group(1)
    src = open("/home/youjun/dataops_ws/run.py").read()
    st = src.index("parser = argparse.ArgumentParser"); en = src.index("args = parser.parse_args()")
    nd = {"argparse": argparse}
    exec("\n".join(l[4:] if l.startswith("    ") else l for l in src[st:en].split("\n")), nd)
    p = nd["parser"]
    a = p.parse_args(["--is_training", "1", "--model_id", "diag", "--model", "Q_S_Mamba", "--data", "custom"])
    for m in re.finditer(r"(\w+)=('([^']*)'|True|False|None|[\d.e\-]+)", ns):
        k, v = m.group(1), m.group(2)
        if v == "True": a.__dict__[k] = True
        elif v == "False": a.__dict__[k] = False
        elif v == "None": a.__dict__[k] = None
        else:
            try: a.__dict__[k] = int(v)
            except ValueError:
                try: a.__dict__[k] = float(v)
                except ValueError: a.__dict__[k] = v.strip("'")
    return a

from data_provider.data_factory import data_provider

def load(mid, mname, dev):
    cfg = cfg_from_log(f"logs/{mid}.log"); cfg.model = mname
    M = importlib.import_module(f"model.{mname}").Model
    model = M(cfg).to(dev)
    ck = glob.glob(f"checkpoints/{mid}*/checkpoint.pth")[0]
    sd = torch.load(ck, map_location=dev)
    sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd); model.eval()
    return cfg, model

def preds(cfg, model, dev):
    _, loader = data_provider(cfg, "test")
    P = []; T = []
    with torch.no_grad():
        for bx, by, bxm, bym in loader:
            bx = bx.float().to(dev); bm = bxm.float().to(dev)
            by = by.float().to(dev); bmy = bym.float().to(dev)
            dec = torch.zeros_like(by[:, -cfg.pred_len:]).float()
            dec = torch.cat([by[:, :cfg.label_len], dec], dim=1).to(dev)
            o = model(bx, bm, dec, bmy)
            o = o[0] if isinstance(o, (list, tuple)) else o
            P.append(o[:, -cfg.pred_len:, -1].cpu().numpy())  # OT=最后一列
            T.append(by[:, -cfg.pred_len:, -1].cpu().numpy())
    return np.concatenate(P), np.concatenate(T)

dev = "cuda:0"
# ---- ETTh1 96 ----
cfq, mq = load("qf2_ETTh1_96_q_base_s2024", "Q_S_Mamba", dev)
cfs, ms = load("bsl_ETTh1_96_s2024", "S_Mamba", dev)
Pq, Tr = preds(cfq, mq, dev)
Ps, _ = preds(cfs, ms, dev)
qmse = ((Pq - Tr) ** 2).mean(1); smse = ((Ps - Tr) ** 2).mean(1)
med_q = np.median(qmse); iqr_q = np.percentile(qmse, 75) - np.percentile(qmse, 25)
delta = smse - qmse
cand = np.where(delta > 0)[0]
if len(cand) == 0:
    print("ETTh1-96: 无 Δ>0 窗口，需按任务单兜底试 720"); sys.exit(2)
in_typ = np.abs(qmse[cand] - med_q) <= 0.5 * max(iqr_q, 1e-9)
pool = cand[in_typ]
pool = pool if len(pool) else cand
j = int(pool[np.argmax(delta[pool])])
imp = (1 - qmse[j] / smse[j]) * 100
print(f"ETTh1-96 窗 {j}: qcc={qmse[j]:.4f} sm={smse[j]:.4f} Δ={delta[j]:.4f} 提升={imp:.1f}% "
      f"(med_q={med_q:.4f}, 中位数附近候选 {len(pool)} 个)")
# 仅当提升足够明显才出图，否则提示换 720
if delta[j] <= 0.001:
    print("Δ 过小，建议 720 兜底")
    sys.exit(3)
tt = np.arange(Pq.shape[1]); L = Pq.shape[1]
fig, ax = plt.subplots(figsize=(6.0, 2.8))
ax.plot(tt, Tr[j], 'k-', lw=1.6, label="Ground truth (OT)")
ax.plot(tt, Ps[j], color="#4C72B0", lw=1.2, label="S-Mamba")
ax.plot(tt, Pq[j], color="#C44E52", lw=1.4, label="QCCK-M")
ax.set_xlabel("step"); ax.set_ylabel("OT"); ax.set_title("ETTh1, horizon 96")
ax.legend(fontsize=8)
ax.text(0.01, 0.03,
        f"window MSE   QCCK-M {qmse[j]:.4f} / S-Mamba {smse[j]:.4f}",
        transform=ax.transAxes, fontsize=8)
fig.tight_layout(); fig.savefig(f"{F}/prediction_case.png", dpi=200)
with open(f"{R}/case_info.txt", "w") as fh:
    fh.write(f"Dataset=ETTh1\nHorizon=96\nWindow_index={int(j)}\n")
    fh.write(f"S-Mamba_MSE={smse[j]:.6f}\nQCCK-M_MSE={qmse[j]:.6f}\n")
    fh.write(f"delta_MSE(SM-QCC)={delta[j]:.6f}\n")
    fh.write(f"improvement_pct={(qmse[j]/smse[j]-1)*100:.1f} (SM相对QCC)\n")
    fh.write(f"channel=OT\nselection_rule=Δ>0 且 QCC误差在全窗中位数±0.5IQR内、取Δ最大\n")
    fh.write(f"n_test_windows={len(qmse)}\nqcc_median_window_mse={med_q:.6f}\n")
print("saved", f"{F}/prediction_case.png", f"{R}/case_info.txt")
