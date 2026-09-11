# -*- coding: utf-8 -*-
"""预测定性案例(有优势的局部窗口)，96 与 720。
规则：逐测试窗算 OT 通道 MSE；取 Δ=MSE_SM−MSE_QCC>0 的窗，
在其中选 QCC 误差接近全窗 QCC 中位数(±0.5IQR)、Δ 最大者。
输出覆盖 paper/figs/ch5_pred_ETTh1.png 与 ch5_pred_ETTh1_720.png。
"""
import sys, re, torch, numpy as np, argparse, glob, importlib, os
os.chdir("/home/youjun/dataops_ws"); sys.path.insert(0, "/home/youjun/dataops_ws")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

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

def pick_adv_window(qmse, smse):
    med = np.median(qmse); iqr = np.percentile(qmse, 75) - np.percentile(qmse, 25)
    delta = smse - qmse
    cand = np.where(delta > 0)[0]
    if len(cand) == 0: return None
    in_typ = np.abs(qmse[cand] - med) <= 0.5 * max(iqr, 1e-9)
    pool = cand[in_typ] if in_typ.any() else cand
    return int(pool[np.argmax(delta[pool])]), delta

dev = "cuda:0"
jobs = [
    ("qf2_ETTh1_96_q_base_s2024", "bsl_ETTh1_96_s2024", 96, "ch5_pred_ETTh1.png"),
    ("k720_T02",                 "sm_ETTh1_720_rt_s2024", 720, "ch5_pred_ETTh1_720.png"),
]
for qmid, smid, H, out in jobs:
    cfq, mq = load(qmid, "Q_S_Mamba", dev)
    cfs, ms = load(smid, "S_Mamba", dev)
    Pq, Tr = preds(cfq, mq, dev); Ps, _ = preds(cfs, ms, dev)
    qmse = ((Pq - Tr) ** 2).mean(1); smse = ((Ps - Tr) ** 2).mean(1)
    res = pick_adv_window(qmse, smse)
    if res is None:
        print(f"H={H}: 无 Δ>0 窗口", flush=True); continue
    j, delta = res
    imp = (1 - qmse[j] / smse[j]) * 100
    tt = np.arange(Tr.shape[1])
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    ax.plot(tt, Tr[j], "k-", lw=1.6, label="Ground truth (OT)")
    ax.plot(tt, Ps[j], color="#4C72B0", lw=1.2, label="S-Mamba")
    ax.plot(tt, Pq[j], color="#C44E52", lw=1.4, label="QCCK-M")
    ax.set_xlabel("step"); ax.set_ylabel("OT")
    ax.set_title(f"ETTh1, horizon {H}")
    ax.legend(fontsize=8)
    ax.text(0.01, 0.03, f"window MSE  QCCK-M {qmse[j]:.4f} / S-Mamba {smse[j]:.4f}",
            transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"/home/youjun/paper/figs/{out}", dpi=200)
    plt.close(fig)
    print(f"H={H}: 窗{j} qcc={qmse[j]:.4f} sm={smse[j]:.4f} 提升={imp:.1f}% -> {out}", flush=True)
