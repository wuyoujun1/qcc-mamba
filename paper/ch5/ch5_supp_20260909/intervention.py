# -*- coding: utf-8 -*-
"""干预实验：把选定变量对的耦合权重在推理时置零，测测试 MSE 变化。
干预对象 = 模型真正使用的消息权重 K_n（K_n *= mask）。用法：
  python intervention.py <qcc_log_id> <f_npz> <out_tag>
"""
import os, sys, re, torch, numpy as np, argparse, glob, importlib
_WS = os.environ.get("QCC_WS", "/home/youjun/dataops_ws")  # 工作副本路径（含 run.py/logs/checkpoints）
os.chdir(_WS); sys.path.insert(0, "/home/youjun/dataops_ws")
import qcc.quantum_mix as qm
from torch.utils.data import DataLoader

SRC = open("run.py").read(); ST = SRC.index("parser = argparse.ArgumentParser"); EN = SRC.index("args = parser.parse_args()")
def cfg_from_log(log):
    txt = open(log, encoding="utf-8", errors="ignore").read()
    ns = re.search(r"Namespace\((.+?)\)\n", txt, re.S).group(1)
    nd = {"argparse": argparse}; exec("\n".join(l[4:] if l.startswith("    ") else l for l in SRC[ST:EN].split("\n")), nd)
    p = nd["parser"]
    a = p.parse_args(["--is_training", "1", "--model_id", "d", "--model", "Q_S_Mamba", "--data", "custom"])
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
def load(mid, dev):
    cfg = cfg_from_log(f"logs/{mid}.log"); cfg.model = "Q_S_Mamba"
    M = importlib.import_module("model.Q_S_Mamba").Model
    model = M(cfg).to(dev)
    ck = glob.glob(f"checkpoints/{mid}*/checkpoint.pth")[0]
    sd = torch.load(ck, map_location=dev); sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd); model.eval()
    return cfg, model

def set_mask(model, mask, dev):
    for mod in model.modules():
        if isinstance(mod, qm.QuantumMixLayer):
            mod.intervention_mask = None if mask is None else torch.as_tensor(mask, dtype=torch.float32, device=dev)

@torch.no_grad()
def eval_mse(cfg, model, loader, dev, mask=None):
    set_mask(model, mask, dev)
    tot = 0.0; n = 0
    MAXB = int(os.environ.get("INT_MAXB", "0"))
    for bi, (bx, by, bxm, bym) in enumerate(loader):
        if MAXB and bi >= MAXB: break
        bx = bx.float().to(dev); bm = bxm.float().to(dev); by = by.float().to(dev); bmy = bym.float().to(dev)
        dec = torch.cat([by[:, :cfg.label_len], torch.zeros_like(by[:, -cfg.pred_len:])], dim=1).to(dev)
        o = model(bx, bm, dec, bmy)
        o = o[0] if isinstance(o, (list, tuple)) else o
        err = (o[:, -cfg.pred_len:, :] - by[:, -cfg.pred_len:, :]) ** 2
        tot += float(err.sum()); n += err.numel()
    return tot / max(n, 1)

def pair_mask(V, pairs):
    m = np.ones((V, V), dtype=np.float32)
    for i, j in pairs:
        m[i, j] = 0.0; m[j, i] = 0.0
    return m

if __name__ == "__main__":
    qmid, fnpz, tag = sys.argv[1], sys.argv[2], sys.argv[3]
    dev = "cuda:0" if len(sys.argv) < 5 else f"cuda:{sys.argv[4]}"
    cfg, model = load(qmid, dev)
    _, loader0 = data_provider(cfg, "test")
    ds = loader0.dataset
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
    if fnpz == "-":
        # 用模型自身的平均 K（前 20 批）来排名变量对，避免依赖外部文件
        Kacc = None; nb = 0
        for bx, by, bxm, bym in loader:
            bx = bx.float().to(dev); bm = bxm.float().to(dev); by = by.float().to(dev); bmy = bym.float().to(dev)
            dec = torch.cat([by[:, :cfg.label_len], torch.zeros_like(by[:, -cfg.pred_len:])], dim=1).to(dev)
            _ = model(bx, bm, dec, bmy)
            K = model._last_K
            if K is None: continue
            K = K.detach().float().mean(0).cpu().numpy()
            Kacc = K if Kacc is None else Kacc + K; nb += 1
            if nb >= 20: break
        F = Kacc / nb; V = F.shape[0]
        np.savez(f"/home/youjun/paper/ch5_supp_20260909/Kmean_{tag}.npz", F=F)
    else:
        F = np.load(fnpz)["F"]; V = F.shape[0]
    ii = np.triu_indices(V, 1); vals = F[ii]
    order = np.argsort(-vals)
    top = [(ii[0][k], ii[1][k]) for k in order[:3]]
    bot = [(ii[0][k], ii[1][k]) for k in order[-3:]]
    print(f"[{tag}] V={V} top3={top} bottom3={bot}", flush=True)
    base = eval_mse(cfg, model, loader, dev, None)
    print(f"[{tag}] baseline MSE={base:.6f}", flush=True)
    # 对照①：屏蔽所有非对角（=关掉跨变量混合）
    alloff = pair_mask(V, [(a, b) for a in range(V) for b in range(a + 1, V)])
    m = eval_mse(cfg, model, loader, dev, alloff)
    print(f"[{tag}] no_offdiag MSE={m:.6f}  Δ={(m-base)/base*100:+.2f}%", flush=True)
    # 对照②：直接关掉门控 γ（H'=H，完全旁路混合）
    gates = []
    for mod in model.modules():
        if isinstance(mod, qm.QuantumMixLayer):
            gates.append(mod.gate); mod.gate = False
    m = eval_mse(cfg, model, loader, dev, None)
    for mod, g in zip([x for x in model.modules() if isinstance(x, qm.QuantumMixLayer)], gates):
        mod.gate = g
    print(f"[{tag}] gate_off MSE={m:.6f}  Δ={(m-base)/base*100:+.2f}%", flush=True)
    res = {}
    for name, pairs in [("top3", top), ("bottom3", bot)]:
        m = eval_mse(cfg, model, loader, dev, pair_mask(V, pairs))
        res[name] = (m - base) / base * 100
        print(f"[{tag}] {name} MSE={m:.6f}  Δ={res[name]:+.2f}%", flush=True)
    rng = np.random.default_rng(0); rd = []
    for s in range(5):
        idx = rng.choice(len(vals), 3, replace=False)
        pairs = [(ii[0][k], ii[1][k]) for k in idx]
        m = eval_mse(cfg, model, loader, dev, pair_mask(V, pairs))
        rd.append((m - base) / base * 100)
    res["random3"] = float(np.mean(rd))
    print(f"[{tag}] random3 Δ={res['random3']:+.2f}% (5 seeds {['%.2f'%x for x in rd]})", flush=True)
    per = []
    for k in order:
        i, j = ii[0][k], ii[1][k]
        m = eval_mse(cfg, model, loader, dev, pair_mask(V, [(i, j)]))
        per.append((float(F[i, j]), (m - base) / base * 100))
        print(f"[{tag}] pair({i},{j}) f={F[i,j]:.4f} Δ={(m-base)/base*100:+.3f}%", flush=True)
    per = np.array(per)
    if len(per) > 2:
        cc = np.corrcoef(per[:, 0], per[:, 1])[0, 1]
        print(f"[{tag}] corr(f, ΔMSE) = {cc:+.3f}  (n={len(per)} pairs)", flush=True)
    np.savez(f"/home/youjun/paper/ch5_supp_20260909/intervention_{tag}.npz",
             base=base, top3_delta=res.get("top3"), bottom3_delta=res.get("bottom3"),
             random3_delta=res.get("random3"), per_pair=per)
    print(f"[{tag}] DONE", flush=True)
