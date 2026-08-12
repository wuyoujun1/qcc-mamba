"""对比修复版 dual 与既有 baseline/h_only/s_only（ECL×96，2 seed 均值）。"""
import re, glob, os, sys

def parse_log(path):
    txt = open(path).read()
    m = re.findall(r"Test MSE \(normalized\): ([\d.e+-]+)", txt)
    mm = re.findall(r"Test MAE \(normalized\): ([\d.e+-]+)", txt)
    km = re.search(r"K stats: diag_mean=([\d.]+) offdiag_mean=([\d.]+) offdiag_std=([\d.]+)", txt)
    am = re.findall(r"α=([\d.]+)", txt)
    return {
        "mse_norm": float(m[-1]) if m else None,
        "mae_norm": float(mm[-1]) if mm else None,
        "k_off": float(km.group(2)) if km else None,
        "k_off_std": float(km.group(3)) if km else None,
        "alpha_end": float(am[-1]) if am else None,
    }

# 既有结果（p1f 批次，2 seed 均值）
rows = [r for r in __import__("csv").DictReader(open("results/summary.csv")) if r["status"] == "ok"]
def old_mean(cfg):
    vals = [float(r["mse_norm"]) for r in rows if r["dataset"] == "electricity" and r["config"] == cfg and r["L"] == "96"]
    return sum(vals) / len(vals), len(vals)
print(f"{'config':22s} {'mse_norm':>9s} {'mae_norm':>9s} {'K_off':>7s} {'K_std':>7s} {'alpha':>7s}")
for cfg, label in [("baseline_smamba", "baseline（纯S-Mamba）"), ("phase1", "dual（现状，旧核）"),
                   ("ablation_h_only", "h_only"), ("ablation_s_only", "s_only")]:
    v, n = old_mean(cfg)
    print(f"{label:22s} {v:9.4f} ({n} seed)")

for lg in sorted(glob.glob("logs/fix_*.log")):
    p = parse_log(lg)
    name = os.path.basename(lg).replace(".log", "")
    print(f"{name:22s} {p['mse_norm'] if p['mse_norm'] is not None else '--':>9} {p['mae_norm'] if p['mae_norm'] else '--':>9} "
          f"{p['k_off'] if p['k_off'] else '--':>7} {p['k_off_std'] if p['k_off_std'] else '--':>7} {p['alpha_end'] if p['alpha_end'] else '--':>7}")
