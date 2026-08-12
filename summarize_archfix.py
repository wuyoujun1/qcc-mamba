#!/usr/bin/env python3
"""波次 2 结果 vs 三臂对比表(baseline_smamba / h_only / s_only)。

用法:
  python summarize_archfix.py                # 全部已完成的格子
  python summarize_archfix.py etth1 weather  # 只看某些数据集
"""
import pandas as pd
import re
import sys

df = pd.read_csv("results/summary.csv")
df = df[(df.status == "ok") & (df.phase.str.startswith("p1f"))]

datasets = sys.argv[1:] if len(sys.argv) > 1 else ["etth1", "electricity", "weather",
                                                   "chinaaqi", "metr_la", "pems_bay"]
variants = ["combo", "hp_channel"]

def arm_mean(ds, L, cfg):
    m = df[(df.dataset == ds) & (df.L == L) & (df.config == cfg)]["mse_norm"].mean()
    return m if pd.notna(m) else float("nan")

def variant_mean(ds, L, v):
    vals = []
    for s in [42, 2024]:
        try:
            txt = open(f"logs/archfix/arch_{v}_{ds}_{L}_{s}.log").read()
            ms = re.findall(r"Test MSE \(normalized\): ([\d.]+)", txt)
            if ms:
                vals.append(float(ms[-1]))
        except FileNotFoundError:
            pass
    return (sum(vals) / len(vals)) if len(vals) == 2 else float("nan")

print(f"{'cell':14s} {'h_only':>8s} {'s_only':>8s} {'baseline':>9s} | "
      f"{'combo':>8s} {'hp_ch':>8s}  最优(需全齐)")
rows = []
for ds in datasets:
    for L in [96, 192, 336, 720]:
        scores = {"h_only": arm_mean(ds, L, "ablation_h_only"),
                  "s_only": arm_mean(ds, L, "ablation_s_only"),
                  "baseline": arm_mean(ds, L, "baseline_smamba")}
        for v in variants:
            scores[v] = variant_mean(ds, L, v)
        if all(pd.notna(x) for x in scores.values()):
            best = min(scores, key=scores.get)
            rows.append((ds, L, scores, best))
            print(f"{ds}:{L:<8d} {scores['h_only']:>8.4f} {scores['s_only']:>8.4f} {scores['baseline']:>9.4f} | "
                  f"{scores['combo']:>8.4f} {scores['hp_channel']:>8.4f}  {best}")
        else:
            # 变体未跑齐也显示
            print(f"{ds}:{L:<8d} {scores['h_only']:>8.4f} {scores['s_only']:>8.4f} {scores['baseline']:>9.4f} | "
                  f"{scores['combo'] if pd.notna(scores['combo']) else '-':>8} "
                  f"{scores['hp_channel'] if pd.notna(scores['hp_channel']) else '-':>8}  (未齐)")

print(f"\n共 {len(rows)} 个格子完整; 最优计数: "
      + ", ".join(f"{k}={sum(1 for r in rows if r[3] == k)}" for k in
                  ["combo", "hp_channel", "h_only", "s_only", "baseline"]))
