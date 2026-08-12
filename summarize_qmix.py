#!/usr/bin/env python3
"""量子混合主干结果汇总（只对比新架构变体：plain / qmix / qmix_sin / qmix_soft / qmix_head / qmix_full）。

用法:
  python summarize_qmix.py                    # 全部已完成的格子
  python summarize_qmix.py etth1 weather      # 只看某些数据集
"""
import re
import sys

DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["etth1", "weather", "chinaaqi",
                                                   "electricity", "pems_bay"]
VARIANTS = ["plain", "qmix", "qmix_sin", "qmix_soft", "qmix_head", "qmix_full"]


def variant_mean(ds, L, v):
    vals = []
    for s in [42, 2024]:
        try:
            txt = open(f"logs/qmix/qm_{v}_{ds}_{L}_{s}.log").read()
            ms = re.findall(r"Test MSE \(normalized\): ([\d.]+)", txt)
            if ms:
                vals.append(float(ms[-1]))
        except FileNotFoundError:
            pass
    return (sum(vals) / len(vals)) if len(vals) == 2 else None


hdr = f"{'cell':14s} " + " ".join(f"{v:>9s}" for v in VARIANTS) + "   最优"
print(hdr)
for ds in DATASETS:
    for L in [96, 192, 336, 720]:
        s = {v: variant_mean(ds, L, v) for v in VARIANTS}
        if all(x is not None for x in s.values()):
            best = min(s, key=s.get)
            row = f"{ds}:{L:<8d} " + " ".join(f"{s[v]:>9.4f}" for v in VARIANTS)
            print(row + f"  {best}")
        else:
            fmt = lambda v: f"{s[v]:>9.4f}" if s[v] is not None else f"{'-':>9s}"
            row = f"{ds}:{L:<8d} " + " ".join(fmt(v) for v in VARIANTS)
            print(row + "  (未齐)")
