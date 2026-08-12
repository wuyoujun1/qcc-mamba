#!/usr/bin/env python3
"""量子混合主干结果汇总（只对比新架构变体：plain / qmix / qmix_sin / qmix_soft / qmix_head / qmix_full）。

用法:
  python summarize_qmix.py                    # 全部已完成的格子
  python summarize_qmix.py etth1 weather      # 只看某些数据集
"""
import os
import re
import sys

# 变体单一事实来源：从 run_qmix.py 导入，避免与运行器不同步
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_qmix import VARIANTS as RUN_VARIANTS

DATASETS = sys.argv[1:] if len(sys.argv) > 1 else ["etth1", "weather", "chinaaqi",
                                                   "electricity", "pems_bay"]
VARIANTS = list(RUN_VARIANTS.keys())


def variant_mean(ds, L, v):
    """返回 (均值, seed 数)。1-seed 快速验证模式下单 seed 也返回（表格会标注）。"""
    vals = []
    for s in [42, 2024]:
        try:
            txt = open(f"logs/qmix/qm_{v}_{ds}_{L}_{s}.log").read()
            ms = re.findall(r"Test MSE \(normalized\): ([\d.]+)", txt)
            if ms:
                vals.append(float(ms[-1]))
        except FileNotFoundError:
            pass
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


hdr = f"{'cell':14s} " + " ".join(f"{v:>9s}" for v in VARIANTS) + "   最优"
print(hdr)
all_done = True
for ds in DATASETS:
    for L in [96, 192, 336, 720]:
        s = {v: variant_mean(ds, L, v) for v in VARIANTS}
        vals = {v: s[v][0] for v in VARIANTS}
        counts = {v: s[v][1] for v in VARIANTS}
        if all(v is not None for v in vals.values()):
            best = min(vals, key=vals.get)
            row = f"{ds}:{L:<8d} " + " ".join(f"{vals[v]:>9.4f}" for v in VARIANTS)
            n1 = sum(1 for v in VARIANTS if counts[v] == 1)
            print(row + f"  {best}" + ("  *" if n1 == len(VARIANTS) else ""))
        else:
            fmt = lambda v: f"{vals[v]:>9.4f}" if vals[v] is not None else f"{'-':>9s}"
            row = f"{ds}:{L:<8d} " + " ".join(fmt(v) for v in VARIANTS)
            print(row + "  (未齐)")
            all_done = False
if all_done:
    print("\n(所有格子的变体均 2-seed 完成)")
else:
    print("\n(未标注的行 = 部分变体缺失;行尾 * = 全 1-seed 快速验证)")
