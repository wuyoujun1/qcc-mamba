"""评估指标与统计显著性。

对应文档：experiment-design.md §九
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import scipy.stats as stats
import torch


def paired_t_test(
    baseline: Sequence[float],
    candidate: Sequence[float],
    alternative: str = "less",
) -> Dict[str, float]:
    """配对 t 检验：candidate 是否显著低于 baseline（MSE 更低）。

    Args:
        baseline: 基线模型在各 seed 上的 MSE。
        candidate: 候选模型在各 seed 上的 MSE。
        alternative: "less" 表示 candidate < baseline（单侧）。

    Returns:
        {t_stat, p_value, mean_diff}
    """
    baseline = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    diff = baseline - candidate
    t_stat, p_value = stats.ttest_rel(baseline, candidate, alternative=alternative)
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
    }


def metric_table(
    results: Dict[str, List[float]],
    baseline_name: str = "none",
) -> str:
    """把多组实验结果格式化为 Markdown 表格。

    results: {method_name: [mse_seed1, mse_seed2, ...]}
    """
    lines = []
    lines.append("| Method | MSE mean ± std | vs baseline (ΔMSE) | p-value |")
    lines.append("|--------|----------------|--------------------|---------|")

    baseline = np.asarray(results.get(baseline_name, []), dtype=np.float64)
    for name, values in results.items():
        arr = np.asarray(values, dtype=np.float64)
        mean = arr.mean()
        std = arr.std(ddof=1)
        if name == baseline_name:
            lines.append(f"| {name} | {mean:.6f} ± {std:.6f} | — | — |")
        else:
            if len(baseline) > 0 and len(baseline) == len(arr):
                test = paired_t_test(baseline, arr, alternative="less")
                delta = test["mean_diff"]
                p = test["p_value"]
                lines.append(
                    f"| {name} | {mean:.6f} ± {std:.6f} | {delta:+.6f} | {p:.4f} |"
                )
            else:
                lines.append(f"| {name} | {mean:.6f} ± {std:.6f} | N/A | N/A |")
    return "\n".join(lines)


def format_results(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """返回结构化结果，便于后续绘图。"""
    out = {}
    for name, values in results.items():
        arr = np.asarray(values, dtype=np.float64)
        out[name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
    return out


__all__ = ["paired_t_test", "metric_table", "format_results"]
