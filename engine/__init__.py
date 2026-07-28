"""训练 / 评估 / 频谱分析。"""
from .train import train_one_epoch, evaluate
from .evaluate import paired_t_test, metric_table, format_results
from .spectrum import (
    kernel_spectrum_over_vars,
    plot_kernel_spectrum_over_vars,
    fourier_spectrum_diff,
    plot_kernel_spectrum,
)

__all__ = [
    "train_one_epoch",
    "evaluate",
    "paired_t_test",
    "metric_table",
    "format_results",
    "kernel_spectrum_over_vars",
    "plot_kernel_spectrum_over_vars",
    "fourier_spectrum_diff",
    "plot_kernel_spectrum",
]
