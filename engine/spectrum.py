"""核矩阵频谱分析（诊断工具）。

对应文档：experiment-design.md §十 E7

注意：QCCBlock 输出的核矩阵 K 形状为 (B, V, V)，描述的是**变量之间**的相似度，
而非时间轴上的周期。因此以下函数都是对**变量维度**做谱分析，属于诊断性工具，
不能直接用来论证"24h/168h/8760h 电力周期"。电力周期卖点由显式周期特征
（PeriodicTimeFeatures / fourier_high_freq）支撑。
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch


def kernel_spectrum_over_vars(K: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """对核矩阵 K (B, V, V) 的变量维度做 FFT，返回频谱幅度。

    Args:
        K: (B, V, V) 核矩阵（实数对称）。
        dt: 采样间隔（变量轴上的虚拟间隔，仅用于频率单位）。

    Returns:
        spectrum: (B, V, V//2+1) 频谱幅度。

    Note:
        变量轴 FFT 是诊断工具，用于观察 K 在变量排序上的平滑性/结构性，
        与电力时间周期无关。
    """
    K_np = K.detach().cpu().numpy()
    spectrum = np.abs(np.fft.rfft(K_np, axis=-1))  # (B, V, V//2+1)
    return torch.from_numpy(spectrum).float()


def plot_kernel_spectrum_over_vars(
    K: torch.Tensor,
    save_path: Optional[str] = None,
):
    """绘制核矩阵在变量维度上的平均频谱（诊断图）。

    Args:
        K: (B, V, V) 核矩阵。
        save_path: 保存路径。
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_kernel_spectrum_over_vars")

    K_np = K.detach().cpu().numpy()
    B, V, _ = K_np.shape

    # 对 batch 和第二个维度取平均
    spec = np.abs(np.fft.rfft(K_np.mean(axis=1), axis=-1)).mean(axis=0)  # (V//2+1,)
    freqs = np.fft.rfftfreq(V, d=1.0)

    plt.figure(figsize=(10, 4))
    plt.plot(freqs[1:], spec[1:])
    plt.xlabel("Frequency (1/sample, over variable axis)")
    plt.ylabel("Amplitude")
    plt.title("Kernel Matrix Spectrum over Variable Axis (diagnostic)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# 保留旧名作为兼容别名（带 Deprecation 语义）
def fourier_spectrum_diff(K: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
    """Deprecated: use kernel_spectrum_over_vars instead."""
    return kernel_spectrum_over_vars(K, dt)


def plot_kernel_spectrum(
    K: torch.Tensor,
    periods: Optional[list] = None,
    save_path: Optional[str] = None,
):
    """Deprecated: use plot_kernel_spectrum_over_vars instead."""
    return plot_kernel_spectrum_over_vars(K, save_path)


__all__ = [
    "kernel_spectrum_over_vars",
    "plot_kernel_spectrum_over_vars",
    "fourier_spectrum_diff",  # 兼容旧名
    "plot_kernel_spectrum",   # 兼容旧名
]
