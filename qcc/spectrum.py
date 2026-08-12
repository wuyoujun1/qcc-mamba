"""频谱特征提取：rfft → 三层对齐 → 重采样 → S (B, V, 2M)。

全 detach 确定性函数，不返回梯度。

三层对齐：
1. 时间轴对齐：δ̂_v = 互相关(vs 共识时钟) → φ̃ = φ − 2πf·δ̂_v
2. 频率轴对齐：f̂_peak = argmax A → f̃ = f/f̂_peak → 重采样 M 点
3. 幅度归一化（消融档）：Ã = A/A_max

对应文档：IDEA_DualAE_QCC.md §2
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SpectrumFeature(nn.Module):
    """频谱特征提取（三层对齐，全 detach 确定性函数）。

    Args:
        M: 重采样点数（默认 32）。
        sample_range: 采样区间，"0_2" 表示 [0,2]，"0_1" 表示 [0,1]。
        amp_normalize: 是否做幅度归一化（消融开关，默认 False）。
        time_align: 是否做时间轴对齐（默认 True）。
        freq_align: 是否做频率轴对齐（默认 True）。
    """

    def __init__(
        self,
        M: int = 32,
        sample_range: str = "0_2",
        amp_normalize: bool = False,
        time_align: bool = True,
        freq_align: bool = True,
    ):
        super().__init__()
        self.M = M
        self.sample_range = sample_range
        self.amp_normalize = amp_normalize
        self.time_align = time_align
        self.freq_align = freq_align

        # 解析采样区间
        if sample_range == "0_2":
            self.range_start, self.range_end = 0.0, 2.0
        elif sample_range == "0_1":
            self.range_start, self.range_end = 0.0, 1.0
        else:
            raise ValueError(f"sample_range must be '0_2' or '0_1', got {sample_range}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, V) → S: (B, V, 2M) float32，全 detach。

        流程：
        1. rfft → 幅度谱 A, 相位谱 φ
        2. 时间轴对齐（可选）：δ̂_v = 互相关(vs 共识时钟) → φ̃ = φ + 2πf·δ̂_v
        3. 频率轴对齐（可选）：f̂_peak = argmax A → f̃ = f/f̂_peak → 重采样 M 点
        4. 幅度归一化（消融，可选）：Ã = A/A_max
        5. 拼接 S = [Ã; φ̃] ∈ R^{2M}
        """
        B, L, V = x.shape
        device = x.device

        # 1. rfft → 幅度谱 A, 相位谱 φ
        # rfft 输出 (B, V, L//2+1) 复数
        x_perm = x.permute(0, 2, 1)  # (B, V, L)
        X = torch.fft.rfft(x_perm, dim=-1)  # (B, V, L//2+1) complex
        A = X.abs()  # (B, V, L//2+1) 幅度谱
        phi = X.angle()  # (B, V, L//2+1) 相位谱

        # 频率格点（cycles/sample 归一化，范围 [0, 0.5]）
        # k/L 对应第 k 个 bin 的频率（cycles/sample）
        freq_bins = torch.arange(A.shape[-1], device=device).float() / L  # (L//2+1,)

        # 2. 时间轴对齐（可选）
        if self.time_align:
            delta_hat = self._estimate_time_shift(x)  # (B, V)
            phi = self._align_phase(phi, delta_hat, freq_bins)  # (B, V, L//2+1)

        # 3. 频率轴对齐（可选）
        if self.freq_align:
            f_peak = self._find_peak_frequency(A)  # (B, V)
            A, phi, freq_bins_aligned = self._align_frequency(
                A, phi, freq_bins, f_peak
            )  # (B, V, L//2+1)
            # 使用用户指定的采样区间
            range_start, range_end = self.range_start, self.range_end
        else:
            freq_bins_aligned = freq_bins
            # 无频率对齐时，强制限制在 [0, 0.5]（避免超出实际频率范围）
            range_start, range_end = 0.0, 0.5

        # 4. 幅度归一化（消融，可选）
        if self.amp_normalize:
            A = self._normalize_amplitude(A)  # (B, V, L//2+1)

        # 5. 重采样 M 点
        A_resampled = self._resample_spectrum(A, freq_bins_aligned, range_start, range_end)  # (B, V, M)
        phi_resampled = self._resample_spectrum(phi, freq_bins_aligned, range_start, range_end)  # (B, V, M)

        # 6. 拼接 S = [A; φ] ∈ R^{2M}
        S = torch.cat([A_resampled, phi_resampled], dim=-1)  # (B, V, 2M)

        # 全 detach
        return S.detach()

    def _estimate_time_shift(self, x: torch.Tensor) -> torch.Tensor:
        """估计每个变量相对共识时钟的时移 δ̂_v。

        共识时钟 = 样本内留一均值：ref(t) = (1/(V-1)) Σ_{j≠v} x_j(t)
        δ̂_v = argmax_τ corr(x_v(t), ref(t))，τ ∈ [-MAX_LAG, MAX_LAG]

        Args:
            x: (B, L, V)

        Returns:
            delta_hat: (B, V) 归一化时移（单位：样本数 / L）
        """
        B, L, V = x.shape
        device = x.device

        # 最大搜索滞后（默认 L/4，避免边界效应）
        MAX_LAG = L // 4

        # 计算留一均值（共识时钟）
        # ref[b, t] = (1/(V-1)) Σ_{j≠v} x[b, t, j]
        x_sum = x.sum(dim=-1, keepdim=True)  # (B, L, 1)
        ref = (x_sum - x) / (V - 1)  # (B, L, V) 留一均值

        # 归一化互相关（频域加速）
        # corr[b, v, τ] = Σ_t x[b, t, v] * ref[b, t+τ, v]
        # 用 FFT 加速：corr = IFFT(FFT(x) * conj(FFT(ref)))
        x_fft = torch.fft.rfft(x, dim=1, n=2*L)  # (B, 2L, V)
        ref_fft = torch.fft.rfft(ref, dim=1, n=2*L)  # (B, 2L, V)
        corr = torch.fft.irfft(x_fft * ref_fft.conj(), dim=1, n=2*L)  # (B, 2L, V)

        # 双向搜索 [-MAX_LAG, MAX_LAG]
        # corr 的前 L 个元素对应 τ ∈ [0, L-1]
        # corr 的后 L 个元素对应 τ ∈ [-L, -1]（循环移位）
        # 我们需要提取 [-MAX_LAG, MAX_LAG] 范围
        corr_pos = corr[:, :MAX_LAG+1, :]  # τ ∈ [0, MAX_LAG]
        corr_neg = corr[:, -MAX_LAG:, :]   # τ ∈ [-MAX_LAG, -1]
        corr_bidir = torch.cat([corr_neg, corr_pos], dim=1)  # (B, 2*MAX_LAG+1, V)

        # 取峰值滞后
        tau_bidir = corr_bidir.argmax(dim=1)  # (B, V) 在 [-MAX_LAG, MAX_LAG] 中的索引
        tau_hat = tau_bidir - MAX_LAG  # 转换为实际滞后值 [-MAX_LAG, MAX_LAG]

        # 归一化到 [-1, 1]（相对于 L）
        delta_hat = tau_hat.float() / L  # (B, V)

        return delta_hat

    def _align_phase(
        self,
        phi: torch.Tensor,
        delta_hat: torch.Tensor,
        freq_bins: torch.Tensor,
    ) -> torch.Tensor:
        """相位去趋势：φ̃ = φ + 2πf·δ̂_v。

        时移定理：x(t-δ) ⟷ X(f)·e^{-i2πfδ}
        如果变量 v 比共识时钟晚到 δ̂_v，其相位谱会有额外线性项 -2πfδ̂_v
        为了对齐，需要加上这个项：φ̃ = φ + 2πfδ̂_v

        Args:
            phi: (B, V, L//2+1) 原始相位谱
            delta_hat: (B, V) 归一化时移（相对于 L，范围 [-1, 1]）
            freq_bins: (L//2+1,) 频率格点（cycles/sample，范围 [0, 0.5]）

        Returns:
            phi_aligned: (B, V, L//2+1) 对齐后相位谱
        """
        # freq_bins: (F,) → (1, 1, F)  cycles/sample
        # delta_hat: (B, V) → (B, V, 1)  归一化时移
        freq = freq_bins.unsqueeze(0).unsqueeze(0)  # (1, 1, F)
        delta = delta_hat.unsqueeze(-1)  # (B, V, 1)

        # 相位去趋势：φ̃ = φ + 2πfδ̂_v
        # freq 是 cycles/sample（即 k/L），delta 是归一化时移（相对于 L）
        # 实际时移 τ = delta * L（采样点数）
        # 相位补偿 = 2π * freq * τ = 2π * (k/L) * (delta * L) = 2π * k * delta
        # 其中 k = freq * L，所以 = 2π * freq * delta * L
        L = (phi.shape[-1] - 1) * 2  # 从频率 bin 数恢复原始序列长度
        phi_aligned = phi + 2 * math.pi * freq * delta * L  # (B, V, F)

        # 包裹到 [-π, π]
        phi_aligned = (phi_aligned + math.pi) % (2 * math.pi) - math.pi

        return phi_aligned

    def _find_peak_frequency(self, A: torch.Tensor) -> torch.Tensor:
        """找幅度谱主峰频率 f̂_peak。

        Args:
            A: (B, V, L//2+1) 幅度谱

        Returns:
            f_peak: (B, V) 主峰频率（cycles/sample，范围 (0, 0.5]）
        """
        L = (A.shape[-1] - 1) * 2  # 原始序列长度
        # 忽略 DC 分量（k=0）和 k=1（防去趋势后残余低频伪峰）
        # 与前提验证脚本口径一致
        A_skip_low = A[:, :, 2:]  # (B, V, L//2-1)
        peak_idx = A_skip_low.argmax(dim=-1)  # (B, V)
        # peak_idx 对应第 (peak_idx+2) 个 bin，频率 = (peak_idx+2)/L cycles/sample
        f_peak = (peak_idx + 2).float() / L  # (B, V) cycles/sample
        return f_peak

    def _align_frequency(
        self,
        A: torch.Tensor,
        phi: torch.Tensor,
        freq_bins: torch.Tensor,
        f_peak: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """频率轴对齐：f̃ = f / f̂_peak。

        Args:
            A: (B, V, L//2+1) 幅度谱
            phi: (B, V, L//2+1) 相位谱
            freq_bins: (L//2+1,) 原始频率格点
            f_peak: (B, V) 主峰频率

        Returns:
            A_aligned: (B, V, L//2+1) 对齐后幅度谱（在 f̃ 格点上）
            phi_aligned: (B, V, L//2+1) 对齐后相位谱
            freq_tilde: (B, V, L//2+1) 归一化频率格点 f̃
        """
        B, V, F = A.shape
        device = A.device

        # freq_bins: (F,) → (1, 1, F)
        freq = freq_bins.unsqueeze(0).unsqueeze(0)  # (1, 1, F)
        # f_peak: (B, V) → (B, V, 1)
        fp = f_peak.unsqueeze(-1)  # (B, V, 1)

        # 归一化频率 f̃ = f / f̂_peak
        # 防止 f_peak = 0 除零
        fp_safe = fp.clamp(min=1e-6)
        freq_tilde = freq / fp_safe  # (B, V, F)

        # 注意：这里不实际重排数据，只是改变频率坐标
        # 后续重采样会在 f̃ 坐标上取 M 个均匀点
        return A, phi, freq_tilde

    def _normalize_amplitude(self, A: torch.Tensor) -> torch.Tensor:
        """幅度归一化：Ã = A / A_max（每条变量独立）。

        Args:
            A: (B, V, L//2+1) 幅度谱

        Returns:
            A_norm: (B, V, L//2+1) 归一化幅度谱
        """
        A_max = A.max(dim=-1, keepdim=True).values  # (B, V, 1)
        A_max_safe = A_max.clamp(min=1e-6)
        return A / A_max_safe

    def _resample_spectrum(
        self,
        values: torch.Tensor,
        freq_tilde: torch.Tensor,
        range_start: float = None,
        range_end: float = None,
    ) -> torch.Tensor:
        """在 f̃ 轴上重采样 M 个均匀点（向量化实现）。

        使用 grid_sample 做双线性插值，避免 Python 循环。

        Args:
            values: (B, V, F) 幅度谱或相位谱
            freq_tilde: (B, V, F) 归一化频率格点 f̃
            range_start: 采样区间起点（默认使用 self.range_start）
            range_end: 采样区间终点（默认使用 self.range_end）

        Returns:
            resampled: (B, V, M) 重采样后的值
        """
        B, V, F = values.shape
        device = values.device

        # 兼容无频率对齐时的 1-D 频率坐标（freq_bins），广播到 (B, V, F)
        if freq_tilde.dim() == 1:
            freq_tilde = freq_tilde.unsqueeze(0).unsqueeze(0).expand(B, V, F)

        # 使用传入的参数或默认值
        if range_start is None:
            range_start = self.range_start
        if range_end is None:
            range_end = self.range_end

        # 目标采样点（在 f̃ 坐标上均匀）
        target_grid = torch.linspace(
            range_start, range_end, self.M, device=device
        )  # (M,)

        # 对 freq_tilde 按频率排序（每个样本、每个变量独立排序）
        # 排序后 freq_tilde 单调递增，便于插值
        sorted_idx = freq_tilde.argsort(dim=-1)  # (B, V, F)
        freq_sorted = torch.gather(freq_tilde, dim=-1, index=sorted_idx)  # (B, V, F)
        values_sorted = torch.gather(values, dim=-1, index=sorted_idx)  # (B, V, F)

        # 对每个目标采样点，找到左右邻居并插值
        # searchsorted 要求 target 前 N-1 维与 boundaries 匹配，广播到 (B, V, M)
        target = target_grid.view(1, 1, self.M).expand(B, V, self.M)  # (B, V, M)

        # 找到每个 target 在 freq_sorted 中的位置（左侧索引）
        # searchsorted 要求 freq_sorted 单调递增
        # freq_sorted: (B, V, F), target: (B, V, M)
        # 输出: (B, V, M) 左侧索引
        left_idx = torch.searchsorted(freq_sorted, target, right=False) - 1
        left_idx = left_idx.clamp(min=0, max=F - 2)  # 防止越界

        right_idx = left_idx + 1  # (B, V, M)

        #  gather 左右邻居的频率和值
        left_freq = torch.gather(freq_sorted, dim=-1, index=left_idx)  # (B, V, M)
        right_freq = torch.gather(freq_sorted, dim=-1, index=right_idx)  # (B, V, M)
        left_val = torch.gather(values_sorted, dim=-1, index=left_idx)  # (B, V, M)
        right_val = torch.gather(values_sorted, dim=-1, index=right_idx)  # (B, V, M)

        # 线性插值权重
        denom = (right_freq - left_freq).clamp(min=1e-6)
        weight = ((target - left_freq) / denom).clamp(0, 1)  # (B, V, M)

        # 插值
        resampled = left_val + weight * (right_val - left_val)  # (B, V, M)

        return resampled


__all__ = ["SpectrumFeature"]
