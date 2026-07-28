"""数据预处理：RevIN 归一化 + 周期特征 + 高频 Fourier 备选。

RevIN (Reversible Instance Normalization, Kim et al. ICLR 2022)：
- 训练时按变量实例归一化（零均值单位方差）
- 反归一化用训练集统计量（这里采用可学习仿射 fallback，对单变量严格等价）

对应文档：experiment-design.md §七
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class RevIN(nn.Module):
    """可逆实例归一化（per-variable）。"""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
        # 注册为 non-persistent buffer，使其随 model.to(device) 迁移
        self.register_buffer("mean", torch.zeros(1, 1, num_features), persistent=False)
        self.register_buffer("stdev", torch.ones(1, 1, num_features), persistent=False)

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """x: (B, L, V) 或 (B, V)。"""
        if mode == "norm":
            self._get_statistics(x)
            return self._normalize(x)
        if mode == "denorm":
            return self._denormalize(x)
        raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")

    def _get_statistics(self, x: torch.Tensor) -> None:
        # 按 L 维求 mean / std（x 通常为 (B, L, V)）
        dim2reduce = tuple(range(1, x.ndim - 1))
        mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()
        # 写入 buffer，保持 device 一致
        self.mean = mean
        self.stdev = stdev

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.affine_bias) / (self.affine_weight + self.eps)
        return x * self.stdev + self.mean


class PeriodicTimeFeatures(nn.Module):
    """周期时间特征：sin/cos(hour/24), sin/cos(dow/7)。

    输入：x_mark (B, L, 2) [hour, dow]（0-23, 0-6）
    输出：周期特征 (B, L, 4) 或拼接后 (B, L, 4+d_extra)
    """

    def __init__(self, include_high_freq: bool = False, high_freq_periods=(24, 12, 8, 6)):
        super().__init__()
        self.include_high_freq = include_high_freq
        self.high_freq_periods = high_freq_periods

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        """x_mark: (B, L, [hour, dow, ...]) — 至少 2 维。"""
        hour = x_mark[..., 0:1]  # (B, L, 1)
        dow = x_mark[..., 1:2]
        feats = [
            torch.sin(2 * math.pi * hour / 24),
            torch.cos(2 * math.pi * hour / 24),
            torch.sin(2 * math.pi * dow / 7),
            torch.cos(2 * math.pi * dow / 7),
        ]
        if self.include_high_freq:
            for p in self.high_freq_periods:
                feats.append(torch.sin(2 * math.pi * hour / p))
                feats.append(torch.cos(2 * math.pi * hour / p))
        return torch.cat(feats, dim=-1)


def fourier_high_freq(B: int, L: int, t: Optional[torch.Tensor] = None,
                       device=None, periods=(24, 12, 8, 6, 4)) -> torch.Tensor:
    """spectral bias fallback：显式构造高频 Fourier 特征。

    Args:
        B, L: batch / seq。
        t: 外部时间索引 (B, L)，None 时用 0..L-1。
        periods: 周期（小时）。

    Returns:
        (B, L, 2 * len(periods)) 实数特征。
    """
    if device is None:
        device = "cpu"
    if t is None:
        t = torch.arange(L, device=device).float().unsqueeze(0).expand(B, -1)  # (B, L)
    feats = []
    for p in periods:
        ang = 2 * math.pi * t / p
        feats.append(torch.sin(ang))
        feats.append(torch.cos(ang))
    return torch.stack(feats, dim=-1)  # (B, L, 2P)


def add_time_features(ts_index: np.ndarray, freq: str = "h") -> np.ndarray:
    """从 pandas DateTimeIndex 抽出 (hour, dow[, month])。

    Args:
        ts_index: 长度 T 的 DatetimeIndex。
        freq: "h"/"t"/"15min" — 仅用于判定是否取 minute。
    """
    hour = ts_index.hour.values.astype(np.float32)
    dow = ts_index.dayofweek.values.astype(np.float32)
    feats = [hour, dow]
    if freq in ("t", "15min", "min"):
        minute = ts_index.minute.values.astype(np.float32)
        feats.insert(0, minute)
    return np.stack(feats, axis=-1)  # (T, F)


__all__ = [
    "RevIN",
    "PeriodicTimeFeatures",
    "fourier_high_freq",
    "add_time_features",
]
