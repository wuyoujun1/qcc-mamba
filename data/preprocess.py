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


class IntegerTimeFeatures(nn.Module):
    """整数时间特征（与 S-Mamba 官方一致）。

    输入：x_mark (B, L, 4) [month, day, weekday, hour]（整数）
    输出：整数特征 (B, L, 4) 直接透传，作为伪变量 token 进 backbone

    注意：与官方 timeenc=0 一致，不做 sin/cos 编码
    """

    def __init__(self):
        super().__init__()

    def forward(self, x_mark: torch.Tensor) -> torch.Tensor:
        """x_mark: (B, L, 4) [month, day, weekday, hour] — 整数。"""
        # 直接透传整数特征，不做任何变换
        return x_mark.float()


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
    """从 pandas DateTimeIndex 抽出整数时间特征（与 S-Mamba 官方一致）。

    返回：(T, F) 数组
    - 小时级（freq='h'）：4 列 [month, day, weekday, hour]
    - 分钟级（freq='t'/'15min'/'min'）：5 列 [month, day, weekday, hour, minute//15]

    Args:
        ts_index: 长度 T 的 DatetimeIndex。
        freq: 频率字符串，'h' 为小时级，'t'/'15min'/'min' 为分钟级
    """
    month = ts_index.month.values.astype(np.float32)      # 1-12
    day = ts_index.day.values.astype(np.float32)          # 1-31
    weekday = ts_index.dayofweek.values.astype(np.float32)  # 0-6 (Monday=0)
    hour = ts_index.hour.values.astype(np.float32)        # 0-23

    feats = [month, day, weekday, hour]

    # 分钟级数据追加 minute//15 特征（与官方 Dataset_ETT_minute 一致）
    if freq in ("t", "15min", "min"):
        minute_bin = (ts_index.minute.values // 15).astype(np.float32)  # 0-3
        feats.append(minute_bin)

    return np.stack(feats, axis=-1)  # (T, 4) 或 (T, 5)


__all__ = [
    "RevIN",
    "IntegerTimeFeatures",
    "fourier_high_freq",
    "add_time_features",
]
