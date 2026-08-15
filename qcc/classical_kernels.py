"""经典核对照（RBF / Periodic / RFF / 无旁路）+ 工厂函数。

接口：H: (B, V, d) → K: (B, V, V)，与量子核严格对齐。

用于 E1 决定性实验：仅替换 kernel_fn，其余结构不动。

对应文档：experiment-design.md §4.4
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch


# ---------------------------------------------------------------------- #
# RBF
# ---------------------------------------------------------------------- #
def rbf_kernel(H: torch.Tensor, gamma: Optional[float] = None) -> torch.Tensor:
    """RBF 核：K[i,j] = exp(-γ ||h_i - h_j||²)。

    Args:
        H: (B, V, d) 浮点张量；复数（量子态）时拼接 [Re; Im] 后算距离。
        gamma: 带宽倒数。默认 1/d。
    """
    if H.is_complex():
        # cdist 不支持复数：实部虚部拼接（欧氏距离等价于复空间距离）
        H = torch.cat([H.real, H.imag], dim=-1)
    if gamma is None:
        gamma = 1.0 / H.shape[-1]
    # cdist 输出 (B, V, V)
    dist2 = torch.cdist(H, H).pow(2)
    return torch.exp(-gamma * dist2)


# ---------------------------------------------------------------------- #
# Periodic
# ---------------------------------------------------------------------- #
def periodic_kernel(
    H: torch.Tensor, period: float = 24.0, lengthscale: float = 1.0
) -> torch.Tensor:
    """GP 周期核：K[i,j] = exp(-2 sin²(π|h_i - h_j|/p) / l²)。

    注：当 H 不含时间索引时退化为对特征差的 sin 变换（与 S-Mamba 提取的
    token 特征对齐，不引入额外信息，避免泄漏未来）。
    """
    dist = torch.cdist(H, H)
    return torch.exp(-2.0 * torch.sin(math.pi * dist / period).pow(2) / (lengthscale ** 2))


# ---------------------------------------------------------------------- #
# Random Fourier Features
# ---------------------------------------------------------------------- #
class RFFCache:
    """缓存 RFF 随机权重 W / b，避免每 batch 重采样（保证可复现性）。"""

    def __init__(self, d: int, D_rff: int, gamma: float, device, seed: int = 0):
        g = torch.Generator(device=device).manual_seed(seed)
        W = torch.randn(d, D_rff, device=device, generator=g) * math.sqrt(2 * gamma)
        b = torch.rand(D_rff, device=device, generator=g) * (2 * math.pi)
        self.W = W
        self.b = b


_RFF_CACHE: dict = {}


def rff_kernel(
    H: torch.Tensor,
    D_rff: int = 256,
    gamma: Optional[float] = None,
    *,
    cache_key: Optional[str] = None,
    seed: int = 0,
) -> torch.Tensor:
    """Random Fourier Features 近似 RBF：K[i,j] = z(h_i)ᵀ z(h_j)。

    z(h) = sqrt(2/D) cos(W h + b)，W~N(0, 2γ I)，b~U[0, 2π]。

    Args:
        H: (B, V, d) 浮点张量。
        D_rff: 随机特征维度。
        gamma: 带宽倒数。默认 1/d。
        cache_key: 若指定，按该 key 缓存 W/b（同一实验内可复现）。
        seed: 随机数种子。不同 seed 应产生不同的 W/b。
    """
    if gamma is None:
        gamma = 1.0 / H.shape[-1]
    B, V, d = H.shape
    key = cache_key or f"rff_{d}_{D_rff}_{gamma}_seed{seed}"
    if key not in _RFF_CACHE or _RFF_CACHE[key].W.device != H.device:
        if key in _RFF_CACHE:
            del _RFF_CACHE[key]  # 主动清理旧缓存，防止跨 device 隐式迁移
        _RFF_CACHE[key] = RFFCache(d, D_rff, gamma, H.device, seed=seed)
    W = _RFF_CACHE[key].W
    b = _RFF_CACHE[key].b
    Z = math.sqrt(2.0 / D_rff) * torch.cos(H @ W + b)  # (B, V, D_rff)
    return torch.einsum("bvi,bwi->bvw", Z, Z)


# ---------------------------------------------------------------------- #
# 无旁路（E1 第 ⑥ 组：α→0 的等效设置）
# ---------------------------------------------------------------------- #
def no_bypass(H: torch.Tensor) -> torch.Tensor:
    """返回单位矩阵（K = I），等价于消息传递仅依赖自身。"""
    V = H.shape[1]
    eye = torch.eye(V, device=H.device, dtype=H.dtype).unsqueeze(0).expand(H.shape[0], -1, -1)
    return eye


# ---------------------------------------------------------------------- #
# 工厂：按字符串名构建 kernel_fn
# ---------------------------------------------------------------------- #
def make_kernel(
    name: str,
    d_token: int,
    D_rff: int = 256,
    cache_key: Optional[str] = None,
    seed: int = 0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """根据名字返回 kernel 函数（输入 H 或 ψ，输出 K）。"""
    name = name.lower()
    if name in ("quantum", "qkcs", "q"):
        from .kernel import quantum_kernel
        return quantum_kernel
    if name in ("linear_imag", "qdir"):
        from .kernel import linear_overlap_kernel
        return lambda psi: linear_overlap_kernel(psi, "imag")
    if name in ("linear_real", "qreal"):
        from .kernel import linear_overlap_kernel
        return lambda psi: linear_overlap_kernel(psi, "real")
    if name == "rbf":
        return rbf_kernel
    if name == "periodic":
        return periodic_kernel
    if name in ("rff", "fourier"):
        gamma = 1.0 / d_token
        return lambda H, _D=D_rff, _g=gamma, _k=cache_key, _s=seed: rff_kernel(
            H, D_rff=_D, gamma=_g, cache_key=_k, seed=_s
        )
    if name in ("none", "no_bypass", "identity"):
        return no_bypass
    if name in ("mps",):
        raise ValueError(
            "MPS has trainable parameters and should be built as MPSBypass, "
            "not via make_kernel. Use qcc.mps_kernel.MPSBypass directly."
        )
    raise ValueError(f"Unknown kernel: {name}")


__all__ = [
    "rbf_kernel",
    "periodic_kernel",
    "rff_kernel",
    "no_bypass",
    "make_kernel",
]
