"""量子核矩阵 K = |⟨φ|φ⟩|²（保真度）。

接口：psi: (B, V, 2^N) → K: (B, V, V)。

性质：
- 对角线恒为 1（酉演化保范 → |⟨φ_i|φ_i⟩|² = 1）
- 对称正定（保真度）
- 可用作频谱分析的 Gram 矩阵

对应文档：experiment-design.md §4.3
"""
from __future__ import annotations

import torch


def quantum_kernel(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """量子核矩阵：K[b,i,j] = |⟨φ(h_i)|φ(h_j)⟩|² = |Σ_k conj(ψ_i[k])·ψ_j[k]|²。

    Args:
        psi: 量子态矢量，形状 (B, V, 2^N) 复数。
        eps: 数值稳定项。

    Returns:
        核矩阵，形状 (B, V, V) 实数，值域 [0, 1]。
    """
    if psi.is_complex():
        inner = torch.einsum("bvi,bwi->bvw", psi.conj(), psi)
    else:
        # 实数回退：直接内积
        inner = torch.einsum("bvi,bwi->bvw", psi, psi)
    K = inner.abs().pow(2)
    # 数值稳定：只在非对角线加 eps，保持对角线为 1（酉演化保范）
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype).unsqueeze(0)
    K = K + eps * (1 - eye)
    return K


def quantum_kernel_normalized(psi: torch.Tensor) -> torch.Tensor:
    """对 psi 先归一化（防止数值漂移）再算保真度。"""
    norms = psi.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    psi_n = psi / norms.clamp_min(1e-12)
    return quantum_kernel(psi_n)


def kernel_diag_check(K: torch.Tensor, atol: float = 1e-4) -> bool:
    """核对角线是否全 ≈ 1（酉演化保范）。"""
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype)
    err = (K - eye.unsqueeze(0)).abs().max().item()
    return err < atol
