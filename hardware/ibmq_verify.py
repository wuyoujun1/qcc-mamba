"""真机验证占位（E8 附录）。

当前 QCC-Mamba 使用 QKCS 经典模拟，真机验证为可选附录。本模块提供接口占位，
后续可接入 PennyLane / Qiskit 进行 N=2-3 小规模核值一致性验证。
"""
from __future__ import annotations

from typing import Callable

import torch


def pennylane_kernel(
    H: torch.Tensor,
    n_qubits: int,
    n_layers: int,
    entangle_topo: str = "linear",
    encode_gate: str = "R_Y",
) -> torch.Tensor:
    """使用 PennyLane 计算量子核（TODO）。"""
    raise NotImplementedError(
        "pennylane_kernel requires PennyLane installation. "
        "This is optional for E8 hardware appendix."
    )


def verify_qkcs_vs_pennylane(
    H: torch.Tensor,
    n_qubits: int = 2,
    atol: float = 1e-5,
) -> dict:
    """验证 QKCS 与 PennyLane 核值一致性（TODO）。"""
    raise NotImplementedError(
        "verify_qkcs_vs_pennylane requires PennyLane installation."
    )


def verify_against_pennylane(
    feature_map: Callable,
    H: torch.Tensor,
    atol: float = 1e-5,
) -> dict:
    """通用 PennyLane 一致性验证（TODO）。"""
    raise NotImplementedError(
        "verify_against_pennylane requires PennyLane installation."
    )


__all__ = [
    "pennylane_kernel",
    "verify_qkcs_vs_pennylane",
    "verify_against_pennylane",
]
