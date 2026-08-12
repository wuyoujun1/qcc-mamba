"""QCC-Mamba: S-Mamba + Quantum-kernel Cross-variable Channel（量子混合主干）。

主入口包：暴露给上层脚本的统一接口。
2026-08-11 重构：QCCBlock / MPSBypass（旁路）已移除，量子核进主干（QuantumMixLayer）。
"""
from .qcc.quantum_mix import QuantumMixLayer
from .qcc.feature_map import EntanglingFeatureMap
from .qcc.kernel import quantum_kernel
from .qcc.classical_kernels import rbf_kernel, periodic_kernel, rff_kernel, no_bypass
from .qcc.message_passing import message_passing
from .model.qcc_mamba import QCCMamba

__all__ = [
    "QuantumMixLayer",
    "EntanglingFeatureMap",
    "quantum_kernel",
    "rbf_kernel",
    "periodic_kernel",
    "rff_kernel",
    "no_bypass",
    "message_passing",
    "QCCMamba",
]
