"""量子核模块：feature map / kernel / classical kernel / message passing / quantum mix / spectrum。

2026-08-11 重构：QCCBlock（旁路修正）已删除，由 QuantumMixLayer（主干内量子混合）替代。
"""
from .feature_map import EntanglingFeatureMap
from .kernel import quantum_kernel
from .classical_kernels import (
    rbf_kernel,
    periodic_kernel,
    rff_kernel,
    no_bypass,
    make_kernel,
)
from .message_passing import message_passing
from .quantum_mix import QuantumMixLayer
from .spectrum import SpectrumFeature

__all__ = [
    "EntanglingFeatureMap",
    "quantum_kernel",
    "rbf_kernel",
    "periodic_kernel",
    "rff_kernel",
    "no_bypass",
    "make_kernel",
    "message_passing",
    "QuantumMixLayer",
    "SpectrumFeature",
]
