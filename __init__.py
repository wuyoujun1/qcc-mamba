"""QCC-Mamba: S-Mamba + Quantum-kernel Cross-variable Channel.

主入口包：暴露给上层脚本的统一接口。
"""
from .qcc.qcc_block import QCCBlock
from .qcc.feature_map import EntanglingFeatureMap
from .qcc.kernel import quantum_kernel
from .qcc.classical_kernels import rbf_kernel, periodic_kernel, rff_kernel, no_bypass
from .qcc.message_passing import message_passing
from .qcc.mps_kernel import MPSBypass, MPSLayer
from .model.qcc_mamba import QCCMamba

__all__ = [
    "QCCBlock",
    "EntanglingFeatureMap",
    "quantum_kernel",
    "rbf_kernel",
    "periodic_kernel",
    "rff_kernel",
    "no_bypass",
    "MPSBypass",
    "MPSLayer",
    "message_passing",
    "QCCMamba",
]
