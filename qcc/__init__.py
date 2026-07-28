"""量子核 QCC 旁路模块：feature map / kernel / classical kernel / message passing / block。"""
from .feature_map import EntanglingFeatureMap
from .kernel import quantum_kernel
from .classical_kernels import (
    rbf_kernel,
    periodic_kernel,
    rff_kernel,
    no_bypass,
    make_kernel,
)
from .mps_kernel import MPSBypass, MPSLayer
from .message_passing import message_passing
from .qcc_block import QCCBlock

__all__ = [
    "EntanglingFeatureMap",
    "quantum_kernel",
    "rbf_kernel",
    "periodic_kernel",
    "rff_kernel",
    "no_bypass",
    "make_kernel",
    "MPSBypass",
    "MPSLayer",
    "message_passing",
    "QCCBlock",
]
