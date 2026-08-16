"""Quantum kernel modules ported from qcc-mamba (quantum mixing for S-Mamba)."""
from .feature_map import EntanglingFeatureMap
from .kernel import quantum_kernel
from .classical_kernels import (rbf_kernel, periodic_kernel, rff_kernel, no_bypass, make_kernel)
from .quantum_mix import QuantumMixLayer
from .spectrum import SpectrumFeature

__all__ = ["EntanglingFeatureMap", "quantum_kernel", "rbf_kernel", "periodic_kernel",
           "rff_kernel", "no_bypass", "make_kernel", "QuantumMixLayer", "SpectrumFeature"]
