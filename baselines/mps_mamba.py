"""MPS-Mamba 基线占位。

当前 E1/E2/E3 使用 qcc_mamba.model.QCCMamba 配合 use_qcc=False 的 MPSBypass 作为
MPS 对照，因此本模块暂不实现完整独立模型。如需扩展独立 MPS-Mamba，可在此实现。
"""
from __future__ import annotations


class MPSMamba:
    """占位：独立 MPS-Mamba 模型（TODO）。"""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "MPSMamba standalone model is not implemented yet. "
            "Use qcc_mamba.model.QCCMamba with use_qcc=False for MPS bypass."
        )


__all__ = ["MPSMamba"]
