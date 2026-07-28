"""QCC 核心模块单元测试。"""
from __future__ import annotations

import torch

from qcc import (
    EntanglingFeatureMap,
    QCCBlock,
    quantum_kernel,
    rbf_kernel,
    rff_kernel,
)
from qcc.mps_kernel import MPSBypass
from backbone.interface import MockBackbone
from model.qcc_mamba import QCCMamba


def test_feature_map_unit_norm():
    """feature map 输出态矢量 L2 范数应为 1。"""
    fmap = EntanglingFeatureMap(n_qubits=4, n_layers=2, d_token=8)
    h = torch.randn(2, 5, 8)
    psi = fmap(h)
    norms = psi.abs().pow(2).sum(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print("✅ test_feature_map_unit_norm passed")


def test_kernel_diag():
    """量子核矩阵对角线应为 1。"""
    fmap = EntanglingFeatureMap(n_qubits=4, n_layers=2, d_token=8)
    h = torch.randn(2, 5, 8)
    psi = fmap(h)
    K = quantum_kernel(psi)
    diag = torch.diagonal(K, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-4)
    print("✅ test_kernel_diag passed")


def test_qcc_block_forward():
    """QCCBlock forward 输出维度正确。"""
    block = QCCBlock(d_token=16, horizon=24, n_qubits=4, n_layers=2)
    H = torch.randn(2, 5, 16)
    y_main = torch.randn(2, 24, 5)
    y, K, correction = block(H, y_main)
    assert y.shape == (2, 24, 5)
    assert K.shape == (2, 5, 5)
    assert correction.shape == (2, 24, 5)
    print("✅ test_qcc_block_forward passed")


def test_classical_kernels_shape():
    """经典核输出形状正确。"""
    H = torch.randn(2, 5, 16)
    for fn in [rbf_kernel, rff_kernel]:
        K = fn(H)
        assert K.shape == (2, 5, 5)
    print("✅ test_classical_kernels_shape passed")


def test_mps_bypass_forward():
    """MPSBypass 与 QCCBlock 同接口。"""
    block = MPSBypass(d_token=16, horizon=24)
    H = torch.randn(2, 5, 16)
    y_main = torch.randn(2, 24, 5)
    y, K, correction = block(H, y_main)
    assert y.shape == (2, 24, 5)
    assert K.shape == (2, 5, 5)
    assert correction.shape == (2, 24, 5)
    print("✅ test_mps_bypass_forward passed")


def test_qccmamba_forward():
    """端到端 QCCMamba forward 输出维度正确（使用 MockBackbone 避免依赖 mamba-ssm）。"""
    mock_backbone = MockBackbone(num_var=5, lookback=32, horizon=8, d_model=16, d_token=16)
    model = QCCMamba(
        backbone=mock_backbone,
        num_var=5,
        lookback=32,
        horizon=8,
        d_token=16,
        use_qcc=True,
        n_qubits=4,
        n_layers=2,
        use_periodic_feat=False,
    )
    x = torch.randn(2, 32, 5)
    y, y_main, K = model(x)
    assert y.shape == (2, 8, 5)
    assert y_main.shape == (2, 8, 5)
    assert K.shape == (2, 5, 5)
    print("✅ test_qccmamba_forward passed")


if __name__ == "__main__":
    test_feature_map_unit_norm()
    test_kernel_diag()
    test_qcc_block_forward()
    test_classical_kernels_shape()
    test_mps_bypass_forward()
    test_qccmamba_forward()
    print("\nAll basic tests passed!")
