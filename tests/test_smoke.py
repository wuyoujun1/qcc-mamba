"""冒烟测试：验证 DualAE-QCC 架构维度流转正确。

运行：
    python tests/test_smoke.py

验证：
    1. 频谱模块输出维度 (B, V, 2M)
    2. 双阶段编码维度流转：H(B,V,512) → ψ(B,V,1024)
    3. 核矩阵维度 (B, V, V)
    4. 最终预测维度 (B, H, V)
    5. K 矩阵对角线 ≈ 1（量子态归一化）
    6. θ 尺度正常（不爆炸）
"""
import torch
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qcc.spectrum import SpectrumFeature
from qcc.feature_map import EntanglingFeatureMap
from qcc.quantum_mix import QuantumMixLayer
from model.qcc_mamba import QCCMamba
from backbone.interface import MockBackbone


def test_spectrum_module():
    """测试频谱模块输出维度。"""
    print("=" * 60)
    print("Test 1: Spectrum Module")
    print("=" * 60)
    
    B, L, V = 2, 96, 10
    M = 32
    
    spectrum = SpectrumFeature(M=M, sample_range="0_2")
    x = torch.randn(B, L, V)
    
    S = spectrum(x)
    
    print(f"Input: x = {x.shape}")
    print(f"Output: S = {S.shape}")
    print(f"Expected: ({B}, {V}, {2*M})")
    
    assert S.shape == (B, V, 2*M), f"Expected {(B, V, 2*M)}, got {S.shape}"
    assert S.dtype == torch.float32, f"Expected float32, got {S.dtype}"
    assert not S.requires_grad, "S should be detached (no gradient)"
    
    print("✓ Spectrum module output dimension correct")
    print("✓ S is detached (no gradient)")
    print()


def test_feature_map():
    """测试双阶段编码维度流转。"""
    print("=" * 60)
    print("Test 2: Dual-Stage Feature Map")
    print("=" * 60)
    
    B, V, d_token = 2, 10, 512
    N = 10
    M = 32
    
    fmap = EntanglingFeatureMap(n_qubits=N, d_token=d_token, M=M)
    
    H = torch.randn(B, V, d_token)
    S = torch.randn(B, V, 2*M)
    
    psi = fmap(H, S)
    
    print(f"Input: H = {H.shape}, S = {S.shape}")
    print(f"Output: ψ = {psi.shape}")
    print(f"Expected: ({B}, {V}, {2**N})")
    
    assert psi.shape == (B, V, 2**N), f"Expected {(B, V, 2**N)}, got {psi.shape}"
    assert psi.dtype == torch.complex64, f"Expected complex64, got {psi.dtype}"
    
    # 检查量子态归一化
    norm = psi.abs().pow(2).sum(dim=-1)
    print(f"Quantum state norm: min={norm.min():.4f}, max={norm.max():.4f}, mean={norm.mean():.4f}")
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-4), "Quantum states should be normalized"
    
    print("✓ Feature map output dimension correct")
    print("✓ Quantum states are normalized")
    print()


def test_qcc_block():
    """测试量子混合层（QuantumMixLayer，2026-08-11 重构替代 QCCBlock）。"""
    print("=" * 60)
    print("Test 3: Quantum Mix Layer")
    print("=" * 60)

    B, V, d_token = 2, 10, 512
    N = 10
    M = 32

    qmix = QuantumMixLayer(d_token=d_token, n_qubits=N, M=M)

    H = torch.randn(B, V, d_token)
    S = torch.randn(B, V, 2*M)

    Hp, K = qmix(H, S)

    print(f"Input: H = {H.shape}, S = {S.shape}")
    print(f"Output: Hp = {Hp.shape}, K = {K.shape}")
    print(f"Expected: Hp = ({B}, {V}, {d_token}), K = ({B}, {V}, {V})")

    assert Hp.shape == (B, V, d_token), f"Expected Hp shape {(B, V, d_token)}, got {Hp.shape}"
    assert K.shape == (B, V, V), f"Expected K shape {(B, V, V)}, got {K.shape}"
    
    # 检查 K 矩阵对角线
    diag = torch.diagonal(K, dim1=-2, dim2=-1)
    print(f"K matrix diagonal: min={diag.min():.4f}, max={diag.max():.4f}, mean={diag.mean():.4f}")
    assert torch.allclose(diag, torch.ones_like(diag), atol=1e-3), "K diagonal should be ≈ 1"
    
    # 检查 K 矩阵对称性
    assert torch.allclose(K, K.transpose(-1, -2), atol=1e-5), "K should be symmetric"
    
    # 检查 γ 参数
    gamma = qmix.gamma
    print(f"γ (theta_S_scale): {gamma.item():.4f}")
    assert 0.1 <= gamma.item() <= 2.0, f"γ should be in [0.1, 2.0], got {gamma.item()}"
    
    print("✓ QCC block output dimensions correct")
    print("✓ K matrix diagonal ≈ 1")
    print("✓ K matrix is symmetric")
    print("✓ γ parameter is valid")
    print()


def test_full_model():
    """测试完整模型。"""
    print("=" * 60)
    print("Test 4: Full Model (QCCMamba)")
    print("=" * 60)
    
    B, L, V = 2, 96, 10
    H_pred = 96
    d_token = 512
    N = 10
    M = 32
    
    if not torch.cuda.is_available():
        print("⚠ Mamba 核仅支持 CUDA，跳过 Test 4（量子混合主干）")
        print()
        return

    device = "cuda"
    model = QCCMamba(
        num_var=V,
        lookback=L,
        horizon=H_pred,
        d_token=d_token,
        n_qubits=N,
        spectrum_M=M,
        qmix_layers=1,  # 量子混合进主干（默认 SMambaBackbone，需 GPU）
        use_spectrum=True,
        use_H=True,
        use_S=True,
    ).to(device)

    x = torch.randn(B, L, V, device=device)

    y, y_main, K = model(x)

    print(f"Input: x = {x.shape}")
    print(f"Output: y = {y.shape}, y_main = {y_main.shape}, K = {K.shape}")
    print(f"Expected: y = ({B}, {H_pred}, {V})")

    assert y.shape == (B, H_pred, V), f"Expected y shape {(B, H_pred, V)}, got {y.shape}"
    assert y_main.shape == (B, H_pred, V), f"Expected y_main shape {(B, H_pred, V)}, got {y_main.shape}"
    assert K.shape == (B, V, V), f"Expected K shape {(B, V, V)}, got {K.shape}"

    # 检查梯度流
    loss = y.pow(2).mean()
    loss.backward()

    # 检查关键参数是否有梯度（量子路径端到端可训练）
    assert model.spectrum is not None, "Spectrum module should exist"
    assert model.backbone.quantum_mix_layers is not None, "qmix layers should exist"
    qmix0 = model.backbone.quantum_mix_layers[0]
    assert qmix0.fmap.proj_H.weight.grad is not None, "proj_H should have gradient"
    assert qmix0.fmap.proj_S.weight.grad is not None, "proj_S should have gradient"
    
    print("✓ Full model output dimensions correct")
    print("✓ Gradient flows correctly")
    print()


def test_ablation_modes():
    """测试消融模式。"""
    print("=" * 60)
    print("Test 5: Ablation Modes")
    print("=" * 60)
    
    B, L, V = 2, 96, 10
    H_pred = 96
    
    # H-only 模式
    print("Testing H-only mode (use_H=True, use_S=False)...")
    model_h_only = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        use_spectrum=False, use_H=True, use_S=False,
        backbone=MockBackbone(
            num_var=V, lookback=L, horizon=H_pred, d_model=512, d_token=512,
        ),
    )
    x = torch.randn(B, L, V)
    y, _, _ = model_h_only(x)
    assert y.shape == (B, H_pred, V)
    print("✓ H-only mode works")

    # S-only 模式
    print("Testing S-only mode (use_H=False, use_S=True)...")
    model_s_only = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        use_spectrum=True, use_H=False, use_S=True,
        backbone=MockBackbone(
            num_var=V, lookback=L, horizon=H_pred, d_model=512, d_token=512,
        ),
    )
    y, _, _ = model_s_only(x)
    assert y.shape == (B, H_pred, V)
    print("✓ S-only mode works")

    # 无对齐模式
    print("Testing no-align mode (time_align=False, freq_align=False)...")
    model_no_align = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        use_spectrum=True, spectrum_time_align=False, spectrum_freq_align=False,
        backbone=MockBackbone(
            num_var=V, lookback=L, horizon=H_pred, d_model=512, d_token=512,
        ),
    )
    y, _, _ = model_no_align(x)
    assert y.shape == (B, H_pred, V)
    print("✓ No-align mode works")
    
    print()


def main():
    """运行所有冒烟测试。"""
    print("\n" + "=" * 60)
    print("DualAE-QCC Smoke Tests")
    print("=" * 60 + "\n")
    
    try:
        test_spectrum_module()
        test_feature_map()
        test_qcc_block()
        test_full_model()
        test_ablation_modes()
        
        print("=" * 60)
        print("✅ All smoke tests passed!")
        print("=" * 60)
        return 0
    except Exception as e:
        print("\n" + "=" * 60)
        print("❌ Smoke test failed!")
        print("=" * 60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
