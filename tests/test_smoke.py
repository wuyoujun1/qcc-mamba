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
from backbone.dual_path_backbone import DualPathBackbone


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


def test_delay_in_s():
    """P0-1（2026-08-14）：δ̂ 时滞入 S → S ∈ R^{2M+1}，端到端可训练。"""
    print("=" * 60)
    print("Test 6: Delay-in-S (P0-1)")
    print("=" * 60)

    B, L, V = 2, 96, 10
    H_pred = 96
    M = 32
    if not torch.cuda.is_available():
        print("⚠ Mamba 核仅支持 CUDA，跳过 Test 6（delay_in_s）")
        print()
        return

    device = "cuda"
    model = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        d_token=512, n_qubits=2, spectrum_M=M,
        qmix_layers=1, spectrum_inject=True,  # 同时覆盖 fmap 与注入两条 S 消费路径
        use_spectrum=True, use_H=True, use_S=True,
        delay_in_s=True,
    ).to(device)

    x = torch.randn(B, L, V, device=device)
    y, y_main, K = model(x)
    assert y.shape == (B, H_pred, V), f"y shape {y.shape}"
    assert K.shape == (B, V, V), f"K shape {K.shape}"

    # S 应为 (B, V, 2M+1)：δ̂ 通道存在
    S = model.spectrum(x)
    assert S.shape == (B, V, 2 * M + 1), f"S shape {S.shape}, expected (B, V, {2*M+1})"
    # δ̂ 是 detach 的（不参与图），且已归一化到 [-0.25, 0.25]
    assert S[..., -1].abs().max() <= 0.26, f"δ̂ 超出归一化范围: {S[..., -1].abs().max()}"

    loss = y.pow(2).mean()
    loss.backward()
    qmix0 = model.backbone.quantum_mix_layers[0]
    assert qmix0.fmap.proj_S.weight.grad is not None, "proj_S should have gradient"
    assert model.backbone.spectrum_inject.weight.grad is not None, "inject should have gradient"
    print(f"✓ delay_in_s=True: S={tuple(S.shape)}，前向/反向通过")

    # 默认关闭路径形状不变（回归保护）
    model_off = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        d_token=512, n_qubits=2, spectrum_M=M,
        qmix_layers=1, spectrum_inject=True,
        use_spectrum=True, use_H=True, use_S=True,
        delay_in_s=False,
    ).to(device)
    y_off, _, _ = model_off(x)
    S_off = model_off.spectrum(x)
    assert S_off.shape == (B, V, 2 * M), f"S_off shape {S_off.shape}"
    print(f"✓ delay_in_s=False: S={tuple(S_off.shape)}（回归不变）")
    print()


def test_dual_path_backbone():
    """Test 7: 双路径主干维度流转 + 梯度完整性（P2-1，2026-08-15）。"""
    print("=" * 60)
    print("Test 7: Dual Path Backbone (P2-1)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ Mamba 核仅支持 CUDA，跳过 Test 7（双路径主干）")
        print()
        return

    B, L, V = 2, 96, 10
    H_pred = 96
    d_token = 64  # 小参数版（省显存）

    device = "cuda"
    model = DualPathBackbone(
        num_var=V, lookback=L, horizon=H_pred,
        d_token=d_token, n_feats=4,
        dp_fusion="add", gate=True, gate_init=0.05,
        kernel_T=0.1, offdiag=True, n_qubits=2,
        delay_in_s=True,
    ).to(device)

    x = torch.randn(B, L, V + 4, device=device)  # 含 4 列时间特征
    S = torch.randn(B, V, 2 * 32 + 1, device=device)  # delay_in_s → 2M+1

    out = model(x, S)

    print(f"Input: x = {x.shape}, S = {S.shape}")
    print(f"Output: H = {out.H.shape}, y_main = {out.y_main.shape}, K = {out.K.shape}")

    assert out.H.shape == (B, V, d_token), f"H shape {out.H.shape}"
    assert out.y_main.shape == (B, H_pred, V), f"y_main shape {out.y_main.shape}"
    assert out.K.shape == (B, V, V), f"K shape {out.K.shape}"
    assert out.qmix_out is not None and out.qmix_out.shape == (B, V, d_token), "qmix_out"

    # 梯度完整性（覆盖 γ 饿死回归：msg_proj/proj_S/W_q/_gate_raw/in_proj 都必须有梯度）
    loss = out.y_main.pow(2).mean()
    loss.backward()
    vp = model.var_path
    assert vp.msg_proj.weight.grad is not None, "msg_proj 无梯度（γ 饿死?）"
    assert vp.qmix.fmap.proj_S.weight.grad is not None, "proj_S 无梯度"
    assert vp.qmix.W_q.weight.grad is not None, "W_q 无梯度"
    assert model._gate_raw.grad is not None, "_gate_raw 无梯度"
    assert model.time_path.in_proj.weight.grad is not None, "in_proj 无梯度"

    print("✓ Dual path output dimensions correct")
    print("✓ Gradients flow through var path + gate + time path")
    print()


def test_dual_path_structure_equivalence():
    """Test 8: 结构保证不更差 — γ=0 时 add 融合 ≡ 纯时间路径（精确验证）。"""
    print("=" * 60)
    print("Test 8: Dual Path γ=0 ≡ time_only")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ Mamba 核仅支持 CUDA，跳过 Test 8（结构等价）")
        print()
        return

    B, L, V = 2, 96, 10
    H_pred = 96
    d_token = 64

    device = "cuda"
    model = DualPathBackbone(
        num_var=V, lookback=L, horizon=H_pred,
        d_token=d_token, n_feats=0,
        dp_fusion="add", gate=True, gate_init=0.0,  # γ=0
        kernel_T=0.1, offdiag=True, n_qubits=2,
    ).to(device)
    model.eval()

    x = torch.randn(B, L, V, device=device)
    S = torch.randn(B, V, 2 * 32, device=device)

    with torch.no_grad():
        y_gate0 = model(x, S).y_main                       # H = H_time + 0·H_var = H_time
        H_time = model.time_path(x)
        y_pure = model.pred_head(H_time).transpose(1, 2)   # 纯时间路径过同一 head

    assert torch.allclose(y_gate0, y_pure, atol=1e-6), "γ=0 时 add 融合应 ≡ 纯时间路径"

    print("✓ γ=0 → H ≡ H_time（结构保证不更差成立）")
    print()


def test_dual_path_full_model():
    """Test 9: 完整模型（CUDA）：QCCMamba(dual_path=True) 前向反向 + 周期特征拆分。"""
    print("=" * 60)
    print("Test 9: Full Model (dual_path, CUDA)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ Mamba 核仅支持 CUDA，跳过 Test 9（双路径全模型）")
        print()
        return

    B, L, V = 2, 96, 10
    H_pred = 96
    d_token = 64

    device = "cuda"
    model = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        d_token=d_token,
        dual_path=True, dp_fusion="add",
        gate=True, gate_init=0.05,
        kernel_T=0.1, offdiag=True, n_qubits=2,
        delay_in_s=True,
        use_periodic_feat=True,   # V+4 拆分路径
        use_spectrum=True, use_S=True,
    ).to(device)

    x = torch.randn(B, L, V, device=device)
    x_mark = torch.randint(0, 24, (B, L, 4), device=device).float()

    y, y_main, K = model(x, x_mark=x_mark)

    print(f"Output: y = {y.shape}, K = {K.shape}")
    assert y.shape == (B, H_pred, V), f"y shape {y.shape}"
    assert K.shape == (B, V, V), f"K shape {K.shape}"

    loss = y.pow(2).mean()
    loss.backward()
    assert model.backbone.time_path.in_proj.weight.grad is not None, "in_proj 无梯度"
    assert model.backbone.var_path.msg_proj.weight.grad is not None, "msg_proj 无梯度"

    print("✓ Full dual-path model forward/backward OK（含周期特征拆分）")
    print()


def test_qk_path():
    """Test 10: QK-Path（量子核独立预测通道）：维度 + 梯度 + γ=0 结构等价。"""
    print("=" * 60)
    print("Test 10: QK-Path (量子核独立预测通道)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ Mamba 核仅支持 CUDA，跳过 Test 10（QK-Path）")
        print()
        return

    B, L, V = 2, 96, 10
    H_pred = 96
    d_token = 64

    device = "cuda"
    model = QCCMamba(
        num_var=V, lookback=L, horizon=H_pred,
        d_token=d_token,
        qk_path=True, qk_gate_init=0.05,
        kernel_T=0.1, offdiag=True, n_qubits=2,
        use_spectrum=True, use_S=True,
    ).to(device)

    x = torch.randn(B, L, V, device=device)
    y, y_main, K = model(x)

    assert y.shape == (B, H_pred, V), f"y shape {y.shape}"
    assert K.shape == (B, V, V), f"K shape {K.shape}"
    assert model.qk_mix is not None and model.qk_head is not None

    loss = y.pow(2).mean()
    loss.backward()
    assert model._qk_gate_raw.grad is not None, "_qk_gate_raw 无梯度"
    assert model.qk_mix.fmap.proj_S.weight.grad is not None, "qk proj_S 无梯度"
    assert model.qk_head.weight.grad is not None, "qk_head 无梯度"
    assert model.backbone.pred_head.weight.grad is not None, "主干 head 无梯度"

    # γ=0 → y 精确等于 plain 主干输出（结构保证不更差）
    model.eval()
    with torch.no_grad():
        model._qk_gate_raw.fill_(0.0)
        y_g0, _, _ = model(x)
        x_norm = model.revin(x, mode="norm")  # 与 model(x) 内部相同的 RevIN 路径
        y_plain = model.revin(model.backbone(x_norm, S=model.spectrum(x_norm)).y_main,
                              mode="denorm")
    assert torch.allclose(y_g0, y_plain, atol=1e-6), "γ=0 时 QK-Path 应精确等于 plain"

    print("✓ QK-Path 维度/梯度/γ=0 结构等价全部通过")
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
        test_delay_in_s()
        test_dual_path_backbone()
        test_dual_path_structure_equivalence()
        test_dual_path_full_model()
        test_qk_path()

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
