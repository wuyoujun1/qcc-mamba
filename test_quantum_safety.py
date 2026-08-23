#!/usr/bin/env python3
"""
量子模块安全自测脚本 - 验证修复后的代码不会触发内核故障

测试策略：
1. CPU 模式：完整前向+反向传播（不触发 CUDA）
2. GPU 模式：小 batch 测试 + 错误检测
3. 压力测试：中等规模数据验证稳定性

使用方法：
    python test_quantum_safety.py
"""

import sys
import torch
import torch.nn as nn
from qcc.spectrum import SpectrumFeature
from qcc.feature_map import EntanglingFeatureMap
from qcc.quantum_mix import QuantumMixLayer


def check_cuda_error(stage: str):
    """检查 CUDA 错误"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        err = torch.cuda.cudart().cudaPeekAtLastError()
        if err != 0:
            print(f"❌ {stage}: CUDA 错误 {err}")
            return False
        print(f"✓ {stage}: CUDA 状态正常")
    return True


def test_spectrum_feature(device='cpu'):
    """测试 SpectrumFeature（包含 searchsorted 修复）"""
    print("\n" + "="*60)
    print(f"测试 1: SpectrumFeature (device={device})")
    print("="*60)

    B, V, L = 4, 7, 96  # 小 batch 测试
    x = torch.randn(B, L, V, device=device)

    try:
        # 创建模块
        sf = SpectrumFeature(L=L, V=V, M=32, device=device).to(device)
        print(f"✓ 模块创建成功")

        # 前向传播
        S = sf(x)
        print(f"✓ 前向传播成功，输出形状: {S.shape}")
        assert S.shape == (B, V, 64), f"形状错误: {S.shape}"

        # 反向传播
        loss = S.sum()
        loss.backward()
        print(f"✓ 反向传播成功")

        # 检查梯度
        has_grad = all(p.grad is not None for p in sf.parameters())
        print(f"✓ 梯度计算完成: {has_grad}")

        return check_cuda_error("SpectrumFeature")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_feature_map(device='cpu'):
    """测试 QuantumFeatureMap（包含复数 einsum 修复）"""
    print("\n" + "="*60)
    print(f"测试 2: QuantumFeatureMap (device={device})")
    print("="*60)

    B, V, d = 4, 7, 32
    N = 2  # 2 量子比特

    try:
        # 创建模块
        fmap = EntanglingFeatureMap(d_in=d, N=N, use_S=True, M=32, device=device).to(device)
        print(f"✓ 模块创建成功")

        # 准备输入
        H = torch.randn(B, V, d, device=device)
        S = torch.randn(B, V, 64, device=device)

        # 前向传播（包含复数运算）
        psi = fmap(H, S)
        print(f"✓ 前向传播成功，输出形状: {psi.shape}")
        assert psi.shape == (B, V, 4), f"形状错误: {psi.shape}"
        assert psi.is_complex(), "输出应该是复数"

        # 反向传播
        loss = psi.abs().sum()
        loss.backward()
        print(f"✓ 反向传播成功")

        # 检查梯度
        has_grad = all(p.grad is not None for p in fmap.parameters())
        print(f"✓ 梯度计算完成: {has_grad}")

        return check_cuda_error("QuantumFeatureMap")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quantum_mix_layer(device='cpu'):
    """测试 QuantumMixLayer（完整流程 + CUDA 安全检查）"""
    print("\n" + "="*60)
    print(f"测试 3: QuantumMixLayer (device={device})")
    print("="*60)

    B, V, d = 4, 7, 32

    try:
        # 创建模块（使用 vd 变体配置）
        layer = QuantumMixLayer(
            d=d,
            n_qubits=2,
            qmix_norm='softmax',
            kernel_T=0.1,
            offdiag=True,
            use_S=True,
            M=32,
            device=device
        ).to(device)
        print(f"✓ 模块创建成功")

        # 准备输入
        H = torch.randn(B, V, d, device=device)
        S = torch.randn(B, V, 64, device=device)

        # 前向传播
        H_out, K = layer(H, S)
        print(f"✓ 前向传播成功")
        print(f"  - H_out 形状: {H_out.shape}")
        print(f"  - K 形状: {K.shape}")
        assert H_out.shape == (B, V, d), f"H_out 形状错误: {H_out.shape}"
        assert K.shape == (B, V, V), f"K 形状错误: {K.shape}"

        # 反向传播
        loss = H_out.sum() + K.sum()
        loss.backward()
        print(f"✓ 反向传播成功")

        # 检查梯度
        has_grad = all(p.grad is not None for p in layer.parameters())
        print(f"✓ 梯度计算完成: {has_grad}")

        return check_cuda_error("QuantumMixLayer")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_stress(device='cpu'):
    """压力测试：中等规模数据"""
    print("\n" + "="*60)
    print(f"测试 4: 压力测试 (device={device})")
    print("="*60)

    B, V, L, d = 8, 14, 192, 64  # 中等规模

    try:
        # 创建完整模型
        sf = SpectrumFeature(L=L, V=V, M=32, device=device).to(device)
        layer = QuantumMixLayer(
            d=d,
            n_qubits=2,
            qmix_norm='softmax',
            kernel_T=0.1,
            offdiag=True,
            use_S=True,
            M=32,
            device=device
        ).to(device)
        print(f"✓ 模块创建成功")

        # 准备输入
        x = torch.randn(B, L, V, device=device)
        H = torch.randn(B, V, d, device=device)

        # 完整前向传播
        S = sf(x)
        H_out, K = layer(H, S)
        print(f"✓ 完整前向传播成功")

        # 反向传播
        loss = H_out.sum()
        loss.backward()
        print(f"✓ 反向传播成功")

        return check_cuda_error("压力测试")
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("量子模块安全自测")
    print("="*60)

    # 检查 PyTorch 版本
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 测试设备列表
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')

    all_passed = True

    for device in devices:
        print(f"\n{'#'*60}")
        print(f"# 设备: {device.upper()}")
        print(f"{'#'*60}")

        tests = [
            ("SpectrumFeature", test_spectrum_feature),
            ("QuantumFeatureMap", test_quantum_feature_map),
            ("QuantumMixLayer", test_quantum_mix_layer),
            ("压力测试", test_stress),
        ]

        for name, test_fn in tests:
            if not test_fn(device):
                all_passed = False
                print(f"\n⚠️  {name} 测试失败，继续其他测试...")

    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有测试通过！修复有效，可以安全运行训练。")
        return 0
    else:
        print("❌ 部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
