#!/usr/bin/env python3
"""量子核修复自测脚本 — 轻量级验证，不触发内核故障。

测试内容：
1. searchsorted 严格单调修复
2. 复数 einsum 反向传播修复
3. 复数张量梯度链连续性修复
4. FFT 内存压力优化

运行方式：
    python test_quantum_fix.py

预期结果：
- 所有测试通过
- 无 CUDA 错误
- 无内存溢出
"""
import torch
import sys
import traceback

def test_searchsorted_fix():
    """测试 searchsorted 严格单调修复"""
    print("=" * 60)
    print("测试 1: searchsorted 严格单调修复")
    print("=" * 60)
    
    from qcc.spectrum import SpectrumFeature
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 构造包含重复频率的输入（触发原始 bug）
    B, L, V = 2, 96, 7
    x = torch.randn(B, L, V, device=device)
    
    # 创建 SpectrumFeature（使用小参数减少内存）
    spec = SpectrumFeature(M=16).to(device)
    
    try:
        # 前向传播（SpectrumFeature 是 @torch.no_grad 的确定性函数，不需要梯度）
        S = spec(x)
        print(f"✓ 前向传播成功，输出形状: {S.shape}")

        # 验证输出无 NaN/Inf
        assert not torch.isnan(S).any(), "输出包含 NaN"
        assert not torch.isinf(S).any(), "输出包含 Inf"
        print(f"✓ 输出无 NaN/Inf")

        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_complex_einsum_fix():
    """测试复数 einsum 反向传播修复"""
    print("\n" + "=" * 60)
    print("测试 2: 复数 einsum 反向传播修复")
    print("=" * 60)
    
    from qcc.kernel import quantum_kernel, linear_overlap_kernel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构造复数输入
    B, V, D = 2, 7, 4
    psi = torch.randn(B, V, D, dtype=torch.complex64, device=device)
    psi.requires_grad = True
    
    try:
        # 测试 quantum_kernel
        K1 = quantum_kernel(psi)
        loss1 = K1.sum()
        loss1.backward()
        print(f"✓ quantum_kernel 反向传播成功")
        
        # 测试 linear_overlap_kernel
        psi.grad = None
        K2 = linear_overlap_kernel(psi, mode="imag")
        loss2 = K2.sum()
        loss2.backward()
        print(f"✓ linear_overlap_kernel 反向传播成功")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_feature_map_gradient_fix():
    """测试复数张量梯度链连续性修复"""
    print("\n" + "=" * 60)
    print("测试 3: 复数张量梯度链连续性修复")
    print("=" * 60)

    from qcc.feature_map import EntanglingFeatureMap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造输入：H 是 (B, V, d_token)，S 是 (B, V, 2M)
    B, V, D = 2, 7, 64
    M = 16
    H = torch.randn(B, V, D, device=device, requires_grad=True)
    S = torch.randn(B, V, 2*M, device=device)

    # 创建 EntanglingFeatureMap
    fmap = EntanglingFeatureMap(n_qubits=2, n_layers=2, d_token=D, M=M).to(device)

    try:
        # 前向传播
        psi = fmap(H, S)
        print(f"✓ 前向传播成功，输出形状: {psi.shape}, dtype: {psi.dtype}")

        # 反向传播
        loss = psi.abs().sum()
        loss.backward()
        print(f"✓ 反向传播成功")

        # 检查梯度是否有效
        if H.grad is not None:
            print(f"✓ 梯度有效，梯度形状: {H.grad.shape}")
        else:
            print(f"✗ 梯度为空")
            return False

        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_fft_memory_optimization():
    """测试 FFT 内存压力优化"""
    print("\n" + "=" * 60)
    print("测试 4: FFT 内存压力优化")
    print("=" * 60)
    
    from qcc.spectrum import SpectrumFeature
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 使用较大的输入（压力测试）
    B, L, V = 4, 192, 14
    x = torch.randn(B, L, V, device=device)
    
    spec = SpectrumFeature(M=32).to(device)
    
    try:
        # 记录初始内存
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            mem_before = torch.cuda.memory_allocated()
        
        # 前向传播
        S = spec(x)
        
        # 记录峰值内存
        if device.type == "cuda":
            mem_peak = torch.cuda.max_memory_allocated()
            mem_after = torch.cuda.memory_allocated()
            print(f"✓ 前向传播成功")
            print(f"  初始内存: {mem_before / 1e6:.2f} MB")
            print(f"  峰值内存: {mem_peak / 1e6:.2f} MB")
            print(f"  最终内存: {mem_after / 1e6:.2f} MB")
        else:
            print(f"✓ 前向传播成功（CPU 模式，跳过内存统计）")
        
        # 反向传播（SpectrumFeature 是 @torch.no_grad 的确定性函数，输出不需要梯度）
        # 验证输出无 NaN/Inf 即可
        assert not torch.isnan(S).any(), "输出包含 NaN"
        assert not torch.isinf(S).any(), "输出包含 Inf"
        print(f"✓ 输出无 NaN/Inf")
        
        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def test_quantum_mix_layer():
    """测试完整的 QuantumMixLayer"""
    print("\n" + "=" * 60)
    print("测试 5: 完整 QuantumMixLayer")
    print("=" * 60)
    
    from qcc.quantum_mix import QuantumMixLayer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构造输入（模拟 S-Mamba 的 hidden states）
    B, L, D = 2, 96, 64
    H = torch.randn(B, L, D, device=device)
    
    # 创建 QuantumMixLayer（使用小参数）
    layer = QuantumMixLayer(
        d_token=D,
        n_qubits=2,
        n_layers=2,
        M=16,
        norm_type="softmax",
        kernel_T=0.1,
        offdiag=True,
    ).to(device)
    
    try:
        # 构造 S（频谱特征）— QuantumMixLayer 的 fmap.M 存储 M 值
        M = layer.fmap.M
        S = torch.randn(B, L, 2*M, device=device)

        # 前向传播
        H_out = layer(H, S)
        print(f"✓ 前向传播成功，输出形状: {H_out.shape}")

        # 反向传播
        loss = H_out.sum()
        loss.backward()
        print(f"✓ 反向传播成功")

        return True
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print("量子核修复自测脚本")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    results = []
    
    # 运行所有测试
    results.append(("searchsorted 严格单调修复", test_searchsorted_fix()))
    results.append(("复数 einsum 反向传播修复", test_complex_einsum_fix()))
    results.append(("复数张量梯度链连续性修复", test_feature_map_gradient_fix()))
    results.append(("FFT 内存压力优化", test_fft_memory_optimization()))
    results.append(("完整 QuantumMixLayer", test_quantum_mix_layer()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{status}: {name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！修复有效。")
        return 0
    else:
        print("\n❌ 部分测试失败，请检查修复。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
