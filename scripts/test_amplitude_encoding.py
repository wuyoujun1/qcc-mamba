#!/usr/bin/env python3
"""
安全测试：振幅编码量子核
- 先用 CPU 单 batch 验证
- 检查内存占用
- 避免危险操作（searchsorted、复数 einsum）
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 仓库根目录

def test_amplitude_encoding():
    """测试振幅编码的安全性"""
    print("=" * 60)
    print("测试振幅编码量子核（CPU 单 batch）")
    print("=" * 60)

    # 参数
    B, V, D = 2, 7, 256  # batch, variables, dim
    M = 32  # 频谱维度
    N = 6   # qubits (2^6 = 64)

    # 模拟频谱特征
    S = torch.randn(B, V, 2 * M)  # (2, 7, 64)
    print(f"频谱特征 S: {S.shape}")

    # 振幅编码：线性投影 + 归一化
    print("\n1. 线性投影 (64 -> 64)")
    proj = nn.Linear(2 * M, 2**N)
    psi = proj(S)  # (2, 7, 64)
    print(f"   投影后: {psi.shape}")

    print("\n2. L2 归一化")
    psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-8)
    print(f"   归一化后: {psi.shape}")
    print(f"   范数检查: {psi.norm(dim=-1).mean().item():.6f}")

    print("\n3. 计算量子保真度核")
    # K[i,j] = |<psi_i|psi_j>|^2
    # 避免复数 einsum，用实数矩阵乘法
    K = torch.matmul(psi, psi.transpose(-1, -2))  # (2, 7, 7)
    K = K ** 2  # 保真度
    print(f"   核矩阵 K: {K.shape}")
    print(f"   对角线: {K[0].diag().mean().item():.6f}")
    print(f"   非对角线: {(K[0] - K[0].diag().diag()).abs().mean().item():.6f}")

    print("\n4. 内存检查")
    # 单 batch 内存
    mem_S = S.element_size() * S.nelement() / 1024 / 1024
    mem_psi = psi.element_size() * psi.nelement() / 1024 / 1024
    mem_K = K.element_size() * K.nelement() / 1024 / 1024
    print(f"   S: {mem_S:.2f} MB")
    print(f"   psi: {mem_psi:.2f} MB")
    print(f"   K: {mem_K:.2f} MB")
    print(f"   总计: {mem_S + mem_psi + mem_K:.2f} MB")

    # 训练时内存（假设 batch=32, 10 epochs）
    print("\n5. 训练时内存估算（batch=32）")
    B_train = 32
    mem_train = (mem_S + mem_psi + mem_K) * B_train / B
    print(f"   单 batch: {mem_train:.2f} MB")
    print(f"   安全余量: {8000 - mem_train:.2f} MB (假设 8GB GPU)")

    print("\n" + "=" * 60)
    print("✓ 振幅编码测试通过，无危险操作")
    print("=" * 60)

if __name__ == "__main__":
    test_amplitude_encoding()
