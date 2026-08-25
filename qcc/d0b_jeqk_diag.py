"""D0b 诊断（2026-08-24）：验证 s36 铁证 —— JEQK 核矩阵是否 κ 不敏感（浓度→常数核）。

动机：s36 的 JEQK κ∈{0.5,1,2,4} 给出完全相同的 MSE 0.37556×4，强烈暗示
quantum_reduced_kernel 对带宽不敏感（核 ≈ 常数）。理论机制：跨变量纠缠的单配性
使每个变量的约化密度矩阵 ρ_i 坍缩向最大混合态 → ‖ρ_i−ρ_j‖²_HS≈0 → K≈1。

本脚本用真实 ETTh1 数据 + 真实组件（SpectrumFeature + JointEntanglingFeatureMap
+ quantum_reduced_kernel）复现这个诊断：
1. ρ_i 的纯度（<1 = 混合；≈1/2^N 的倒数归一化 = 近最大混合）
2. ρ 间 HS 距离 d² 分布（min/max/median/mean）
3. K 对 κ∈{0.5,1,2,4} 的实际变化（非对角 min/max/mean/std）
4. 对比：逐变量独立编码（s35 的对照，非 joint）同批数据下的 K 分布

运行：cd /mnt/train_data/dataops_ws && PYTHONDONTWRITEBYTECODE=1 /tmp/qcc-env/bin/python qcc/d0b_jeqk_diag.py
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from qcc.joint_feature_map import JointEntanglingFeatureMap
from qcc.kernel import quantum_reduced_kernel, _fidelity
from qcc.spectrum import SpectrumFeature

torch.manual_seed(0)
np.random.seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={DEVICE}")


def load_etth1_batch(L: int = 96, V: int = 7, B: int = 8, stride: int = 240) -> torch.Tensor:
    """读真实 ETTh1，取 B 个长度 L 的实例归一化窗口，返回 (B, L, V)。"""
    df = pd.read_csv("dataset/ETT-small/ETTh1.csv")
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"][:V]
    raw = df[cols].values.astype(np.float32)
    xs = []
    for b in range(B):
        seg = raw[b * stride : b * stride + L]
        mean = seg.mean(axis=0, keepdims=True)
        std = seg.std(axis=0, keepdims=True) + 1e-5
        xs.append((seg - mean) / std)
    return torch.from_numpy(np.stack(xs)).float().to(DEVICE)


def rho_reduced(psi_joint: torch.Tensor, n_vars: int, n_qubits: int) -> torch.Tensor:
    """变量约化密度矩阵 ρ_i = Tr_{¬i}|Ψ⟩⟨Ψ|，返回 (B, V, 2^N, 2^N)（复数）。"""
    B = psi_joint.shape[0]
    T = n_vars * n_qubits
    psi_r = psi_joint.reshape(B, *([2] * T))
    rhos = []
    for i in range(n_vars):
        var_axes = [1 + i * n_qubits + q for q in range(n_qubits)]
        others = [a for a in range(1, 1 + T) if a not in var_axes]
        psi_p = psi_r.permute([0] + others + var_axes)
        rest = psi_p.reshape(B, -1, 1 << n_qubits)
        rho = torch.einsum("boi,boj->bij", rest, rest.conj())
        rhos.append(rho)
    return torch.stack(rhos, dim=1)  # (B, V, 2^N, 2^N)


def main() -> None:
    V, nq, n_layers, M, d_token = 7, 2, 2, 32, 512
    B = 8

    # ---- 真实 S（确定性函数，与管线一致）----
    x = load_etth1_batch(B=B)
    S = SpectrumFeature(M=M)(x)  # (B, V, 2M)
    print(f"S: {tuple(S.shape)}  S.abs().mean={S.abs().mean().item():.4f}")

    # ---- 真实联合编码组件（与 s36 相同配置）----
    jf = JointEntanglingFeatureMap(
        n_qubits=nq, n_vars=V, n_layers=n_layers, d_token=d_token, M=M,
        entangle_topo="var_linear", use_H=True, use_S=True,
    ).to(DEVICE)
    lnH = nn.LayerNorm(d_token).to(DEVICE)   # 与管线 pre_ln 一致
    lnS = nn.LayerNorm(2 * M).to(DEVICE)     # 与管线 s_ln 一致
    H = lnH(torch.randn(B, V, d_token, device=DEVICE))   # 语义分布与 H_time 一致（LayerNorm 后）
    S_scaled = 0.5 * lnS(S.float())                      # gamma=0.5 · s_ln(S)
    print(f"H: {tuple(H.shape)}  H.std={H.std().item():.4f}  S_scaled.std={S_scaled.std().item():.4f}")

    # ---- 联合态 Ψ ----
    psi = jf(H, S_scaled)  # (B, 2^(V·nq))
    norm = psi.abs().pow(2).sum(-1).sqrt()
    print(f"joint state dim=2^{V*nq},  unit-norm err max={((norm-1).abs().max().item()):.2e}")

    # ---- ρ_i 纯度诊断 ----
    rho = rho_reduced(psi, V, nq)                       # (B, V, 4, 4) complex
    pur = torch.einsum("bvij,bvji->bv", rho, rho.conj()).real  # Tr(ρ²)
    print("\n[ρ_i 纯度 Tr(ρ²)]  max-mixed=0.25(2⁻²N),  pure=1.0")
    print(f"  per-batch mean/min/max = {pur.mean().item():.4f} / {pur.min().item():.4f} / {pur.max().item():.4f}")

    # ---- ρ_i 间 HS 距离 d² 分布 ----
    rho_f = torch.cat([rho.real, rho.imag], dim=-1).reshape(B, V, -1)  # (B,V,32)
    d2 = torch.cdist(rho_f, rho_f).pow(2)  # (B, V, V)
    off = d2.clone()
    for b in range(B):
        off[b].fill_diagonal_(float("nan"))
    offv = off[~torch.isnan(off)]
    print("\n[ρ_i HS 距离 d² 分布（非对角）]")
    print(f"  min={offv.min().item():.2e}  median={offv.median().item():.2e}  mean={offv.mean().item():.2e}  max={offv.max().item():.2e}")
    print(f"  距中位数的比值 range/median={((offv.max()-offv.min()).item()/(offv.median().item()+1e-12)):.2e}")

    # ---- K 对 κ 的敏感性（核心判据）----
    print("\n[K = exp(-κ·d²) 对 κ 的敏感性]  非对角 min/max/mean/std")
    for k in (0.5, 1.0, 2.0, 4.0):
        K = quantum_reduced_kernel(psi, V, nq, kappa=k)  # (B,V,V)
        Ko = K.clone()
        for b in range(B):
            Ko[b].fill_diagonal_(float("nan"))
        kv = Ko[~torch.isnan(Ko)]
        # 与 κ=0.5 的差值（k 敏感度）
        if k == 0.5:
            ref = kv.clone()
        delta = (kv - ref).abs().mean().item()
        print(f"  κ={k}: min={kv.min().item():.4f} max={kv.max().item():.4f} mean={kv.mean().item():.4f} "
              f"std={kv.std().item():.4f}  |Δvs κ0.5| mean={delta:.2e}")

    # ---- 选择性（同一批，非对角 top/mean 比）----
    K1 = quantum_reduced_kernel(psi, V, nq, kappa=1.0)
    top = torch.topk(K1, 2, dim=-1).values  # (B,V,2)
    print(f"\n[选择性 κ=1]  非对角 top2 mean={top.mean().item():.4f}  vs 全非对角 mean={offv.mean().item():.4f}")

    # ---- 对照：逐变量独立编码（s35 设置）同批数据 ----
    from qcc.feature_map import EntanglingFeatureMap
    fm = EntanglingFeatureMap(
        n_qubits=nq, n_layers=n_layers, d_token=d_token, M=M,
        entangle_topo="linear", use_H=True, use_S=True,
    ).to(DEVICE)
    psi_ind = fm(H, S_scaled)  # (B, V, 2^N)
    f = _fidelity(psi_ind).clamp(1e-8, 1 - 1e-6)
    d_ind = torch.acos(f.sqrt())
    d2_ind = d_ind * d_ind
    offi = d2_ind.clone()
    for b in range(B):
        offi[b].fill_diagonal_(float("nan"))
    offiv = offi[~torch.isnan(offi)]
    print("\n[对照：逐变量独立编码 2-qubit]  非对角 d² 分布")
    print(f"  min={offiv.min().item():.2e}  median={offiv.median().item():.2e}  mean={offiv.mean().item():.2e}  max={offiv.max().item():.2e}")
    print(f"  ρ 纯度对照：独立态应为纯态（无跨变量纠缠），熵=0")


if __name__ == "__main__":
    main()


def harness_sensitivity() -> None:
    """验证 harness 扼杀核值：T=0.1 softmax+offdiag 下 K_n 是否 κ 无关（近硬选择）。"""
    import torch
    from qcc.joint_feature_map import JointEntanglingFeatureMap
    from qcc.kernel import quantum_reduced_kernel
    torch.manual_seed(0)
    V, nq, n_layers, M, d_token, B = 7, 2, 2, 32, 512, 8
    x = load_etth1_batch(B=B)
    S = SpectrumFeature(M=M)(x)
    jf = JointEntanglingFeatureMap(n_qubits=nq, n_vars=V, n_layers=n_layers, d_token=d_token,
                                   M=M, entangle_topo="var_linear").to(DEVICE)
    lnH = nn.LayerNorm(d_token).to(DEVICE)
    lnS = nn.LayerNorm(2 * M).to(DEVICE)
    H = lnH(torch.randn(B, V, d_token, device=DEVICE))
    S_scaled = 0.5 * lnS(S.float())
    psi = jf(H, S_scaled)
    eye = torch.eye(V, device=DEVICE).unsqueeze(0)

    print("\n[harness 扼杀诊断]  K_n = softmax((K-I)/T)，offdiag")
    for T in (0.1, 0.3, 1.0):
        for k in (0.5, 4.0):
            K = quantum_reduced_kernel(psi, V, nq, kappa=k)
            K_n = torch.softmax((K - eye) / T, dim=-1)
            diag_w = K_n * eye  # 对角权重
            # 有效参与数（香农熵 → e^H）
            ent = -(K_n * torch.log(K_n.clamp_min(1e-12))).sum(-1).exp().mean().item()
            print(f"  T={T}: κ={k}: 对角权重mean={diag_w.mean().item():.4f} 有效变量数={ent:.2f}")
        K05 = torch.softmax((quantum_reduced_kernel(psi, V, nq, kappa=0.5) - eye) / T, dim=-1)
        K4 = torch.softmax((quantum_reduced_kernel(psi, V, nq, kappa=4.0) - eye) / T, dim=-1)
        print(f"  T={T}: K_n(κ0.5) vs K_n(κ4) 差 mean={((K05 - K4).abs().mean().item()):.5f}")


if __name__ == "__main__":
    main()
    harness_sensitivity()
