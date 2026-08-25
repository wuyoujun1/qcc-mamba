"""跨变量纠缠联合编码 feature map（JEQK，2026-08-24）。

把 V 个变量的 token **联合编码进一个 V·nq qubit 的纠缠态** Ψ ∈ C^{2^(V·nq)}，
纠缠层 CNOT 跨越变量边界 → 变量间真实量子关联（当前逐变量独立编码没有的）。

动机（针对 7 种核函数全输 rbf 的根因）：
- 现有架构每个变量独立编码成 nq-qubit 态，K_ij = |⟨ψ_i|ψ_j⟩|² 是乘积态上的
  经典可比相似度，没有任何跨变量纠缠 → 经典核(rbf)在相同特征上必然追平
- JEQK：核从**约化密度矩阵** ρ_i = Tr_{¬i}|Ψ⟩⟨Ψ| 算（变量 i 的局部态，
  蕴含与其余所有变量的纠缠关联），这是经典在多项式成本下算不出来的量

结构（复用 EntanglingFeatureMap 的门机制）：
- 首层：proj_H(H) → 每变量 nq 角度 → ⊗ 乘积态（变量身份）
- 重上传：proj_S(S) → 每变量 nq 角度 → 逐 qubit 旋转 + 跨变量 CNOT
- 纠缠拓扑："var_linear" = 全部 V·nq qubit 线性链（相邻变量自然跨接）

数学：|Ψ(h,s)⟩ = Π_l [ U_ent · (⊗_i ⊗_q RZ(φ)RY(θ)) ] U_ent ⊗_i ⊗_q RZ(φ)RY(θ)|0⟩^⊗(V·nq)

数值：V=7,nq=2 → 2^14=16384 维，每 batch ~6e7 flops（很便宜）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

def _rz_ry_composite(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """复合 RZ(φ) @ RY(θ) 2x2 门（= EntanglingFeatureMap._rz_ry_matrix，模块级副本）。"""
    c = torch.cos(theta / 2)
    s = torch.sin(theta / 2)
    e_neg = torch.exp(-1j * phi / 2)
    e_pos = torch.exp(1j * phi / 2)
    row0 = torch.stack([e_neg * c, -e_neg * s], dim=-1)
    row1 = torch.stack([e_pos * s, e_pos * c], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _apply_gate_to_qubit(
    psi: torch.Tensor, gate: torch.Tensor, qubit: int, n_total: int
) -> torch.Tensor:
    """对 psi (..., 2^n_total) 的第 qubit 位施加 2x2 gate（左乘 gate@psi，列向量约定）。

    独立于 EntanglingFeatureMap._apply_single_qubit（那里用 self.N=单变量 qubit 数，
    联合态需要 n_total=V·nq）。gate 前导维与 psi 前导维对齐（如 (B, 2, 2)）。
    """
    psi = psi.unflatten(-1, [2] * n_total)          # (B, q0, q1, ..., q{T-1})
    target_dim = -(n_total - qubit)
    psi = torch.movedim(psi, target_dim, -1).contiguous()  # 目标 qubit 移到最后: (B, others..., target)

    # 修复（2026-08-24 根因）：matmul(gate, psi) 会把 psi 的倒数第二轴（非目标 qubit）
    # 当作矩阵行收缩，门错打在上一 qubit。改用 einsum 显式收缩 target 轴（psi 的最后轴）：
    #   gate: (B, 2, 2) → 'boi'（o=输出, i=输入=target）
    #   psi:  (B, others..., i) → 'b...i'，ellipsis 覆盖其余 qubit 轴
    #   result 'b...o'：Σ_i gate[b,o,i]·psi[b,others,i] —— 门只作用于 target
    if gate.dim() == 3:
        psi = torch.einsum("boi,b...i->b...o", gate, psi)
    else:  # gate (2,2) 标量 batch
        psi = torch.einsum("oi,b...i->b...o", gate, psi)
    psi = torch.movedim(psi, -1, target_dim).contiguous()
    return psi.flatten(-n_total)


class JointEntanglingFeatureMap(nn.Module):
    """跨变量纠缠联合编码。

    Args:
        n_qubits: 每个变量分配的 qubit 数 N。
        n_vars: 变量数 V。
        n_layers: 数据重上传层数 D。
        d_token: backbone token 维度。
        M: 频谱采样点数（S 维度 = 2M）。
        entangle_topo: "var_linear"（全部 qubit 线性链，跨变量）| "var_ring"（环）| "none"。
        use_H / use_S: 首层用 H、重上传用 S（同双阶段主线）。
        angle_norm: "clamp" | "sphere"。
    """

    def __init__(
        self,
        n_qubits: int = 2,
        n_vars: int = 7,
        n_layers: int = 2,
        d_token: int = 256,
        M: int = 32,
        entangle_topo: str = "var_linear",
        use_H: bool = True,
        use_S: bool = True,
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
        delay_in_s: bool = False,
        s_input_dim: int | None = None,
    ):
        super().__init__()
        if entangle_topo not in ("var_linear", "var_ring", "none"):
            raise ValueError(f"joint entangle_topo must be var_linear|var_ring|none, got {entangle_topo}")
        if not use_H and not use_S:
            raise ValueError("At least one of use_H or use_S must be True")
        if angle_norm not in ("clamp", "sphere"):
            raise ValueError(f"angle_norm must be clamp|sphere, got {angle_norm}")
        if n_qubits * n_vars > 22:
            raise ValueError(f"total qubits {n_qubits*n_vars} > 22 → state dim 2^{n_qubits*n_vars} too large")

        self.N = n_qubits           # 每变量 qubit 数
        self.V = n_vars             # 变量数
        self.D = n_layers
        self.T = n_qubits * n_vars   # 总 qubit 数
        self.dim = 1 << self.T       # 2^(V·nq)
        self.d_token = d_token
        self.entangle_topo = entangle_topo
        self.use_H = use_H
        self.use_S = use_S
        self.angle_norm = angle_norm
        self.angle_radius = angle_radius
        self.required_dim = self.N * 2  # 每变量角度数 = N qubit × 2 角（proj 输出 (B,V,2N)）

        # 跨变量纠缠置换（var_linear = 全链，天然跨变量边界）
        perm = self._build_entangle_perm()
        self.register_buffer("ent_perm", perm.long(), persistent=False)

        if use_H:
            self.proj_H = nn.Linear(d_token, self.required_dim, bias=True)
            nn.init.xavier_uniform_(self.proj_H.weight)
            if self.proj_H.bias is not None:
                nn.init.zeros_(self.proj_H.bias)

        if use_S:
            if s_input_dim is None:
                s_input_dim = 2 * M + (1 if delay_in_s else 0)
            self.proj_S = nn.Linear(s_input_dim, self.required_dim, bias=True)
            nn.init.xavier_uniform_(self.proj_S.weight)
            if self.proj_S.bias is not None:
                nn.init.zeros_(self.proj_S.bias)

    def _build_entangle_perm(self) -> torch.Tensor:
        """var_linear：全 V·nq qubit 线性链 CNOT(k, k+1)，跨变量边界。"""
        if self.entangle_topo == "none":
            return torch.arange(self.dim, dtype=torch.long)
        if self.entangle_topo == "var_linear":
            pairs = [(i, i + 1) for i in range(self.T - 1)]
        else:  # var_ring
            pairs = [(i, (i + 1) % self.T) for i in range(self.T)]
        perm = torch.arange(self.dim, dtype=torch.long)
        for (c, t) in pairs:
            perm = self._cnot_perm(perm, c, t)
        return perm

    def _cnot_perm(self, perm: torch.Tensor, control: int, target: int) -> torch.Tensor:
        new = perm.clone()
        for x in range(self.dim):
            if (x >> (self.T - 1 - control)) & 1:
                y = x ^ (1 << (self.T - 1 - target))
                new[y] = perm[x]
        return new

    def _compute_angles(self, feats: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        """feats: (B, V, d) → angles: (B, V, N, 2)（每变量 N qubit × 2 角）。"""
        B, V, _ = feats.shape
        proj_ang = proj(feats)  # (B, V, 2N)
        if self.angle_norm == "sphere":
            proj_ang = proj_ang / proj_ang.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            proj_ang = proj_ang * self.angle_radius
        else:
            proj_ang = torch.clamp(proj_ang, -math.pi, math.pi)
        return proj_ang.reshape(B, V, self.N, 2)

    def forward(self, h: torch.Tensor, s: torch.Tensor | None = None) -> torch.Tensor:
        """联合编码。h: (B,V,d)，s: (B,V,2M)。返回 Ψ: (B, 2^(V·N)) 复数。

        首层：|0⟩^⊗T 逐 qubit 旋转（用 H 角度，变量身份）。
        重上传：逐 qubit 旋转（用 S 角度）+ 跨变量 CNOT，共 D-1 次。
        """
        if h.dim() != 3:
            raise ValueError(f"h must be (B, V, d), got {tuple(h.shape)}")
        B, V, _ = h.shape
        if V != self.V:
            raise ValueError(f"n_vars mismatch: got V={V}, expected {self.V}")

        if self.use_H:
            h_angles = self._compute_angles(h, self.proj_H)  # (B, V, N, 2)
        else:
            h_angles = torch.zeros(B, V, self.N, 2, device=h.device, dtype=h.dtype)

        if self.use_S:
            if s is None:
                raise ValueError("s must be provided when use_S=True")
            s_angles = self._compute_angles(s, self.proj_S)  # (B, V, N, 2)
        else:
            s_angles = h_angles

        # 首层：从 |0⟩^⊗T 逐 qubit 旋转（= 乘积态构造，变量身份）
        psi = torch.zeros(B, self.dim, dtype=torch.cfloat, device=h.device)
        psi[:, 0] = 1.0
        for i in range(self.V):
            for q in range(self.N):
                t = i * self.N + q  # 全局 qubit 索引
                theta_i = math.pi * h_angles[:, i, q, 0]
                phi_i = math.pi * h_angles[:, i, q, 1]
                gate = _rz_ry_composite(theta_i, phi_i)
                psi = _apply_gate_to_qubit(psi, gate, t, self.T)

        # 重上传层（D-1 次）：旋转 + 跨变量纠缠
        for _ in range(1, self.D):
            for i in range(self.V):
                for q in range(self.N):
                    t = i * self.N + q
                    theta_i = math.pi * s_angles[:, i, q, 0]
                    phi_i = math.pi * s_angles[:, i, q, 1]
                    gate = _rz_ry_composite(theta_i, phi_i)
                    psi = _apply_gate_to_qubit(psi, gate, t, self.T)
            if self.entangle_topo != "none":
                psi = psi.index_select(-1, self.ent_perm)

        return psi

    def check_unit_norm(self, h: torch.Tensor, s: torch.Tensor | None = None,
                        atol: float = 1e-5) -> torch.Tensor:
        psi = self.forward(h, s)
        return psi.abs().pow(2).sum(dim=-1).sqrt()


__all__ = ["JointEntanglingFeatureMap"]
