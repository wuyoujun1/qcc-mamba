"""时频双阶段对齐量子编码 feature map（QKCS 经典模拟，无训练量子参数）。

DualAE-QCC 架构：
    首层：|ψ⟩ = ⊗ RZ(πφ_Hi)RY(πθ_Hi)|0⟩        θ_H = proj_H(H)
    重上传：|ψ⟩ = U_ent · (⊗ RZ(πφ_Si)RY(πθ_Si)) · |ψ⟩  ×(D−1)  θ_S = proj_S(S)

数学定义：
    |φ(h,s)⟩ = Π_{l=2..D} [ U_ent · ( ⊗_{i=1..N} RZ(π·φ_Si) RY(π·θ_Si) ) ]
               · U_ent · ( ⊗_{i=1..N} RZ(π·φ_Hi) RY(π·θ_Hi) ) |0⟩^⊗N

QKCS 实现要点：
- 态矢量 ψ ∈ C^{B×V×2^N}，N=10 时 1024 维复向量，BLAS/GPU 友好
- **第一层旋转**：乘积态优化，直接构造 ⊗_i |ψ_i⟩（用 H 角度）
- **后续层旋转**：态已纠缠，逐比特 movedim + matmul（用 S 角度）
- **CNOT 纠缠层**：预计算 2^N 维置换索引，一次 index_select 完成
- **双投影**：proj_H(d_token → 2N) + proj_S(2M → 2N)，Xavier 初始化

对应文档：IDEA_DualAE_QCC.md §1
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _ry_matrix(theta: torch.Tensor) -> torch.Tensor:
    """R_Y(θ) = [[cos(θ/2), -sin(θ/2)],[sin(θ/2), cos(θ/2)]]。

    Args:
        theta: 任意形状 (...) 的实数张量。

    Returns:
        复数矩阵，形状 (..., 2, 2)。
    """
    half = theta / 2
    c = torch.cos(half)
    s = torch.sin(half)
    real = torch.stack(
        [
            torch.stack([c, s], dim=-1),
            torch.stack([-s, c], dim=-1),
        ],
        dim=-2,
    )  # (..., 2, 2) 实数
    return real.to(torch.cfloat)


class EntanglingFeatureMap(nn.Module):
    """时频双阶段对齐量子编码 feature map。

    编码策略：
        - 首层：用 H（backbone 语义特征）编码"变量身份"
        - 重上传：用 S（对齐后频谱特征）做"调制"
        - "身份固定 + 调制变化"

    消融开关：
        - use_H=True, use_S=True → 双阶段（主线）
        - use_H=True, use_S=False → 仅H（= 旧 QCC 但 d=512, N=10）
        - use_H=False, use_S=True → 仅S

    reupload_source 消融（重上传层用什么角度）：
        - 'S'（默认）→ 重上传用 S 角度（双阶段主线）
        - 'H' → 重上传也用 H 角度（"S 调制"净贡献）
        - 'alternate' → 重上传层 H/S 交替（调制方式敏感性）

    Args:
        n_qubits: N 量子比特数，特征空间维度 2^N。默认 10。
        n_layers: D 数据重上传层数。默认 2。
        d_token: backbone 输出维度。默认 512。
        M: 频谱采样点数（S 的维度 = 2M）。默认 32。
        entangle_topo: 纠缠拓扑 "linear" / "ring" / "none"。
        use_H: 首层是否用 H 编码（默认 True）。
        use_S: 重上传是否用 S 编码（默认 True）。
        reupload_source: 重上传层角度来源 'S' / 'H' / 'alternate'（默认 'S'）。
        angle_norm: 角度归一化方式。"clamp"（默认）= 原版 clamp(±π)（保留原行为）；
            "sphere" = 球面归一化（proj 先除 L2 范数再乘 angle_radius）。
        angle_radius: 球面归一化半径（仅 angle_norm="sphere" 时生效）。默认 1.0。
    """

    SUPPORTED_TOPO = ("linear", "ring", "none")
    SUPPORTED_REUPLOAD = ("S", "H", "alternate")
    SUPPORTED_ANGLE_NORM = ("clamp", "sphere")

    def __init__(
        self,
        n_qubits: int = 10,
        n_layers: int = 2,
        d_token: int = 512,
        M: int = 32,
        entangle_topo: str = "linear",
        use_H: bool = True,
        use_S: bool = True,
        reupload_source: str = "S",
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
    ):
        super().__init__()
        if entangle_topo not in self.SUPPORTED_TOPO:
            raise ValueError(f"entangle_topo must be one of {self.SUPPORTED_TOPO}")
        if n_qubits < 2 or n_qubits > 14:
            raise ValueError(f"n_qubits must be in [2, 14] for QKCS memory, got {n_qubits}")
        if not use_H and not use_S:
            raise ValueError("At least one of use_H or use_S must be True")
        if reupload_source not in self.SUPPORTED_REUPLOAD:
            raise ValueError(f"reupload_source must be one of {self.SUPPORTED_REUPLOAD}, got {reupload_source}")
        if angle_norm not in self.SUPPORTED_ANGLE_NORM:
            raise ValueError(f"angle_norm must be one of {self.SUPPORTED_ANGLE_NORM}, got {angle_norm}")
        if angle_radius <= 0:
            raise ValueError(f"angle_radius must be > 0, got {angle_radius}")

        self.N = n_qubits
        self.D = n_layers
        self.d_token = d_token
        self.M = M
        self.entangle_topo = entangle_topo
        self.dim = 1 << n_qubits
        self.use_H = use_H
        self.use_S = use_S
        self.reupload_source = reupload_source
        self.angle_norm = angle_norm
        self.angle_radius = angle_radius

        # 每 qubit 2 角度（RZ·RY 完整 Bloch 球）
        self.angles_per_qubit = 2
        self.required_dim = n_qubits * self.angles_per_qubit  # = 2N

        # 预计算纠缠层 U_ent 的总置换索引
        perm = self._build_entangle_perm(entangle_topo)
        self.register_buffer("ent_perm", perm.long(), persistent=False)

        # 双投影层（Xavier 初始化）
        # proj_H: backbone 语义特征 → 首层角度
        if use_H:
            self.proj_H = nn.Linear(d_token, self.required_dim, bias=True)
            nn.init.xavier_uniform_(self.proj_H.weight)
            if self.proj_H.bias is not None:
                nn.init.zeros_(self.proj_H.bias)

        # proj_S: 频谱特征 → 重上传角度
        if use_S:
            self.proj_S = nn.Linear(2 * M, self.required_dim, bias=True)
            nn.init.xavier_uniform_(self.proj_S.weight)
            if self.proj_S.bias is not None:
                nn.init.zeros_(self.proj_S.bias)

    # ------------------------------------------------------------------ #
    # 纠缠拓扑：把 U_ent 表达为 2^N 维置换索引
    # ------------------------------------------------------------------ #
    def _build_entangle_perm(self, topo: str) -> torch.Tensor:
        """返回长度 2^N 的置换索引 perm，使 psi_new = psi[..., perm] 等价于 U_ent。"""
        if topo == "none":
            return torch.arange(self.dim, dtype=torch.long)

        if topo == "linear":
            pairs = [(i, i + 1) for i in range(self.N - 1)]
        else:  # ring
            pairs = [(i, (i + 1) % self.N) for i in range(self.N)]

        perm = torch.arange(self.dim, dtype=torch.long)
        for (c, t) in pairs:
            perm = self._cnot_perm(perm, c, t)
        return perm

    def _cnot_perm(self, perm: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """单个 CNOT 对置换的更新：控制位为 1 时翻转目标位。"""
        new = perm.clone()
        for x in range(self.dim):
            if (x >> (self.N - 1 - control)) & 1:  # 控制位为 1（高位在左）
                y = x ^ (1 << (self.N - 1 - target))  # 翻转目标位
                new[y] = perm[x]
        return new

    # ------------------------------------------------------------------ #
    # 复合 RZ·RY 2x2 矩阵
    # ------------------------------------------------------------------ #
    def _rz_ry_matrix(self, theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """复合 RZ(φ) @ RY(θ) 门。返回 (..., 2, 2) 复数。"""
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        e_neg = torch.exp(-1j * phi / 2)
        e_pos = torch.exp(1j * phi / 2)
        row0 = torch.stack([e_neg * c, -e_neg * s], dim=-1)
        row1 = torch.stack([e_pos * s, e_pos * c], dim=-1)
        return torch.stack([row0, row1], dim=-2)

    # ------------------------------------------------------------------ #
    # 单 qubit 旋转（movedim helper）
    # ------------------------------------------------------------------ #
    def _apply_single_qubit(self, psi: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        """对 psi (B, V, 2^N) 在第 qubit 个比特轴上施加 2x2 batched 矩阵。

        注意：必须左乘 gate @ psi（列向量约定，与首层 RZ(φ)RY(θ)|0⟩ 一致）。
        若写成 psi @ gate 会应用 gate 的转置，对复合门 RZ·RY 顺序颠倒（RY 与 RZ 不对易）。
        """
        psi = psi.unflatten(-1, [2] * self.N)  # (B, V, 2, 2, ..., 2)
        target_dim = -(self.N - qubit)
        psi = torch.movedim(psi, target_dim, -1)
        n_extra = psi.dim() - gate.dim()
        if n_extra > 0:
            gate = gate.view(gate.shape[0], gate.shape[1], *([1] * n_extra), 2, 2)
        psi = torch.matmul(gate, psi)
        psi = torch.movedim(psi, -1, target_dim)
        return psi.flatten(-self.N)  # (B, V, 2^N)

    # ------------------------------------------------------------------ #
    # 旋转层（用于重上传，作用于已纠缠态）
    # ------------------------------------------------------------------ #
    def _apply_rot_layer(self, psi: torch.Tensor, angles: torch.Tensor) -> torch.Tensor:
        """对每个 qubit i 施加 RZ(φ_i)·RY(θ_i)。

        angles: (B, V, N, 2)，[..., i, 0]=θ_i, [..., i, 1]=φ_i（未乘 π）
        """
        for i in range(self.N):
            theta_i = math.pi * angles[..., i, 0]  # (B, V)
            phi_i = math.pi * angles[..., i, 1]    # (B, V)
            gate = self._rz_ry_matrix(theta_i, phi_i)
            psi = self._apply_single_qubit(psi, gate, i)
        return psi

    # ------------------------------------------------------------------ #
    # 首层乘积态构造（RZ·RY 完整 Bloch 球）
    # ------------------------------------------------------------------ #
    def _product_state_rz_ry(self, angles: torch.Tensor) -> torch.Tensor:
        """首层乘积态：|ψ⟩ = ⊗_i RZ(φ_i) RY(θ_i) |0⟩。

        angles: (B, V, N, 2)，[..., 0]=θ（未乘 π），[..., 1]=φ（未乘 π）
        返回 (B, V, 2^N) cfloat。
        """
        B, V, N, _ = angles.shape
        theta = math.pi * angles[..., 0]  # (B, V, N)
        phi = math.pi * angles[..., 1]    # (B, V, N)

        # RZ(φ) RY(θ) |0⟩ = e^{-iφ/2} cos(θ/2) |0⟩ + e^{iφ/2} sin(θ/2) |1⟩
        amp_0 = torch.cos(theta / 2) * torch.exp(-1j * phi / 2)  # (B, V, N) 复数
        amp_1 = torch.sin(theta / 2) * torch.exp(1j * phi / 2)   # (B, V, N) 复数

        vec = torch.stack([amp_0, amp_1], dim=-1)  # (B, V, N, 2) 复数
        psi = vec[:, :, 0]
        for i in range(1, N):
            psi = (psi.unsqueeze(-1) * vec[:, :, i].unsqueeze(-2)).reshape(B, V, 1 << (i + 1))
        return psi

    # ------------------------------------------------------------------ #
    # 角度计算
    # ------------------------------------------------------------------ #
    def _compute_H_angles(self, h: torch.Tensor) -> torch.Tensor:
        """H → 角度：proj_H + （clamp 或 球面归一化 × 半径）+ reshape。

        h: (B, V, d_token) → (B, V, N, 2)
        """
        B, V, _ = h.shape
        h_proj = self.proj_H(h)  # (B, V, 2N)
        if self.angle_norm == "sphere":
            h_proj = h_proj / h_proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            h_proj = h_proj * self.angle_radius
        else:
            h_proj = torch.clamp(h_proj, -math.pi, math.pi)
        return h_proj.reshape(B, V, self.N, 2)

    def _compute_S_angles(self, s: torch.Tensor) -> torch.Tensor:
        """S → 角度：proj_S + （clamp 或 球面归一化 × 半径）+ reshape。

        s: (B, V, 2M) → (B, V, N, 2)
        """
        B, V, _ = s.shape
        s_proj = self.proj_S(s)  # (B, V, 2N)
        if self.angle_norm == "sphere":
            s_proj = s_proj / s_proj.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            s_proj = s_proj * self.angle_radius
        else:
            s_proj = torch.clamp(s_proj, -math.pi, math.pi)
        return s_proj.reshape(B, V, self.N, 2)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(
        self,
        h: torch.Tensor,
        s: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """双阶段量子编码。

        Args:
            h: (B, V, d_token) backbone 输出的 token 表征。
            s: (B, V, 2M) 频谱特征（全 detach）。use_S=True 时必须提供。

        Returns:
            ψ: (B, V, 2^N) 复数态矢量。
        """
        if h.dim() != 3:
            raise ValueError(f"h must be (B, V, d), got {tuple(h.shape)}")
        B, V, _ = h.shape

        # 计算角度
        if self.use_H:
            h_angles = self._compute_H_angles(h)  # (B, V, N, 2)
        else:
            # 仅S模式：首层用零角度（恒等 = |0⟩^⊗N）
            h_angles = torch.zeros(B, V, self.N, 2, device=h.device, dtype=h.dtype)

        if self.use_S:
            if s is None:
                raise ValueError("s must be provided when use_S=True")
            s_angles = self._compute_S_angles(s)  # (B, V, N, 2)
        else:
            # 仅H模式：重上传也用 H 角度（= 旧 QCC 行为）
            s_angles = h_angles

        # 首层：乘积态（用 H 角度），无纠缠
        psi = self._product_state_rz_ry(h_angles)

        # 重上传层（D-1 次）：先旋转后纠缠
        # reupload_source 控制重上传层角度来源（use_S=False 时 s_angles==h_angles，等价重上传 H）
        for l in range(1, self.D):
            if self.reupload_source == "alternate":
                layer_angles = s_angles if (l % 2 == 1) else h_angles
            elif self.reupload_source == "H":
                layer_angles = h_angles
            else:  # "S"（主线）
                layer_angles = s_angles
            psi = self._apply_rot_layer(psi, layer_angles)
            if self.entangle_topo != "none":
                psi = psi.index_select(-1, self.ent_perm)

        return psi

    # ------------------------------------------------------------------ #
    # 单元测试辅助
    # ------------------------------------------------------------------ #
    def check_unit_norm(self, h: torch.Tensor, s: Optional[torch.Tensor] = None,
                        atol: float = 1e-5) -> torch.Tensor:
        """返回逐样本的 L2 范数（应全 ≈ 1）。"""
        psi = self.forward(h, s)
        return psi.abs().pow(2).sum(dim=-1).sqrt()  # (B, V)


__all__ = ["EntanglingFeatureMap"]
