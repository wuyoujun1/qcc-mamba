"""纠缠数据编码 feature map（QKCS 经典模拟，无训练量子参数）。

数学定义（无变分层，固定电路）：
    |φ(h)⟩ = V(h) |0⟩^⊗N
    V(h) = Π_{l=1..D} [ U_ent · ( ⊗_{i=1..N} R_Y(π · h[i mod d]) ) ]
    U_ent = Π_{i=1..N-1} CNOT_{i,i+1}        # linear 拓扑

QKCS 实现要点：
- 态矢量 ψ ∈ C^{B×V×2^N}，N=8 时仅 256 维复向量，BLAS/GPU 友好
- **第一层旋转**：因初态是 |0⟩^N 乘积态，可直接用外积构造 ⊗_i [cos(θ_i/2), sin(θ_i/2)]
- **后续层旋转**：态已纠缠，必须对每个 qubit 轴逐比特施加 R_Y（movedim + batched matmul）
- **CNOT 纠缠层**：预计算 2^N 维置换索引，一次 index_select 完成整个 U_ent

对应文档：experiment-design.md §4.2 / §4.7
"""
from __future__ import annotations

import math

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
    """纠缠数据编码 feature map（QKCS 经典模拟，无训练量子参数）。

    Args:
        n_qubits: N 量子比特数，特征空间维度 2^N。默认 8。
        n_layers: D 数据重上传层数。默认 2。
        d_token:  输入 token 维度 d，默认 128。
        entangle_topo: 纠缠拓扑，仅 "linear" / "ring" / "none" 生效。
            - "linear": Π_{i=0}^{N-2} CNOT_{i,i+1}
            - "ring":   Π_{i=0}^{N-1} CNOT_{i,(i+1)%N}
            - "none":   恒等（用于消融"无纠缠"对照）
        encode_gate: 数据编码旋转门，仅 "R_Y" / "R_X" / "R_Z" 生效。
    """

    SUPPORTED_TOPO = ("linear", "ring", "none")
    SUPPORTED_GATE = ("R_Y", "R_X", "R_Z")

    def __init__(
        self,
        n_qubits: int = 8,
        n_layers: int = 2,
        d_token: int = 128,
        entangle_topo: str = "linear",
        encode_gate: str = "R_Y",
    ):
        super().__init__()
        if entangle_topo not in self.SUPPORTED_TOPO:
            raise ValueError(f"entangle_topo must be one of {self.SUPPORTED_TOPO}")
        if encode_gate not in self.SUPPORTED_GATE:
            raise ValueError(f"encode_gate must be one of {self.SUPPORTED_GATE}")
        if n_qubits < 2 or n_qubits > 14:
            raise ValueError(f"n_qubits must be in [2, 14] for QKCS memory, got {n_qubits}")
        if d_token < n_qubits:
            raise ValueError(f"d_token ({d_token}) must be >= n_qubits ({n_qubits})")

        self.N = n_qubits
        self.D = n_layers
        self.d = d_token
        self.d_token = d_token  # 兼容旧代码引用
        self.entangle_topo = entangle_topo
        self.encode_gate = encode_gate
        self.dim = 1 << n_qubits  # 2^N

        # 预计算整个纠缠层 U_ent 的总置换索引
        perm = self._build_entangle_perm(entangle_topo)
        # 显式 long 化，避免 buffer 类型推导为 int
        self.register_buffer("ent_perm", perm.long(), persistent=False)

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
    # 编码门：R_Y / R_X / R_Z
    # ------------------------------------------------------------------ #
    def _gate_matrix(self, theta: torch.Tensor) -> torch.Tensor:
        """根据 self.encode_gate 返回 2x2 复数矩阵 batch。"""
        if self.encode_gate == "R_Y":
            return _ry_matrix(theta)
        if self.encode_gate == "R_X":
            # R_X(θ) = cos(θ/2) I + i sin(θ/2) X
            half = theta / 2
            c = torch.cos(half)
            s = torch.sin(half)
            zero = torch.zeros_like(c)
            real = torch.stack(
                [torch.stack([c, zero], -1), torch.stack([zero, c], -1)], -2
            )
            imag = torch.stack(
                [torch.stack([zero, -s], -1), torch.stack([s, zero], -1)], -2
            )
            return real.to(torch.cfloat) + 1j * imag.to(torch.cfloat)
        # R_Z
        # R_Z(θ) = diag(e^{-iθ/2}, e^{iθ/2})
        half = theta / 2
        e_neg = torch.exp(-1j * half).to(torch.cfloat)
        e_pos = torch.exp(1j * half).to(torch.cfloat)
        zero = torch.zeros_like(e_neg)
        row0 = torch.stack([e_neg, zero], -1)
        row1 = torch.stack([zero, e_pos], -1)
        return torch.stack([row0, row1], -2)

    # ------------------------------------------------------------------ #
    # 旋转层：作用于所有 N 个 qubit（movedim helper）
    # ------------------------------------------------------------------ #
    def _apply_single_qubit(self, psi: torch.Tensor, gate: torch.Tensor, qubit: int) -> torch.Tensor:
        """对 psi (B, V, 2^N) 在第 qubit 个比特轴上施加 2x2 batched 矩阵。

        实现思路：movedim 把目标轴移到最后一维 → 与 gate 做 batched matmul → 移回原位。
        """
        # 把 qubit 对应的轴移到最后一维（中间插入 2 维）
        # psi: (B, V, 2, 2, ..., 2) 共 N 个 2
        psi = psi.unflatten(-1, [2] * self.N)  # (B, V, 2, 2, ..., 2)
        # 目标轴在 -1-0=-1-N+qubit+1? 严谨算：unflatten 后轴顺序是 -N,...,-1
        # 第 qubit 个轴在 unflatten 之后位于第 -(N - qubit) 维
        target_dim = -(self.N - qubit)
        psi = torch.movedim(psi, target_dim, -1)  # (B, V, 2, ..., 2 [target last])
        # gate: (B, V, 2, 2) -> reshape to (B, V, 1, ..., 1, 2, 2) 以匹配 psi 的 batch 维度
        n_extra = psi.dim() - gate.dim()  # N - 2 个额外 batch 维度
        if n_extra > 0:
            gate = gate.view(gate.shape[0], gate.shape[1], *([1] * n_extra), 2, 2)
        psi = torch.matmul(psi, gate)  # gate 右乘
        # 移回
        psi = torch.movedim(psi, -1, target_dim)
        return psi.flatten(-self.N)  # (B, V, 2^N)

    def _apply_rot_layer(self, psi: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """对每个 qubit i 施加 R_*(π·h[i])。psi: (B,V,2^N) -> (B,V,2^N)。"""
        for i in range(self.N):
            theta = math.pi * h[..., i]  # (B, V)；要求 d_token >= N（已在 __init__ 断言）
            gate = self._gate_matrix(theta)  # (B, V, 2, 2) 复数
            psi = self._apply_single_qubit(psi, gate, i)
        return psi

    def _apply_rot_layer_product_state(self, h: torch.Tensor) -> torch.Tensor:
        """首层乘积态优化：|ψ⟩ = ⊗_i [cos(θ_i/2), sin(θ_i/2)]。

        只在初态 |0⟩^⊗N 上用，外积得到 product state 的精确 2^N 维态矢量。
        """
        B, V, _ = h.shape
        # theta: (B, V, N)；当 d_token < N 时用 `i % d_token` 循环复用
        theta = math.pi * h[..., : self.N]  # 先取前 N 个；若 d_token < N 会因越界抛错
        # 兼容：把 d_token < N 走 fallback 路径
        if self.d_token < self.N:
            idx = torch.arange(self.N, device=h.device) % self.d_token
            theta = math.pi * h[..., idx]
        c = torch.cos(theta / 2)
        s = torch.sin(theta / 2)
        # 每个 qubit 一个 2 维向量 [c, s]
        # 外积构造：先 (B, V, N, 2)
        vec = torch.stack([c, s], dim=-1)  # (B, V, N, 2)
        # 展平 N 个 qubit 为 2^N：fold+matmul
        psi = vec[:, :, 0]  # (B, V, 2)
        for i in range(1, self.N):
            # psi (B,V,2^i)  与  vec[...,i,:] (B,V,2) 做外积 → (B,V,2^(i+1))
            psi = (
                psi.unsqueeze(-1) * vec[:, :, i].unsqueeze(-2)
            ).reshape(B, V, 1 << (i + 1))
        return psi.to(torch.cfloat)  # (B, V, 2^N)

    # ------------------------------------------------------------------ #
    # Forward
    # ------------------------------------------------------------------ #
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, V, d) → ψ: (B, V, 2^N) 复数态矢量。

        注意：调用方应当先对 h 做 LayerNorm（防止 D>2 时激活爆炸）。
        """
        if h.dim() != 3:
            raise ValueError(f"h must be (B, V, d), got {tuple(h.shape)}")
        B, V, _ = h.shape
        device = h.device

        # 数值稳定：限幅到 [-π, π] 防止 π·h 极大（保持分布尾部）
        h = torch.clamp(h, -math.pi, math.pi)

        # 首层：R_Y 时可用乘积态外积构造优化；R_X/R_Z 从 |0⟩^N 逐比特 matmul
        if self.encode_gate == "R_Y":
            psi = self._apply_rot_layer_product_state(h)  # (B, V, 2^N) 复数
        else:
            B, V, _ = h.shape
            psi = torch.zeros(B, V, self.dim, dtype=torch.cfloat, device=device)
            psi[..., 0] = 1.0  # |0⟩^⊗N
            psi = self._apply_rot_layer(psi, h)

        # 每一层都加纠缠 U_ent（包括 D=1），与文档电路结构一致
        if self.entangle_topo != "none":
            psi = psi.index_select(-1, self.ent_perm)

        # 后续 D-1 层：逐比特 movedim + matmul，每层 rotation + CNOT
        for l in range(1, self.D):
            psi = self._apply_rot_layer(psi, h)  # 旋转
            if self.entangle_topo != "none":
                psi = psi.index_select(-1, self.ent_perm)  # CNOT 等价置换
        return psi

    # ------------------------------------------------------------------ #
    # 单元测试辅助
    # ------------------------------------------------------------------ #
    def check_unit_norm(self, h: torch.Tensor, atol: float = 1e-5) -> torch.Tensor:
        """返回逐样本的 L2 范数（应全 ≈ 1）。"""
        psi = self.forward(h)
        return psi.abs().pow(2).sum(dim=-1).sqrt()  # (B, V)
