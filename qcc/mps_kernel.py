"""MPS (Matrix Product State) 张量网络旁路。

数学背景：
    Schollwöck (2011) 证明：顺序量子电路的纠缠能力由层数 D 决定，
    MPS bond_dim = 2^D 编码相同的纠缠表达能力。
    QCC(n_qubits=8, n_layers=2) 的 Hilbert 空间 = 2^8 = 256 维，
    MPS(bond_dim=4) 的键空间 = 4 维——两者纠缠能力等价，但函数空间不等价。
    本模块的 MPSBypass 用 bond_dim 控制投影维度，配合 RBF 非线性核，
    提供与 QCC 进行有效对比的经典基线（线性内积是 trivial 弱基线，
    不能作为 QCC 的对手）。

关键设计（修订版）：
    1. W 的输出维度 = bond_dim（而不是 d_token），bond_dim 才有意义
    2. kernel_type 支持 {linear, rbf, poly}，默认 rbf
    3. 与 QCCBlock 保持同一 forward 接口 (H, y_main) -> (y, K, correction_raw)

对应文档：experiment-design.md §五 / §六
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .message_passing import message_passing


def _inv_softplus(target: float, lo: float = -20.0) -> float:
    """求 x 使 softplus(x) = target。target<=0 时返回 lo（softplus(lo)≈0）。"""
    if target <= 0:
        return lo
    return math.log(math.expm1(target))


class MPSBypass(nn.Module):
    """MPS 旁路（修订版：bond_dim 生效 + 可选非线性核）。

    形式：
        Hw = W(H)                              # (B, V, bond_dim)  ← 关键：bond_dim 决定秩
        K   = kernel(Hw, Hw)                    # (B, V, V)
        H'  = (1/V) K · (W_q H)                 # message passing
        y   = y_main + α · Projection(LN(H + H'))

    与 QCCBlock 的唯一区别：K 来自"经典可训练投影 + 可选 RBF/Poly 非线性"，
                            而非量子 feature map。
    当 bond_dim = 2^n_layers 且 kernel_type='rbf' 时，本旁路
    的纠缠表达能力等价于 n_layers 层顺序量子电路（Schollwöck 2011）。
    注：等价的是纠缠能力（entanglement expressivity），不是完整函数空间
    ——QCC 的 Hilbert 空间维度（2^N）通常远大于 MPS 的键空间（bond_dim）。
    """

    def __init__(
        self,
        d_token: int = 128,
        horizon: int = 96,
        bond_dim: int = 4,           # 默认改 4：纠缠能力等价于 QCC(n_qubits=8, n_layers=2)
        alpha0: float = 0.1,
        use_layer_norm: bool = True,
        pre_norm: bool = True,
        kernel_type: str = "rbf",    # 新增：linear/rbf/poly
        rbf_sigma: float = 1.0,      # 新增：RBF 带宽（可学习）
        poly_degree: int = 2,        # 新增：多项式次数
        poly_constant: float = 1.0,  # 新增：多项式常数项
    ):
        super().__init__()
        if kernel_type not in ("linear", "rbf", "poly"):
            raise ValueError(f"kernel_type must be linear/rbf/poly, got {kernel_type}")

        self.d_token = d_token
        self.horizon = horizon
        self.bond_dim = bond_dim
        self.pre_norm = pre_norm
        self.kernel_type = kernel_type
        self.poly_degree = poly_degree
        self.poly_constant = poly_constant

        if pre_norm:
            self.pre_ln = nn.LayerNorm(d_token)

        # 关键修复：W 输出维度 = bond_dim（不再是 d_token）
        # 这样 bond_dim=4 + RBF 的纠缠能力对齐 QCC(n_qubits=8, n_layers=2)
        self.W = nn.Linear(d_token, bond_dim, bias=False)

        # 跨变量消息映射
        self.W_q = nn.Linear(d_token, d_token, bias=False)

        # RBF 带宽（可学习，softplus 保证 > 0），init 使 sigma == rbf_sigma
        if kernel_type == "rbf":
            self._sigma_raw = nn.Parameter(torch.tensor(_inv_softplus(rbf_sigma - 1e-4)))

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(d_token)

        # 可学习融合系数 α（softplus 保证 > 0），init 使 softplus(_alpha_raw) == alpha0
        self._alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha0)))

        # 跨变量特征 → 预测修正
        self.proj = nn.Linear(d_token, horizon)

    @property
    def sigma(self) -> torch.Tensor:
        """RBF 带宽（保证 > 0）。"""
        return torch.nn.functional.softplus(self._sigma_raw) + 1e-4

    @property
    def alpha(self) -> torch.Tensor:
        """非负融合系数。"""
        return torch.nn.functional.softplus(self._alpha_raw)

    def compute_kernel(self, H: torch.Tensor) -> torch.Tensor:
        """根据 kernel_type 计算核矩阵 K[b, i, j]。

        Args:
            H: (B, V, d_token) backbone 输出。

        Returns:
            K: (B, V, V) 核矩阵。
        """
        Hw = self.W(H)  # (B, V, bond_dim)

        if self.kernel_type == "linear":
            # 线性内积 K = Hw @ Hw^T  (rank ≤ bond_dim)
            return torch.einsum("bvi,bwi->bvw", Hw, Hw)

        if self.kernel_type == "poly":
            # 多项式核 K = (Hw @ Hw^T + c)^d
            inner = torch.einsum("bvi,bwi->bvw", Hw, Hw)
            return (inner + self.poly_constant) ** self.poly_degree

        # rbf: K[b,i,j] = exp(-||H_i - H_j||^2 / (2σ^2 + eps))
        # 分母加 eps 防止 σ→0 时数值爆炸
        dist_sq = torch.cdist(Hw, Hw, p=2.0).pow(2)  # (B, V, V)
        return torch.exp(-dist_sq / (2.0 * self.sigma.pow(2) + 1e-6))

    def forward(self, H: torch.Tensor, y_main: torch.Tensor):
        """H: (B, V, d), y_main: (B, H, V) → (y, K, correction_raw)。"""
        H_in = self.pre_ln(H) if self.pre_norm else H
        K = self.compute_kernel(H_in)
        Hp = message_passing(K, H_in, self.W_q.weight)
        qcc_out = Hp
        if self.use_layer_norm:
            qcc_out = self.ln(H_in + qcc_out)
        correction_raw = self.proj(qcc_out).transpose(1, 2)
        y = y_main + self.alpha * correction_raw
        return y, K, correction_raw


class MPSLayer(nn.Module):
    """简化 MPS 层（保留给 make_kernel 兼容；推荐用 MPSBypass 做正式对比）。"""

    def __init__(self, d_token: int, bond_dim: int = 4, kernel_type: str = "rbf"):
        super().__init__()
        self.bond_dim = bond_dim
        self.kernel_type = kernel_type
        self.W = nn.Linear(d_token, bond_dim, bias=False)
        if kernel_type == "rbf":
            # init 使 softplus(_sigma_raw) + 1e-4 == 1.0
            self._sigma_raw = nn.Parameter(torch.tensor(_inv_softplus(1.0 - 1e-4)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self._sigma_raw) + 1e-4

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        Hw = self.W(H)
        if self.kernel_type == "linear":
            return torch.einsum("bvi,bwi->bvw", Hw, Hw)
        dist_sq = torch.cdist(Hw, Hw, p=2.0).pow(2)
        return torch.exp(-dist_sq / (2.0 * self.sigma.pow(2) + 1e-6))


__all__ = ["MPSBypass", "MPSLayer"]
