"""MPS (Matrix Product State) 张量网络旁路。

MPS 是经典张量网络，是 QCC 的最强经典张量对手（challange Q2/Q7）。
这里实现一个简化但可训练的版本：site-specific 线性映射 + 内积核，
与 QCCBlock 保持同一 forward 接口 (H, y_main) -> (y, K)，便于 E1 六组对照。

对应文档：experiment-design.md §五 / §六
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .message_passing import message_passing


class MPSBypass(nn.Module):
    """MPS 旁路，与 QCCBlock 同接口。

    形式：K[b,i,j] = (H[b,i] @ W) · (H[b,j] @ W)
          H' = (1/V) K · (W_q H)
          y = y_main + α · Projection(LN(H + H'))

    与 QCCBlock 的唯一区别：K 来自可训练线性映射的内积，而非量子 feature map。
    """

    def __init__(
        self,
        d_token: int = 128,
        horizon: int = 96,
        bond_dim: int = 8,
        alpha0: float = 0.1,
        use_layer_norm: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.d_token = d_token
        self.horizon = horizon
        self.pre_norm = pre_norm

        if pre_norm:
            self.pre_ln = nn.LayerNorm(d_token)

        # MPS site 映射：可训练
        self.W = nn.Linear(d_token, d_token, bias=False)

        # 跨变量消息映射
        self.W_q = nn.Linear(d_token, d_token, bias=False)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(d_token)

        self.alpha = nn.Parameter(torch.tensor(alpha0))
        self.proj = nn.Linear(d_token, horizon)

    def compute_kernel(self, H: torch.Tensor) -> torch.Tensor:
        """K[b,i,j] = (H_i W) · (H_j W)。"""
        Hw = self.W(H)  # (B, V, d)
        return torch.einsum("bvi,bwi->bvw", Hw, Hw)

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
    """简化 MPS 层（保留给 make_kernel 兼容，但推荐使用 MPSBypass）。"""

    def __init__(self, d_token: int, bond_dim: int = 8):
        super().__init__()
        self.bond_dim = bond_dim
        self.W = nn.Linear(d_token, d_token, bias=False)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        """H: (B, V, d) → K: (B, V, V)。"""
        Hw = self.W(H)
        return torch.einsum("bvi,bwi->bvw", Hw, Hw)


__all__ = ["MPSBypass", "MPSLayer"]
