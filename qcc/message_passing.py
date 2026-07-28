"""跨变量消息传递：H' = (1/V) K · (W_q H)。

与 GAT 同构，但权重来自核矩阵而非 softmax。

对应文档：experiment-design.md §4.5
"""
from __future__ import annotations

import torch
import torch.nn as nn


def message_passing(K: torch.Tensor, H: torch.Tensor, W_q: torch.Tensor) -> torch.Tensor:
    """H'[v] = (1/V) Σ_u K[v,u]·(W_q H[u])。

    Args:
        K: 核矩阵 (B, V, V)。
        H: token 表征 (B, V, d)。
        W_q: 可学习线性映射 (d, d)。

    Returns:
        聚合后的 token 表征 (B, V, d)。
    """
    HW = torch.einsum("bvd,de->bve", H, W_q)  # (B, V, d)
    out = torch.einsum("bvw,bwe->bve", K, HW)  # (B, V, d)
    return out / K.shape[1]


class MessagePassing(nn.Module):
    """带可学习 W_q 的消息传递包装。"""

    def __init__(self, d_token: int):
        super().__init__()
        self.W_q = nn.Linear(d_token, d_token, bias=False)

    def forward(self, K: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return message_passing(K, H, self.W_q.weight)
