"""S-Mamba Embedding 层（从官方仓库适配）。
来源：https://github.com/sci-m-wang/S-D-Mamba

只保留 DataEmbedding_inverted（S-Mamba 用的反转嵌入），
删去标准 DataEmbedding / TemporalEmbedding / PositionalEmbedding 等未使用的类。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DataEmbeddingInverted(nn.Module):
    """反转嵌入：把 (B, L, V) → (B, V, E)，序列长度 L 被线性映射为 d_model。

    与标准 Transformer 嵌入不同，这里"反转"了变量和序列维度：
    - 标准：每个时间点是一个 token（L 个 token，每个含 V 维特征）
    - 反转：每个变量是一个 token（V 个 token，每个含 L 维特征）
    这使得 Mamba/注意力作用于变量间（inter-variate）而非时序间。
    """

    def __init__(self, c_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, V) → (B, V, d_model)"""
        x = x.permute(0, 2, 1)  # (B, V, L)
        x = self.value_embedding(x)  # (B, V, d_model)
        return self.dropout(x)


__all__ = ["DataEmbeddingInverted"]
