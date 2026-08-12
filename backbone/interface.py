"""S-Mamba 主干统一接口。

本模块定义 BackboneOutput 数据类与 BaseBackbone 抽象接口，并提供：
- MockBackbone：基于 BiLSTM 的轻量 backbone，用于快速验证与 E1 决定性实验。
- 正式 SMambaBackbone 见 smamba_backbone.py。

对应文档：experiment-design.md §三
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneOutput:
    """主干输出。

    H: (B, V, d) — 每个变量的 token 表征
    y_main: (B, H_pred, V) — 主预测头输出
    K: (B, V, V) — 量子核矩阵（最后一个量子混合层，无可选；无则 None）
    """

    H: torch.Tensor
    y_main: torch.Tensor
    K: Optional[torch.Tensor] = None


class BaseBackbone(nn.Module):
    """所有 backbone 的抽象接口。"""

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None) -> BackboneOutput:
        """x: (B, L, V) → BackboneOutput(H, y_main)。

        S: (B, V, 2M) 对齐频谱特征（量子混合 / 频谱注入需要时提供，无则忽略）。
        """
        raise NotImplementedError


# ---------------------------------------------------------------------- #
# Mock Backbone：BiLSTM + 两个头，用于快速验证与 E1 决定性实验
# ---------------------------------------------------------------------- #
class MockBackbone(BaseBackbone):
    """轻量 backbone，接口与 S-Mamba 对齐。

    结构：
        x (B, L, V)
        → RevIN（外部做，这里可选）
        → Linear(V → d_model)
        → BiLSTM(L, d_model)
        → H_token (B, V, d)   # 通过把 L 维度平均池化后投影到 V 个 token
        → 预测头 (B, H, V)
    """

    def __init__(
        self,
        num_var: int = 321,
        lookback: int = 720,
        horizon: int = 96,
        d_model: int = 512,
        d_token: int = 256,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.d_token = d_token

        # 输入投影：V -> d_model
        self.input_proj = nn.Linear(num_var, d_model)

        # BiLSTM 编码
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # 把序列表征 (B, L, d_model) 映射到 V 个变量 token (B, V, d_token)
        # 先做 d_model -> d_token，再把 L 维线性投影到 V 维（每个变量拿到不同时间组合）
        self.dim_proj = nn.Linear(d_model, d_token)
        self.token_proj = nn.Linear(lookback, num_var)  # (B, d_token, L) -> (B, d_token, V)

        # 主预测头：H (B, V, d_token) -> y_main (B, H, V)
        self.pred_head = nn.Linear(d_token, horizon)

    def forward(self, x: torch.Tensor, S: Optional[torch.Tensor] = None) -> BackboneOutput:
        """x: (B, L, V) → BackboneOutput。

        S: 忽略（Mock 不实用频谱）。
        注意：为了与真实 S-Mamba 接口一致，token 维度是 V 优先。
        """
        B, L, V = x.shape
        # 输入投影
        z = self.input_proj(x)  # (B, L, d_model)
        # LSTM 编码
        z, _ = self.lstm(z)  # (B, L, d_model)
        # 投影到 d_token
        z = self.dim_proj(z)  # (B, L, d_token)
        # 把 L 维映射到 V 维：每个变量获得不同的时间组合 → H (B, V, d_token)
        H = self.token_proj(z.transpose(1, 2)).transpose(1, 2)  # (B, V, d_token)
        # 主预测头：(B, V, d_token) -> (B, V, H) -> (B, H, V)
        y_main = self.pred_head(H).transpose(1, 2)  # (B, H, V)
        return BackboneOutput(H=H, y_main=y_main)


__all__ = ["BackboneOutput", "BaseBackbone", "MockBackbone"]
