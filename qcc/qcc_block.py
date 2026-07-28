"""QCC 旁路块：feature map → kernel → message passing → 残差+LN → 预测修正。

对外接口：
    forward(H, y_main) -> (y, K)
        H: (B, V, d) backbone 输出的 token 表征
        y_main: (B, H_pred, V) backbone 的主预测
        y: (B, H_pred, V) 融合预测
        K: (B, V, V) 核矩阵（供 E7 频谱分析）

决定性实验切换：仅替换 kernel_fn；feature map 在经典核时自动跳过。
对应文档：experiment-design.md §4.6
"""
from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn

from .classical_kernels import no_bypass
from .feature_map import EntanglingFeatureMap
from .kernel import quantum_kernel
from .message_passing import message_passing


class QCCBlock(nn.Module):
    """量子核跨变量旁路。

    Args:
        d_token: backbone 输出的 token 维度。
        horizon: 预测步长 H。
        n_qubits: 量子比特数 N。<=0 时退化为不构造 feature map。
        n_layers: 数据重上传层数 D。
        entangle_topo: 纠缠拓扑 "linear" / "ring" / "none"。
        encode_gate: 编码门 "R_Y" / "R_X" / "R_Z"。
        kernel_fn: 核函数。默认量子核；E1 切换为 rbf_kernel / rff_kernel 等。
        use_fmap: 是否启用量子 feature map。False 时仅用经典 kernel_fn。
        alpha0: 旁路融合强度 α 初始值（可学习）。
        use_layer_norm: 是否在消息传递后加 LayerNorm。
        pre_norm: 是否在 feature map 前对 H 做 LayerNorm（推荐 True）。
    """

    def __init__(
        self,
        d_token: int = 128,
        horizon: int = 96,
        n_qubits: int = 8,
        n_layers: int = 2,
        entangle_topo: str = "linear",
        encode_gate: str = "R_Y",
        kernel_fn: Optional[Callable] = None,
        use_fmap: bool = True,
        alpha0: float = 0.1,
        use_layer_norm: bool = True,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.d_token = d_token
        self.horizon = horizon
        self.use_fmap = use_fmap
        self.pre_norm = pre_norm

        if pre_norm:
            self.pre_ln = nn.LayerNorm(d_token)

        if use_fmap:
            self.fmap = EntanglingFeatureMap(
                n_qubits=n_qubits,
                n_layers=n_layers,
                d_token=d_token,
                entangle_topo=entangle_topo,
                encode_gate=encode_gate,
            )

        # 决定性实验：传入具体核函数
        self.kernel_fn = kernel_fn if kernel_fn is not None else quantum_kernel

        # 可学习 W_q：跨变量消息映射
        self.W_q = nn.Linear(d_token, d_token, bias=False)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(d_token)

        # 可学习融合系数 α
        self.alpha = nn.Parameter(torch.tensor(alpha0))

        # 跨变量特征 → 预测修正 (B, V, d) → (B, V, H)
        self.proj = nn.Linear(d_token, horizon)

    # ------------------------------------------------------------------ #
    def compute_kernel(self, H_norm: torch.Tensor) -> torch.Tensor:
        """根据 use_fmap 选择 feature map + 核 / 经典核。"""
        if self.use_fmap:
            psi = self.fmap(H_norm)
            return self.kernel_fn(psi)
        # 经典核直接吃 H
        return self.kernel_fn(H_norm)

    def forward(self, H: torch.Tensor, y_main: torch.Tensor):
        """H: (B, V, d), y_main: (B, H, V) → (y, K, correction_raw)。

        correction_raw: (B, H, V) 未乘 α 的原始修正量，用于辅助损失监督残差。
        """
        H_in = self.pre_ln(H) if self.pre_norm else H
        K = self.compute_kernel(H_in)
        Hp = message_passing(K, H_in, self.W_q.weight)
        qcc_out = Hp
        if self.use_layer_norm:
            qcc_out = self.ln(H_in + qcc_out)  # 残差 + LN
        # 预测修正：(B, V, d) → (B, V, H) → (B, H, V)
        correction_raw = self.proj(qcc_out).transpose(1, 2)
        y = y_main + self.alpha * correction_raw
        return y, K, correction_raw


__all__ = ["QCCBlock"]
