"""DualAE-QCC 旁路块：feature map → kernel → message passing → 残差+LN → 预测修正。

DualAE 架构：
    - 首层用 H（backbone 语义特征）编码"变量身份"
    - 重上传用 S（对齐后频谱特征）做"调制"
    - S 路输出 = γ · proj_S(S_norm)，γ 可学习（init=0.5, clamp [0.1, 2]）

对外接口：
    forward(H, y_main, S=None) -> (y, K, correction_raw)
        H: (B, V, d) backbone 输出的 token 表征
        y_main: (B, H_pred, V) backbone 的主预测
        S: (B, V, 2M) 频谱特征（全 detach，可选）
        y: (B, H_pred, V) 融合预测
        K: (B, V, V) 核矩阵（供频谱分析）
        correction_raw: (B, H_pred, V) 未乘 α 的原始修正量

对应文档：IDEA_DualAE_QCC.md §1
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from .classical_kernels import make_kernel, no_bypass
from .feature_map import EntanglingFeatureMap
from .kernel import quantum_kernel
from .message_passing import message_passing


def _inv_softplus(target: float, lo: float = -20.0) -> float:
    """求 x 使 softplus(x) = target。target<=0 时返回 lo（softplus(lo)≈0）。"""
    if target <= 0:
        return lo
    return math.log(math.expm1(target))


class QCCBlock(nn.Module):
    """DualAE-QCC 量子核跨变量旁路。

    Args:
        d_token: backbone 输出的 token 维度（默认 512）。
        horizon: 预测步长 H。
        n_qubits: 量子比特数 N（默认 10）。
        n_layers: 数据重上传层数 D。
        M: 频谱采样点数（S 的维度 = 2M，默认 32）。
        entangle_topo: 纠缠拓扑 "linear" / "ring" / "none"。
        kernel_fn: 核函数。默认量子核。
        use_fmap: 是否启用量子 feature map。
        alpha0: 旁路融合强度 α 初始值（可学习）。
        theta_S_scale0: S 路调制强度 γ 初始值（可学习，默认 0.5）。
        use_layer_norm: 是否在消息传递后加 LayerNorm。
        pre_norm: 是否在 feature map 前对 H 做 LayerNorm。
        use_H: 首层是否用 H 编码（消融开关）。
        use_S: 重上传是否用 S 编码（消融开关）。
    """

    def __init__(
        self,
        d_token: int = 512,
        horizon: int = 96,
        n_qubits: int = 10,
        n_layers: int = 2,
        M: int = 32,
        entangle_topo: str = "linear",
        kernel_fn: Optional[Callable] = None,
        use_fmap: bool = True,
        alpha0: float = 0.1,
        theta_S_scale0: float = 0.5,
        use_layer_norm: bool = True,
        pre_norm: bool = True,
        use_H: bool = True,
        use_S: bool = True,
    ):
        super().__init__()
        self.d_token = d_token
        self.horizon = horizon
        self.use_fmap = use_fmap
        self.pre_norm = pre_norm
        self.use_H = use_H
        self.use_S = use_S

        if pre_norm:
            self.pre_ln = nn.LayerNorm(d_token)

        if use_fmap:
            self.fmap = EntanglingFeatureMap(
                n_qubits=n_qubits,
                n_layers=n_layers,
                d_token=d_token,
                M=M,
                entangle_topo=entangle_topo,
                use_H=use_H,
                use_S=use_S,
            )

        # 决定性实验：传入具体核函数（支持字符串名或 callable）
        if kernel_fn is None:
            self.kernel_fn = quantum_kernel
        elif isinstance(kernel_fn, str):
            self.kernel_fn = make_kernel(kernel_fn, d_token)
        else:
            self.kernel_fn = kernel_fn

        # 可学习 W_q：跨变量消息映射
        self.W_q = nn.Linear(d_token, d_token, bias=False)

        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln = nn.LayerNorm(d_token)

        # 可学习融合系数 α（softplus 保证 > 0），init 使 softplus(_alpha_raw) == alpha0
        self._alpha_raw = nn.Parameter(torch.tensor(_inv_softplus(alpha0)))

        # S 路调制强度 γ（可学习标量，init=0.5, clamp [0.1, 2]）
        # 只在 use_S=True 时使用
        if use_S:
            self._gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(theta_S_scale0)))
            # S 路 LayerNorm（对 S 做归一化后再投影）
            self.s_ln = nn.LayerNorm(2 * M)

        # 跨变量特征 → 预测修正 (B, V, d) → (B, V, H)
        self.proj = nn.Linear(d_token, horizon)

    @property
    def alpha(self) -> torch.Tensor:
        """非负融合系数。"""
        return torch.nn.functional.softplus(self._alpha_raw)

    @property
    def gamma(self) -> torch.Tensor:
        """S 路调制强度 γ（clamp 到 [0.1, 2]）。"""
        if not self.use_S:
            return torch.tensor(1.0, device=self._alpha_raw.device)
        g = torch.nn.functional.softplus(self._gamma_raw)
        return g.clamp(min=0.1, max=2.0)

    # ------------------------------------------------------------------ #
    def compute_kernel(
        self,
        H_norm: torch.Tensor,
        S: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """根据 use_fmap 选择 feature map + 核 / 经典核。

        Args:
            H_norm: (B, V, d) 归一化后的 backbone 特征。
            S: (B, V, 2M) 频谱特征（可选，use_S=True 时必须提供）。
        """
        if self.use_fmap:
            # S 路处理：LayerNorm + γ 缩放
            if self.use_S and S is not None:
                S_norm = self.s_ln(S)  # (B, V, 2M)
                S_scaled = self.gamma * S_norm  # γ 调制
            else:
                S_scaled = None
            psi = self.fmap(H_norm, S_scaled)
            return self.kernel_fn(psi)
        # 经典核直接吃 H
        return self.kernel_fn(H_norm)

    def forward(
        self,
        H: torch.Tensor,
        y_main: torch.Tensor,
        S: Optional[torch.Tensor] = None,
    ):
        """DualAE-QCC 前向传播。

        Args:
            H: (B, V, d) backbone 输出的 token 表征。
            y_main: (B, H_pred, V) backbone 的主预测。
            S: (B, V, 2M) 频谱特征（全 detach，可选）。

        Returns:
            y: (B, H_pred, V) 融合预测。
            K: (B, V, V) 核矩阵。
            correction_raw: (B, H_pred, V) 未乘 α 的原始修正量。
        """
        H_in = self.pre_ln(H) if self.pre_norm else H
        K = self.compute_kernel(H_in, S)
        Hp = message_passing(K, H_in, self.W_q.weight)
        qcc_out = Hp
        if self.use_layer_norm:
            qcc_out = self.ln(H_in + qcc_out)  # 残差 + LN
        # 预测修正：(B, V, d) → (B, V, H) → (B, H, V)
        correction_raw = self.proj(qcc_out).transpose(1, 2)
        y = y_main + self.alpha * correction_raw
        return y, K, correction_raw


__all__ = ["QCCBlock"]
