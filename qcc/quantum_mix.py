"""量子核跨变量混合层（主干内量子混合，2026-08-11 重构替代旁路 QCCBlock）。

设计：
    探针诊断（analyze_bypass 时代）证明旁路修正机制失效：
      - S 信号死在 LN(H+Hp) 残差结构（修正量对 S 的敏感度 0.02~6%）
      - corr 与残差相关 ≈ 0，α 冻结在初值
      - 主干预测对跨变量输入零响应（非对角/对角 ≈ 1e-6，两个数据集实测）
    重构：抛弃旁路，量子核进主干 —— K 作为主干的跨变量混合算子：
      H' = LN(H + (1/V)·K·H·W_q)
    语义 H 编码变量身份 + 对齐频谱 S 调制角度 → 保真度核 K（逐样本自适应）。
    proj_H / proj_S 端到端训练，K 随表征演化。

forward(H, S) -> (H', K)
    H: (B, V, d) 主干变量 token
    S: (B, V, 2M) 对齐频谱特征（全 detach，确定性函数）
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from .classical_kernels import make_kernel
from .feature_map import EntanglingFeatureMap
from .kernel import quantum_kernel


def _inv_softplus(target: float, lo: float = -20.0) -> float:
    """求 x 使 softplus(x) = target。target<=0 时返回 lo（softplus(lo)≈0）。"""
    if target <= 0:
        return lo
    return math.log(math.expm1(target))


class QuantumMixLayer(nn.Module):
    """量子核跨变量混合层。

    Args:
        d_token: 主干 token 维度 d。
        n_qubits: 量子比特数 N。
        n_layers: 数据重上传层数 D。
        M: 频谱采样点数（S 的维度 = 2M）。
        entangle_topo: 纠缠拓扑 "linear" / "ring" / "none"。
        kernel_fn: 核函数，默认量子核。
        use_fmap: 是否启用量子 feature map（False 时核直接吃 H）。
        theta_S_scale0: S 路调制强度 γ 初始值（可学习，clamp [0.1, 2]）。
        pre_norm: 是否在 feature map 前对 H 做 LayerNorm。
        use_H: 首层是否用 H 编码变量身份。
        use_S: 重上传是否用 S 调制。
        reupload_source: 重上传层角度来源 'S' / 'H' / 'alternate'。
        angle_norm / angle_radius: 角度归一化。
    """

    def __init__(
        self,
        d_token: int = 512,
        n_qubits: int = 8,
        n_layers: int = 2,
        M: int = 32,
        entangle_topo: str = "linear",
        kernel_fn: Optional[Callable] = None,
        use_fmap: bool = True,
        theta_S_scale0: float = 0.5,
        pre_norm: bool = True,
        use_H: bool = True,
        use_S: bool = True,
        reupload_source: str = "S",
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
        norm_type: str = "avg",
        output_mode: str = "residual",
        kernel_T: float = 1.0,
        topk: int = 0,
    ):
        """消息传递归一化与输出模式（2026-08-11 晚，运输修复；2026-08-12 选择性修复）。

        norm_type:
            "avg"     —— 原版 (1/V)·K·H·W_q（跨变量信号被 V 稀释，灵敏度 1e-6 的根因）
            "softmax" —— 行 softmax 归一化 K，GAT 式加权平均（Hp 幅度 ≈ H，信号不被淹没）
        output_mode:
            "residual" —— H' = LN(H + Hp)，用于主干内混合层
            "raw"      —— 返回 LN(Hp)，用于预测头级量子聚合（方案 1：K 最接近损失）
        kernel_T: 保真度核温度（选择性修复）：
            高维量子态浓度使 softmax(K) 近均匀（行熵 1.85/1.95），消息传递退化为全局平均。
            softmax(K/T) 且 T<1 可放大 0.004 级差异，恢复 K 的尖峰/选择性。T=1 不生效。
        topk: softmax 后仅保留每行最大的 topk 个耦合并重归一化（0 = 不启用）。
            让 K 明确做"变量选择"，其余变量权重归零。
        """
        super().__init__()
        self.use_fmap = use_fmap
        self.pre_norm = pre_norm
        self.use_H = use_H
        self.use_S = use_S
        self.norm_type = norm_type
        self.output_mode = output_mode
        self.kernel_T = kernel_T
        self.topk = topk
        if norm_type not in ("avg", "softmax"):
            raise ValueError(f"norm_type must be 'avg' or 'softmax', got {norm_type}")
        if output_mode not in ("residual", "raw"):
            raise ValueError(f"output_mode must be 'residual' or 'raw', got {output_mode}")
        if kernel_T <= 0:
            raise ValueError(f"kernel_T must be > 0, got {kernel_T}")
        if topk < 0:
            raise ValueError(f"topk must be >= 0, got {topk}")

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
                reupload_source=reupload_source,
                angle_norm=angle_norm,
                angle_radius=angle_radius,
            )

        if kernel_fn is None:
            self.kernel_fn = quantum_kernel
        elif isinstance(kernel_fn, str):
            self.kernel_fn = make_kernel(kernel_fn, d_token)
        else:
            self.kernel_fn = kernel_fn

        # 可学习 W_q：跨变量消息映射
        self.W_q = nn.Linear(d_token, d_token, bias=False)
        self.ln = nn.LayerNorm(d_token)

        # S 路调制强度 γ（可学习标量，init=0.5, clamp [0.1, 2]）
        if use_S:
            self._gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(theta_S_scale0)))
            self.s_ln = nn.LayerNorm(2 * M)

    @property
    def gamma(self) -> torch.Tensor:
        """S 路调制强度 γ（clamp 到 [0.1, 2]）。"""
        if not self.use_S:
            return torch.tensor(1.0, device=self.W_q.weight.device)
        g = torch.nn.functional.softplus(self._gamma_raw)
        return g.clamp(min=0.1, max=2.0)

    def forward(
        self,
        H: torch.Tensor,
        S: Optional[torch.Tensor] = None,
    ):
        """量子核跨变量混合：H' = LN(H + (1/V)·K·H·W_q)。

        Args:
            H: (B, V, d) 变量 token。
            S: (B, V, 2M) 对齐频谱特征（use_S=True 时必须提供）。

        Returns:
            H': (B, V, d) 混合后 token。
            K: (B, V, V) 保真度核矩阵（供可解释性分析）。
        """
        # 量子路径必须在 fp32 下运行：AMP 下复数张量变 ComplexHalf，CUDA 不支持
        with torch.autocast(device_type=H.device.type, enabled=False):
            H_in = self.pre_ln(H.float()) if self.pre_norm else H.float()
            if self.use_fmap:
                if self.use_S and S is not None:
                    S_scaled = self.gamma * self.s_ln(S.float())  # γ 调制
                else:
                    S_scaled = None
                psi = self.fmap(H_in, S_scaled)
                K = self.kernel_fn(psi)
            else:
                K = self.kernel_fn(H_in)
            HW = torch.einsum("bvd,de->bve", H_in, self.W_q.weight)
            if self.norm_type == "softmax":
                # GAT 式行归一化：K[v,:] 和为 1，Hp 幅度 ≈ H，跨变量信号不被 1/V 稀释
                # kernel_T < 1：放大保真度差异（0.004 级 → 0.04 级），恢复核选择性
                K_n = torch.softmax(K / self.kernel_T, dim=-1)
                if self.topk > 0:
                    # 变量选择：仅保留每行 topk 个耦合，其余归零后重归一化
                    kth = torch.topk(K_n, self.topk, dim=-1).values[:, :, -1:]
                    K_n = K_n * (K_n >= kth)
                    K_n = K_n / K_n.sum(-1, keepdim=True).clamp_min(1e-8)
                Hp = torch.einsum("bvw,bwe->bve", K_n, HW)
            else:
                Hp = torch.einsum("bvw,bwe->bve", K, HW) / K.shape[1]  # (1/V)·K·H·W_q
            if self.output_mode == "raw":
                # 预测头级聚合：返回 LN(Hp)，供 head 直接拼接放大
                return self.ln(Hp), K
            return self.ln(H_in + Hp), K


__all__ = ["QuantumMixLayer"]
