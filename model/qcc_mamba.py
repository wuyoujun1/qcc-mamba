"""端到端模型：DualAE-QCC-Mamba。

数据流：
    x (B, L, V)
    → RevIN norm
    → [可选] 拼接周期时间特征 → (B, L, V+F)
    → Backbone → H (B, V, d), y_main (B, H, V)
    → 频谱模块（detach）→ S (B, V, 2M)
    → QCC / MPS 旁路(H, y_main, S) → y (B, H, V), K (B, V, V)
    → RevIN denorm

对应文档：IDEA_DualAE_QCC.md §1
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from backbone.interface import BaseBackbone
from backbone.smamba_backbone import SMambaBackbone
from data.preprocess import IntegerTimeFeatures, RevIN
from qcc import QCCBlock, SpectrumFeature
from qcc.mps_kernel import MPSBypass


class QCCMamba(nn.Module):
    """端到端 DualAE-QCC-Mamba 模型。

    Args:
        backbone: 已实例化的 backbone（BaseBackbone 子类）。
        num_var: 变量数 V。
        horizon: 预测步长 H。
        d_token: backbone 输出 token 维度（默认 512）。
        use_qcc: True 用 QCCBlock，False 用 MPSBypass。
        n_qubits: QCC 量子比特数 N（默认 10）。
        n_layers: QCC 数据重上传层数 D。
        entangle_topo: QCC 纠缠拓扑。
        kernel_fn: QCC 经典核函数（E1 切换用）。若 use_qcc=True 且为 None，用 quantum_kernel。
        use_fmap: QCC 是否使用量子 feature map。
        alpha0: 旁路融合系数初始值。
        theta_S_scale0: S 路调制强度 γ 初始值（默认 0.5）。
        beta: 辅助损失权重。
        use_periodic_feat: 是否在输入中拼接 sin/cos 时间特征。
        revin_affine: RevIN 是否使用可学习仿射。
        # 频谱模块参数
        use_spectrum: 是否启用频谱模块（默认 True）。
        spectrum_M: 频谱采样点数 M（默认 32）。
        spectrum_range: 采样区间 "0_2" 或 "0_1"（默认 "0_2"）。
        spectrum_amp_normalize: 是否做幅度归一化（消融，默认 False）。
        spectrum_time_align: 是否做时间轴对齐（默认 True）。
        spectrum_freq_align: 是否做频率轴对齐（默认 True）。
        # 消融开关
        use_H: 首层是否用 H 编码（默认 True）。
        use_S: 重上传是否用 S 编码（默认 True）。
    """

    def __init__(
        self,
        backbone: Optional[BaseBackbone] = None,
        num_var: int = 321,
        lookback: int = 720,
        horizon: int = 96,
        d_token: int = 512,
        use_qcc: bool = True,
        n_qubits: int = 10,
        n_layers: int = 2,
        entangle_topo: str = "linear",
        kernel_fn: Optional[callable] = None,
        use_fmap: bool = True,
        alpha0: float = 0.1,
        theta_S_scale0: float = 0.5,
        beta: float = 0.1,
        use_periodic_feat: bool = True,
        revin_affine: bool = True,
        bond_dim: int = 8,
        kernel_type: str = "rbf",
        # 频谱模块参数
        use_spectrum: bool = True,
        spectrum_M: int = 32,
        spectrum_range: str = "0_2",
        spectrum_amp_normalize: bool = False,
        spectrum_time_align: bool = True,
        spectrum_freq_align: bool = True,
        # 消融开关
        use_H: bool = True,
        use_S: bool = True,
    ):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.horizon = horizon
        self.d_token = d_token
        self.use_qcc = use_qcc
        self.beta = beta
        self.use_periodic_feat = use_periodic_feat
        self.kernel_type = kernel_type
        self.use_spectrum = use_spectrum
        self.use_H = use_H
        # P2-3 防护：use_S=True 但关闭频谱模块时自动降级为仅 H（否则 S=None 崩溃）
        if use_S and not use_spectrum:
            print("[QCCMamba] 警告: use_S=True 但 use_spectrum=False，自动降级 use_S=False（仅 H 模式）")
            use_S = False
        self.use_S = use_S

        # RevIN：实例归一化
        self.revin = RevIN(num_features=num_var, affine=revin_affine)

        # 整数时间特征（与 S-Mamba 官方一致，timeenc=0）
        if use_periodic_feat:
            self.periodic = IntegerTimeFeatures()
            # backbone 输入维度会变成 V + 4（month, day, weekday, hour）
            backbone_in_dim = num_var + 4
        else:
            self.periodic = None
            backbone_in_dim = num_var

        # Backbone: 默认使用 S-Mamba 官方实现
        if backbone is None:
            backbone = SMambaBackbone(
                num_var=num_var,          # 原始变量数 V（用于正确切片）
                lookback=lookback,
                horizon=horizon,
                d_model=d_token,
                use_norm=False,           # 外层已有 RevIN
            )
        self.backbone = backbone

        # 频谱模块（可选，全 detach 确定性函数）
        if use_spectrum and use_S:
            self.spectrum = SpectrumFeature(
                M=spectrum_M,
                sample_range=spectrum_range,
                amp_normalize=spectrum_amp_normalize,
                time_align=spectrum_time_align,
                freq_align=spectrum_freq_align,
            )
        else:
            self.spectrum = None

        # 旁路：QCC or MPS
        if use_qcc:
            self.qcc = QCCBlock(
                d_token=d_token,
                horizon=horizon,
                n_qubits=n_qubits,
                n_layers=n_layers,
                M=spectrum_M,
                entangle_topo=entangle_topo,
                kernel_fn=kernel_fn,
                use_fmap=use_fmap,
                alpha0=alpha0,
                theta_S_scale0=theta_S_scale0,
                pre_norm=True,
                use_H=use_H,
                use_S=use_S,
            )
        else:
            self.qcc = MPSBypass(
                d_token=d_token,
                horizon=horizon,
                bond_dim=bond_dim,
                alpha0=alpha0,
                pre_norm=True,
                kernel_type=kernel_type,
            )

    def _prepare_input(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, L, V)；可选 x_mark: (B, L, 4) [month, day, weekday, hour]。返回 backbone 输入。"""
        if self.use_periodic_feat and x_mark is not None:
            # IntegerTimeFeatures 直接透传整数特征，不做变换
            pf = self.periodic(x_mark)  # (B, L, 4) 整数特征
            x_in = torch.cat([x, pf], dim=-1)  # (B, L, V+4)
        else:
            x_in = x
        return x_in

    def forward(
        self,
        x: torch.Tensor,
        x_mark: Optional[torch.Tensor] = None,
        return_norm: bool = False,
    ):
        """DualAE-QCC 前向传播。

        Args:
            x: (B, L, V) 输入序列。
            x_mark: (B, L, F_t) 时间特征。
            return_norm: 是否同时返回归一化空间的预测（用于训练损失）。

        Returns:
            y: (B, H, V) 反归一化最终预测。
            y_main: (B, H, V) 反归一化主预测。
            K: (B, V, V) 核矩阵（若使用旁路）。
            (可选) y_norm, y_main_norm, correction_norm: 归一化空间预测与原始修正量。
        """
        # 1) RevIN norm
        x_norm = self.revin(x, mode="norm")  # (B, L, V)

        # 2) 拼接周期特征
        x_in = self._prepare_input(x_norm, x_mark)  # (B, L, V or V+4)

        # 3) Backbone
        out = self.backbone(x_in)
        H, y_main_norm = out.H, out.y_main  # (B, V, d), (B, H, V)

        # 4) 频谱模块（detach，确定性函数）
        S = None
        if self.spectrum is not None:
            S = self.spectrum(x_norm)  # (B, V, 2M)，全 detach

        # 5) QCC / MPS 旁路
        y_norm, K, correction_norm = self.qcc(H, y_main_norm, S)  # (B, H, V), (B, V, V), (B, H, V)

        # 6) RevIN denorm
        y_main = self.revin(y_main_norm, mode="denorm")
        y = self.revin(y_norm, mode="denorm")

        if return_norm:
            return y, y_main, K, y_norm, y_main_norm, correction_norm
        return y, y_main, K

    def compute_loss(
        self,
        y: torch.Tensor,
        y_main: torch.Tensor,
        y_true: torch.Tensor,
        y_norm: torch.Tensor,
        y_main_norm: torch.Tensor,
        y_true_norm: torch.Tensor,
        correction_norm: torch.Tensor,
    ) -> torch.Tensor:
        """总损失 = MSE(y, y_true) + β · MSE(correction, y_true_norm - y_main_norm)。

        辅助损失只监督旁路应补充的残差，避免与主损失冗余。
        """
        loss_main = nn.functional.mse_loss(y, y_true)
        residual_norm = y_true_norm - y_main_norm
        loss_qcc = nn.functional.mse_loss(correction_norm, residual_norm)
        return loss_main + self.beta * loss_qcc


__all__ = ["QCCMamba"]
