"""端到端模型：量子混合主干（Quantum-Mixed Backbone，2026-08-11 重构）。

数据流：
    x (B, L, V)
    → RevIN norm
    → [可选] 拼接周期时间特征 → (B, L, V+F)
    → DataEmbeddingInverted → 变量 token X0 (B, V, d)
    → [可选] 频谱注入: X0 += proj_in(S)              # 周期结构进主干第一层
    → 每层: X_l = Mamba(X_{l-1}); X_l, K_l = QuantumMix(X_l, S)
    → H = X_L → pred_head → y_main (B, H, V)          # 直接预测,无旁路修正
    → RevIN denorm

重构背景（探针诊断，见 qcc/quantum_mix.py docstring）：
    原旁路架构（dual + α·corr）中 S 信号死在残差结构、corr 与残差相关≈0、
    主干预测对跨变量输入零响应（非对角/对角 ≈ 1e-6）。
    重构：抛弃旁路，量子核进主干 —— K 作为主干的跨变量混合算子，
    proj_H/proj_S 端到端训练，K 随表征演化。量子核雷打不动。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from backbone.dual_path_backbone import DualPathBackbone
from backbone.interface import BaseBackbone
from backbone.smamba_backbone import SMambaBackbone
from data.preprocess import IntegerTimeFeatures, RevIN
from qcc import QuantumMixLayer, SpectrumFeature


class QCCMamba(nn.Module):
    """端到端量子混合主干模型。

    Args:
        backbone: 已实例化的 backbone（BaseBackbone 子类，默认 SMambaBackbone）。
        num_var: 变量数 V。
        lookback: 输入窗口 L。
        horizon: 预测步长 H。
        d_token: backbone 输出 token 维度（默认 512）。
        qmix_layers: 量子混合层数（插在每层 Mamba 之后，默认 0 = 纯主干）。
        qmix_norm: 主干量子混合的消息传递归一化，"avg"(1/V) 或 "softmax"(GAT 式)。
        head_agg: 预测头级量子聚合（方案 1：K 最接近损失，跨变量信息零稀释）。
        spectrum_inject: 是否把频谱特征注入 embedding 输出（默认 False）。
        n_qubits: 量子比特数 N（默认 8）。
        n_layers: 量子数据重上传层数 D。
        entangle_topo: 量子纠缠拓扑。
        kernel_fn: 核函数（默认量子核）。
        use_fmap: 是否启用量子 feature map。
        theta_S_scale0: S 路调制强度 γ 初始值（默认 0.5）。
        use_periodic_feat: 是否在输入中拼接 sin/cos 时间特征。
        revin_affine: RevIN 是否使用可学习仿射。
        # 频谱模块参数
        use_spectrum: 是否启用频谱模块（默认 True）。
        spectrum_M: 频谱采样点数 M（默认 32）。
        spectrum_range: 采样区间 "0_2" 或 "0_1"（默认 "0_2"）。
        spectrum_amp_normalize: 是否做幅度归一化（默认 False）。
        spectrum_time_align: 是否做时间轴对齐（默认 True）。
        spectrum_freq_align: 是否做频率轴对齐（默认 True）。
        # 量子编码消融开关
        use_H: 首层是否用 H 编码（默认 True）。
        use_S: 重上传是否用 S 编码（默认 True）。
        reupload_source: 重上传层角度来源 'S' / 'H' / 'alternate'（默认 'S'）。
    """

    def __init__(
        self,
        backbone: Optional[BaseBackbone] = None,
        num_var: int = 321,
        lookback: int = 720,
        horizon: int = 96,
        d_token: int = 512,
        qmix_layers: int = 0,
        qmix_norm: str = "avg",
        head_agg: bool = False,
        spectrum_inject: bool = False,
        kernel_T: float = 1.0,
        topk: int = 0,
        offdiag: bool = False,
        gate: bool = False,
        gate_init: float = 0.0,
        hp_scale: float = 1.0,
        aux_loss: bool = False,
        aux_beta: float = 0.1,
        kernel_sup: float = 0.0,
        n_qubits: int = 8,
        n_layers: int = 2,
        entangle_topo: str = "linear",
        kernel_fn: Optional[callable] = None,
        use_fmap: bool = True,
        theta_S_scale0: float = 0.5,
        use_periodic_feat: bool = True,
        revin_affine: bool = True,
        # 频谱模块参数
        use_spectrum: bool = True,
        spectrum_M: int = 32,
        spectrum_range: str = "0_2",
        spectrum_amp_normalize: bool = False,
        spectrum_time_align: bool = True,
        spectrum_freq_align: bool = True,
        # P0-1（2026-08-14）：δ̂ 时滞入 S → S = [Ã; φ̃; δ̂] ∈ R^{2M+1}
        delay_in_s: bool = False,
        # 量子编码消融开关
        use_H: bool = True,
        use_S: bool = True,
        reupload_source: str = "S",
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
        # P2-1 双路径（2026-08-15）：时间 SSM 单向 + 量子核独占跨变量
        dual_path: bool = False,
        dp_time_layers: int = 2,
        dp_time_dim: int = 256,
        dp_time_pool: str = "mean",
        dp_var_embed: bool = True,
        dp_msg: str = "S",
        dp_fusion: str = "add",
        # QK-Path（2026-08-15）：量子核独立预测通道，与主干并行融合
        qk_path: bool = False,
        qk_gate_init: float = 0.05,
        qk_use_H: bool = False,
        qk_norm: str = "softmax",
    ):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.horizon = horizon
        self.d_token = d_token
        self.qmix_layers = qmix_layers
        self.use_periodic_feat = use_periodic_feat
        self.use_H = use_H
        self.use_S = use_S

        if qmix_layers > 0 and not (use_spectrum and use_S):
            raise ValueError("qmix_layers>0 需要 use_spectrum=True 且 use_S=True（S 特征驱动量子核）")
        if dual_path and dp_fusion != "time_only" and not (use_spectrum and use_S):
            raise ValueError("dual_path 变量路径需要 use_spectrum=True 且 use_S=True（S 特征驱动量子核）")

        # RevIN：实例归一化
        self.revin = RevIN(num_features=num_var, affine=revin_affine)

        # 整数时间特征（与 S-Mamba 官方一致，timeenc=0）
        if use_periodic_feat:
            self.periodic = IntegerTimeFeatures()
            backbone_in_dim = num_var + 4
        else:
            self.periodic = None
            backbone_in_dim = num_var

        # 频谱模块（detach，确定性函数）
        self.spectrum = None
        if use_spectrum and use_S:
            self.spectrum = SpectrumFeature(
                M=spectrum_M,
                sample_range=spectrum_range,
                amp_normalize=spectrum_amp_normalize,
                time_align=spectrum_time_align,
                freq_align=spectrum_freq_align,
                delay_in_s=delay_in_s,
            )

        # 量子混合层（进主干，每层 Mamba 之后插一层）
        qmix_modules = None
        if qmix_layers > 0:
            qmix_modules = nn.ModuleList(
                [
                    QuantumMixLayer(
                        d_token=d_token,
                        n_qubits=n_qubits,
                        n_layers=n_layers,
                        M=spectrum_M,
                        entangle_topo=entangle_topo,
                        kernel_fn=kernel_fn,
                        use_fmap=use_fmap,
                        theta_S_scale0=theta_S_scale0,
                        pre_norm=True,
                        use_H=use_H,
                        use_S=use_S,
                        reupload_source=reupload_source,
                        angle_norm=angle_norm,
                        angle_radius=angle_radius,
                        norm_type=qmix_norm,
                        output_mode="residual",
                        kernel_T=kernel_T,
                        topk=topk,
                        offdiag=offdiag,
                        gate=gate,
                        gate_init=gate_init,
                        hp_scale=hp_scale,
                        delay_in_s=delay_in_s,
                    )
                    for _ in range(qmix_layers)
                ]
            )

        # 方案 1：预测头级量子聚合（K 最接近损失，跨变量信息零稀释）
        head_qmix = None
        if head_agg:
            if not (use_spectrum and use_S):
                raise ValueError("head_agg=True 需要 use_spectrum=True 且 use_S=True")
            head_qmix = QuantumMixLayer(
                d_token=d_token,
                n_qubits=n_qubits,
                n_layers=n_layers,
                M=spectrum_M,
                entangle_topo=entangle_topo,
                kernel_fn=kernel_fn,
                use_fmap=use_fmap,
                theta_S_scale0=theta_S_scale0,
                pre_norm=True,
                use_H=use_H,
                use_S=use_S,
                reupload_source=reupload_source,
                angle_norm=angle_norm,
                angle_radius=angle_radius,
                norm_type="softmax",  # GAT 式加权平均
                output_mode="raw",    # 返回 LN(Hp)，head 拼接后可直接放大
                kernel_T=kernel_T,
                topk=topk,
                offdiag=offdiag,
                gate=gate,
                gate_init=gate_init,
                hp_scale=hp_scale,
                delay_in_s=delay_in_s,
            )

        # 频谱注入投影（P0-1: delay_in_s 时 S 多 1 维 δ̂）
        inject = None
        if spectrum_inject and use_spectrum and use_S:
            inject = nn.Linear(2 * spectrum_M + (1 if delay_in_s else 0), d_token)

        # Backbone：默认 S-Mamba（量子混合/频谱注入只在默认主干内插值）；
        # dual_path=True 时用双路径主干（时间 SSM 单向 + 量子核独占跨变量）
        if dual_path:
            if backbone is not None:
                raise ValueError("dual_path=True 时不能传自定义 backbone")
            backbone = DualPathBackbone(
                num_var=num_var,
                lookback=lookback,
                horizon=horizon,
                d_token=d_token,
                n_feats=4 if use_periodic_feat else 0,
                dp_time_layers=dp_time_layers,
                dp_time_dim=dp_time_dim,
                dp_time_pool=dp_time_pool,
                dp_var_embed=dp_var_embed,
                dp_msg=dp_msg,
                dp_fusion=dp_fusion,
                M=spectrum_M,
                n_qubits=n_qubits,
                n_layers=n_layers,
                entangle_topo=entangle_topo,
                kernel_fn=kernel_fn,
                kernel_T=kernel_T,
                topk=topk,
                offdiag=offdiag,
                gate=gate,
                gate_init=gate_init,
                theta_S_scale0=theta_S_scale0,
                angle_norm=angle_norm,
                angle_radius=angle_radius,
                delay_in_s=delay_in_s,
                use_H=use_H,
            )
        elif backbone is None:
            backbone = SMambaBackbone(
                num_var=num_var,
                lookback=lookback,
                horizon=horizon,
                d_model=d_token,
                use_norm=False,  # 外层已有 RevIN
                quantum_mix_layers=qmix_modules,
                spectrum_inject=inject,
                head_qmix=head_qmix,
            )
        elif qmix_layers > 0 or spectrum_inject or head_agg:
            raise ValueError("qmix/spectrum_inject/head_agg 需要默认 SMambaBackbone（自定义 backbone 无插值点）")
        self.backbone = backbone

        # QK-Path（2026-08-15）：量子核独立预测通道 —— 主干保持 plain 最强形态，
        # 量子核在预测头端独立出预测 y_qk，与 y_main 融合：y = y_main + γ·y_qk。
        # 与"改表示"路线（qmix 注入/双路径融合）的本质区别：量子核有自己的预测目标和
        # 梯度路径（K 坍缩直接伤害自己的预测），不存在"被主干吸收"的梯度路径。
        # el 诊断（2026-08-15）：保真度核编码坍缩（offdiag_std 0.009）丢结构，
        # rbf/有向核保留坐标/相位信息有选择性（std 0.8/赢面）——kernel_fn 可换。
        self.qk_mix = None
        self.qk_head = None
        self.qk_gate = None
        if qk_path:
            if not (use_spectrum and use_S):
                raise ValueError("qk_path=True 需要 use_spectrum=True 且 use_S=True")
            self.qk_mix = QuantumMixLayer(
                d_token=d_token,
                n_qubits=n_qubits,
                n_layers=n_layers,
                M=spectrum_M,
                entangle_topo=entangle_topo,
                kernel_fn=kernel_fn,
                use_fmap=use_fmap,
                theta_S_scale0=theta_S_scale0,
                pre_norm=True,
                use_H=qk_use_H,   # False: K 纯 S 驱动（频谱结构，避免与主干 H 重合）
                use_S=True,
                reupload_source="S",
                angle_norm=angle_norm,
                angle_radius=angle_radius,
                norm_type=qk_norm,
                output_mode="raw",   # 返回 LN(K_n @ (H@W_q)), K
                kernel_T=kernel_T,
                topk=topk,
                offdiag=offdiag,
                gate=False,
                delay_in_s=delay_in_s,
            )
            self.qk_head = nn.Linear(d_token, horizon, bias=True)
            # 名字含 gate_raw → build_optimizer 自动放入门控组（gate_lr 生效）
            self._qk_gate_raw = nn.Parameter(torch.full((), float(qk_gate_init)))

        # 辅助残差损失（qdir_aux，2026-08-13）：给量子混合分支直接学习目标
        # L = MSE(y, y_true) + β·MSE(aux_head(LN(Hp)), (y_true - y_main).detach())
        # 目的：Hp 学会指向残差 → γ 门控才有关联信号可开（端到端梯度弱的老问题）
        self.aux_loss = aux_loss
        self.aux_beta = aux_beta
        if aux_loss:
            self.aux_head = nn.Linear(d_token, horizon, bias=True)

        # 核监督（qkern，2026-08-13）：直接教量子核学习数据的跨变量相关结构
        # L += λ·MSE(K_batch均值, |corr(x_norm窗口)|)——K 不再靠端到端梯度"猜"，
        # 而是被明确告知数据里的相关结构；相关矩阵是归一化结构量，比水平可迁移
        self.kernel_sup = kernel_sup

    def _prepare_input(self, x: torch.Tensor, x_mark: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x: (B, L, V)；可选 x_mark: (B, L, 4) [month, day, weekday, hour]。返回 backbone 输入。"""
        if self.use_periodic_feat and x_mark is not None:
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
        """量子混合主干前向传播。

        Returns:
            y: (B, H, V) 反归一化预测（= y_main，无旁路）。
            y_main: (B, H, V) 主预测（与 y 相同，保留签名兼容）。
            K: (B, V, V) 最后一个量子混合层的核矩阵（无量子混合时为 None）。
            (可选) y_norm, y_main_norm, None: 归一化空间预测与占位 correction_norm。
        """
        x_norm = self.revin(x, mode="norm")  # (B, L, V)

        x_in = self._prepare_input(x_norm, x_mark)  # (B, L, V or V+4)

        S = None
        if self.spectrum is not None:
            S = self.spectrum(x_norm)  # (B, V, 2M)，全 detach

        out = self.backbone(x_in, S=S)
        H, y_norm, K = out.H, out.y_main, out.K  # (B, V, d), (B, H, V), (B, V, V)
        self._last_qmix_out = out.qmix_out  # (B, V, d) 或 None（aux 损失用）

        # QK-Path：量子核独立预测通道（norm 空间融合，γ=0 → 精确 plain）
        if self.qk_mix is not None and S is not None:
            Hagg, K_qk = self.qk_mix(H, S)                     # (B, V, d), (B, V, V)
            y_qk = self.qk_head(Hagg).transpose(1, 2)          # (B, horizon, V)
            g = self._qk_gate_raw.clamp(0.0, 2.0)
            y_norm = y_norm + g * y_qk
            K = K_qk  # 核统计/K 监督用 qk 的 K

        self._last_K = K                     # (B, V, V) 或 None（核监督用）
        self._last_x_norm = x_norm           # (B, L, V)（核监督目标用）

        y = self.revin(y_norm, mode="denorm")

        if return_norm:
            return y, y, K, y_norm, y_norm, None
        return y, y, K

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
        """总损失 = MSE(y, y_true)（无旁路）；aux_loss 时加 β·MSE(aux_head(LN(Hp)), 残差)。"""
        loss = nn.functional.mse_loss(y, y_true)
        if self.aux_loss and self._last_qmix_out is not None:
            aux_pred = self.aux_head(self._last_qmix_out)          # (B, V, H)
            residual = (y_true - y_main).detach().transpose(1, 2)  # (B, V, H)
            loss = loss + self.aux_beta * nn.functional.mse_loss(aux_pred, residual)
        if self.kernel_sup > 0 and self._last_K is not None and self._last_x_norm is not None:
            # 核监督：K（批均值）→ 数据窗口的 |相关矩阵|（两者均 [0,1]、对角 1）
            xf = self._last_x_norm.double().reshape(-1, self._last_x_norm.shape[-1])
            corr = torch.corrcoef(xf.T).abs().float()              # (V, V)
            Kb = self._last_K.mean(0)                              # (V, V)
            loss = loss + self.kernel_sup * nn.functional.mse_loss(Kb, corr)
        return loss


__all__ = ["QCCMamba"]
