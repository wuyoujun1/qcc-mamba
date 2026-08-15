"""双路径主干（P2-1，2026-08-15）：时间 SSM 单向 + 量子核独占跨变量。

架构（ins.md P2-1，DeMa 双路径分工哲学，保留量子核/频谱对齐元件）：
    x (B, L, V) → RevIN → 对齐频谱 S (B, V, 2M)   [SpectrumFeature 已有，全 detach]
      ├→ 时间路径 TimePath：per-variable 时间 SSM，变量间零通信 → H_time (B, V, d)
      ├→ 变量路径 VarPath：S 驱动量子编码 → K (B, V, V) → 行 softmax 归一化
      │     → 聚合频谱消息 → LN(H_var) (B, V, d)   [复用 QuantumMixLayer(use_H=False, output_mode="raw")]
      └→ 融合：H = H_time + γ·H_var（γ 可学习 init=gate_init，γ=0 → H ≡ H_time）
        → pred_head → RevIN denorm

动机：现有主干（反转嵌入 + 双向 Mamba 扫变量轴）本身已做跨变量建模，
    量子混合层是冗余的第二跨变量通道，增量被主干吸收（"加了不如没加"）。
    本架构中量子核是唯一跨变量通道；K 读 S（频域结构，主干拿不到）而非 H（信息重合）；
    softmax 行归一化后消息幅度 O(1)，不存在 1/V 稀释（hp_scale_v 不再需要）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .interface import BackboneOutput, BaseBackbone
from qcc.quantum_mix import QuantumMixLayer


class TimePath(nn.Module):
    """per-variable 时间 SSM：每个变量一条独立时间序列，变量间零通信。

    输入 x_in: (B, L, V_eff)，后 n_feats 列是周期时间特征（对每个变量广播进行内通道）。
    输出 H_time: (B, V, d_token)。
    """

    def __init__(
        self,
        num_var: int,
        d_time: int = 256,
        d_token: int = 512,
        n_layers: int = 2,
        d_state: int = 16,
        n_feats: int = 0,
        pool: str = "mean",
        var_embed: bool = True,
    ):
        super().__init__()
        self.num_var = num_var
        self.n_feats = n_feats
        self.pool = pool
        self.in_proj = nn.Linear(1 + n_feats, d_time)
        self.var_emb = nn.Parameter(torch.zeros(num_var, d_time)) if var_embed else None
        self.layers = nn.ModuleList(
            [
                Mamba(d_model=d_time, d_state=d_state, d_conv=2, expand=1)
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(d_time) for _ in range(n_layers)])
        self.out_proj = nn.Linear(d_time, d_token)

    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """x_in: (B, L, V_eff) → H_time: (B, V, d_token)。"""
        B, L, Ve = x_in.shape
        V = self.num_var
        xv = x_in[:, :, :V]  # (B, L, V) 变量列
        xm = x_in[:, :, V:]  # (B, L, F) 时间特征列
        z = xv.permute(0, 2, 1).unsqueeze(-1)  # (B, V, L, 1)
        if xm.shape[-1] > 0:
            # 时间特征广播进每个变量行（共享上下文，非跨变量消息）
            xm_b = xm.unsqueeze(1).expand(B, V, L, xm.shape[-1])  # (B, V, L, F)
            z = torch.cat([z, xm_b], dim=-1)  # (B, V, L, 1+F)
        z = self.in_proj(z)  # (B, V, L, d_time)
        if self.var_emb is not None:
            z = z + self.var_emb[None, :, None, :]  # 变量身份（zero init，先不区分）
        # fold V 进 batch：per-channel SSM 标准做法，同一模块实例 = 共享权重
        z = z.reshape(B * V, L, z.shape[-1]).contiguous()
        for m, norm in zip(self.layers, self.norms):
            z = norm(z + m(z))  # pre-norm 残差
        z = z.reshape(B, V, L, z.shape[-1])
        h = z.mean(dim=2) if self.pool == "mean" else z[:, :, -1, :]  # (B, V, d_time)
        return self.out_proj(h)  # (B, V, d_token)


class VarPath(nn.Module):
    """变量路径：量子核独占跨变量。

    K 纯由 S 驱动（QuantumMixLayer use_H=False → fmap 首层零角度、重上传纯 proj_S(S)）；
    消息 = msg_proj(S)（频谱嵌入，可训练）。梯度只经 msg_proj / W_q / proj_S。
    """

    def __init__(
        self,
        d_token: int = 512,
        M: int = 32,
        n_qubits: int = 2,
        n_layers: int = 2,
        entangle_topo: str = "linear",
        kernel_fn=None,
        kernel_T: float = 0.1,
        topk: int = 0,
        offdiag: bool = True,
        theta_S_scale0: float = 0.5,
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
        delay_in_s: bool = False,
    ):
        super().__init__()
        s_dim = 2 * M + (1 if delay_in_s else 0)
        self.msg_proj = nn.Linear(s_dim, d_token)  # 频谱消息投影
        self.qmix = QuantumMixLayer(
            d_token=d_token,
            n_qubits=n_qubits,
            n_layers=n_layers,
            M=M,
            entangle_topo=entangle_topo,
            kernel_fn=kernel_fn,
            use_fmap=True,
            theta_S_scale0=theta_S_scale0,
            pre_norm=True,
            use_H=False,  # K 纯由 S 驱动（首层零角度）
            use_S=True,
            reupload_source="S",
            angle_norm=angle_norm,
            angle_radius=angle_radius,
            norm_type="softmax",   # GAT 式行归一化，消息幅度 O(1)，无 1/V 稀释
            output_mode="raw",     # 返回 LN(Hp), K
            kernel_T=kernel_T,
            topk=topk,
            offdiag=offdiag,
            gate=False,
            delay_in_s=delay_in_s,
        )

    def forward(self, S: torch.Tensor, H_time: torch.Tensor, msg: str = "S"):
        """S: (B, V, 2M(+1)) 全 detach → H_var: (B, V, d), K: (B, V, V)。

        msg: 消息源 'S'（频谱嵌入）/ 'H'（时间表示，重合对照）/ 'both'。
        """
        S_f = S.float()
        if msg == "S":
            src = self.msg_proj(S_f)
        elif msg == "H":
            src = H_time
        else:  # both
            src = self.msg_proj(S_f) + H_time
        H_var, K = self.qmix(src, S_f)
        return H_var, K


class DualPathBackbone(BaseBackbone):
    """双路径主干：时间路径 + 量子核变量路径 + 加权融合。

    融合（dp_fusion）：
        "add"       — H = H_time + γ·H_var（γ 可学习 init=gate_init clamp [0,2]；
                      γ=0 → H ≡ H_time，结构保证不更差）
        "concat"    — H = cat([H_time, H_var], -1)，head 输入 2d（消融）
        "time_only" — 无变量路径（新架构的 plain 基线）

    返回 BackboneOutput(H, y_main, K, qmix_out=H_var)——qmix_out 复用使
    aux_loss / kernel_sup 在 QCCMamba.compute_loss 中零改动生效。
    """

    def __init__(
        self,
        num_var: int = 321,
        lookback: int = 720,
        horizon: int = 96,
        d_token: int = 512,
        n_feats: int = 0,
        dp_time_layers: int = 2,
        dp_time_dim: int = 256,
        dp_time_pool: str = "mean",
        dp_var_embed: bool = True,
        dp_msg: str = "S",
        dp_fusion: str = "add",
        M: int = 32,
        n_qubits: int = 2,
        n_layers: int = 2,
        entangle_topo: str = "linear",
        kernel_fn=None,
        kernel_T: float = 0.1,
        topk: int = 0,
        offdiag: bool = True,
        gate: bool = True,
        gate_init: float = 0.05,
        theta_S_scale0: float = 0.5,
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
        delay_in_s: bool = False,
    ):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.horizon = horizon
        self.d_token = d_token
        self.dp_msg = dp_msg
        self.dp_fusion = dp_fusion
        self.gate = gate and dp_fusion == "add"

        self.time_path = TimePath(
            num_var=num_var,
            d_time=dp_time_dim,
            d_token=d_token,
            n_layers=dp_time_layers,
            n_feats=n_feats,
            pool=dp_time_pool,
            var_embed=dp_var_embed,
        )

        self.var_path = None
        if dp_fusion != "time_only":
            self.var_path = VarPath(
                d_token=d_token,
                M=M,
                n_qubits=n_qubits,
                n_layers=n_layers,
                entangle_topo=entangle_topo,
                kernel_fn=kernel_fn,
                kernel_T=kernel_T,
                topk=topk,
                offdiag=offdiag,
                theta_S_scale0=theta_S_scale0,
                angle_norm=angle_norm,
                angle_radius=angle_radius,
                delay_in_s=delay_in_s,
            )
            if self.gate:
                # 名字含 gate_raw → build_optimizer 自动放入门控组（gate_lr 生效）
                self._gate_raw = nn.Parameter(torch.full((), float(gate_init)))

        head_in_dim = d_token * (2 if dp_fusion == "concat" else 1)
        self.pred_head = nn.Linear(head_in_dim, horizon, bias=True)

    @property
    def gate_value(self) -> torch.Tensor:
        """混合门控强度（γ=0 → H ≡ H_time，架构等于 dp_time）。"""
        if not self.gate:
            return torch.tensor(1.0, device=self.pred_head.weight.device)
        return self._gate_raw.clamp(0.0, 2.0)

    def forward(self, x: torch.Tensor, S: torch.Tensor | None = None) -> BackboneOutput:
        """x: (B, L, V_eff)（后 n_feats 列是时间特征）→ BackboneOutput。"""
        B, L, Ve = x.shape
        V = self.num_var

        H_time = self.time_path(x)  # (B, V, d)

        if self.var_path is None:
            y_main = self.pred_head(H_time).transpose(1, 2)  # (B, horizon, V)
            return BackboneOutput(H=H_time, y_main=y_main)

        if S is None:
            raise ValueError("dual_path 变量路径需要 S（对齐频谱特征）")
        H_var, K = self.var_path(S, H_time, msg=self.dp_msg)  # (B, V, d), (B, V, V)

        if self.dp_fusion == "concat":
            H = torch.cat([H_time, H_var], dim=-1)  # (B, V, 2d)
        elif self.gate:
            H = H_time + self.gate_value * H_var  # γ=0 → H ≡ H_time
        else:
            H = H_time + H_var  # 无门控满强度（dp_ng）

        y_main = self.pred_head(H).transpose(1, 2)  # (B, horizon, V)
        return BackboneOutput(H=H, y_main=y_main, K=K, qmix_out=H_var)


__all__ = ["TimePath", "VarPath", "DualPathBackbone"]
