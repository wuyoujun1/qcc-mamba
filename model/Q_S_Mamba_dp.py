"""P2-1 双路径重构（2026-08-17，官方基线移植）。

架构（ins.md P2-1，DeMa 双路径分工哲学，保留量子核/频谱对齐元件）：
    x (B, L, V) → 实例归一化 → 对齐频谱 S (B, V, 2M)   [SpectrumFeature 全 detach]
      ├→ 时间路径 TimePath：per-variable 时间 SSM，变量间零通信 → H_time (B, V, d)
      ├→ 变量路径 VarPath：量子核独占跨变量（K 由 H_time 语义 + S 重上传驱动）
      │     → 聚合消息 → LN(H_var) (B, V, d)   [QuantumMixLayer output_mode="raw"]
      └→ 融合：H = H_time + γ·H_var（γ 可学习 init=dp_gate_init，γ=0 → H ≡ H_time）
        → projector (d → pred_len) → denorm

动机：官方 S-Mamba（反转嵌入 + 双向 Mamba 扫变量轴）本身已做跨变量建模，
    量子混合层作为附加层冗余，增量被主干吸收（add-on 实验只赢 0.3%）。
    本架构主干（TimePath）只做时间建模（SSM 沿时间维，变量间零通信），
    量子核是唯一跨变量通道，K 读 H_time（变量语义）+ S（频域结构，主干拿不到）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from qcc import QuantumMixLayer, SpectrumFeature


class TimePath(nn.Module):
    """per-variable 时间 SSM：每个变量一条独立时间序列，变量间零通信。

    输入 x_enc (B, L, V) + x_mark (B, L, F)（时间特征广播进每个变量行）。
    输出 H_time (B, V, d_token)。
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

    def forward(self, x_enc: torch.Tensor, x_mark: torch.Tensor | None = None) -> torch.Tensor:
        """x_enc: (B, L, V) → H_time: (B, V, d_token)。"""
        B, L, V = x_enc.shape
        z = x_enc.permute(0, 2, 1).unsqueeze(-1)  # (B, V, L, 1)
        if x_mark is not None and self.n_feats > 0:
            xm = x_mark[:, :, : self.n_feats]  # (B, L, F)
            xm_b = xm.unsqueeze(1).expand(B, V, L, self.n_feats)  # (B, V, L, F)
            z = torch.cat([z, xm_b], dim=-1)  # (B, V, L, 1+F)
        z = self.in_proj(z)  # (B, V, L, d_time)
        if self.var_emb is not None:
            z = z + self.var_emb[None, :, None, :]  # 变量身份（zero init）
        # fold V 进 batch：per-channel SSM 标准做法，同一模块实例 = 共享权重
        z = z.reshape(B * V, L, z.shape[-1]).contiguous()
        for m, norm in zip(self.layers, self.norms):
            z = norm(z + m(z))  # pre-norm 残差
        z = z.reshape(B, V, L, z.shape[-1])
        h = z.mean(dim=2) if self.pool == "mean" else z[:, :, -1, :]  # (B, V, d_time)
        return self.out_proj(h)  # (B, V, d_token)


class VarPath(nn.Module):
    """变量路径：量子核独占跨变量。

    K 由 H_time（变量语义，随表征演化）+ S（频谱重上传）驱动；
    消息 = msg_proj(S)（频谱嵌入）/ H_time / both（dp_msg）。
    复用 QuantumMixLayer(output_mode="raw") → LN(Hp), K。
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
        delay_in_s: bool = False,
        angle_groups: int = 1,  # 第五轮：多组角度桥接
        kernel_group_agg: str = "product",
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
            use_H=True,  # K 读变量语义（H 驱动 fmap 首层角度），旧仓库 V2 诊断：H 语义比 S 谱形更有益
            use_S=True,
            reupload_source="S",
            angle_norm=angle_norm,
            angle_radius=1.0,
            norm_type="softmax",  # GAT 式行归一化，消息幅度 O(1)，无 1/V 稀释
            output_mode="raw",    # 返回 LN(Hp), K
            kernel_T=kernel_T,
            topk=topk,
            offdiag=offdiag,
            gate=False,
            delay_in_s=delay_in_s,
            angle_groups=angle_groups,  # 第五轮：多组角度桥接
            kernel_group_agg=kernel_group_agg,
        )

    def forward(self, S: torch.Tensor, H_time: torch.Tensor, msg: str = "S"):
        """S: (B, V, 2M(+1)) → H_var: (B, V, d), K: (B, V, V)。"""
        S_f = S.float()
        if msg == "S":
            src = self.msg_proj(S_f)
        elif msg == "H":
            src = H_time
        else:  # both
            src = self.msg_proj(S_f) + H_time
        H_var, K = self.qmix(src, S_f)
        return H_var, K


class Model(nn.Module):
    """P2-1 双路径模型（官方仓库接口：forecast/forward）。"""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.num_var = configs.enc_in

        self.dp_time_layers = int(getattr(configs, "dp_time_layers", 2))
        self.dp_time_dim = int(getattr(configs, "dp_time_dim", 256))
        self.dp_time_pool = getattr(configs, "dp_time_pool", "mean")
        self.dp_var_embed = bool(getattr(configs, "dp_var_embed", True))
        self.dp_msg = getattr(configs, "dp_msg", "S")
        self.dp_fusion = getattr(configs, "dp_fusion", "add")

        # n_feats：官方 timeenc=0 的 x_mark 特征数（freq='h' → 4）
        self.n_feats = 4 if getattr(configs, "use_dp_feats", False) else 0

        # 频谱（全 detach，确定性函数）
        self.spectrum = SpectrumFeature(
            M=configs.spectrum_M,
            sample_range=configs.spectrum_range,
            amp_normalize=configs.spectrum_amp_normalize,
            time_align=configs.spectrum_time_align,
            freq_align=configs.spectrum_freq_align,
            delay_in_s=configs.delay_in_s,
        )

        # 时间路径
        self.time_path = TimePath(
            num_var=self.num_var,
            d_time=self.dp_time_dim,
            d_token=configs.d_model,
            n_layers=self.dp_time_layers,
            d_state=configs.d_state,
            n_feats=self.n_feats,
            pool=self.dp_time_pool,
            var_embed=self.dp_var_embed,
        )

        # 变量路径（量子核）
        self.var_path = None
        if self.dp_fusion != "time_only":
            self.var_path = VarPath(
                d_token=configs.d_model,
                M=configs.spectrum_M,
                n_qubits=configs.n_qubits,
                n_layers=configs.qmix_n_layers,
                entangle_topo=configs.entangle_topo,
                kernel_fn=configs.kernel_fn,
                kernel_T=configs.kernel_T,
                topk=configs.topk,
                offdiag=configs.offdiag,
                theta_S_scale0=configs.theta_S_scale0,
                angle_norm=configs.angle_norm,
                delay_in_s=configs.delay_in_s,
                angle_groups=int(getattr(configs, "angle_groups", 1)),  # 第五轮：多组角度桥接
                kernel_group_agg=getattr(configs, "kernel_group_agg", "product"),
            )
            gate_init = float(getattr(configs, "dp_gate_init", 0.05))
            self._gate_raw = nn.Parameter(torch.full((), gate_init))
        else:
            gate_init = float(getattr(configs, "dp_gate_init", 0.05))
            self._gate_raw = nn.Parameter(torch.full((), gate_init))

        # 投影头：每变量 d → pred_len
        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)
        self._last_K = None
        self._last_S = None

    @property
    def gate_value(self) -> torch.Tensor:
        """融合门控 γ（clamp [0, 2]；γ=0 → H ≡ H_time）。"""
        return self._gate_raw.clamp(0.0, 2.0)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # --- 实例归一化（与官方一致） ---
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        V = self.num_var

        # --- 频谱 S：归一化后的 x_enc 上计算，内部全 detach ---
        S = self.spectrum(x_enc)  # (B, V, 2M) 或 (B, V, 2M+1)

        # --- 时间路径 ---
        xm = x_mark_enc if self.n_feats > 0 else None
        H_time = self.time_path(x_enc, xm)  # (B, V, d)

        # --- 变量路径 + 融合 ---
        if self.var_path is not None:
            H_var, K = self.var_path(S, H_time, msg=self.dp_msg)  # (B, V, d), (B, V, V)
            H = H_time + self.gate_value * H_var
            self._last_K = K
            self._last_S = S
        else:
            H = H_time

        # --- 投影 + denorm ---
        dec_out = self.projector(H).permute(0, 2, 1)[:, :, :V]  # (B, pred_len, V)
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len:, :]  # [B, pred_len, V]


__all__ = ["Model"]
