import torch
import torch.nn as nn

from model.S_Mamba import Model as _S_Mamba
from qcc import QuantumMixLayer, SpectrumFeature


class Model(_S_Mamba):
    """Official S-Mamba + optional quantum mixing (qcc port).

    qmix_layers=0 -> forward path bit-identical to official Model
    (no spectrum/qmix modules created; module tree and state_dict
    interchangeable with the official S-Mamba).
    """

    def __init__(self, configs):
        super().__init__(configs)
        self.V = configs.enc_in  # 前 V 个 token 是变量（后续是时间特征 token）
        q = int(getattr(configs, "qmix_layers", 0))
        self.qmix_layers = q
        # P1-1: 量子核直接读频谱 S（不读主干 H）
        self.use_S_only = getattr(configs, "qmix_use_S_only", False)
        # hp_scale: 7/V 归一化（旧仓库赢家关键）
        hp_scale = 7.0 / configs.enc_in if getattr(configs, "hp_scale_v", False) else 1.0
        if q > 0:
            if q > configs.e_layers:
                raise ValueError(f"qmix_layers={q} must be <= e_layers={configs.e_layers}")
            # S 是确定性函数（内部全 detach），只依赖输入
            self.spectrum = SpectrumFeature(
                M=configs.spectrum_M,
                sample_range=configs.spectrum_range,
                amp_normalize=configs.spectrum_amp_normalize,
                time_align=configs.spectrum_time_align,
                freq_align=configs.spectrum_freq_align,
                delay_in_s=configs.delay_in_s,
            )
            self.qmix = nn.ModuleList([
                QuantumMixLayer(
                    d_token=configs.d_model,
                    n_qubits=configs.n_qubits,
                    n_layers=configs.qmix_n_layers,
                    M=configs.spectrum_M,
                    entangle_topo=configs.entangle_topo,
                    kernel_fn=configs.kernel_fn,
                    use_fmap=bool(getattr(configs, "qmix_use_fmap", 1)),
                    theta_S_scale0=configs.theta_S_scale0,
                    pre_norm=True,
                    use_H=bool(configs.qmix_use_H),
                    use_S=True,
                    reupload_source="S",
                    angle_norm=configs.angle_norm,
                    angle_radius=1.0,
                    norm_type=configs.qmix_norm,
                    output_mode="residual",
                    kernel_T=configs.kernel_T,
                    topk=configs.topk,
                    offdiag=configs.offdiag,
                    gate=configs.qmix_gate,
                    gate_init=configs.qmix_gate_init,
                    gate_per_var=bool(getattr(configs, "qmix_gate_pv", False)),
                    gate_pv_init=getattr(configs, "qmix_gate_pv_init", 3.0),
                    gate_pv_src=bool(getattr(configs, "qmix_gate_pv_src", False)),
                    gate_pv_src_init=getattr(configs, "qmix_gate_pv_src_init", -3.0),
                    hp_scale=hp_scale,
                    delay_in_s=configs.delay_in_s,
                    fixed_s_scale=bool(getattr(configs, "qmix_fixed_s_scale", False)),
                    kernel_kappa=getattr(configs, "kernel_exp", 1.0),
                    kernel_power=getattr(configs, "kernel_power", 2.0),
                    kernel_exp_learn=getattr(configs, "kernel_exp_learn", False),
                    kernel_power_learn=getattr(configs, "kernel_power_learn", False),
                    kernel_phase=getattr(configs, "kernel_phase", 1.0),
                    joint_encoding=bool(getattr(configs, "qmix_joint", 0)),
                    joint_topo=getattr(configs, "joint_topo", "var_linear"),
                    n_vars=configs.enc_in,
                ) for _ in range(q)
            ])
            self._last_K = None  # (B, V, V) 诊断用
            self._last_K_norm = None  # QF-1: 归一化消息权重 (B, V, V)，对齐损失用
            self._last_S = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # --- 官方实例归一化（逐位一致） ---
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # N = enc_in（时间特征不在 x_enc 里）
        V = self.V

        # --- 频谱 S：实例归一化后的 x_enc 上计算，内部全 detach ---
        S = None
        if self.qmix_layers > 0:
            S = self.spectrum(x_enc)  # (B, V, 2M) 或 (B, V, 2M+1)

        # --- embedding: (B, L, 7) -> (B, 11, d_model)（7 变量 + 4 时间特征 token） ---
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        if self.qmix_layers > 0:
            # P1-1: 量子核直接读频谱 S（不读主干 H）
            use_S_only = getattr(self, 'use_S_only', False)
            # 手动循环官方 EncoderLayer（与 Encoder.forward 相同操作顺序）
            for i, layer in enumerate(self.encoder.attn_layers):
                enc_out, _ = layer(enc_out, attn_mask=None)
                if i < len(self.qmix):
                    enc_var = enc_out[:, :V, :]                     # (B, 7, d)
                    enc_mixed, K = self.qmix[i](enc_var, S, use_S_only=use_S_only)  # (B, 7, d), (B, 7, 7)
                    enc_out = torch.cat([enc_mixed, enc_out[:, V:, :]], dim=1)
                    self._last_K = K
                    self._last_K_norm = getattr(self.qmix[i], '_last_K_norm', None)
                    self._last_S = S
            if self.encoder.norm is not None:
                enc_out = self.encoder.norm(enc_out)
        else:
            enc_out, attns = self.encoder(enc_out, attn_mask=None)   # 官方原路径

        # --- projector + 裁剪 + denorm（逐位一致） ---
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    # forward() 继承官方：返回普通 tensor（exp 代码依赖 outputs[:, -pred_len:, :]）
