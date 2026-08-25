"""Frequency-Token S-Mamba（2026-08-23，第一轮筛选候选）。

动机（历史结论）：
  - add-on 量子混合与官方双向 Mamba 跨变量建模冗余（只赢 ~0.3%，且赢家是经典 rbf 核）。
  - P2-1 双路径（量子独占跨变量）失败（+80%）。
  - 频域双轴对齐频谱 S 是主干拿不到的独立信息（Mamba 时域扫变量轴，看不见频域结构）。
设计：
  把对齐频谱 S (B, V, 2M) 投影成"频率 token"，拼进反向嵌入的 token 序列
  [变量token × V ; 频率token × V ; 时间特征token × F]，让 Mamba 跨变量扫描时
  同时读到时域表征与频域结构（主干本身不冗余、可端到端训练）。
  freq_tokens=1: 仅频率 token（纯频域信息增益测试）
  freq_tokens=2: 频率 token + 量子核跨变量混合（量子核读 S，保留差异化能力）
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.S_Mamba import Model as _S_Mamba
from qcc import QuantumMixLayer, SpectrumFeature


class Model(_S_Mamba):
    def __init__(self, configs):
        super().__init__(configs)
        self.V = configs.enc_in
        self.ft = int(getattr(configs, "freq_tokens", 0))  # 0/1/2
        if self.ft > 0:
            self.spectrum = SpectrumFeature(
                M=configs.spectrum_M,
                sample_range=configs.spectrum_range,
                amp_normalize=configs.spectrum_amp_normalize,
                time_align=configs.spectrum_time_align,
                freq_align=configs.spectrum_freq_align,
                delay_in_s=configs.delay_in_s,
            )
            s_dim = 2 * configs.spectrum_M + (1 if configs.delay_in_s else 0)
            self.freq_proj = nn.Linear(s_dim, configs.d_model)
            nn.init.xavier_uniform_(self.freq_proj.weight)

        q = int(getattr(configs, "qmix_layers", 0))
        self.qmix_layers = q
        self.use_S_only = getattr(configs, "qmix_use_S_only", False)
        if self.ft >= 2 and q > 0:
            if q > configs.e_layers:
                raise ValueError(f"qmix_layers={q} must be <= e_layers={configs.e_layers}")
            hp_scale = 7.0 / configs.enc_in if getattr(configs, "hp_scale_v", False) else 1.0
            self.qmix = nn.ModuleList([
                QuantumMixLayer(
                    d_token=configs.d_model,
                    n_qubits=configs.n_qubits,
                    n_layers=configs.qmix_n_layers,
                    M=configs.spectrum_M,
                    entangle_topo=configs.entangle_topo,
                    kernel_fn=configs.kernel_fn,
                    use_fmap=True,
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
                    hp_scale=hp_scale,
                    delay_in_s=configs.delay_in_s,
                    fixed_s_scale=bool(getattr(configs, "qmix_fixed_s_scale", False)),
                ) for _ in range(q)
            ])
            self._last_K = None
            self._last_S = None

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # --- 官方实例归一化 ---
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        V = self.V

        S = None
        if self.ft > 0:
            S = self.spectrum(x_enc)  # (B, V, 2M(+1))，全 detach

        # --- embedding: (B, V+F, d) ---
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        # --- 频率 token 拼接：[var × V ; freq × V ; time × F] ---
        if self.ft > 0:
            freq_tok = self.freq_proj(S.float())  # (B, V, d)
            enc_out = torch.cat([enc_out[:, :V], freq_tok, enc_out[:, V:]], dim=1)

        # --- 编码器（量子核可选，作用在变量 token 上） ---
        if self.ft >= 2 and self.qmix_layers > 0:
            for i, layer in enumerate(self.encoder.attn_layers):
                enc_out, _ = layer(enc_out, attn_mask=None)
                if i < len(self.qmix):
                    enc_var = enc_out[:, :V, :]
                    enc_mixed, K = self.qmix[i](enc_var, S, use_S_only=self.use_S_only)
                    enc_out = torch.cat([enc_mixed, enc_out[:, V:, :]], dim=1)
                    self._last_K = K
                    self._last_S = S
            if self.encoder.norm is not None:
                enc_out = self.encoder.norm(enc_out)
        else:
            enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # --- projector + 裁剪 + denorm（前 V 个 token 是变量） ---
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    # forward() 继承官方


__all__ = ["Model"]
