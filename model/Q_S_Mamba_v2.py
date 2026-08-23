"""
Q_S_Mamba_v2: 双路径重构架构

核心思想：
- 时间路径：单向 SSM，只做单变量时间建模
- 变量路径：量子核独占跨变量混合
- 两路径并列，最后融合

这是 ins.md 中 P2-1 的实现，但更激进：
1. 完全去掉双向扫描
2. 量子核直接作用于频谱特征（不是主干隐状态）
3. 可学习融合权重
"""

import torch
import torch.nn as nn
from ..layers.Embed import DataEmbedding
from ..layers.SelfAttention_Family import FullAttention, AttentionLayer
from ..layers.Autoformer_EncDec import EncoderLayer
from mamba_ssm.modules.mamba2 import Mamba2
from ..layers.Embed import PatchEmbedding
from ..layers.StandardNorm import Normalize
from qcc.quantum_mix import QuantumMixLayer
from qcc.spectrum import SpectrumFeature


class TemporalPath(nn.Module):
    """时间路径：单向 SSM，只做单变量时间建模"""
    def __init__(self, d_model, d_state, d_ff, dropout=0.1):
        super().__init__()
        self.mamba = Mamba2(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=2,
            ngroups=1,
            attn_layer_idx=(),
            attn_cfg=dict(softmax=False),
            seq_idx=None,
            scan_groups=1,
            scan_impl='causal',
            z_activation=None,
            use_bias=False,
            device=None,
            dtype=torch.float32,
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, D)
        residual = x
        x = self.mamba(x)
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x)
        return x


class CrossVarPath(nn.Module):
    """变量路径：量子核独占跨变量混合"""
    def __init__(self, d_model, configs):
        super().__init__()
        self.qmix = QuantumMixLayer(
            d_model=d_model,
            n_qubits=configs.n_qubits,
            n_layers=configs.qmix_n_layers,
            norm=configs.qmix_norm,
            T=configs.kernel_T,
            offdiag=configs.offdiag,
            entangle_topo=configs.entangle_topo,
            kernel_fn=configs.kernel_fn,
            angle_norm=configs.angle_norm,
            topk=getattr(configs, 'topk', 0),
            gate=getattr(configs, 'qmix_gate', False),
            gate_init=getattr(configs, 'qmix_gate_init', 1.0),
            fixed_s_scale=getattr(configs, 'qmix_fixed_s_scale', False),
            hp_scale=7.0 / configs.enc_in if getattr(configs, 'hp_scale_v', False) else 1.0,
            delay_in_s=configs.delay_in_s,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, S):
        # x: (B, L, D), S: (B, V, 2M)
        residual = x
        x = self.qmix(x, S)
        x = x + residual
        x = self.norm(x)
        return x


class FusionLayer(nn.Module):
    """融合层：可学习加权融合两路径"""
    def __init__(self, d_model):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x_temporal, x_crossvar):
        # 可学习加权融合
        x = self.alpha * x_temporal + self.beta * x_crossvar
        x = self.norm(x)
        return x


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.enc_in = configs.enc_in

        # 频谱特征
        self.spec = SpectrumFeature(
            M=configs.spectrum_M,
            time_align=configs.spectrum_time_align,
            freq_align=configs.spectrum_freq_align,
            freq_range=configs.spectrum_range,
            delay_in_s=configs.delay_in_s,
        )

        # Embedding
        self.enc_embedding = PatchEmbedding(
            configs.enc_in,
            configs.d_model,
            patch_len=8,
            stride=4,
            dropout=configs.dropout,
        )

        # 双路径重构
        self.temporal_path = TemporalPath(
            d_model=configs.d_model,
            d_state=configs.d_state,
            d_ff=configs.d_ff,
            dropout=configs.dropout,
        )
        self.crossvar_path = CrossVarPath(
            d_model=configs.d_model,
            configs=configs,
        )
        self.fusion = FusionLayer(configs.d_model)

        # Encoder（单向 SSM）
        self.encoder = nn.ModuleList([
            TemporalPath(
                d_model=configs.d_model,
                d_state=configs.d_state,
                d_ff=configs.d_ff,
                dropout=configs.dropout,
            )
            for _ in range(configs.e_layers)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(configs.d_model) if configs.model_norm else None

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = nn.GELU()
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalizer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev

        # 计算频谱特征
        S = self.spec(x_enc)

        # Embedding
        x_enc = self.enc_embedding(x_enc, x_mark_enc)

        # 时间路径：单向 SSM
        x_temporal = x_enc
        for layer in self.encoder:
            x_temporal = layer(x_temporal)

        # 变量路径：量子核独占跨变量
        x_crossvar = self.crossvar_path(x_enc, S)

        # 融合
        x = self.fusion(x_temporal, x_crossvar)

        if self.norm is not None:
            x = self.norm(x)

        # Predict
        dec_out = self.projection(x)
        dec_out = dec_out[:, -self.pred_len:, :]

        # De-Normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None
