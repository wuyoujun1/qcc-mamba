"""S-Mamba 官方实现的统一 Backbone 封装。

适配自 https://github.com/sci-m-wang/S-D-Mamba
输出 BackboneOutput(H, y_main) 以对接 QCC-Mamba 旁路。

数据流：
    x: (B, L, V)
    → DataEmbeddingInverted: (B, V, E)      # 每个变量一个 token
    → SMambaEncoder: (B, V, E)              # H = enc_out
    → pred_head: (B, V, E) → (B, V, H) → (B, H, V)  # y_main
"""
from __future__ import annotations

import torch
import torch.nn as nn
from mamba_ssm import Mamba

from .interface import BackboneOutput, BaseBackbone
from .smamba_embed import DataEmbeddingInverted
from .smamba_encdec import SMambaEncoder, SMambaEncoderLayer


class SMambaBackbone(BaseBackbone):
    """S-Mamba backbone 的统一封装。

    Args:
        num_var: 变量数 V（即序列中变量的数量）。
        lookback: 输入窗口 L。
        horizon: 预测步长 H。
        d_model: 模型隐藏维度（也是 QCC 旁路的 token 维度 d_token）。
        d_state: SSM 状态扩展因子（默认 16）。
        e_layers: 编码器层数（默认 2）。
        d_ff: FFN 隐藏维度（默认 4 * d_model）。
        dropout: dropout 概率（默认 0.1）。
        activation: 激活函数（默认 "gelu"）。
        use_norm: 是否使用 Instance Normalization（默认 False，因为外层已有 RevIN）。
        classify_strategy: 分类策略（默认 None，不使用）。
    """

    def __init__(
        self,
        num_var: int = 321,
        lookback: int = 720,
        horizon: int = 96,
        d_model: int = 512,
        d_state: int = 16,
        e_layers: int = 2,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_norm: bool = False,
        classify_strategy: str | None = None,
        quantum_mix_layers: nn.ModuleList | None = None,
        spectrum_inject: nn.Module | None = None,
        head_qmix: nn.Module | None = None,
    ):
        super().__init__()
        self.num_var = num_var
        self.lookback = lookback
        self.horizon = horizon
        self.d_model = d_model
        self.use_norm = use_norm
        self.classify_strategy = classify_strategy
        d_ff = d_ff or 4 * d_model

        # (B, L, V) → (B, V, d_model)
        self.enc_embedding = DataEmbeddingInverted(
            c_in=lookback, d_model=d_model, dropout=dropout,
        )

        # Encoder: 双向 Mamba 堆叠
        self.encoder = SMambaEncoder(
            [
                SMambaEncoderLayer(
                    Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1),
                    Mamba(d_model=d_model, d_state=d_state, d_conv=2, expand=1),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # 预测头：(B, V, d_model) → (B, V, horizon) → (B, horizon, V)
        # 方案 1（2026-08-11 晚）：head_qmix 时拼接量子聚合特征，输入维度 2d
        self.head_qmix = head_qmix
        head_in_dim = d_model * (2 if head_qmix is not None else 1)
        self.pred_head = nn.Linear(head_in_dim, horizon, bias=True)

        # 量子混合（可选，2026-08-11 重构）：每层 Mamba 后插一层 QuantumMixLayer
        self.quantum_mix_layers = quantum_mix_layers
        # 频谱注入（可选）：S (B, V, 2M) → (B, V, d_model)，加到 embedding 输出
        self.spectrum_inject = spectrum_inject

    def forward(self, x: torch.Tensor, S: torch.Tensor | None = None) -> BackboneOutput:
        """x: (B, L, V) → BackboneOutput(H, y_main)

        Args:
            x: 输入序列，形状 (B, L, V_eff)。如果外部拼接了时间特征，
               V_eff = num_var + n_feats。H 会包含 num_var 个变量 token；
               如果 x 额外带了时间特征（如 sin/cos），H 维度会相应放大。
               本函数使用 self.num_var（构造时指定的原始变量数）正确切片。
            S: 对齐频谱特征 (B, V, 2M)（量子混合层 / 频谱注入需要时提供）。
        """
        B, L, V_eff = x.shape
        V = self.num_var  # 原始变量数（构造时指定）

        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = x / stdev

        # Embedding: (B, L, V_eff) → (B, V_eff, d_model)
        enc_out = self.enc_embedding(x)  # (B, V_eff, d_model)

        # 频谱注入：周期结构进主干第一层（只注入前 V 个变量 token，S 是 (B,V,2M)）
        if self.spectrum_inject is not None and S is not None:
            enc_out[:, :V, :] = enc_out[:, :V, :] + self.spectrum_inject(S.float())

        K_last = None
        qmix_last = None  # 无量子混合时保持 None（plain）——2026-08-14 修：之前在 if 分支内导致 plain 崩溃
        if self.quantum_mix_layers is not None and len(self.quantum_mix_layers) > 0:
            # 量子混合主干：每层 Mamba 后插量子核跨变量混合
            # 量子混合只作用于 V 个变量 token（fmap 的 S 角度需要与 H 匹配），
            # 时间特征 token 保持不动，与混合后结果拼接继续下一层 Mamba。
            layers = self.encoder.attn_layers
            for i, layer in enumerate(layers):
                enc_out, _ = layer(enc_out, attn_mask=None)
                if i < len(self.quantum_mix_layers):
                    enc_var = enc_out[:, :V, :]
                    enc_mixed, K = self.quantum_mix_layers[i](enc_var, S)
                    enc_out = torch.cat([enc_mixed, enc_out[:, V:, :]], dim=1)
                    K_last = K
                    qmix_last = self.quantum_mix_layers[i]._last_Hp  # (B, V, d) aux 用
            if self.encoder.norm is not None:
                enc_out = self.encoder.norm(enc_out)
        else:
            enc_out = self.encoder(enc_out, attn_mask=None)

        # 取前 V 个变量 token（去掉时间特征 token）
        H = enc_out[:, :V, :]  # (B, V, d_model)

        # 方案 1：预测头级量子聚合 —— K 直接在输出端加权别的变量的 token，零稀释
        if self.head_qmix is not None and S is not None:
            Hagg, K_head = self.head_qmix(H, S)  # (B, V, d) = LN(softmax(K)·H·W_q)
            K_last = K_head
            head_in = torch.cat([H, Hagg], dim=-1)  # (B, V, 2d)
        else:
            head_in = H

        # 预测头：(B, V, [2]d) → (B, H, V)
        y_main = self.pred_head(head_in).transpose(1, 2)  # (B, horizon, V)

        if self.use_norm:
            y_main = y_main * (stdev[:, 0, :V].unsqueeze(1).repeat(1, self.horizon, 1))
            y_main = y_main + (means[:, 0, :V].unsqueeze(1).repeat(1, self.horizon, 1))

        return BackboneOutput(H=H, y_main=y_main, K=K_last, qmix_out=qmix_last)


__all__ = ["SMambaBackbone"]
