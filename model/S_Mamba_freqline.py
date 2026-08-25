"""S-Mamba + 频率主线旁路（FITS 风格，2026-08-23）。

动机：
  官方 S-Mamba 是倒置模型（变量作 token，Mamba 扫变量轴），主干拿不到时间轴频域结构。
  FreDF 只在 loss 监督频域（模型本身看不到周期）。Q_S_Mamba_ft 把频谱投影成 token
  注入已失败（s1 全被拒，复杂度过高）。本项目用 FITS（ICLR'24）思路做**轻量频率旁路**：

    y = y_main + freq_scale · iFFT(W_freq ⊙ rFFT(x_enc))

  - W_freq：可学习的复数低频滤波（低通，只保留最低比例 bin）
  - 对输入做 rFFT → 逐 bin 缩放 → iFFT 取后 pred_len 步 → 与主干输出叠加
  - 主干 Mamba 完全不变，旁路是纯并行、端到端可训练
  - freq_mainline 默认 0（=官方 S-Mamba 行为），1 开启

  旁路显式建模周期结构（对 ECL/weather 这类强周期数据应有效）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from model.S_Mamba import Model as _S_Mamba


class FreqMainline(nn.Module):
    """FITS 风格低频主线：学习可微复数低通滤波，预测周期主线。"""

    def __init__(self, seq_len: int, pred_len: int, n_variates: int,
                 lowpass_ratio: float = 0.2, learnable: bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_variates = n_variates
        n_bins = seq_len // 2 + 1  # rFFT bin 数
        keep = max(1, int(n_bins * lowpass_ratio))
        self.keep = keep
        # 复权重：低通，超出 keep 的 bin 权重恒 0（非学习）
        # 可学习部分只作用在低频 bin 上（防高频噪声过拟合）
        self.freq_scale = nn.Parameter(torch.ones(1, keep, n_variates, dtype=torch.float32))
        if not learnable:
            self.freq_scale.requires_grad_(False)
        # 可学习门控：模型自己决定主线旁路叠加多大权重（初始接近 0，防淹没主干）
        self.gate = nn.Parameter(torch.zeros(1, 1, n_variates))

    def forward(self, x_enc):
        # x_enc: (B, L, N)，时间轴 rFFT
        X = torch.fft.rfft(x_enc, dim=1)  # (B, n_bins, N)
        B, nb, N = X.shape
        # 低频加权
        Xc = torch.zeros_like(X)
        Xc[:, : self.keep, :] = X[:, : self.keep, :] * self.freq_scale.to(X.dtype)
        # iFFT 回时域 → 取最后 pred_len 步
        x_freq = torch.fft.irfft(Xc, n=self.seq_len, dim=1)  # (B, L, N)
        line = x_freq[:, -self.pred_len :, :]  # (B, S, N)
        # 沿变量轴归一化，再经门控缩放（幅度稳定，主干主导）
        line = line / (torch.sqrt(torch.var(line, dim=1, keepdim=True, unbiased=False) + 1e-5) + 1e-5)
        return torch.tanh(self.gate) * line


class Model(_S_Mamba):
    def __init__(self, configs):
        super().__init__(configs)
        self.freq_mainline = bool(getattr(configs, "freq_mainline", 0))
        self.freq_mainline_scale = getattr(configs, "freq_mainline_scale", 1.0)
        self.freq_lowpass = getattr(configs, "freq_lowpass_mainline", 0.2)
        if self.freq_mainline:
            self.mainline = FreqMainline(
                seq_len=configs.seq_len,
                pred_len=configs.pred_len,
                n_variates=configs.enc_in,
                lowpass_ratio=self.freq_lowpass,
                learnable=getattr(configs, "freq_mainline_learnable", True),
            )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        dec_out = super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        if self.freq_mainline:
            line = self.mainline(x_enc)
            # 门控内含缩放（tanh(gate)∈[-1,1]），不再用固定 scale
            dec_out = dec_out + line
        return dec_out
