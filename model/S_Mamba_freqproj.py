"""S-Mamba + 频域基投影（frequency-aware projector，2026-08-23）。

动机：
  FreDF 已在 loss 层面给频域监督（2-3% 稳定增益）。但 S-Mamba 的 projector 是
  nn.Linear(d_model, pred_len)——直接把变量表征线性映射到预测序列，没有利用序列
  本身的结构。历史架构尝试（量子核 token 注入、freqline 旁路、amp_phase 监督）
  全失败：要么破坏主干学习，要么门控被模型关闭。

  本方向改造**投影层本身**（不是加旁路，主干 encoder 完全不变）：
    y_pred = DCT⁻¹( Linear(enc_out) )    其中 DCT⁻¹ 是可学习的 DCT 基矩阵

  - 把"线性映射到 pred_len 时域点"换成"映射到 K 个频域系数，再用 DCT 基重构时域"
  - DCT 基向量显式编码不同频率分量 → 强制预测是平滑频率成分叠加，天然对齐周期结构
  - 频域系数由主干输出经一个线性层产生（可学习，端到端）
  - 与 FreDF 频域监督天然互补：FreDF 在 loss 惩罚频率差，本项目在架构上提供频率基
  - freqproj_k 控制保留的频率系数个数（K < pred_len 时是低通约束；K = pred_len 时是全频）

  为什么这次可能有效（vs 之前失败）：
  - 不破坏主干：encoder 零改动，只换 projector
  - 不可被忽略：这是唯一的投影路径，模型无法关闭
  - 强先验：对强周期数据（ECL/ETT）DCT 基是有效归纳偏置
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from model.S_Mamba import Model as _S_Mamba


def dct_basis(n: int, k: int) -> torch.Tensor:
    """离散余弦变换基矩阵 (k, n)：第 i 行 = 频率 i 的 DCT 基在 n 个时域点上的采样。
    DCT-II 正交归一化基：phi_i(t) = sqrt(1/n) for i=0; sqrt(2/n)*cos(pi*i*(t+0.5)/n) else。
    返回 (k, n)，可直接 y_time = coeff @ basis。
    """
    t = np.arange(n) + 0.5
    basis = np.zeros((k, n))
    for i in range(k):
        if i == 0:
            basis[i] = np.sqrt(1.0 / n) * np.ones(n)
        else:
            basis[i] = np.sqrt(2.0 / n) * np.cos(np.pi * i * t / n)
    return torch.tensor(basis, dtype=torch.float32)


class FreqProjector(nn.Module):
    """频域基投影：Linear(d_model, K) → DCT 基重构 (K, pred_len) → 时域预测。"""

    def __init__(self, d_model: int, pred_len: int, k: int, learnable: bool = True):
        super().__init__()
        self.pred_len = pred_len
        self.k = k
        # 固定 DCT 基 (k, pred_len) —— 显式频率结构，不训练
        self.register_buffer('basis', dct_basis(pred_len, k))
        if learnable:
            # 允许微调基：加一个小的可学习扰动（初始化 0）
            self.basis_fix = nn.Parameter(torch.zeros(k, pred_len))
        else:
            self.basis_fix = None
        self.proj = nn.Linear(d_model, k, bias=True)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, enc_out):
        # enc_out: (B, N, d_model)
        coeff = self.proj(enc_out)  # (B, N, k)
        basis = self.basis
        if self.basis_fix is not None:
            basis = basis + self.basis_fix
        # (B, N, k) @ (k, pred_len) -> (B, N, pred_len)
        out = coeff @ basis
        return out  # (B, N, pred_len)


class Model(_S_Mamba):
    def __init__(self, configs):
        super().__init__(configs)
        self.freqproj = bool(getattr(configs, "freqproj", 0))
        k = int(getattr(configs, "freqproj_k", 0))
        if self.freqproj:
            self.projector = FreqProjector(
                d_model=configs.d_model,
                pred_len=configs.pred_len,
                k=k if k > 0 else configs.pred_len,
                learnable=getattr(configs, "freqproj_learnable", True),
            )
