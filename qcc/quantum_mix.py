"""量子核跨变量混合层（主干内量子混合，2026-08-11 重构替代旁路 QCCBlock）。

设计：
    探针诊断（analyze_bypass 时代）证明旁路修正机制失效：
      - S 信号死在 LN(H+Hp) 残差结构（修正量对 S 的敏感度 0.02~6%）
      - corr 与残差相关 ≈ 0，α 冻结在初值
      - 主干预测对跨变量输入零响应（非对角/对角 ≈ 1e-6，两个数据集实测）
    重构：抛弃旁路，量子核进主干 —— K 作为主干的跨变量混合算子：
      H' = LN(H + (1/V)·K·H·W_q)
    语义 H 编码变量身份 + 对齐频谱 S 调制角度 → 保真度核 K（逐样本自适应）。
    proj_H / proj_S 端到端训练，K 随表征演化。

forward(H, S) -> (H', K)
    H: (B, V, d) 主干变量 token
    S: (B, V, 2M) 对齐频谱特征（全 detach，确定性函数）
"""
from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

from .classical_kernels import make_kernel
from .feature_map import EntanglingFeatureMap
from .joint_feature_map import JointEntanglingFeatureMap
from .kernel import (
    quantum_kernel,
    quantum_reduced_kernel,
    quantum_entanglement_kernel,
)


def _inv_softplus(target: float, lo: float = -20.0) -> float:
    """求 x 使 softplus(x) = target。target<=0 时返回 lo（softplus(lo)≈0）。"""
    if target <= 0:
        return lo
    return math.log(math.expm1(target))


class QuantumMixLayer(nn.Module):
    """量子核跨变量混合层。

    Args:
        d_token: 主干 token 维度 d。
        n_qubits: 量子比特数 N。
        n_layers: 数据重上传层数 D。
        M: 频谱采样点数（S 的维度 = 2M）。
        entangle_topo: 纠缠拓扑 "linear" / "ring" / "none"。
        kernel_fn: 核函数，默认量子核。
        use_fmap: 是否启用量子 feature map（False 时核直接吃 H）。
        theta_S_scale0: S 路调制强度 γ 初始值（可学习，clamp [0.1, 2]）。
        pre_norm: 是否在 feature map 前对 H 做 LayerNorm。
        use_H: 首层是否用 H 编码变量身份。
        use_S: 重上传是否用 S 调制。
        reupload_source: 重上传层角度来源 'S' / 'H' / 'alternate'。
        angle_norm / angle_radius: 角度归一化。
    """

    def __init__(
        self,
        d_token: int = 512,
        n_qubits: int = 8,
        n_layers: int = 2,
        M: int = 32,
        entangle_topo: str = "linear",
        kernel_fn: Optional[Callable] = None,
        use_fmap: bool = True,
        theta_S_scale0: float = 0.5,
        pre_norm: bool = True,
        use_H: bool = True,
        use_S: bool = True,
        reupload_source: str = "S",
        angle_norm: str = "clamp",
        angle_radius: float = 1.0,
        norm_type: str = "avg",
        output_mode: str = "residual",
        kernel_T: float = 1.0,
        topk: int = 0,
        offdiag: bool = False,
        gate: bool = False,
        gate_init: float = 0.0,
        gate_per_var: bool = False,
        gate_pv_init: float = 3.0,
        gate_pv_src: bool = False,
        gate_pv_src_init: float = -3.0,
        hp_scale: float = 1.0,
        delay_in_s: bool = False,
        fixed_s_scale: bool = False,  # P1-1b: S 固定尺度进 fmap（不可压制）
        amplitude_encoding: bool = False,  # 第四轮：振幅编码（零信息损失）
        angle_groups: int = 1,  # 第五轮：多组角度桥接（把 2M 维 S 拆 G 组，每组独立 2N 角度）
        kernel_group_agg: str = "product",  # 多组核聚合: product | mean
        kernel_kappa: float = 1.0,  # quantum_exp 固定带宽 κ
        kernel_power: float = 2.0,  # quantum_power 固定指数 p
        kernel_exp_learn: bool = False,  # quantum_exp 带宽 κ 可学习（自调带宽）
        kernel_power_learn: bool = False,  # quantum_power 指数 p 可学习
        kernel_phase: float = 1.0,  # quantum_phase_exp 相位强度 λ
        joint_encoding: bool = False,  # JEQK：跨变量纠缠联合编码（2026-08-24）
        joint_topo: str = "var_linear",  # 联合纠缠拓扑 var_linear|var_ring|none
        n_vars: int = 7,  # 变量数 V（联合编码用）
    ):
        """消息传递归一化与输出模式（2026-08-11 晚，运输修复；2026-08-12 选择性修复）。

        norm_type:
            "avg"     —— 原版 (1/V)·K·H·W_q（跨变量信号被 V 稀释，灵敏度 1e-6 的根因）
            "softmax" —— 行 softmax 归一化 K，GAT 式加权平均（Hp 幅度 ≈ H，信号不被淹没）
        output_mode:
            "residual" —— H' = LN(H + Hp)，用于主干内混合层
            "raw"      —— 返回 LN(Hp)，用于预测头级量子聚合（方案 1：K 最接近损失）
        kernel_T: 保真度核温度（选择性修复）：
            高维量子态浓度使 softmax(K) 近均匀（行熵 1.85/1.95），消息传递退化为全局平均。
            softmax(K/T) 且 T<1 可放大 0.004 级差异，恢复 K 的尖峰/选择性。T=1 不生效。
        topk: softmax 后仅保留每行最大的 topk 个耦合并重归一化（0 = 不启用）。
            让 K 明确做"变量选择"，其余变量权重归零。
        offdiag: 去对角（2026-08-12 根因级修复）：
            保真度核 diag=1 的性质使 softmax(K/T) 的对角权重随 T 减小趋近 1
            （T=0.1 时 95.7%，T=0.01 时 99.999%）——消息传递退化为恒等映射，
            跨变量信息从未进入。offdiag=True 时对 (K-I) 做 softmax，跨变量权重占主导
            （T=0.1 时对角仅 9.5%）。
        gate: 混合门控（2026-08-13 轮 C）：输出 H' = H + γ·LN(Hp)，γ 可学习 init=0
            clamp [0,2]。γ=0 时输出 ≡ 输入（架构数学上等于 plain，结构保证不更差）；
            仅在 Hp 与损失梯度相关（有真实信号）时 γ 才打开。消除轮 B 的"灾难尾"
            （变体赢 2-3 个 cell 同时输 6-8 个 cell）。
        """
        super().__init__()
        self.use_fmap = use_fmap
        self.pre_norm = pre_norm
        self.use_H = use_H
        self.use_S = use_S
        self.norm_type = norm_type
        self.output_mode = output_mode
        self.kernel_T = kernel_T
        self.topk = topk
        self.offdiag = offdiag
        self.gate = gate
        self.hp_scale = hp_scale
        self.amplitude_encoding = amplitude_encoding
        self.angle_groups = int(angle_groups)
        self.kernel_group_agg = kernel_group_agg
        self.joint_encoding = joint_encoding
        self.n_qubits = n_qubits
        self.n_vars = n_vars
        self.kernel_kappa = kernel_kappa
        self._last_K_norm = None  # QF-1: 归一化消息权重 K_n（softmax/offdiag 后），供对齐损失使用
        # D2 纠缠耦合核（joint 分支专用）：kernel_fn=qmi/qent → 用 ρ_ij 纠缠核替代约化态相似度核
        _kf0 = kernel_fn.lower() if isinstance(kernel_fn, str) else ""
        self.joint_kernel_mode = {
            "quantum_entangle": "mi",
            "qmi": "mi",
            "qent": "mi",
            "quantum_entangle_hs": "hs_ent",
            "qent_hs": "hs_ent",
        }.get(_kf0)
        if self.angle_groups < 1:
            raise ValueError(f"angle_groups must be >= 1, got {angle_groups}")
        if kernel_group_agg not in ("product", "mean"):
            raise ValueError(f"kernel_group_agg must be 'product' or 'mean', got {kernel_group_agg}")
        if norm_type not in ("avg", "softmax", "l1"):
            raise ValueError(f"norm_type must be 'avg' or 'softmax' or 'l1', got {norm_type}")
        if output_mode not in ("residual", "raw"):
            raise ValueError(f"output_mode must be 'residual' or 'raw', got {output_mode}")
        if kernel_T <= 0:
            raise ValueError(f"kernel_T must be > 0, got {kernel_T}")
        if topk < 0:
            raise ValueError(f"topk must be >= 0, got {topk}")

        if pre_norm:
            self.pre_ln = nn.LayerNorm(d_token)

        if use_fmap:
            # 第三轮改进：更深的 S 投影网络
            deep_s_proj = getattr(self, '_deep_s_proj', False)
            if deep_s_proj:
                # 多层 MLP 替代单层线性
                s_input_dim = 2 * M + (1 if delay_in_s else 0)
                s_hidden_dim = n_qubits * 4  # 中间层维度
                self._deep_s_proj_net = nn.Sequential(
                    nn.Linear(s_input_dim, s_hidden_dim),
                    nn.ReLU(),
                    nn.Linear(s_hidden_dim, n_qubits * 2),  # 输出 2N 维（N 实部 + N 虚部）
                )
                # 初始化
                for m in self._deep_s_proj_net:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            else:
                self._deep_s_proj_net = None

            # 多组角度桥接（第五轮）：把 S 拆 G 组，每组独立 fmap + 独立 2N 角度。
            # 单组 2N 角度是瓶颈（64 维 S → 2N 维角度信息被压）；G 组 → 总角度 G·2N，
            # 每组只吃 S 的 1/G 切片，proj_S: (2M/G) → 2N，缓解单组压缩。
            # delay_in_s 的 δ̂ 通道不参与分组（1 维无法整除），作为公共特征附加到每组。
            if self.angle_groups > 1 and use_S:
                s_core_dim = 2 * M  # 频谱部分
                if s_core_dim % self.angle_groups != 0:
                    raise ValueError(
                        f"S core dim {s_core_dim} not divisible by angle_groups {self.angle_groups}"
                    )
                self.group_s_dim = s_core_dim // self.angle_groups + (1 if delay_in_s else 0)
                self.group_fmaps = nn.ModuleList(
                    [
                        EntanglingFeatureMap(
                            n_qubits=n_qubits,
                            n_layers=n_layers,
                            d_token=d_token,
                            M=M,
                            entangle_topo=entangle_topo,
                            use_H=use_H,
                            use_S=use_S,
                            reupload_source=reupload_source,
                            angle_norm=angle_norm,
                            angle_radius=angle_radius,
                            delay_in_s=False,  # δ̂ 已并入分组切片
                            s_input_dim=self.group_s_dim,
                        )
                        for _ in range(self.angle_groups)
                    ]
                )
            else:
                self.group_fmaps = None
                self.group_s_dim = 0

            self.fmap = EntanglingFeatureMap(
                n_qubits=n_qubits,
                n_layers=n_layers,
                d_token=d_token,
                M=M,
                entangle_topo=entangle_topo,
                use_H=use_H,
                use_S=use_S,
                reupload_source=reupload_source,
                angle_norm=angle_norm,
                angle_radius=angle_radius,
                delay_in_s=delay_in_s,  # P0-1: δ̂ 通道 → proj_S 输入 2M+1（此前漏传导致 65 维 S 崩溃）
            )

            # JEQK：跨变量纠缠联合编码（2026-08-24）
            if joint_encoding:
                self.joint_fmap = JointEntanglingFeatureMap(
                    n_qubits=n_qubits,
                    n_vars=n_vars,
                    n_layers=n_layers,
                    d_token=d_token,
                    M=M,
                    entangle_topo=joint_topo,
                    use_H=use_H,
                    use_S=use_S,
                    angle_norm=angle_norm,
                    angle_radius=angle_radius,
                    delay_in_s=delay_in_s,
                )

        if kernel_fn is None:
            self.kernel_fn = quantum_kernel
        elif isinstance(kernel_fn, str):
            kf = kernel_fn.lower()
            if kf in ("quantum_exp_pair", "qpair"):
                # 每变量对可学习带宽（metric learning）：对齐损失能直接调 κ_ij
                from .kernel import QuantumGeodesicPairwiseKernel
                self.kernel_fn = QuantumGeodesicPairwiseKernel(self.n_vars, kappa0=kernel_kappa)
            elif kf in ("quantum_exp", "qexp", "quantum_geodesic") and kernel_exp_learn:
                # 可学习带宽版：κ 随训练自适应（rbf 固定 γ 做不到）
                from .kernel import QuantumGeodesicKernel
                self.kernel_fn = QuantumGeodesicKernel(kappa0=kernel_kappa)
            elif kf in ("quantum_power", "qpow") and kernel_power_learn:
                from .kernel import QuantumPowerKernel
                self.kernel_fn = QuantumPowerKernel(power0=kernel_power)
            elif kf in ("free", "free_couple"):
                # D0a-v2：自由学习耦合矩阵（核轴天花板测试），非 joint 路径直接替换核
                from .kernel import FreeCoupleKernel
                self.kernel_fn = FreeCoupleKernel(self.n_vars)
            elif self.joint_kernel_mode is not None:
                # D2：joint 分支专用，核在 forward 按 joint_kernel_mode 计算，不走 make_kernel
                self.kernel_fn = quantum_kernel  # 占位（joint 分支不调用 self.kernel_fn）
            else:
                self.kernel_fn = make_kernel(
                    kf, d_token, kappa=kernel_kappa, power=kernel_power, phase=kernel_phase
                )
        else:
            self.kernel_fn = kernel_fn

        # 可学习 W_q：跨变量消息映射
        self.W_q = nn.Linear(d_token, d_token, bias=False)
        self.ln = nn.LayerNorm(d_token)

        # 混合门控 γ（可学习标量，init=gate_init（默认 0 → 输出 ≡ 输入），clamp [0, 2]）
        if gate:
            self._gate_raw = nn.Parameter(torch.full((), float(gate_init)))

        # P2: 每变量消息门控（2026-08-24）：每个目标变量 v 学习是否接受跨变量消息。
        # 大 V 弱耦合数据集（ECL/traffic）的 qmix 税 = K_n 近均匀 → Hp≈全局平均噪声；
        # 每变量 sigmoid 门让独立变量学会关掉注入（gate_pv→0），耦合变量保留量子信号。
        if gate and gate_per_var:
            self.gate_per_var = True
            self._gate_pv_raw = nn.Parameter(torch.full((int(n_vars),), float(gate_pv_init)))
        else:
            self.gate_per_var = False
        # P2: 每源门控 —— 可学习 topk：每个源变量 w 是否参与跨变量消息（K 列选择性）
        if gate and gate_pv_src:
            self.gate_pv_src = True
            self._gate_src_raw = nn.Parameter(torch.full((int(n_vars),), float(gate_pv_src_init)))
        else:
            self.gate_pv_src = False

        # S 路调制强度 γ（可学习标量，init=0.5, clamp [0.1, 2]）
        if use_S:
            self._gamma_raw = nn.Parameter(torch.tensor(_inv_softplus(theta_S_scale0)))
            self.fixed_s_scale = fixed_s_scale
            # P0-1: delay_in_s 时 S 多 1 维 δ̂ 时滞通道
            self.s_ln = nn.LayerNorm(2 * M + (1 if delay_in_s else 0))

        # 振幅编码投影层（第四轮：零信息损失）
        if amplitude_encoding and use_S:
            s_dim = 2 * M + (1 if delay_in_s else 0)
            target_dim = 2 ** n_qubits
            self.amp_proj = nn.Linear(s_dim, target_dim, bias=False)
            # Xavier 初始化
            nn.init.xavier_uniform_(self.amp_proj.weight)

    @property
    def gate_value(self) -> torch.Tensor:
        """混合门控强度（γ=0 → H' ≡ H，架构等于 plain）。"""
        if not self.gate:
            return torch.tensor(1.0, device=self.W_q.weight.device)
        return self._gate_raw.clamp(0.0, 2.0)

    @property
    def gamma(self) -> torch.Tensor:
        """S 路调制强度 γ（clamp 到 [0.1, 2]）。"""
        if not self.use_S:
            return torch.tensor(1.0, device=self.W_q.weight.device)
        g = torch.nn.functional.softplus(self._gamma_raw)
        return g.clamp(min=0.1, max=2.0)

    def forward(
        self,
        H: torch.Tensor,
        S: Optional[torch.Tensor] = None,
        use_S_only: bool = False,  # P1-1: 量子核直接读频谱 S（不读主干 H）
    ):
        """量子核跨变量混合：H' = LN(H + (1/V)·K·H·W_q)。

        Args:
            H: (B, V, d) 变量 token。
            S: (B, V, 2M) 对齐频谱特征（use_S=True 时必须提供）。
            use_S_only: P1-1 模式，量子核直接读频谱 S（不读主干 H）。

        Returns:
            H': (B, V, d) 混合后 token。
            K: (B, V, V) 保真度核矩阵（供可解释性分析）。
        """
        # 安全检查：检测 CUDA 错误
        if H.device.type == 'cuda':
            torch.cuda.synchronize()

        # 量子路径必须在 fp32 下运行：AMP 下复数张量变 ComplexHalf，CUDA 不支持
        with torch.autocast(device_type=H.device.type, enabled=False):
            H_in = self.pre_ln(H.float()) if self.pre_norm else H.float()

            # JEQK：跨变量纠缠联合编码（2026-08-24）
            if self.joint_encoding:
                if S is not None:
                    if self.fixed_s_scale:
                        S_scaled = S.float() / (S.float().abs().max(dim=1, keepdim=True).values + 1e-8)
                    else:
                        S_scaled = self.gamma * self.s_ln(S.float())
                else:
                    S_scaled = None
                psi_joint = self.joint_fmap(H_in, S_scaled)  # (B, 2^(V·nq))
                if self.joint_kernel_mode is not None:
                    # D2：直接度量 i-j 量子纠缠耦合（经典核拿不到的通道）
                    K = quantum_entanglement_kernel(
                        psi_joint, self.n_vars, self.n_qubits, self.kernel_kappa,
                        self.joint_kernel_mode,
                    )  # (B, V, V)
                else:
                    K = quantum_reduced_kernel(
                        psi_joint, self.n_vars, self.n_qubits, self.kernel_kappa
                    )  # (B, V, V)
            # 第四轮：振幅编码（零信息损失，避免复数运算）
            elif self.amplitude_encoding and use_S_only and S is not None:
                # S 归一化
                if self.fixed_s_scale:
                    S_scaled = S.float() / (S.float().abs().max(dim=1, keepdim=True).values + 1e-8)
                else:
                    S_scaled = self.gamma * self.s_ln(S.float())

                # 振幅编码：线性投影 + L2 归一化（实数向量，不用复数）
                psi = self.amp_proj(S_scaled)  # (B, V, 2^N)
                psi = psi / (psi.norm(dim=-1, keepdim=True) + 1e-8)  # L2 归一化

                # 量子保真度核：K[i,j] = |<psi_i|psi_j>|^2 = (psi_i · psi_j)^2
                # 实数内积，避免复数 einsum
                inner = torch.matmul(psi, psi.transpose(-1, -2))  # (B, V, V)
                K = inner ** 2  # 保真度
            elif self.use_fmap:
                if use_S_only and S is not None:
                    # P1-1: 量子核直接读频谱 S（不读主干 H）
                    # 让量子核编码主干拿不到的频域结构
                    if self.fixed_s_scale:
                        S_scaled = S.float() / (S.float().abs().max(dim=1, keepdim=True).values + 1e-8)
                    else:
                        S_scaled = self.gamma * self.s_ln(S.float())
                    # 2026-08-23 修复：S-only 走完整 fmap（含 CNOT 纠缠层），
                    # 而非手动构造无纠缠乘积态。乘积态保真度核数学上等价于经典核
                    # （量子核 vs rbf 消融无差异的根因）。用 H 编码变量身份 + S 调制，
                    # 经纠缠 feature map 生成真纠缠量子态。
                    if self.fixed_s_scale:
                        S_scaled_enc = S.float() / (S.float().abs().max(dim=1, keepdim=True).values + 1e-8)
                    else:
                        S_scaled_enc = self.gamma * self.s_ln(S.float())
                    psi = self.fmap(H_in, S_scaled_enc)  # (B, V, 2^N)，含纠缠
                    self._last_psi = psi
                    K = self.kernel_fn(psi)
                elif self.use_S and S is not None:
                    if self.fixed_s_scale:
                        # P1-1b: 固定尺度归一化（按变量行 max），去掉可学习 γ/s_ln —— 优化器无法压塌
                        S_scaled = S.float() / (S.float().abs().max(dim=1, keepdim=True).values + 1e-8)
                    else:
                        S_scaled = self.gamma * self.s_ln(S.float())  # γ 调制
                else:
                    S_scaled = None
                if not use_S_only:
                    if self.group_fmaps is not None:
                        # 多组角度桥接：频谱 2M 切成 G 块；δ̂（如有）附加到每组
                        s_core = S_scaled[..., : 2 * self.fmap.M]
                        delay = S_scaled[..., 2 * self.fmap.M :] if self.fmap.M * 2 < S_scaled.shape[-1] else None
                        s_chunks = torch.chunk(s_core, self.angle_groups, dim=-1)
                        if delay is not None:
                            s_chunks = [torch.cat([c, delay], dim=-1) for c in s_chunks]
                        K = None
                        for fm, s_chunk in zip(self.group_fmaps, s_chunks):
                            psi_g = fm(H_in, s_chunk)  # (B, V, 2^N)
                            K_g = self.kernel_fn(psi_g)  # (B, V, V)
                            if K is None:
                                K = K_g
                            elif self.kernel_group_agg == "product":
                                K = K * K_g
                            else:
                                K = K + K_g
                        if self.kernel_group_agg == "mean":
                            K = K / self.angle_groups
                    else:
                        psi = self.fmap(H_in, S_scaled)
                        self._last_psi = psi
                        K = self.kernel_fn(psi)
            else:
                K = self.kernel_fn(H_in)

            # 安全检查：feature map 后检测 CUDA 错误
            if H.device.type == 'cuda':
                torch.cuda.synchronize()

            self._last_K = K  # 原始保真度核（对称、对角≈1）供解释性验证
            HW = torch.einsum("bvd,de->bve", H_in, self.W_q.weight)
            if self.norm_type == "softmax":
                # GAT 式行归一化：K[v,:] 和为 1，Hp 幅度 ≈ H，跨变量信号不被 1/V 稀释
                # kernel_T < 1：放大保真度差异（0.004 级 → 0.04 级），恢复核选择性
                K_soft = K
                if self.offdiag:
                    # 去对角（根因级）：保真度核 diag=1 使 softmax 对角权重趋近 1（恒等映射）。
                    # 修正（2026-09-07）：仅减单位阵会在 softmax 中留下 e^0 的对角权重(≈1/Z)，
                    # 与“消息只分配给其他变量”的理论不一致；这里把对角 logit 直接置为极大负值，
                    # 使对角权重严格为 0，行只在其余变量上归一化。
                    K_soft = K.clone()
                    diag_idx = torch.arange(K.shape[-1], device=K.device)
                    K_soft[:, diag_idx, diag_idx] = K_soft[:, diag_idx, diag_idx].detach().min() - 1e6
                K_n = torch.softmax(K_soft / self.kernel_T, dim=-1)
                if self.gate_pv_src:
                    # P2: 每源门控（可学习 topk）：弱耦合源变量 w 的跨变量贡献被关闭，
                    # 其余源重归一化 —— 缓解大 V 弱耦合下 K 退化为全局平均（ECL/traffic 税）
                    src_g = torch.sigmoid(self._gate_src_raw).unsqueeze(0).unsqueeze(0)  # (1,1,V) 列
                    K_n = K_n * src_g
                    K_n = K_n / K_n.sum(-1, keepdim=True).clamp_min(1e-8)
                self._last_K_norm = K_n
                if self.topk > 0:
                    # 变量选择：仅保留每行 topk 个耦合，其余归零后重归一化
                    kth = torch.topk(K_n, self.topk, dim=-1).values[:, :, -1:]
                    K_n = K_n * (K_n >= kth)
                    K_n = K_n / K_n.sum(-1, keepdim=True).clamp_min(1e-8)
                self._last_K_norm = K_n  # QF-1: 暴露下游真正用的归一化权重
                Hp = torch.einsum("bvw,bwe->bve", K_n, HW)
            elif self.norm_type == "l1":
                # 有符号权重归一化（轮 C 线性核专用）：K 可为负（虚部反对称=有向），
                # 不用 softmax（会破坏符号），按行 |K| 归一化保留方向
                K_n = K / K.abs().sum(-1, keepdim=True).clamp_min(1e-8)
                Hp = torch.einsum("bvw,bwe->bve", K_n, HW)
            else:
                Hp = torch.einsum("bvw,bwe->bve", K, HW) / K.shape[1]  # (1/V)·K·H·W_q
            if self.output_mode == "raw":
                # 预测头级聚合：返回 LN(Hp)，供 head 直接拼接放大
                return self.ln(Hp), K
            # 暴露 LN(Hp) 供辅助残差损失（qdir_aux）与探针使用
            self._last_Hp = self.ln(Hp)
            Hp_g = self.hp_scale * self.ln(Hp)   # 固定缩放（轮 G：防 V 大时强信号发散）
            if self.gate:
                # 门控模式（轮 C）：γ=0 → H' ≡ H（结构保证不更差），γ>0 时才注入混合
                gv = self.gate_value
                if self.gate_per_var:
                    # P2: 每变量接受门控 —— 弱耦合变量关闭跨变量消息（消除大 V 平均化税）
                    gv = gv * torch.sigmoid(self._gate_pv_raw).unsqueeze(0).unsqueeze(-1)  # (1, V, 1)
                return H + gv * Hp_g, K
            return self.ln(H_in + Hp_g), K


__all__ = ["QuantumMixLayer"]
