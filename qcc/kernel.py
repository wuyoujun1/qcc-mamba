"""量子核矩阵 K = |⟨φ|φ⟩|²（保真度）。

接口：psi: (B, V, 2^N) → K: (B, V, V)。

性质：
- 对角线恒为 1（酉演化保范 → |⟨φ_i|φ_i⟩|² = 1）
- 对称正定（保真度）
- 可用作频谱分析的 Gram 矩阵

对应文档：experiment-design.md §4.3
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


def _fidelity(psi: torch.Tensor) -> torch.Tensor:
    """原始保真度 f[b,i,j] = |⟨φ_i|φ_j⟩|²（不加数值稳定项，供下游核函数复用）。"""
    if psi.is_complex():
        # 修复：clone 共轭视图，避免 einsum 反向传播时的内存共享问题
        psi_conj = psi.conj().clone()
        inner = torch.einsum("bvi,bwi->bvw", psi_conj, psi)
    else:
        # 实数回退：直接内积
        inner = torch.einsum("bvi,bwi->bvw", psi, psi)
    return inner.abs().pow(2)


def quantum_kernel(psi: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """量子核矩阵：K[b,i,j] = |⟨φ(h_i)|φ(h_j)⟩|² = |Σ_k conj(ψ_i[k])·ψ_j[k]|²。

    Args:
        psi: 量子态矢量，形状 (B, V, 2^N) 复数。
        eps: 数值稳定项。

    Returns:
        核矩阵，形状 (B, V, V) 实数，值域 [0, 1]。
    """
    K = _fidelity(psi)
    # 数值稳定：只在非对角线加 eps，保持对角线为 1（酉演化保范）
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype).unsqueeze(0)
    K = K + eps * (1 - eye)
    return K


def quantum_geodesic_kernel(psi: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """量子测地核（Bures-RBF）：K[b,i,j] = exp(-κ · d_ij²)。

    d_ij = arccos(√f_ij) 是保真度诱导的 Bures 角（Fubini-Study 测地线距离）。
    这是把量子核装回 rbf 同款"带宽旋钮"的根因修复（2026-08-24）：
    - 原保真度核 f=|⟨φ|φ⟩|² 非对角挤在小值区（0.004~0.8），softmax(K-I)/T 选择性弱，
      rbf 的 exp(-γ‖ψ-ψ'‖²) 凭 γ=1/d 自动校准而胜出 —— 浓度问题
    - 测地核用**真量子几何距离**而非欧氏距离，且 κ 提供显式带宽控制：
        κ→0：K→均匀 1（全局平均）；κ 大：局部近邻核（选择性/尖峰）
    - 对角恒 1（arccos(1)=0 → exp(0)=1），与保真度核一致

    数值坑（2026-08-24 实测发现）：上界 clamp 若用 1-1e-8，float32 中
    sqrt(1-1e-8) 舍入为精确 1.0，acos(1.0)=0 反向梯度 = -1/sqrt(0) = -inf，
    乘 d·exp(-κd²)=0 得 0×(-inf)=NaN，训练全炸。必须 clamp 到 1-1e-6
    使 sqrt 严格 < 1（对角线损失 ≤2κe-6，可忽略）。
    """
    f = _fidelity(psi).clamp(1e-8, 1.0 - 1e-6)
    d = torch.acos(f.sqrt())          # Bures 角 ∈ (0, π/2]
    return torch.exp(-kappa * d * d)


def quantum_pqk_kernel(psi: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Projected Quantum Kernel（投影量子核，Huang et al. 2021）：
    K = exp(-κ · Σ_q ‖ρ_q(i) − ρ_q(j)‖²_HS)，ρ_q = 单 qubit 约化密度矩阵。

    动机（针对保真度核的两大缺陷）：
    - 保真度 |⟨ψ|ψ'⟩|² 在维数增长时指数浓度（off-diag 挤成 0，Gram 近 I）——
      PQK 用局部边缘（1-RDM）代替全局重叠，正是文献给出的根因解法
    - 保真度对局部酉变换不变（|⟨Uψ_i|Uψ_j⟩|²=|⟨ψ_i|ψ_j⟩|²，相位/基底信息被洗掉）；
      1-RDM 距离‖ρ_q(i)−ρ_q(j)‖_HS 依赖测量基底，**保留了局部相干/相位结构**，
      这是经典 rbf 与保真度都拿不到的信息
    - 文献实证：数据受限区 PQK 可赢经典 RBF（几何差 ∝ √N）

    数值：1-RDM = Tr_{¬q}(|ψ⟩⟨ψ|)，HS 距离 = ‖[Re;Im] 平铺‖² 的 cdist。
    对角：同态距离 0 → K=1。对称，配 softmax/offdiag 机器直接可用。
    """
    B, V, D = psi.shape
    N = int(round(math.log2(D)))
    psi_r = psi.reshape(B, V, *([2] * N))
    rhos = []
    for q in range(N):
        rest = torch.movedim(psi_r, 2 + q, -1).reshape(B, V, -1, 2)
        rho_q = torch.einsum("bvoi,bvoj->bvij", rest, rest.conj())  # (B,V,2,2)
        rhos.append(rho_q)
    rho = torch.stack(rhos, dim=2)  # (B,V,N,2,2)
    # [Re;Im] 平铺 → cdist 平方 = Σ_q ‖ρ_q(i)-ρ_q(j)‖²_HS
    rho_f = torch.cat([rho.real, rho.imag], dim=-1).reshape(B, V, -1)
    d2 = torch.cdist(rho_f, rho_f).pow(2)
    return torch.exp(-kappa * d2)


def quantum_pqk_geo_kernel(psi: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """PQK × 测地核混合：K = exp(-κ·(d²_pqk + d²_geo))。

    d²_pqk = 1-RDM HS 距离（局部相干统计），d²_geo = Bures 角²（全局几何距离）。
    乘积 = 距离和。同时给模型"局部（基底敏感）+ 全局（酉不变）"两路相似度。
    """
    B, V, D = psi.shape
    N = int(round(math.log2(D)))
    psi_r = psi.reshape(B, V, *([2] * N))
    rhos = []
    for q in range(N):
        rest = torch.movedim(psi_r, 2 + q, -1).reshape(B, V, -1, 2)
        rho_q = torch.einsum("bvoi,bvoj->bvij", rest, rest.conj())
        rhos.append(rho_q)
    rho = torch.stack(rhos, dim=2)
    rho_f = torch.cat([rho.real, rho.imag], dim=-1).reshape(B, V, -1)
    d2_pqk = torch.cdist(rho_f, rho_f).pow(2)
    # 测地距离
    inner = torch.einsum("bvi,bwi->bvw", psi.conj().clone(), psi)
    f = inner.abs().pow(2).clamp(1e-8, 1.0 - 1e-6)
    d_geo = torch.acos(f.sqrt())
    d2_geo = d_geo * d_geo
    return torch.exp(-kappa * (d2_pqk + d2_geo))


def quantum_phase_geodesic_kernel(
    psi: torch.Tensor, kappa: float = 1.0, lam: float = 1.0
) -> torch.Tensor:
    """相位感知量子测地核：K = exp(-κ·d²)·(1 + λ·ph)，d=Bures 角，ph=Im⟨ψ_i|ψ_j⟩ 归一化。

    双通道设计（2026-08-24，针对 rbf 的物理盲区）：
    - 局部性通道 base = exp(-κ·d²)：Bures 角测地距离，提供选择性（同 quantum_exp）
    - 方向性通道 ph = Im⟨ψ_i|ψ_j⟩ / rowmax|Im| ∈ [-1,1]：**rbf 唯一无法计算的信息**。
      rbf 的 exp(-γ‖ψ-ψ'‖²) 只含 Re⟨ψ|ψ'⟩（‖ψ-ψ'‖² = 2-2Re⟨⟩）；保真度 |⟨⟩|² = Re²+Im²
      无符号地丢掉相位。而 Im⟨⟩ 反对称（K_ij = −K_ji），编码跨变量相位对齐/时滞方向。
    - 组合后有向（K_ij ≠ K_ji），必须配 norm_type='l1' 保留符号；λ 控相位强度。
    - 对角：Im⟨ψ_i|ψ_i⟩=0 → ph=0 → K=exp(0)=1。
    """
    psi_conj = psi.conj().clone()
    inner = torch.einsum("bvi,bwi->bvw", psi_conj, psi)
    f = inner.abs().pow(2).clamp(1e-8, 1.0 - 1e-6)
    d = torch.acos(f.sqrt())
    base = torch.exp(-kappa * d * d)
    ph = inner.imag
    ph_n = ph / (ph.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8))  # 行归一化 [-1,1]
    return base * (1.0 + lam * ph_n)


def quantum_reduced_kernel(
    psi_joint: torch.Tensor, n_vars: int, n_qubits: int, kappa: float = 1.0
) -> torch.Tensor:
    """跨变量投影量子核（JEQK 专用）：
    K_ij = exp(-κ · ‖ρ_i − ρ_j‖²_HS)，ρ_i = Tr_{¬i}|Ψ⟩⟨Ψ|（变量 i 的约化密度矩阵）。

    与 quantum_pqk_kernel 的区别：那里逐变量独立态做单 qubit 投影；这里输入是
    **全变量联合纠缠态** Ψ (B, 2^(V·nq))，ρ_i 蕴含 i 与其余变量的纠缠关联。
    - 变量间无纠缠（乘积态）时 ρ_i=|ψ_i⟩⟨ψ_i| 纯态，退化为逐变量保真度核
    - 有纠缠时 ρ_i 混合，含跨变量关联 → 经典核在多项式成本下算不出

    Args:
        psi_joint: (B, 2^(V·nq)) 联合纠缠态。
        n_vars: 变量数 V。
        n_qubits: 每变量 qubit 数 N。
        kappa: 带宽。
    """
    B = psi_joint.shape[0]
    T = n_vars * n_qubits
    # psi_joint: (B, 2^T) → reshape (B, 2, ..., 2)：仅 1 个前导维（B）
    psi_r = psi_joint.reshape(B, *([2] * T))
    rhos = []
    for i in range(n_vars):
        var_axes = [1 + i * n_qubits + q for q in range(n_qubits)]
        others = [a for a in range(1, 1 + T) if a not in var_axes]
        order = [0] + others + var_axes  # 待迹除的位放最后
        psi_p = psi_r.permute(order)
        rest = psi_p.reshape(B, -1, 1 << n_qubits)  # (B, 2^(T-N), 2^N)
        rho = torch.einsum("boi,boj->bij", rest, rest.conj())  # 部分迹 → (B, 2^N, 2^N)
        rhos.append(rho)
    rho = torch.stack(rhos, dim=1)  # (B, V, 2^N, 2^N)
    rho_f = torch.cat([rho.real, rho.imag], dim=-1).reshape(B, n_vars, -1)
    d2 = torch.cdist(rho_f, rho_f).pow(2)
    return torch.exp(-kappa * d2)


def _vn_entropy(rho: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """冯·诺依曼熵 S(ρ) = −Tr(ρ ln ρ)，rho: (B, M, M) complex Hermitian → (B,)。"""
    rho_sym = (rho + rho.conj().transpose(-1, -2)) / 2  # 数值对称化，保证 Hermitian
    # jitter 正则化：病态/重复特征值矩阵上 eigvalsh 会报 error code 3 不收敛
    M = rho_sym.shape[-1]
    rho_sym = rho_sym + 1e-8 * torch.eye(M, device=rho.device, dtype=rho.dtype)
    ev = torch.linalg.eigvalsh(rho_sym).clamp_min(eps).clamp_max(1.0)   # (B, M) 实特征值，升序；>1 数值误差截断防负熵
    return -(ev * torch.log(ev)).sum(-1)


def _pair_reduced_states(
    psi_joint: torch.Tensor, n_vars: int, n_qubits: int
) -> torch.Tensor:
    """全部两变量约化密度矩阵 ρ_ij = Tr_{¬ij}|Ψ⟩⟨Ψ|。

    Args:
        psi_joint: (B, 2^(V·nq)) 联合纠缠态。
        n_vars: 变量数 V。
        n_qubits: 每变量 qubit 数 N。

    Returns:
        rho_pair: (B, V, V, 2^(2N), 2^(2N)) complex，只填 j≥i 上三角（其余为 0 占位）。
    """
    B = psi_joint.shape[0]
    T = n_vars * n_qubits
    M = 1 << (2 * n_qubits)
    psi_r = psi_joint.reshape(B, *([2] * T))
    out = torch.zeros(B, n_vars, n_vars, M, M, dtype=psi_joint.dtype, device=psi_joint.device)
    for i in range(n_vars):
        for j in range(i + 1, n_vars):  # 严格上三角：ρ_ii 用单变量约化态，避免 keep 重复索引
            keep = [1 + i * n_qubits + q for q in range(n_qubits)] + \
                   [1 + j * n_qubits + q for q in range(n_qubits)]
            others = [a for a in range(1, 1 + T) if a not in keep]
            psi_p = psi_r.permute([0] + others + keep)
            rest = psi_p.reshape(B, -1, M)
            out[:, i, j] = torch.einsum("boi,boj->bij", rest, rest.conj())
    return out


def _variable_reduced_states(
    psi_joint: torch.Tensor, n_vars: int, n_qubits: int
) -> torch.Tensor:
    """单变量约化密度矩阵 ρ_i。返回 (B, V, 2^N, 2^N) complex（与 d0b 诊断同口径）。"""
    B = psi_joint.shape[0]
    T = n_vars * n_qubits
    M = 1 << n_qubits
    psi_r = psi_joint.reshape(B, *([2] * T))
    rhos = []
    for i in range(n_vars):
        keep = [1 + i * n_qubits + q for q in range(n_qubits)]
        others = [a for a in range(1, 1 + T) if a not in keep]
        psi_p = psi_r.permute([0] + others + keep)
        rest = psi_p.reshape(B, -1, M)
        rhos.append(torch.einsum("boi,boj->bij", rest, rest.conj()))
    return torch.stack(rhos, dim=1)


def quantum_entanglement_kernel(
    psi_joint: torch.Tensor,
    n_vars: int,
    n_qubits: int,
    kappa: float = 1.0,
    mode: str = "mi",
) -> torch.Tensor:
    """跨变量纠缠耦合核（D2，2026-08-24）：K[i,j] 直接度量 i-j 间的量子耦合强度。

    与 quantum_reduced_kernel（比单变量约化态 ρ_i 的相似度）的根本区别：
    这里用**两变量约化态 ρ_ij**，直接测"i 和 j 被量子耦合得有多紧"——
    这正是消息传递想要的耦合信号，且是经典核（比单变量特征相似度）在多项式
    成本下拿不到的结构。

    mode:
      'mi'    量子互信息 I(i:j) = S(ρ_i) + S(ρ_j) − S(ρ_ij)，∈[0, log(2^(2N))]。
              归一化到 [0,1]（I(i:j) ≤ 2·min(S_i,S_j) ≤ log(2^(2N))，纯态约束）。
              乘积态（无跨变量纠缠）时 I(i:j)=0 → 不混合；纠缠对 → 高权重。
              对角 = S(ρ_i)/log(2^(2N))（offdiag 机制会去对角）。
      'hs_ent' 纠缠距离 K = 1 − exp(−κ·‖ρ_ij − ρ_i⊗ρ_j‖²_HS)，乘积态→0，纠缠→1。

    真量子信息：ρ_ij 蕴含 i-j 与全系统的纠缠关联；乘积态编码退化为"无耦合"，
    不会像 rbf 那样强行给所有变量分配权重。
    """
    if mode == "mi":
        rho_i = _variable_reduced_states(psi_joint, n_vars, n_qubits)     # (B,V,2^N,2^N)
        rho_pair = _pair_reduced_states(psi_joint, n_vars, n_qubits)      # (B,V,V,M,M) 上三角
        logM = math.log(1 << (2 * n_qubits))
        S_i = _vn_entropy(rho_i)                                          # (B,V)
        K = torch.zeros(n_vars, n_vars, device=psi_joint.device)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):  # 严格上三角（对角用 S_i 单独填）
                S_ij = _vn_entropy(rho_pair[:, i, j])                      # (B,)
                I_ij = (S_i[:, i] + S_i[:, j] - S_ij).clamp_min(0.0) / logM
                K[i, j] = K[j, i] = I_ij.mean()                            # 跨 batch 平均 → (V,V)
        # 对角用单变量熵（均值）；offdiag 机制会去对角
        diag = (S_i / logM).mean(0)
        arange = torch.arange(n_vars, device=psi_joint.device)
        K[arange, arange] = diag
        return K.unsqueeze(0).expand(psi_joint.shape[0], -1, -1)           # (B,V,V)
    if mode == "hs_ent":
        rho_i = _variable_reduced_states(psi_joint, n_vars, n_qubits)      # (B,V,2^N,2^N)
        rho_pair = _pair_reduced_states(psi_joint, n_vars, n_qubits)       # (B,V,V,M,M) 上三角
        Bsz, V = rho_i.shape[0], rho_i.shape[1]
        M = 1 << n_qubits
        rho_ten = torch.zeros_like(rho_pair)
        for i in range(n_vars):
            for j in range(i + 1, n_vars):  # 严格上三角
                # 张量积 ρ_i⊗ρ_j，基序匹配 ρ_ij（行=(i0,i1,j0,j1)）：避免索引收缩
                rho_ten[:, i, j] = torch.einsum("bij,bkl->bikjl", rho_i[:, i], rho_i[:, j]).reshape(Bsz, M * M, M * M)
        d2 = (rho_pair - rho_ten).abs().pow(2).sum(dim=(-2, -1))          # (B,V,V) 仅上三角有值
        d2 = d2 + d2.transpose(-2, -1)                                     # 下三角补=上三角（对称化）
        K_ent = 1.0 - torch.exp(-kappa * d2)
        arange = torch.arange(n_vars, device=psi_joint.device)
        K_ent[:, arange, arange] = 0.0  # 无自纠缠
        return K_ent
    raise ValueError(f"mode must be 'mi' or 'hs_ent', got {mode}")


class FreeCoupleKernel(nn.Module):
    """自由学习耦合矩阵（D0a-v2 核轴天花板测试，2026-08-24）：K = softplus(M)。

    M 是可学习 V×V 矩阵——最一般的**固定**耦合模式（无核函数、无数据依赖）。
    若连它都赢不了 rbf+align，说明固定耦合轴的天花板已到，量子优势必须从
    特征（D7）或数据依赖耦合里拿；若它能赢，说明耦合结构本身是有余量的杠杆，
    量子核（数据依赖）有资格去够它。

    用法：kernel_fn='free'，配 offdiag 消息传递机器（K_soft = K − I → softmax）。
    """

    def __init__(self, n_vars: int, init: float = -5.0):
        super().__init__()
        self.n_vars = int(n_vars)
        # init=-5 → softplus(-5)≈0.007 → 初始几乎不耦合（防噪声），模型自行生长
        self.M = nn.Parameter(torch.full((self.n_vars, self.n_vars), float(init)))

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        B = psi.shape[0]
        K = torch.nn.functional.softplus(self.M)  # (V,V) 正
        return K.unsqueeze(0).expand(B, -1, -1)


def quantum_power_kernel(psi: torch.Tensor, power: float = 2.0) -> torch.Tensor:
    """保真度幂核：K[b,i,j] = f_ij^p。

    p>1 单调锐化对比度（高保真对更突出、低保真对更快趋 0），对角恒 1。
    比测地核简单，但同样给保真度核一个"锐度旋钮"。
    """
    f = _fidelity(psi).clamp(0.0, 1.0)
    return torch.pow(f, power)


class QuantumGeodesicKernel(nn.Module):
    """可学习带宽的量子测地核：K = exp(-softplus(κ_raw) · d²)，d = arccos(√f)。

    带宽 κ = softplus(κ_raw)，init 使 κ≈kappa0，随训练自适应（rbf 固定 γ 做不到）。
    给量子核"自调带宽"，等价于在学习量子度量上的尺度。
    """

    def __init__(self, kappa0: float = 1.0):
        super().__init__()
        self.kappa0 = float(kappa0)
        # init: softplus(raw) = kappa0 → raw = log(exp(kappa0)-1)
        self.kappa_raw = nn.Parameter(torch.tensor(math.log(max(math.expm1(kappa0), 1e-6))))
        self.register_buffer("_min", torch.tensor(1e-6))

    @property
    def kappa(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.kappa_raw).clamp_min(self._min)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        f = _fidelity(psi).clamp(1e-8, 1.0 - 1e-6)  # 上界 1-1e-6：防 sqrt 舍入为 1 致 acos 梯度 NaN
        d = torch.acos(f.sqrt())
        return torch.exp(-self.kappa * d * d)


class QuantumGeodesicPairwiseKernel(nn.Module):
    """每变量对可学习带宽的量子测地核：K_ij = exp(-κ_ij · d²_ij)，d=Bures 角。

    metric learning（2026-08-24，对齐洞察的直接应用）：
    - s37 发现对齐损失塑造编码时 rbf 传递信号更好、量子核把编码变化洗掉
    - 给量子核一个**对齐梯度能直接调的旋钮**：每对变量的带宽 κ_ij（对称矩阵）。
      对齐想让 K_ij 高（目标协动）→ 梯度直接调低 κ_ij；rbf 固定 γ=1/d 没这个能力
    - K_ij = exp(-κ_ij·d²_ij)，对角 d=0 → K=1；κ=softplus(raw)，对称化
    """

    def __init__(self, n_vars: int, kappa0: float = 4.0):
        super().__init__()
        self.n_vars = int(n_vars)
        self.kappa0 = float(kappa0)
        self.raw = nn.Parameter(torch.full((self.n_vars, self.n_vars), math.log(max(math.expm1(kappa0), 1e-6))))
        self.register_buffer("_min", torch.tensor(1e-6))

    @property
    def kappa(self) -> torch.Tensor:
        k = torch.nn.functional.softplus(self.raw).clamp_min(self._min)
        return (k + k.T) / 2  # 对称

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        f = _fidelity(psi).clamp(1e-8, 1.0 - 1e-6)
        d2 = torch.acos(f.sqrt()).pow(2)  # (B, V, V)
        return torch.exp(-self.kappa.unsqueeze(0) * d2)


class QuantumPowerKernel(nn.Module):
    """可学习指数的保真度幂核：K = f^softplus(p_raw)。"""

    def __init__(self, power0: float = 2.0):
        super().__init__()
        self.power0 = float(power0)
        self.p_raw = nn.Parameter(torch.tensor(math.log(max(math.expm1(power0), 1e-6))))
        self.register_buffer("_min", torch.tensor(1e-6))

    @property
    def power(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.p_raw).clamp_min(self._min)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        f = _fidelity(psi).clamp(0.0, 1.0)
        return torch.pow(f, self.power)


def quantum_kernel_normalized(psi: torch.Tensor) -> torch.Tensor:
    """对 psi 先归一化（防止数值漂移）再算保真度。"""
    norms = psi.abs().pow(2).sum(dim=-1, keepdim=True).sqrt()
    psi_n = psi / norms.clamp_min(1e-12)
    return quantum_kernel(psi_n)


def linear_overlap_kernel(psi: torch.Tensor, mode: str = "imag") -> torch.Tensor:
    """线性内积核 K[b,i,j] = ⟨ψ_i|ψ_j⟩ 的实部/虚部（有符号）。

    与保真度 |⟨ψ_i|ψ_j⟩|² 的根本区别（2026-08-13 轮 C，有向混合）：
    - 虚部反对称（K_ij = −K_ji，相位 e^{iφ} 编码时滞/领先方向）→ 混合权重天然有向，
      经典实向量点积无法表达；保真度 abs().pow(2) 恰好把相位（有向信息）扔掉
    - 均值 0、相对波动 O(1) → 无保真度在 2^N 维的浓度问题（diag=1 主导 / softmax 均匀化）
    - 对角线 Im=0 → 恒等映射问题天然不存在，无需 offdiag 补丁
    """
    # 修复：clone 共轭视图，避免 einsum 反向传播时的内存共享问题
    psi_conj = psi.conj().clone()
    inner = torch.einsum("bvi,bwi->bvw", psi_conj, psi)
    if mode == "imag":
        return inner.imag
    if mode == "real":
        return inner.real
    raise ValueError(f"mode must be 'real' or 'imag', got {mode}")


def kernel_diag_check(K: torch.Tensor, atol: float = 1e-4) -> bool:
    """核对角线是否全 ≈ 1（酉演化保范）。"""
    eye = torch.eye(K.size(-1), device=K.device, dtype=K.dtype)
    err = (K - eye.unsqueeze(0)).abs().max().item()
    return err < atol
