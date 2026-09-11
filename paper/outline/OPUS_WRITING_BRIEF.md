# QCC/QCCK-M 论文写作总纲（OPUS_WRITING_BRIEF）

> 用途：论文写作导航地图。**严格按 `PAPER_OUTLINE_TKDE.md` 大纲**。
> 投稿目标：IEEE TKDE（CCF-A），≤12 页，7 章。
> **写作范围：第 2 章（Related Work）→ 第 5 章（Experiments）**。Abstract/Intro/Conclusion 大纲已有但本次不写。
> 行文逻辑与语言风格：对齐施金金团队 6 篇组内论文（见 `STYLE_GUIDE.md`，句式级风格已提炼）。
> 最后更新：2026-08-27

---

## 0. 一句话定位（取自大纲）

实证揭示多变量时序预测主干（S-Mamba）的跨变量耦合缺失（灵敏度 ≈10⁻⁶），提出**量子核作为主干内显式跨变量混合算子**，并用**频域双轴对齐 + 核选择性修复**（温度/topk）使其兑现收益。

**标题**：*Quantum Cross-Variable Coupling Kernels for mamba Multivariate Time Series Forecasting*

**三大贡献**（取自大纲）：
1. **Quantum Cross-Variable Coupling Kernels 模块（QCCM）** —— 第三章
2. **Quantum Cross-Variable Coupling Kernels-mamba 网络（QCCK-M）** —— 第四章
3. 实验效果更优，且具有可解释性

---

## 1. 章节结构（严格按大纲，7 章 TKDE 布局，≤12 页）

| 章 | 标题 | 页数 | 核心内容 | 图 |
|---|---|---|---|---|
| Abstract | — | 0.3 | 150–250 词，6 句式（见大纲） | — |
| I | Introduction | 1.5 | 缺陷驱动，7 段，每段一任务；提三贡献 | — |
| II | Related Work | 1.0 | 2.1 量子态/量子门；2.2 量子核方法；2.3 频域时序方法；2.4 时序预测/跨变量 | — |
| **III** | **QCCM: Quantum Cross-Variable Coupling Module** | 1.5 | **频域对齐(3.2.1) + PAI(3.2.2) + 量子核(3.3)**，PAI 只是 3.2.2 一小部分 | 第三章架构图 |
| **IV** | **QCCK-M**（整体网络） | 2.5 | 4.1 总览（架构图，主干灰色+我们彩色）；4.2 伪代码算法/后处理/loss/量子梯度下降；第三章没讲完的补全 | 第四章架构图 |
| V | Experiments | 3.5 | 5.1 设置；5.2 机制验证（R1-R4，独有）；5.3 主结果；5.4 消融；5.5 异质度-增益；5.6 可解释性 | 多图 |
| VI | Conclusion | 0.5 | 三贡献 + 诚实边界 + 未来 | — |
| Refs | — | 1.0 | 见 references.bib | — |

### ⚠️ 章节归属澄清（修正先前错误）
- **第三章 = QCCM 模块**：是"耦合度量模块"本身，由三环节串行——①频域对齐 ②PAI 编码 ③量子核输出耦合矩阵 K。**PAI 只是 3.2.2 一个小节**，不是整章。
- **第四章 = QCCK-M 网络**：把 QCCM 插入 S-Mamba 主干的完整网络；含整体架构、数据流、算法伪代码、loss、梯度下降。大纲里"第三章没有讲完的"（算法/后处理/loss/量子梯度）在第四章。

---

## 2. 第三章 QCCM 详细蓝图（对应 `PAPER_OUTLINE_TKDE.md` §3 + 第三章架构图）

> 大纲已给出第三章完整中文正文（公式 3.1-1~3.3-4），写时按此英文化 + 插图。

**3.1 Module Overview**
- 动机：长 horizon 单变量信息耗尽，预测质量依赖跨变量耦合刻画；S-Mamba 双向 SSM 隐式建模耦合，实测输出对跨变量输入灵敏度≈10⁻⁶ → 耦合信息在主干表征中缺失。
- QCCM 定义：以主干隐状态 H + 原始序列 x 为输入，输出可解释跨变量耦合矩阵 K ∈ ℝ^(V×V)。
- 三环节串行（见图 X）：(i) 频域对齐(3.2.1) → S ∈ ℝ^(B×V×64)；(ii) PAI(3.2.2) → |ψ_v⟩ ∈ ℂ^32；(iii) 量子核(3.3) → K[i,j]=|⟨ψ_i|ψ_j⟩|² (3.1-1)。

**3.2 Frequency-Domain Alignment and Quantum Encoding**
- **3.2.1 Frequency-Domain Alignment**
  - 两类环境差异：时移（sampling-start offset）+ 周期尺度（dominant-period scale）。
  - rFFT → 幅度谱 A_v(f) + 相位谱 φ_v(f)。
  - 时间轴对齐（相位去趋势）：φ̃_v(f)=φ_v(f)−2πf·δ̂_v (3.2-1)，δ̂_v 由 leave-one-out 互相关最优滞后估。
  - 频率轴对齐（主频归一化）：f̃=f/f̂_peak (3.2-2)。
  - 重采样 M=32 均匀点 + 线性插值 → S_v=[Ã_v; φ̃_v] ∈ ℝ^64 (3.2-3)。
  - **性质 1**：S_v 对时移/尺度不变，且与输入长度 L 无关。全程不参与梯度。
- **3.2.2 Phase–Amplitude Interface (PAI)** ← *只是这一个小节*
  - 振幅通道（内容）：p(h_v)=softmax(W_r·h_v) ∈ ℝ^32 (3.2-4)。
  - 相位通道（上下文）：θ(s_v)=W_θ·s_v ∈ ℝ^32 (3.2-5)，W_θ 列=可学习频率。
  - **定义 1（PAI 编码）**：|ψ_v⟩=Σ_{b∈{0,1}⁵} √p_b(h_v)·e^(i·θ_b(s_v))·|b⟩ (3.2-6)，自动归一。
  - **命题 1（电路可编译性）**：可分离特例 → 单级乘积态 |ψ_v⟩=⊗_{q=1}^5 RZ(η_q)·RY(α_q)|0⟩ (3.2-7)。

**3.3 Quantum Kernel and Kernel Factorization**
- **3.3.1 Fidelity Kernel**：K_raw[i,j]=|Σ_b ω_b·e^(i·Δθ_b)|²=C_ij²+S_ij² (3.3-1)；S_ij 反对称（lead-lag）。
- **3.3.2 Kernel Factorization**：**定理 1** K_raw[i,j]=B(h_i,h_j)²·|M_ij|² (3.3-3)；|M|≤1 相位只能抑制；S_ij 方向项（ComplEx 先例）。
- **3.3.3 Row Normalization**：K_n=softmax((K_raw−I)/kernel_T) (3.3-4)，去对角+温度缩放，diag 置零防坍缩。

---

## 3. 第四章 QCCK-M 详细蓝图（对应大纲 §4 + 第四章架构图）

> 大纲第四章只给要点，需展开。

**4.1 Overview**
- 架构图：主干灰色 + 我们的部分彩色放大。
- 数据流：x → RevIN → tokenize → [频谱注入] → 每层 Mamba → QuantumMix → 预测头。
- QCCM 产出的 K_n 作为跨变量消息混合权重，插入主干每层后。

**4.2 Algorithm & Training**（大纲："伪代码算法，后处理，loss，量子的梯度下降等，第三章没有讲完的"）
- 算法伪代码：前向 QuantumMixLayer（H'=LN(H+gate·LN(K_n·(H·W_q)))）。
- 后处理：预测头/反 RevIN。
- Loss：时域 MSE + FreDF 频域监督（L=l_t+λ·|rfft(pred)−rfft(true)|.mean()）。
- 量子梯度下降：PAI 参数 W_r/W_θ、核带宽 κ、温度 T、门 gate 的端到端反向传播；复向量 autograd。
- 视界自适应旋钮：kernel_T 96/192/336→0.1，720→0.2。
- 公平对照：rbf@nq5 作用于同一 2^5 维复特征 [Re;Im]。

---

## 4. 核心卖点清单（取自大纲 §4，诚信边界按用户指示暂不管）

1. **实证驱动动机**：主干零耦合响应 1e-6 + 残差 59% 可解释（非"我们假设"）。
2. **显式可解释耦合**：K=变量耦合矩阵，可可视化（对标 iTransformer 黑盒注意力）。
3. **新理论点**：浓度定理 ⇒ 量子核选择性缺失（行熵论证）+ 双轴对齐不变性定理。
4. **频域对齐与 FAN 分工**：时间维 vs 变量间、输入层 vs 核层。
5. **诚实边界**（用户指示先不管，但大纲有）：对照组 ΔMSE≈0 证明方法有适用前提。

---

## 5. 资产清单（`/home/youjun/paper/`）

```
/home/youjun/paper/
├── OPUS_WRITING_BRIEF.md          # 本文件
├── STYLE_GUIDE.md                # ⭐ 语言风格指南（句式级，写作必看）
├── RELATED_WORK_CITATIONS.md     # 引用素材库
├── PAPER_OUTLINE_TKDE.md         # ⭐ 大纲（权威，写作以它为准）
├── main.tex                       # IEEEtran 模板（待填充）
├── bib/references.bib            # BibTeX（待补全）
├── refs/  8 篇论文：
│   ├── [组内6篇]
│   │   ├── QSyncFold_BIB2026.pdf        (BIB 2026, Vol.27 Iss.3, bbag234)
│   │   ├── GQHAN.pdf                    (TPAMI, Grover量子硬注意力)
│   │   ├── Quantum_Circuit_Learning_..._Boson_Sampling.pdf  (TKDE投稿范式)
│   │   ├── Pretrained_Quantum-Inspired_..._NLP.pdf  (QPFE, TCYB2024)
│   │   ├── Two_End-to-End_..._Text_Classification.pdf (ICWE, TKDE2023)
│   │   └── tpami_phl.pdf               (PHL, TPAMI2023)
│   └── [外部2篇] DeMa, (S-Mamba见arXiv)
├── figs/
│   ├── fig_ch3_architecture.png    # 第三章 QCCM 架构图
│   └── fig_ch4_architecture.png    # 第四章 QCCK-M 网络图
└── sections/                       # 分章草稿
```

外部资料源（草稿与代码）：
| 资料 | 路径 | 用途 |
|---|---|---|
| **大纲（权威）** | `PAPER_OUTLINE_TKDE.md` | 章节结构以此为准 |
| 第三章中文正文 | 大纲 §3 内嵌 | 第三章主源 |
| 理论推导 | `qcc-mamba/THEORY_QUANTUM_INTERFACE.md` | PAI/核因子化详证（可入附录） |
| 方法架构 | `qcc-mamba/METHOD_ARCHITECTURE.md` | 第四章实现细节 |
| 实验数据 | `qcc-mamba/EXPERIMENT_CHAPTER.md` | 第五章数据 |
| 代码 | `qcc-mamba/qcc/*.py` | 实现细节 |
| 实验图清单 | `dataops_ws/PLOTS_INDEX.md` | 5.2 机制验证图 |

---

## 6. 行文逻辑与语言风格指南（对齐施金金团队 6 篇）

> 你是施金金组成员，论文必须像组里出的。以下从 6 篇提炼。

### 6.1 结构惯例
- **标准 7 章**：Intro → Preliminaries → Method（核心模块）→ Network → Experiment → Conclusion。（GQHAN/Boson/PHL/ICWE 均如此；大纲把 Preliminaries 并入第3章 QCCM 前置为 3.1/3.2 概念铺垫）。
- **Method 章先 Overview 再逐节展开**：每节先一句动机/问题，再定义/构造，再性质/定理，再一句作用收尾。
- **定理-证明体例**：定理→"证明："→简短推导→"∎"；性质用"证明思路："。
- **图先行**：Overview 配架构图（Fig.X），后续节引用。

### 6.2 语言风格
- **IEEE 风格**：第三人称被动，"is proposed"、"we define"；不滥用第一人称。
- **每段一个任务**：大纲要求 Intro 7 段每段一任务，Method 节同理——先讲清"为什么需要这个"，再"怎么构造"。
- **定义驱动**：关键概念显式给 Definition（"Definition 1: PAI 编码"）。
- **公式编号连续**：(3.1-1)(3.2-1)…按章-节编号。
- **诚实但克制**：组内论文（如 GQHAN）会老实说"implements on PennyLane / binary classifications"，不过度声称。大纲的"诚实边界"（对照组≈0）按此风格写，不过度自夸。
- **机制-性能双线**：组内论文实验章先"机制验证/分析"再"性能对比"（GQHAN 的 Experimental Analysis 即此），大纲 5.2 机制验证 R1-R4 完全继承此范式——**这是审稿人最吃的一节，要重点写**。

### 6.3 必引区分（Related Work 要点，取自大纲表格）
- **vs S-Mamba/iTransformer**：变量关系隐式黑盒 vs 我们显式可解释 K。
- **vs FAN**（必引）：FAN 处理时间维非平稳（输入层/取峰/分解），我们处理变量间异质（核层/双轴对齐/均匀采样）。
- **vs 量子核 QuaCK/HAQJSK**：无人在时序处理核选择性/浓度定理；HAQJSK 对齐图顶点，我们对齐频谱双轴。

---

## 7. 下一步执行顺序

1. ✅ 建工作区 + 搬 8 篇论文 + 2 图 + 大纲（完成）
2. ✅ 重写 BRIEF（对齐正确章节结构）（完成）
3. ⬜ 重写 RELATED_WORK_CITATIONS（补 6 篇组内论文 + FAN/QuaCK/HAQJSK）
4. ⬜ 等用户资料/指令，按大纲填 main.tex：Abstract→I→II→III(QCCM)→IV(QCCK-M)→V→VI
5. ⬜ 第五章优先写 5.2 机制验证（审稿人最看重）
6. ⬜ 实验图生成 + 编译验证

> **写作第一原则：以 `PAPER_OUTLINE_TKDE.md` 为准。任何与此大纲冲突的旧文档（METHOD_ARCHITECTURE/EXPERIMENT_CHAPTER 等）以大纲为准。**
