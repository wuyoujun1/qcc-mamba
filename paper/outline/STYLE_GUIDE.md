# 语言风格指南（STYLE_GUIDE）

> 用途：写作时的句式/段级/论证风格参照。从施金金组 6 篇论文逐字提炼，确保写出像组里出的论文。
> 范围：本论文只写 **第 2 章（Related Work）→ 第 5 章（Experiments）**。
> 最后更新：2026-08-27

---

## 0. 总则（用户口授，优先于以下所有句式模板）

- **用英文的思路写中文**：中文初稿按英文学术写作的逻辑组织句子，使中文成为"英文直译底稿"，英文化时逐句机械对应、不返工。
- **每句必须有显式主语 + 谓语(+宾语)**；找不到主语就用"我们"。禁止用"这些/上述/那些"等孤立指代词开头当主语/指路（否则译文是 dangling 的 these/those）。
- **附加信息用非限定定语从句**（该…/其…=which…），名词前不堆一串"…的…的…"长定语，拆成后置从句。
- **句间逻辑显式**：承接/因果/顺序用 then/after/从而/使…能够 等写出。
- 概览层能短句带过不写长句，不写"见 §3.4"式前向指针，一两句讲掉大体过程。
- 最后更新：2026-09-02 补

---

## 1. 整体语域（IEEE/期刊风格）

- **第三人称被动为主**：is proposed / is designed / can be expressed as / is utilized to。少量 "we" 用于提出主张（we propose / we define / we conduct）。
- **正式连接词**：Moreover / Furthermore / Nevertheless / Simultaneously / Consequently / Precisely / Obviously / Hence。组内高频用 "Simultaneously"、"Nevertheless"、"Obviously" 做转折或强调。
- **不口语化**：用 "tremendously" / "substantial" / "constitutes a threat" / "surmount" / "engender" 这类正式词，不用 "a lot of / make / big"。
- **每个概念给完整定义**：关键术语首次出现必给 Definition/Notation，配数学符号。

---

## 2. Introduction 段落范式（7 段，每段一个任务）

> 取自 GQHAN Intro 的句式模板，适配大纲缺陷驱动结构。

**段 1·背景句**：`In recent years, X has developed tremendously [cite]. However, [现状缺陷]. This not only [后果1], but also [后果2].`
> 例(GQHAN): "In recent years, QML has developed tremendously. However, many current QML models treat each quantum data equally and neglect..."

**段 2·引入解法概念**：`The above urgent issues can be effectively addressed by Y. Y was first proposed in [cite], as a [性质] model that [机制]. This unique computing mechanism [好处].`
> 句式："can be effectively addressed by" 是组内高频引出方案句。

**段 3·举例佐证 + 转折缺陷**：`One of the striking cases is that Y achieves [数字] for [任务]. Although very effective, Y is stuck in [困境] that hinders [优化] due to [原因].`

**段 4·已有尝试 + 不足**：`To conquer this dilemma, various strategies such as [A] and [B] have been proposed. Nevertheless, [A 的不足]. [B] may also require a trade-off between [效率] and [精度]. Hence, it is inherently more practical and convenient to [我们的方向].`

**段 5·引入我们的核心机制**：`To achieve this goal, a [量子/复值] scheme inspired by [先例] is attempted to compensate for the above shortcoming. [先例机制简述]. Obviously, the working mechanism of [先例] is somehow similar to [我们的对象], as it also involves [共同点]. However, [单用先例不够]. Thus a dramatic modification of this algorithm is necessary to [目标], which induces the three main propositions of this paper:`

**段 6·贡献清单**（大纲三贡献）：
`1. 提出 QCCK 模块` / `2. 提出 QCCK-M 网络` / `3. 实验效果更优且可解释`

**段 7·贡献详述 + 行文路线**：`The remainder of this paper is organized as follows. Section 2... Section 3... Section 4... Section 5... In Section 6, we draw a conclusion.`
> ICWE 用的就是这个固定模板："theory about ... are introduced in Section 2. We describe ... in Section 3. Experimental results are shown in Section 4. In Section 5, we draw a conclusion."

---

## 3. Related Work 段落范式（§2.1–2.4）

> 取自 ICWE §2 风格。

**章首导引句**：`In this section, we briefly review the related work about [主题] based on [理论]. Elemental notations used in our paper are shown in the Table X at the beginning of this section.`
> 组内惯例：Related Work 章首必放 Notation 表（见 ICWE Table 1）。

**小节开头**：`In [领域], [概念] is often represented by [形式], where [解释].`（用 ICWE 量子力学段的 "a single particle is often represented by a superposition state..." 模板）。

**定义内嵌段**：术语加粗引导，如 `**Density Matrix.** In order to describe the amplitude and phase of quantum particles, a quantum superposition state can be denoted as follows:` → 公式 → 解释。

**区分句（必引 FAN/HAQJSK 用）**：
- `Although [前人] is very effective, [前人] 处理 [A 维] whereas 本文处理 [B 维].`
- `To the best of our knowledge, no prior work has [做 X] in the context of [我们的设定].`（组内用 "no one has" 句式标新）

---

## 4. Method 段落范式（第3章 QCCK / 第4章 QCCK-M）

> 取自 Boson §2.2 + GQHAN §III。

**节首动机句**：`In order to [目标], a [组件名] is [proposed/designed], which [一句话作用].`（ICWE/Boson 高频 "In order to ... is proposed"）
> 大纲 3.2："为消除…环境差异，频域对齐以确定性变换…消除两类差异" → 此句式。

**定义驱动**：
`**Definition 1 (QCCE 编码).** 记 b∈{0,1}⁵ 为...，变量 v 的量子态定义为：` → 公式 (3.2-6) → `由 Σp_b=1 立得 ‖ψ_v‖²=1，即该编码自动归一。`

**命题/定理体例**：
`**Proposition 1 (电路可编译性).** [陈述]. **证明思路：** [推导]. ∎`
`**Theorem 1 (核因子化).** [陈述]. *证明.* [推导]. ∎`
> Boson 用 "证明思路："；GQHAN/PinH用 "*证明.*"；统一用一种。

**性质收尾句**：`This [property] guarantees that [好处]，mitigating [问题] to a certain extent.`（GQHAN "mitigating ... to a certain extent" 是组内惯用让步表达）

**小节衔接**：`Based on the [上节产物] constructed in [X.Y], this section defines [本节对象] and derives [其性质].`（大纲 3.3 开头正是此句）

---

## 5. Experiments 段落范式（第5章）

> 取自 GQHAN §V，**机制-性能双线**是组内范式。

**章首导引 + bullet 分段**：
`In this section, GQHAN is implemented on [平台] to conduct [任务]. Precisely, the experiments are categorized into the subsequent segments:`
- bullet 1: 性能对比（"is evaluated and compared ... to emphasize the merit of GQHAN in accuracy"）
- bullet 2: 可视化（"A visualization of X is performed to represent ..."）
- bullet 3: 鲁棒性/真机（"To elucidate the performance on a real quantum computer, the impact of ... is scrutinized"）
- bullet 4: 参数规模（"The advantages of GQHAN on the parameter scaling ... are highlighted by comparing"）

> **大纲 5.2 机制验证（R1-R4）完全继承此范式**——这是审稿人最看重的一节，用 "is scrutinized / is highlighted by comparing" 这类动词。

**数据集段**：`[Dataset], the recognized and widely adopted benchmark datasets in [领域], consists of [N] test images and [M] training images, respectively, each containing [尺寸].`（GQHAN 原句）
> 适配时序：`ETTh1, a widely adopted benchmark in multivariate forecasting, consists of ...`

**对比表描述**：`As shown in Tab. X, [模型] demonstrates significantly higher [指标], highlighting its capability to [能力] with fewer [资源]. Conversely, to achieve similar accuracy to [本文], [基线] require a large increase in [资源].`（GQHAN 原句，直接套用于 vs S-Mamba 对比）

**诚实让步**（组内范式，非自夸）：
- `Acknowledging the existing constraint on [限制], one strategy is to [做法].`（GQHAN "Acknowledging the existing constraint on the number of public qubits"）
- 大纲的诚实边界（对照组≈0）按此风格：`Acknowledging that [对照组增益≈0], this反而证明 [方法有适用前提].`

---

## 6. 高频句式速查表

| 功能 | 组内句式 | 出处 |
|---|---|---|
| 引出方案 | `The above urgent issues can be effectively addressed by Y.` | GQHAN |
| 强调新颖 | `To the best of our knowledge, no prior work has ...` | 通用 |
| 让步转折 | `Although very effective, X is stuck in a dilemma that hinders ...` | GQHAN |
| 列不足 | `Nevertheless, ... proves challenging, and misguided ... may engender suboptimal ...` | GQHAN |
| 引入我们 | `To achieve this goal, a ... scheme inspired by ... is attempted to compensate ...` | GQHAN |
| 机制类比 | `Obviously, the working mechanism of X is somehow similar to Y, as it also involves ...` | GQHAN |
| 章首导引 | `In this section, we briefly review ... Elemental notations are shown in Table X.` | ICWE |
| 定义引入 | `In order to describe ..., a ... can be denoted as follows:` | ICWE |
| 性质收尾 | `This guarantees that ..., mitigating ... to a certain extent.` | GQHAN |
| 实验分段 | `Precisely, the experiments are categorized into the subsequent segments:` | GQHAN |
| 对比强调 | `demonstrates significantly higher ..., highlighting its capability to ... with fewer ...` | GQHAN |
| 诚实限制 | `Acknowledging the existing constraint on ..., one strategy is to ...` | GQHAN |
| 行文路线 | `The remainder of this paper is organized as follows.` | ICWE |

---

## 7. 公式与符号惯例

- 编号：`(3.2-1)` 章节号-序号（大纲已定，沿用）。
- 量子态记号：$|\psi\rangle$、$|b\rangle$、$\langle\cdot|\cdot\rangle$（Dirac）。
- 矩阵转置/共轭：$M^*$（共轭）、$M^\dagger$（Hermitian）。
- 编码公式：$|\psi_v\rangle=\sum_{b\in\{0,1\}^5}\sqrt{p_b(h_v)}\,e^{i\theta_b(s_v)}|b\rangle$。
- 电路门：$RZ(\cdot)$、$RY(\cdot)$、$\otimes$（张量积）。
- 测量/核：$K[i,j]=|\langle\psi_i|\psi_j\rangle|^2$。

---

## 8. 写作范围与执行

**只写第 2→5 章**：
- 第2章 Related Work（§2.1-2.4 + Notation 表 + 必引区分）
- 第3章 QCCK（3.1 总览 + 3.2 频域对齐 + 3.3 QCCE 编码 + 3.4 量子核，插第三章架构图）
- 第4章 QCCK-M（4.1 总览插第四章架构图 + 4.2 算法/loss/梯度）
- 第5章 Experiments（5.1-5.6，**5.2 机制验证优先且最重**）

**不写**：Abstract、第1章 Intro、第6章 Conclusion、References（大纲已有，不在本次范围）。

> Abstract/Intro/Conclusion 的大纲已存，若需补写随时可加。第3章中文正文大纲已给完整版，英文化即可。
