# 相关工作与引用库（RELATED_WORK_CITATIONS）

> 用途：Related Work / Introduction / Method 定位的引用素材。**严格对齐 `PAPER_OUTLINE_TKDE.md` §2 Related Work 四脉络 + 必引区分表**。
> 写作以大纲为准；本文件提供每条的元数据 + cite key + 定位 + 落点。
> 最后更新：2026-08-27

---

## A. 施金金组 6 篇（本论文范式来源 + 风格范本）

> 投稿人是施金金组成员，这 6 篇是行文逻辑/语言风格的对齐基准，也是方法学引用源。其中 ICWE 与 Boson Sampling 是组里的 TKDE 论文。

### A1. ICWE-QNN/CICWE-QNN — Two End-to-End Quantum-Inspired DNN for Text Classification 【组内·TKDE】
- 作者：Jinjing Shi, Zhenhuan Li, Wei Lai, Fangfang Li, Ronghua Shi, Yanyan Feng, Shichao Zhang
- 发表：*IEEE TKDE*, Vol. 35, No. 4, pp. 4335–4348, Apr 2023
- DOI：10.1109/TKDE.2021.3130598
- 文件：`refs/Two_End-to-End_Quantum-Inspired_Deep_Neural_Networks_for_Text_Classification.pdf`
- cite key：`shi2023icwe`
- 范式贡献：Hilbert 空间量子启发复词嵌入；$|\varphi\rangle=\sum_j r_j e^{ic_j}|e_j\rangle$；义词=多态量子粒子，句子=粒子干涉量子系统。GRU+attention+conv 端到端。
- 落点：复振幅分解三件套来源（Related Work §2.2）；QCCE 的"变量=量子态、跨变量消息=粒子干涉"本体论先例（Method §3.3）。

### A2. QPFE-ERNIE — Pretrained Quantum-Inspired DNN for NLP 【组内·TCYB】
- 作者：Jinjing Shi, Tian Chen, Wei Lai, Shichao Zhang, Xuelong Li
- 发表：*IEEE Trans. Cybernetics*, Vol. 54, No. 10, pp. 5973–5986, Oct 2024
- 文件：`refs/Pretrained_Quantum-Inspired_Deep_Neural_Network_for_Natural_Language_Processing.pdf`
- cite key：`shi2024qpfe`
- 范式贡献：复嵌入层=振幅嵌入（语义）+相位嵌入（短语交互/位置隐式特征）；预训练相位参数作先验加速拟合；$|t\rangle=\sum_i r_i e^{i\theta_i}|e_i\rangle$。
- 落点："振幅=内容/语义、相位=上下文"共识（§2.2）；QCCE 的 s 作确定性先验（detach）类比预训练相位思想（§3.3）。

### A3. MQN — Multimodal Quantum-inspired Network for Emotion Recognition 【组内·ESWA】
- 作者：Zimeng Xiao, Jia Liao, Jinjing Shi*, Shichao Zhang, Xuelong Li
- 发表：*Expert Systems with Applications*, 2025
- 文件：`refs/1-s2.0-S0957417425036796-main.pdf`
- cite key：`xiao2025mqn`
- 范式贡献：CFE（对话导向特征嵌入）振幅+相位嵌入多模态/说话人/位置信息；JIF 联合信息融合；QGRU 演化复依赖；密度矩阵测量。**关键 Eq.4**：$|\varphi\rangle=\sum_j z_j|k_j\rangle=\sum_j r_j e^{i\theta_j}|k_j\rangle$。相位加法通道：$\varphi_j=\text{pos}\cdot w_p^j+\text{spe}\cdot w_s^j+\theta_j$。
- 落点：相位=可学习频率加法通道（§2.2）；QCCE 的 $\theta(s)=W_\theta s$ 推广自 MQN 的离散词级到 SSM 隐状态级（§3.3）。

### A4. PHL — Parameterized Hamiltonian Learning With Quantum Circuit 【组内·TPAMI】
- 作者：Jinjing Shi, Wenxuan Wang, Xiaoping Lou, Shichao Zhang, Xuelong Li
- 发表：*IEEE TPAMI*, Vol. 45, No. 5, pp. 6086–, May 2023
- DOI：10.1109/TPAMI.2022.3203157
- 文件：`refs/tpami_phl.pdf`
- cite key：`shi2023phl`
- 范式贡献：参数化量子电路分解酉算子激发系统演化；迭代更新损失对电路参数梯度准备哈密顿系统；图像分割应用。
- 落点："参数化量子电路+可学习参数"方向（§2.1 量子门/电路）；QCCE 的 RZ∘RY 电路 + 可学习 $W_\theta$ 同源（§3.3）。

### A5. Quantum Circuit Learning With Parameterized Boson Sampling 【组内·TKDE 投稿范式】
- 作者：Jinjing Shi, Yongze Tang, Yuhu Lu, Yanyan Feng, Ronghua Shi, Shichao Zhang
- 文件：`refs/Quantum_Circuit_Learning_With_Parameterized_Boson_Sampling.pdf`
- cite key：`shiBoson`
- 范式贡献：基于 boson sampling 的参数化量子电路结构；梯度优化迭代更新电路参数；两种 circuit loss（KMM/MSE）；方差降至 2.54e-6 / 6.91e-6；电路深度 d 增则复杂度降。
- 落点：**TKDE 投稿的行文范式基准**（结构/语言/公式体例）；量子梯度下降写法（第四章 §4.2）；§2.1 量子电路学习。

### A6. QSyncFold — Quantum NN for Multidimensional Sync-Discovery in Protein Folding 【组内·BIB】
- 作者：Jinjing Shi, Peng Du, Wenwu Zeng, Wenxuan Wang, Shaoliang Peng, Xuelong Li
- 发表：*Briefings in Bioinformatics*, Vol. 27, Issue 3, May 2026, bbag234
- URL：https://academic.oup.com/bib/article/27/3/bbag234/8698685
- DOI：10.1093/bib/bbag234
- 文件：`refs/QSyncFold_BIB2026.pdf`
- cite key：`shi2026qsyncfold`
- 范式贡献：量子神经网络多维同步发现；蛋白质折叠应用。
- 落点：组内最新量子 NN 应用，佐证量子方法泛化性（§2.1/§2.2）。

### A7. GQHAN — Grover-inspired Quantum Hard Attention Network 【组内·TPAMI】
- 作者：Ren-Xin Zhao, Jinjing Shi*
- 发表：*IEEE TPAMI*（vol/issue 待补）
- 文件：`refs/GQHAN.pdf`
- cite key：`zhaoGqhan`
- 范式贡献：Grover 启发量子硬注意力机制 GQHAM（Flexible Oracle 解非可微 + Adaptive Diffusion Operator + QHAS 可视化）；PennyLane 实现；Fashion MNIST/CIFAR-10 二分类。
- 落点：**量子注意力/混合机制的直接前身**，与本文 QCCK 的"量子核做跨变量混合"同范畴（§2.1 量子注意力/§2.2）；可解释性论证范式（K 可视化对标 QHAS）。

> **三件套提炼**（Related Work §2.2 论述依据，自 ICWE/QPFE/MQN 提炼）：
> 1. 复振幅分解 $z_j=r_j e^{i\theta_j}$（振幅=内容，相位=语境）；2. 相位加法通道；3. 测量 $p=|\langle\lambda|\varphi\rangle|^2$。

---

## B. 主干与基线（Mamba / 时序预测，§2.4）

| 方法 | cite key | 出处 | 落点 |
|---|---|---|---|
| S-Mamba | `wang2024smamba` | arXiv:2403.11144, (ICLR25 按大纲) | 主干 + 唯一核心基线；实测零耦合响应 |
| Mamba | `gu2024mamba` | arXiv:2312.00752 | SSM 背景 |
| DeMa | `an2026dema` | arXiv:2601.05527 | 双路径时延感知 Mamba，同期跨变量工作 |
| iTransformer | `liu2024itransformer` | ICLR24 | 变量作 token，黑盒注意力对照 |
| Crossformer | `zeng2023crossformer` | — | 跨变量注意力对照 |
| PatchTST | `nie2023patchtst` | ICLR23 | patch+通道独立 |
| DLinear | `zeng2023dlinear` | AAAI23 | 线性分解 |
| TimesNet | `wu2023timesnet` | ICLR23 | 1D→2D 多周期 |
| SOFTS | `softs2024` | — | %TODO venue |

---

## C. 频域时序方法（§2.3，含必引 FAN）

| 方法 | cite key | 出处 | 落点/区分 |
|---|---|---|---|
| **FAN** ⚠️必引 | `fan2024` | arXiv:2409.20371 | **必须主动区分**：FAN 处理时间维非平稳（输入层/取峰/分解），我们处理变量间异质（核层/双轴对齐/均匀采样）|
| Autoformer | `wu2021autoformer` | NeurIPS21 | 自相关 |
| FEDformer | `zhou2022fedformer` | ICML22 | 频域增强 |
| FreTS | `yi2023frets` | — | 频域时序 |
| FiLM | `film2023` | — | 频率插值 |
| FreDF | `freDF2024` | %TODO | 频域监督 loss（本文用） |

---

## D. 量子核/量子时序（§2.2，含 QuaCK/HAQJSK）

| 方法 | cite key | 出处 | 落点/区分 |
|---|---|---|---|
| QLSTM | `qlstm2022` | ICASSP22 | 量子时序 |
| QRC | `qrc` | — | 量子游走/递归 |
| QGRNN | `qgrnn` | TPAMI | 量子门递归神经网络 |
| **QuaCK** | `quack2024` | arXiv24 | 量子核时序对照 |
| **HAQJSK** | `haqjsk2024` | TKDE24 | **对齐图顶点 vs 我们对齐频谱双轴**；核心区分点 |
| SPE | `spe2026` | EPJ QT 2026, %TODO | 纯相位核 $K=\frac1{2^{2n}}|\sum_j e^{i\Delta_j}|^2$，本文引理2特例 |
| ComplEx | `trouillon2016complex` | ICML16 | 复数共轭建模反对称关系，$S_{ij}$ 方向项先例 |
| RFF | `rahimi2007rff` | NeurIPS07 | Bochner，相位通道=复傅里叶特征，rff_kernel 对照 |
| Huang 2021 几何差 | `huang2021geometric` | %TODO | PQK 可模拟性（用户指示暂不管诚实边界，备引）|

---

## E. 归一化/对齐（§2.3 补充）

| 方法 | cite key | 区分 |
|---|---|---|
| RevIN | `kim2022revin` | 管统计非平稳；我们管结构非平稳（相位/周期漂移）且对齐在核输入层（可证不变）|
| SAN | `san` | — |
| Dish-TS | `dishTS` | — |
| DTW 族/VTLN | `dtw` | — |

---

## F. 引用落点速查（写章节按此插 \cite）

| 论文章节 | 必引 keys |
|---|---|
| I Intro·背景(SSM) | `gu2024mamba`,`wang2024smamba`,`an2026dema`,`liu2024itransformer` |
| I Intro·量子启发动机 | `shi2023icwe`,`shi2024qpfe`,`xiao2025mqn` |
| II §2.1 量子态/量子门 | `shi2023phl`,`shiBoson`,`shi2026qsyncfold`,`zhaoGqhan` |
| II §2.2 量子核方法 | `shi2023icwe`,`shi2024qpfe`,`xiao2025mqn`,`spe2026`,`trouillon2016complex`,`quack2024`,`haqjsk2024`,`zhaoGqhan` |
| II §2.3 频域时序方法 | `fan2024`(必引区分),`wu2021autoformer`,`zhou2022fedformer`,`wu2023timesnet`,`yi2023frets`,`film2023`,`freDF2024`,`kim2022revin` |
| II §2.4 时序预测/跨变量 | `wang2024smamba`,`liu2024itransformer`,`zeng2023crossformer`,`nie2023patchtst`,`zeng2023dlinear`,`wu2023timesnet` |
| III §3.3 QCCE | `xiao2025mqn`(Eq.4同构),`shi2024qpfe`(振幅=语义/相位=上下文),`shi2023icwe`(干涉本体),`rahimi2007rff`(相位=复傅里叶),`spe2026`(纯相位特例),`trouillon2016complex`(方向项) |
| III §3.2 频域对齐 | `fan2024`(区分),`kim2022revin`(区分) |
| IV QCCK-M | `wang2024smamba`,`freDF2024`,`shiBoson`(梯度范式) |
| V 实验·基线 | `wang2024smamba` 及 B 表全部 |

> %TODO：FreDF/SOFTS/FreTS/FiLM/QLSTM/QRC/QGRNN/QuaCK/HAQJSK/SPE/Huang/SAN/Dish-TS/DTW 的精确 venue 与作者——references.bib 中标 %TODO，后续查补（用户指示非关键）。
