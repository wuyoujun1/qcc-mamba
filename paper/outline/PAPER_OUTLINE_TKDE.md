# TKDE 投稿论文大纲（只保留结构）

> 2026-09-11 重写：**原大纲内嵌的第三章旧正文（QCCM / PAI / 因子分解）与 §5 的旧方法口径、已证伪的"浓度定理"全部移除**。
> 正文一律以现行稿为准：第三章 `paper/ch3/第三章修改版1.md`、第四章 `paper/ch4/第四章修改版1.md`、第五章 `paper/ch5/ch5cn_pdf.py`（输出 `第五章初稿_中文_20260909.pdf`）。
> 投稿目标：IEEE TKDE（CCF-A），正文 ≤12 页。

---

## 1. 标题

**Quantum Cross-Variable Coupling Kernels for Mamba Multivariate Time Series Forecasting**

---

## 2. 整体结构（7 章）

```
1. Introduction                    1.5 页
2. Related Work                    1 页
3. Preliminaries                   1.5 页（量子核 / 傅里叶；含一张图，覆盖量子核与频域对齐）
4. Method                          2.5 页（核心）
5. Experiments                     3.5 页（核心）
6. Conclusion                      0.5 页
   References                      1 页
```

当前写作范围：第 2 章 → 第 5 章。大纲章与正文章节的对应：

| 大纲章 | 正文来源 | 内容 |
|---|---|---|
| 3 | `paper/ch3/第三章修改版1.md` | QCCK：频域对齐 / QCCE 编码 / 量子核 |
| 4 | `paper/ch4/第四章修改版1.md` | QCCK-M 网络：整体结构 / 前向算法 / 损失与训练 |
| 5 | `paper/ch5/ch5cn_pdf.py` | 实验：5.1 设置 / 5.2 主结果 / 5.3 机制验证 / 5.4 消融 / 5.5 可解释性 |

---

## 3. 各章骨架

**Abstract（150–250 词）**

> 背景句（多变量预测、长视界、跨变量耦合关键）→ 问题句（现有主干实测对跨变量输入近乎零响应，耦合缺失）→ 方案句（量子核作为主干内显式跨变量混合算子 + 频谱双轴对齐）→ 结果句（机制验证：灵敏度跳升、K 与数据耦合对齐；性能：QCCK-M 优于基线）→ 意义句（可解释耦合矩阵、时移与尺度不变）。

### 1 Introduction（缺陷驱动）

解决经典模型跨变量耦合缺失，提出量子核作为主干内显式跨变量混合算子，并用频域对齐加强其跨变量建模能力。

三条贡献：

1. 提出 **Quantum Cross-Variable Coupling Kernels** 模块（第三章）
2. 提出 **QCCK-Mamba** 网络（第四章）
3. 实验效果更优，并具有可解释性（第五章）

### 2 Related Work

- 2.1 量子态，量子门
- 2.2 量子核方法
- 2.3 频域时序方法
- 2.4 时序预测：s-mamba、mamba、跨变量时序预测

按脉络组织（代表工作 → 我们的位置）：

| 脉络 | 代表工作 | 缺陷/差异（我们的位置） |
|---|---|---|
| **多变量时序预测** | iTransformer（ICLR24）、Crossformer、SOFTS、S-Mamba（ICLR25）、PatchTST、DLinear | 变量关系**隐式/黑盒**；S-Mamba 实测耦合响应极低 |
| **频域时序方法** | Autoformer、FEDformer、TimesNet、FreTS、FiLM、**FAN**（arXiv 2409.20371） | 频域都处理**时间维**；**没人做"变量间频谱对齐"** |
| **归一化/对齐** | RevIN、SAN、Dish-TS、FAN、DTW 族、VTLN | RevIN 管统计非平稳；我们管**结构非平稳**（相位/周期漂移），且对齐在**核输入构造**层 |
| **量子核/量子时序** | QLSTM（ICASSP22）、QRC、QGRNN（TPAMI）、QuaCK（arXiv24）、HAQJSK（TKDE24） | HAQJSK 对齐图顶点，我们对齐**频谱双轴** |
| **变量关系可视化/图建模** | GAT、Crossformer 跨变量注意力、iTransformer 注意力 | 注意力是黑盒；K 是**显式、可解释、逐样本自适应**的耦合矩阵 |

> ⚠️ **必引 FAN**：主动区分——FAN 处理时间维非平稳（输入层、取峰、分解），我们处理变量间异质（核层、双轴对齐、均匀采样）。
> 📚 引用库与划界见 `paper/outline/RELATED_WORK_CITATIONS.md`。

### 3 QCCK

正文：`paper/ch3/第三章修改版1.md`

- 3.1 总览：三环节串行 → 耦合矩阵 K
- 3.2 频域对齐：时间轴对齐（相位去趋势）+ 频率轴对齐（主频归一化）→ 对齐频谱 S
- 3.3 QCCE 编码：振幅通道（主干隐状态）+ 相位通道（对齐频谱）→ 量子态 |ψ_v⟩
- 3.4 量子核：保真度核 → 耦合矩阵 K

### 4 QCCK-M

正文：`paper/ch4/第四章修改版1.md`

- 4.1 总览：架构图（主干灰色 + 我们的部分彩色放大），数据流 x → RevIN → Tokenize → 每层 [S-Mamba 编码器 → QCCK Layer] → 预测头
- 4.2 算法与训练：单层前向算法、损失（时域 + FreDF 频域监督）、训练细节

### 5 Experiments

定版：`paper/ch5/ch5cn_pdf.py`

- 5.1 实验设置
- 5.2 主结果（vs 基线）
- 5.3 机制验证
- 5.4 消融
- 5.5 可解释性

### 6 Conclusion

总结三条贡献 + 诚实边界 + 未来工作（数据重上传深度、可学习对齐等）。

---

## 4. 核心卖点清单

1. **实证驱动的动机**（非假设）：主干对跨变量输入的响应极低，比"我们假设耦合重要"强一个量级；
2. **显式可解释耦合**：K 是可读取、可可视化的变量耦合矩阵，对标 iTransformer 的黑盒注意力；
3. **频域对齐与 FAN 的清晰分工**（时间维 vs 变量间、输入层 vs 核层）；
4. **诚实边界**：弱耦合数据集上增益小，说明方法有适用前提、不添乱。
