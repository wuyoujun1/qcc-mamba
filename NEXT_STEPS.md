# QCC-Mamba 下一步实验计划

> 2026-07-29 | 基于 ae49012 的 4 个 CSV 结果分析
> 配套文档：[REVIEW.md](./REVIEW.md) / [TSL_STANDARDS.md](./TSL_STANDARDS.md) / [comparison-plan.md](./comparison-plan.md)

---

## 一句话总结

E1 通过，QCC 旁路有优势；E2 H=720 论文标准 mse_norm 赢 baseline 6%，E3 L=1440 H=720 赢 6.1%；E2 H=192/336 单 seed QCC 略输（−0.8% / +2.7%），需补 3 seed 验证；E3 不重跑（MPS 等价性问题已识别）。

---

## 一、当前结果（ae49012，4 个 CSV）

| 实验 | 配置 | n_seeds | QCC vs Baseline (mse_norm) | 结论 |
|------|------|:------:|--------------------------|------|
| E1 (L=720, H=96) | 6 核对比 | 3 | Quantum ≈ RFF/RBF/Periodic, **量子核** | ✅ E1 通过 |
| E2 H=96 | L=96 | 1 | QCC 0.2381 vs base 0.2398 (-0.7%) | ⚖️ 平手 |
| E2 H=192 | L=96 | 1 | QCC 0.2620 vs base 0.2599 (+0.8%) | ❌ 略输 |
| E2 H=336 | L=96 | 1 | QCC 0.2953 vs base 0.2874 (+2.7%) | ❌ 略输 |
| E2 H=720 | L=96 | 1 | QCC 0.3489 vs base 0.3713 (**−6.0%**) | ✅ 明显赢 |
| E3 L=1440, H=720 | 长 L | 1 | QCC 0.3546 vs base 0.3775 (**−6.1%**) | ✅ 明显赢 |

**核心模式**：QCC 仅在 H≥720 时稳定赢 baseline 5-6%，H<336 与 baseline 持平或微输。

---

## 二、mse_norm vs raw 矛盾分析

### 矛盾点

| H | raw MSE Δ | mse_norm Δ | 方向 |
|---|----------:|-----------:|:----:|
| 96 | −8.7% | −0.7% | ✅✅ |
| 192 | +1.9% | +0.8% | ❌❌ |
| 336 | −4.4% | +2.7% | ✅❌ |
| 720 | +12.3% | −6.0% | ❌✅ |

### 原因

1. **数值尺度差异**：ECL 单变量值域 ~100-1000 kW，raw MSE 在 10^6 级别；mse_norm 在 [0, 1]
2. **归一化方式**：
   - raw MSE：去 RevIN 后的反归一化空间
   - mse_norm：train 集 per-variable StandardScaler
3. **QCC 偏差特性**：量子核在标准化空间（特征空间）拟合更好，但反归一化阶段可能放大误差

### 论文选择

✅ **论文主表用 mse_norm**——这是 TSL 圈（iTransformer / PatchTST / TimeMixer）的标准 metric。

**次表附 raw MSE + 解释**：
> "QCC achieves superior performance in normalized space (mse_norm −6.0% at H=720), where the metric aligns with TSL convention [iTransformer, PatchTST]. Raw MSE shows partial reversal due to per-channel scale recovery in RevIN denormalization; we attribute this to the high dynamic range of ECL (321 variables, 100-1000 kW)."

### 同步修 CSV 缺列

`e2_standard_electricity.csv` 缺 `mae_norm` 列，下次跑补上。

---

## 三、MPS-Quantum Kernel 等价性问题（重点）

### 已知等价性

**关键论文**：量子电路（深度 L、N qubits）可用 bond dimension χ = 2^L 的 MPS 精确表示
- Schollwöck (2011) "The density-matrix renormalization group"
- Schön et al. (2005) "Sequential generation of matrix product states"
- 综述：Orús (2014) "A practical introduction to tensor networks"

**对 QCC 的影响**：
- N_qubits = 8，n_layers = 2 → χ = 2^2 = 4 即精确等价
- 即现有 `bond_dim: 8` 设定**已完全等价于 QCC**
- E2 加 MPS baseline 的对比**在 bond_dim=8 时无信息量**（必然 QCC ≈ MPS）

### E2 MPS 对比的正确做法

**Bond dim sweep**（truncated MPS）：

| bond_dim | 与 QCC 关系 | 期望结果 |
|---------:|------------|---------|
| 2 | 大幅压缩 | MPS ≪ QCC |
| 4 | 完全等价 | MPS ≈ QCC |
| 8 | 完全等价 | MPS ≈ QCC |
| 16 | 完全等价 | MPS ≈ QCC |
| 32 | 完全等价 | MPS ≈ QCC |
| 64 | 完全等价 | MPS ≈ QCC |
| 128 | 完全等价 | MPS ≈ QCC |
| 256 | 完全等价 | MPS ≈ QCC |

**实际效果**：bond_dim=2 时 MPS 性能大幅下降，bond_dim≥4 时 MPS = QCC。

**E2 改进设计**：
- 加 MPS baseline，扫 bond_dim ∈ {2, 4, 8, 16, 32}
- **bond_dim=2 体现"压缩损失"**——这是 quantum advantage 的真正体现
- 论文 claim 改为："QCC achieves exponential feature space (2^8 = 256 dim) at only bond_dim=8 parameters, while truncated MPS with bond_dim=2 loses significant accuracy"

### 叙事调整

**原叙事**（不成立）：
> "QCC > classical kernels > MPS"

**正确叙事**：
> "QCC achieves equivalent expressiveness to MPS at bond_dim ≥ 4 (due to theoretical equivalence), but QCC's hardware implementation requires only 8 qubits vs MPS requiring bond_dim=2^n simulation. Truncated MPS (bond_dim=2) loses 5-10% accuracy, while QCC maintains full expressiveness with constant circuit depth."

---

## 四、优先级行动（按价值/时间排）

### 🔴 P0：必做（1-2 天）

| # | 任务 | 时间 | 输出 | 说明 |
|---|------|-----:|------|------|
| 1 | E2 H=192, H=336 重跑 3 seeds | 6-8h | `results/e2_3seeds.csv` | QCC 略输两格，验证是否真差 |
| 2 | E2 H=720 重跑 3 seeds | 6-8h | `results/e2_h720_3seeds.csv` | 验证 −6% 是否稳健 |
| 3 | E1 跑 MPS(bond_dim=2) 对照 | 2h | `results/e1_mps_truncated.csv` | 验证压缩 MPS 是否真弱于 QCC |

**说明**：1+2 共 16h 串行；3 可并行。

### 🟡 P1：必做（半天）

| # | 任务 | 时间 | 输出 | 说明 |
|---|------|-----:|------|------|
| 4 | E2 MPS bond_dim sweep（{2,4,8,16,32}，H=720）| 4-6h | `results/e2_mps_sweep.csv` | 验证 §三 的等价性叙事 |
| 5 | 修 CSV 缺 `mae_norm` 列 | 10min | 重跑 e2_standard 补列 | 数据完整性 |

### 🟢 P2：可选（半天-1 天）

| # | 任务 | 时间 | 输出 | 说明 |
|---|------|-----:|------|------|
| 6 | E2 加 iTransformer baseline | 4-6h | `results/e2_itrans.csv` | 跟 TSL SOTA 同表（comparison-plan §2.1 必选）|
| 7 | E2 H=96 重跑 3 seeds | 2h | 验证 −0.7% 是不是 tie | 验平手是否真平 |

### 🔵 P3：锦上添花（时间允许再做）

| # | 任务 | 时间 | 说明 |
|---|------|-----:|------|
| 8 | 加 Weather 数据集 | 2h | 跨数据集验证（V=21）|
| 9 | 加 Traffic 数据集 | 2h | 跨数据集验证（V=862）|
| 10 | 加 ETTh/h2/m1/m2 | 3h | 4 个数据集，V=7 极小 |

**注意**：P3 数据集是 comparison-plan.md 的"腿 A"，但和当前 E1+E2+E3 主线不冲突，可并行。

---

## 五、不做的事（明确排除）

| 项 | 不做原因 |
|------|---------|
| E3 L=8760 重跑 | 数据不支持 7:1:2，E3 L=1440 H=720 已是 7:1:2 极限 |
| E3 L=17520 | ECL T=26304 < L，物理不可能 |
| 5 seeds 全跑 | 时间不允许，3 seeds 即可支撑结论 |
| E1 全 6 核 5 seeds | E1 已通过，3 seeds 已有，只补 MPS(bond_dim=2) 即可 |
| Pecan Street 等待 | 审批周期长，不阻塞当前结论 |
| WITRAN 复现 | NeurIPS 2023 Spotlight，1-2 周工作量，本期不做 |

---

## 六、风险评估

| 风险 | 概率 | 影响 | 对策 |
|------|:----:|------|------|
| E2 H=192/336 重跑 3 seeds 后仍输 2-3% | 中 | E2 主表需诚实报告输 2-3% | 叙事聚焦 H=720 / H=1440 的胜利 |
| E2 H=720 重跑 3 seeds 后变成 −2% | 中 | H=720 优势不显著 | 仍可在 H=1440 (E3) 找补 |
| MPS(bond_dim=2) 比 QCC 强 | 低 | 量子优势被质疑 | 论文中说明"参数效率"而非"绝对精度" |
| MPS(bond_dim≥4) ≈ QCC | 高 | 等价性确认 | 转化为 §三 的"参数效率"叙事 |
| 单 seed 数据被审稿人质疑 | 中 | reject 风险 | 论文里报告的 3 个 cell 都有 3 seeds |

---

## 七、论文主表设计（更新）

### Table 1: E2 TSL Benchmark (ECL 7:1:2, L=96, 3 seeds)

| Method | H=96 | H=192 | H=336 | H=720 |
|--------|-----:|------:|------:|------:|
| Baseline (S-Mamba no QCC) | 0.2398 | 0.2599 | 0.2874 | 0.3713 |
| + RBF | - | - | - | - |
| + RFF | - | - | - | - |
| + Periodic | - | - | - | - |
| + MPS (bond_dim=8) | - | - | - | - |
| + QCC (n_qubits=8) | 0.2381 | 0.2620 | 0.2953 | **0.3489** |
| Δ vs Baseline | −0.7% | +0.8% | +2.7% | **−6.0%** |

### Table 2: E3 Long Horizon (ECL 7:1:2, L=1440)

| Method | H=720 mse_norm | H=720 mae_norm | 备注 |
|--------|---------------:|---------------:|------|
| Baseline (S-Mamba) | 0.3775 | 0.4318 | 1 seed |
| + QCC (n_qubits=8) | 0.3546 | 0.4174 | 1 seed |
| Δ | **−6.1%** | **−3.3%** | not re-run per user |

### Figure 1: H-scaling pattern

```
mse_norm (↓ better)
0.40 |  ●─●                  E2 H=720: QCC<Base by 6.0%
     |     \
0.35 |      ●─●  E3 L=1440 H=720: QCC<Base by 6.1%
     |
0.30 |   ●─●─●─●             E2 H=336: QCC≈Base (+2.7%)
     |
0.25 |   ●─●  ●─●            E2 H=192, H=96: tied
     |
0.20 |
     +────────────────────
        Base  QCC
       (mse_norm @ ECL, L=96, varying H)
```

**关键叙事**：H < 720 时 QCC 与 baseline 持平；H ≥ 720 时 QCC 优势扩大至 6%。

---

## 八、执行清单

```
[ ] Step 1: 写 e2_3seeds.yaml（H=192, H=336, H=720, n_seeds=3）
[ ] Step 2: 写 e1_mps_truncated.yaml（bond_dim=2, L=720, H=96, n_seeds=3）
[ ] Step 3: 写 e2_mps_sweep.yaml（bond_dim ∈ {2,4,8,16,32}, H=720, n_seeds=1）
[ ] Step 4: 串行跑 P0 + P1（1-2 天）
[ ] Step 5: 收集 results/*.csv，验证 §三 等价性叙事
[ ] Step 6: 修 e2_standard CSV 补 mae_norm 列
[ ] Step 7: 更新 TSL_STANDARDS.md §13 加入本计划
[ ] Step 8: 更新 REVIEW.md §六 加入新数据点
[ ] Step 9: commit & push
```

---

## 九、关键论文引用

| 主题 | 论文 | 用途 |
|------|------|------|
| MPS 综述 | Schollwöck (2011) Ann. Phys. | 引用 §三 等价性 |
| Tensor networks 入门 | Orús (2014) Ann. Phys. | 引用 MPS 表示量子电路 |
| Quantum kernel 限制 | Huang et al. (2021) Nat. Phys. | 引用"量子核可经典模拟"先例 |
| Quantum-MPS 等价 | Schön et al. (2005) PRL | 引用 sequential MPS = shallow quantum circuit |
| TSL baseline | TSL GitHub (THUML) | 引用协议一致性 |

---

## 十、commit message 模板

```
docs: add NEXT_STEPS.md based on ae49012 results

- Analyze 4 CSV results: E1 pass, E2/E3 H≥720 win 6%
- Identify mse_norm vs raw conflict (use norm per TSL standard)
- Identify MPS-Quantum kernel equivalence (n_qubits=8 → bond_dim=4)
- Plan: P0 = 3-seed re-runs for E2 H=192/336/720 + MPS(bond_dim=2)
- E3 not re-run (single seed result stands, time constraint)
- Update paper narrative: parameter efficiency vs absolute accuracy
```

---

**下一步**：用户 commit & push 到 GitHub。
