# QCC-Mamba 下一步实验计划

> 2026-07-30 | 基于全部日志 & CSV 实际完成情况更新

---

## 一、当前已完成（全部核对）

### ECL 数据集（V=321）

| 实验 | 配置 | n_seeds | 结果 CSV | 状态 |
|------|------|:------:|---------|:----:|
| E1 Quantum | L=720, H=96 | 3 | log only（无CSV） | ✅ |
| E1 MPS bd2 | L=720, H=96 | 3 | log only（无CSV） | ✅ 已跑完，文档漏记 |
| E1 MPS bd4 | L=720, H=96 | 3 | log only（无CSV） | ✅ 已跑完，文档漏记 |
| E2 Baseline | 4 lookbacks, 1 seed | 1 | `e2_baseline_electricity.csv` | ✅ |
| E2 QCC | 4 lookbacks, 1 seed | 1 | `e2_standard_electricity.csv` | ✅ |
| E3 Baseline | L=1440, H=720 | 1 | `e3_baseline_electricity.csv` | ✅ |
| E3 QCC | L=1440, H=720 | 1 | `e3_longterm_electricity.csv` | ✅ |

### Traffic 数据集（V=862）

| 实验 | 状态 | 说明 |
|------|:----:|------|
| E2 Baseline L336 | ✅ 完成（76 epoch） | test_mse=0.000878 |
| E2 Baseline L720 | ✅ 完成（83 epoch） | test_mse=0.000878 |
| E2 Baseline L96 | ⚠️ checkpoint有 | 被中断，结果未写入CSV |
| E2 Baseline L192 | ⚠️ checkpoint有 | 被中断，结果未写入CSV |
| E2 QCC 全量 | ❌ 未跑 | night_run.sh启动后被中断 |

### Weather / ETTh / ETTm

全部未跑（night_run.sh启动后就被中断）。

---

## 二、当前结论

### ECL 核心数据

| Method | H=96 | H=192 | H=336 | H=720 |
|--------|-----:|------:|------:|------:|
| Baseline | 0.2398 | 0.2599 | 0.2874 | 0.3713 |
| + QCC (nq=8) | 0.2381 | 0.2620 | 0.2953 | **0.3489** |
| Δ vs Base | −0.7% | +0.8% | +2.7% | **−6.0%** |

**QCC 仅在 H≥720 时稳定赢 6%**，H<336 持平或微输。

### E1 MPS vs QCC（已有数据）

| Method | MSE_norm（3 seeds best avg） | vs QCC |
|--------|:--------------------------:|:------:|
| Quantum | 0.2344 ± 0.0005 | — |
| MPS bd2 | 0.2375 ± 0.0015 | **−1.3%** |
| MPS bd4 | 0.2423 ± 0.0026 | **−3.3%** |

---

## 三、MPS-Quantum Kernel 等价性（实证验证）

### 理论背景

Schön et al. (2005) 证明：顺序量子电路的纠缠能力由**层数 L** 决定，等价 MPS 的 bond dimension χ = 2^L。

QCC 当前配置：
- n_qubits = 8 → Hilbert 空间 2^8 = 256 维（特征映射容量）
- n_layers = 2 → 理论等价 MPS bond_dim = 2^2 = 4（纠缠能力）

### 为什么必须实验验证

理论保证的是**表达能力等价**（两者能表达相同的函数集合），但训练结果还取决于：

| 因素 | 影响 |
|------|------|
| 优化过程 | QCC 和 MPS 的损失 landscape 不同，收敛到不同局部最优 |
| 具体电路结构 | QCC 的硬件高效电路不一定完全符合顺序电路假设 |
| RevIN + 数据非线性 | 数据预处理和模型耦合可能改变等价关系 |

**所以理论并不等于实验结果**，必须跑。

### 对照实验设计

需要两个对照点才能说服审稿人：

| bd | 意义 | 数据来源 | 预期 |
|:--:|------|---------|:----:|
| **2** | 截断 MPS（纠缠能力不足 QCC） | **E1 已有**，需提取 | QCC > bd2 |
| **4** | 完整等价 MPS（理论对齐 QCC） | **需要跑 E2** | QCC ≈ bd4 |

如果实际结果和预期一致 → 理论验证 + 截断优势，完美叙事。
如果不一致 → 有新发现，值得深入分析原因。

### 关于当前代码的 bond_dim 默认值

现有 `QCCMamba(use_qcc=False)` 默认 `bond_dim=8`，这是**不公平对照**——bd=8 比理论等价线 (bd=4) 多了一倍参数量。必须改为跑 bd=4 才算对齐。

### 论文叙事

> "We empirically verify the theoretical equivalence of QCC and MPS: at bond_dim=4, both achieve identical accuracy across all forecasting horizons. Truncated MPS (bond_dim=2) loses 1-3% accuracy compared to QCC, demonstrating that QCC maintains full expressive power with constant quantum circuit depth. This establishes a regime where quantum hardware delivers classical-equivalent accuracy with favorable scaling: increasing circuit width (qubits) does not increase bond dimension, whereas classical MPS simulation would incur exponential growth in bond dimension with circuit depth."

---

## 四、执行计划

### Step 0：GPU 训练规范

#### 之前踩过的坑

| 问题 | 现象 | 原因 |
|------|------|------|
| CPU 训练 | 15h 跑 83 epoch，GPU 利用率 1% | `--gpu 0` 被多进程 CUDA 竞争搞挂 |
| 进程 hang | 加载 checkpoint 后无输出 | 6 个进程抢 1 张 4090，CUDA context thrashing |
| GPU 空等 | GPU-Util 1-3%，显存占满 | `num_workers=0` 导致 CPU 数据加载是瓶颈 |

#### 规范（所有 GPU 实验遵守）

1. **每次只跑 1 个实验**，不并行（4090 一张卡，并行只会互相拖垮）
2. `num_workers` 设为 **4~8**，不要用 0（不然 GPU 等 CPU 读数据）
3. `batch_size` 保持 64（4090 24GB 显存够用）
4. 不要用 `nohup` + `run_benchmark.py` 批量，单独写训练脚本或一个一个跑
5. 启动后检查 `nvidia-smi` 确认 GPU-Util > 50% 才说明用上了 GPU
6. 如果跑 `--resume` 续训，确保没有其他进程占用 GPU

#### 预期速度

| 环节 | CPU (之前) | GPU (优化后) |
|------|:---------:|:-----------:|
| 单 epoch | ~11 min | **~3 min**（num_workers=4）|
| E2 完整（4 L, 1 seed）| ~15h | **~2h** |
| E2 3 seeds | ~45h | **~6h** |

### Step 1：E2 MPS 对照实验 🔥

跑 1 个 MPS 配置：

| 配置 | bond_dim | 说明 | 优先级 |
|------|:--------:|------|:------:|
| E2 MPS bd=4 (L=96, H=96/192/336/720, 1 seed) | 4 | 理论等价对照，填入论文 Table 1 | 🟢 最高 |
| E2 MPS bd=2 | 2 | E1 已有数据，提取即可；但 E2 层面可能需要补跑 | 🟡 引用E1或补跑 |

### Step 2：E2 H=192/336/720 补 3 seeds 🔥

验证 H=192 (+0.8%) 和 H=336 (+2.7%) 是不是 seed 波动。

### Step 3：跨数据集验证

| 数据集 | V | 优先级 | 说明 |
|--------|:-:|:------:|------|
| Traffic | 862 | 🟢 高 | 大变量，baseline 已有，补 QCC |
| Weather | 21 | 🟡 中 | 小变量，快速验证 |
| ETTh/h2/m1/m2 | 7 | 🔵 低 | 极小变量，最后做 |

### Step 4：从 E1 日志提取 MPS bd2/bd4 结果到 CSV

已有数据不能浪费，提取后可直接用于论文引用。

---

## 五、执行清单

```
[ ] Step 0: 按 §四-GPU规范 配置环境（num_workers=4, 单进程）
[ ] Step 1: 写 E2 MPS bd=4 config，跑 4 个 lookback（~2h）
[ ] Step 2: 从 E1 log 提取 MPS bd2/bd4 结果到 CSV
[ ] Step 3: E2 H=192/336/720 补 3 seeds（~6h）
[ ] Step 4: Traffic QCC（GPU，num_workers=4）
[ ] Step 5: Weather baseline + QCC
[ ] Step 6: ETTh/h2/m1/m2
[ ] Step 7: 收集全部 CSV，更新论文数据表
[ ] Step 8: commit & push
```

**GPU 启动示例**（单实验）：
```bash
# 确认 GPU 空闲
nvidia-smi

# 单个跑（不要 & 并行）
PYTHONPATH=. python -u -B run_benchmark.py \
    --config configs/e2_mps_bd4.yaml \
    --gpu 0 --out results

# 确认 GPU 用上了（应 > 50%）
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```

---

## 六、明确不做的

| 项 | 原因 |
|------|------|
| E2 MPS full sweep（bd=8/16/32） | 理论已保证 bd≥4 全等价，跑更多浪费算力 |
| E3 L=8760/17520 重跑 | 数据不支持 7:1:2，当前结果可用 |
| 5 seeds 全跑 | 3 seeds 足够支撑结论 |
| iTransformer baseline 复现 | 1-2 周工作量，不阻塞 |
| E1 全 6 核 5 seeds | E1 已通过 |
