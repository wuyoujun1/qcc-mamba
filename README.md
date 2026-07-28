# QCC-Mamba

量子核跨变量相关（Quantum Cross-Correlation, QCC）旁路 + S-Mamba 主干，面向电力超长序列预测。

---

## 1. 项目定位

- **方法**：在 S-Mamba 主干外并联一个 QCC 旁路，用量子核捕获变量间非线性相似度。
- **核心实验**：E1 决定性实验——同一模型只换核，六组对照证明量子核 > 经典核。
- **目标场景**：电力负荷 / 电价 / 新能源出力预测，L ∈ {720, 1440, 8760, 17520}。

详细实验设计见 [`paperidea/experiment-design.md`](./paperidea/experiment-design.md)。

---

## 2. 目录结构

```
qcc_mamba/
├── backbone/          # S-Mamba 主干接口 + MockBackbone（快速验证）
├── qcc/               # 量子核 feature map / kernel / classical kernels / QCCBlock
├── model/             # 端到端 QCCMamba
├── data/              # Dataset / DataLoader / RevIN / 周期特征
├── engine/            # 训练、评估、Fourier 频谱分析
├── configs/           # 实验配置文件
├── tests/             # 单元测试
├── run_e1.py          # E1 一键运行脚本
├── requirements.txt   # 服务器依赖
└── README.md          # 本文件
```

---

## 3. 服务器环境安装（推荐 uv）

```bash
# 1. 进入项目目录
cd qcc_mamba

# 2. 用 uv 安装依赖（服务器推荐）
uv pip install -r requirements.txt

# 如果服务器有 CUDA，先根据 CUDA 版本换 PyTorch index：
# CUDA 12.1
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
# CUDA 11.8
uv pip install torch --index-url https://download.pytorch.org/whl/cu118
```

> 本地不跑训练，代码写好直接上传服务器执行。

---

## 4. 快速开始

### 4.1 单元测试（服务器上）

```bash
PYTHONPATH=. python tests/test_qcc_basic.py
```

### 4.2 E1 决定性实验（最高优先级）

```bash
# 单卡跑全部 6 组方法，3 seeds
PYTHONPATH=. python run_e1.py --config configs/e1_kernel_decisive.yaml --gpu 0

# 只跑量子核和 RFF 快速摸底
PYTHONPATH=. python run_e1.py --config configs/e1_kernel_decisive.yaml --gpu 0 --methods quantum rff
```

### 4.3 配置说明

`configs/e1_kernel_decisive.yaml` 中关键字段：

- `dataset`: 数据集名（electricity / traffic / weather / solar / etth1 / ...）
- `lookback` / `horizon`: 输入窗口和预测步长
- `methods`: 六组对照方法（quantum / rbf / periodic / rff / mps / none）
- `training.epochs` / `patience`: 训练轮数和早停耐心
- `model.use_periodic_feat`: 是否拼接 hour/day sin-cos 周期特征

---

## 5. 关键代码入口

| 文件 | 作用 |
|------|------|
| `qcc/feature_map.py` | 纠缠数据编码 `EntanglingFeatureMap` |
| `qcc/kernel.py` | 量子核 `quantum_kernel` |
| `qcc/classical_kernels.py` | RBF / Periodic / RFF / no_bypass 经典核工厂 |
| `qcc/mps_kernel.py` | MPSBypass（最强经典张量对手） |
| `qcc/qcc_block.py` | QCCBlock：旁路融合结构 |
| `model/qcc_mamba.py` | QCCMamba 端到端模型 |
| `engine/train.py` | 训练循环 + 早停 |
| `engine/evaluate.py` | MSE / MAE / CRPS + 配对 t 检验 |
| `engine/spectrum.py` | Fourier 频谱分析（E7） |
| `run_e1.py` | E1 一键运行 |

---

## 6. 实验优先级（3 天冲刺版）

| 天数 | 实验 | 目的 | 决策点 |
|------|------|------|--------|
| Day 1 下午 | **E1** 六组核对照 | 量子核 vs 经典核生死战 | 赢 → Day 2；输 → 切 E0a 诊断 |
| Day 2 | E2 标准 benchmark + E3 超长序列 | 不退化 + 主战场 SOTA | 验证长序列优势 |
| Day 3 | E4/E5/E6 消融 + E7 频谱 | 归因量子核贡献 | 整理内部报告 |

完整决策树见 [`paperidea/experiment-design.md`](./paperidea/experiment-design.md) §二。

---

## 7. 数据集准备

默认从 `qcc_mamba/../ts_quantum/datasets/` 读取。请在服务器上放置：

```
ts_quantum/datasets/
├── electricity.csv
├── traffic.csv
├── weather.csv
├── solar.csv
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
└── exchange_rate.csv
```

CSV 格式：第一列为时间戳，其余列为变量。

---

## 8. 注意事项

1. **backbone 使用 S-Mamba 官方实现**：已从 GitHub `sci-m-wang/S-D-Mamba` 适配，封装为 `SMambaBackbone`（`backbone/smamba_backbone.py`），与 `BackboneOutput(H, y_main)` 接口对齐。**需要服务器安装 mamba-ssm**：`uv pip install mamba-ssm[causal-conv1d]`。
2. **MockBackbone** 保留在 `backbone/interface.py`，可在本地快速验证 QCC 模块（不依赖 mamba-ssm），但 E1/E2/E3 请使用 SMambaBackbone。
2. **真机附录不参与训练**：`hardware/` 目录预留给 IBM Q / Qiskit 验证，仅用于小 scale（N=2-3）的保真度一致性验证。
3. **决定性实验必须做**：没有 E1 中量子核 > RFF/RBF/periodic 的证据，不要投大规模实验。

---

## 9. 引用

- Schuld, M. "Supervised quantum machine learning models are kernel methods." arXiv:2101.11020 (2021).
- Åsgrim, et al. "Quantum kernels are spectral tensor networks." arXiv:2606.20402 (2026).
- Shen, et al. "Quantum vs classical kernels for financial time series." arXiv:2607.20168 (2026).
