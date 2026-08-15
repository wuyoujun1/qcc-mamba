# DualAE-QCC

时频双阶段对齐量子编码（Dual-stage Aligned Encoding with Quantum Kernel）+ S-Mamba 主干，面向多变量时间序列预测。

---

## 1. 项目定位

- **方法**：在 S-Mamba 主干外并联 DualAE-QCC 旁路，首层用 backbone 语义特征 H 编码"变量身份"，重上传层用对齐后频谱特征 S 做"调制"，通过量子核捕获变量间非线性相似度。
- **核心创新**：三层频谱对齐（时间轴/频率轴/幅度）+ 双阶段量子编码 + L 无关的固定采样。
- **目标场景**：多变量时间序列预测，支持 12 个标准数据集。

详细设计见 [`IDEA_DualAE_QCC.md`](./IDEA_DualAE_QCC.md)。

---

## 2. 目录结构

```
qcc_mamba/
├── backbone/          # S-Mamba 主干接口 + MockBackbone（快速验证）
├── qcc/               # 量子核 feature map / kernel / spectrum / QCCBlock
├── model/             # 端到端 QCCMamba（DualAE-QCC 架构）
├── data/              # Dataset / DataLoader / RevIN / 周期特征
├── engine/            # 训练、评估
├── configs/           # 实验配置文件（dual_*.yaml）
├── tests/             # 单元测试（test_smoke.py）
├── run_dual_ae.py     # DualAE-QCC 统一运行入口
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

# 安装 mamba-ssm（需要 Linux + CUDA）
uv pip install causal-conv1d mamba-ssm
```

> 本地不跑训练，代码写好直接上传服务器执行。

---

## 4. 快速开始

### 4.1 冒烟测试（服务器上）

```bash
PYTHONPATH=. python tests/test_smoke.py
```

验证：
- 频谱模块输出维度 (B, V, 2M)
- 双阶段编码维度流转：H(B,V,512) → ψ(B,V,1024)
- 核矩阵维度 (B, V, V)，对角线 ≈ 1
- 最终预测维度 (B, H, V)
- 梯度流正确

### 4.2 运行 DualAE-QCC 实验

```bash
# 主线实验（H + S 双阶段）
PYTHONPATH=. python run_dual_ae.py --config configs/dual_phase1.yaml --gpu 0

# 消融实验
PYTHONPATH=. python run_dual_ae.py --config configs/dual_ablation_h_only.yaml --gpu 0
PYTHONPATH=. python run_dual_ae.py --config configs/dual_ablation_s_only.yaml --gpu 0
PYTHONPATH=. python run_dual_ae.py --config configs/dual_ablation_no_align.yaml --gpu 0
```

### 4.3 配置说明

`configs/dual_phase1.yaml` 中关键字段：

- `dataset`: 数据集名（electricity / chinaaqi / metr_la / pems_bay / ili / etth1 / ...）
- `lookback` / `horizon`: 输入窗口和预测步长
- `model.use_H` / `model.use_S`: 编码来源消融开关
- `model.spectrum_time_align` / `model.spectrum_freq_align`: 对齐层消融开关
- `model.spectrum_M`: 频谱采样点数（默认 32）
- `training.use_amp`: 是否使用混合精度训练
- `training.accumulation_steps`: 梯度累积步数

### 4.4 量子混合主干实验（2026-08 起主线）

```bash
# 变体矩阵（qdir/qoff 家族 23 个变体 × 数据集 × 4 档，变体列表见 run_qmix.py VARIANTS）
PYTHONPATH=. python run_qmix.py --variant qoff_n2_v --dataset etth1 --lookback 96 --horizon 42 --seed 42

# 结果汇总（读 logs/qmix/，打印全变体矩阵）
PYTHONPATH=. python summarize_qmix.py
```

---

## 5. 关键代码入口

| 文件 | 作用 |
|------|------|
| `qcc/spectrum.py` | 频谱特征提取（rfft → 三层对齐 → 重采样） |
| `qcc/feature_map.py` | 双阶段量子编码 `EntanglingFeatureMap` |
| `qcc/kernel.py` | 量子核 `quantum_kernel` |
| `qcc/qcc_block.py` | QCCBlock：旁路融合结构（含可学习 γ） |
| `model/qcc_mamba.py` | QCCMamba 端到端模型 |
| `data/dataset.py` | Dataset（支持 12 个数据集，含 ILI 周频特判） |
| `data/dataloader.py` | DataLoader 工厂（统一接口） |
| `engine/train.py` | 训练循环（支持 AMP + 梯度累积） |
| `engine/evaluate.py` | MSE / MAE + 配对 t 检验 |
| `run_dual_ae.py` | DualAE-QCC 统一运行入口 |

---

## 6. 支持的数据集

| 数据集 | 变量数 | 频率 | 划分方式 |
|--------|--------|------|----------|
| Electricity (ECL) | 321 | 小时 | 7:1:2 |
| Traffic | 862 | 小时 | 7:1:2 |
| Weather | 21 | 10分钟 | 7:1:2 |
| Solar | 137 | 10分钟 | 7:1:2 |
| Exchange | 8 | 日 | 7:1:2 |
| ChinaAQI | 342 | 小时 | 7:1:2 |
| METR_LA | 207 | 5分钟 | 7:1:2 |
| PEMS_BAY | 325 | 5分钟 | 7:1:2 |
| ETTh1/h2 | 7 | 小时 | 12mo:4mo:4mo |
| ETTm1/m2 | 7 | 15分钟 | 12mo:4mo:4mo |
| ILI | 317 | 周 | 7:1:2 |

数据集默认从 `qcc_mamba/../ts_quantum/datasets/` 读取，CSV 格式（第一列时间戳）。

### 6.1 数据集下载（不入库）

```bash
mkdir -p ~/ts_quantum/datasets && cd ~/ts_quantum/datasets
```

| 数据集 | 来源 | 文件名 |
|--------|------|--------|
| ETTh1/h2, ETTm1/m2, Weather, Electricity, Traffic, Exchange, ILI | [TSLib 官方 dataset 目录](https://github.com/thuml/Time-Series-Library)（README 内 Google Drive 链接） | ETTh1.csv 等（列=变量，行=时间戳） |
| METR_LA / PEMS_BAY | TSLib 同目录 .h5 → 用 `data/dataset.py::load_h5_to_csv()` 转换 | metr_la.csv / pems_bay.csv |
| ChinaAQI | TSLib 数据集链接（部分仓库拆分提供） | china_aqi.csv |

> 服务器现有 `_archive/` 为旧格式备份，不用管。详细实验记录见 [`EXPERIMENT_MAINLINE_20260815.md`](./EXPERIMENT_MAINLINE_20260815.md)。

---

## 7. 注意事项

1. **backbone 使用 S-Mamba 官方实现**：封装为 `SMambaBackbone`（`backbone/smamba_backbone.py`）。**需要服务器安装 mamba-ssm**：`uv pip install mamba-ssm[causal-conv1d]`。
2. **MockBackbone** 保留在 `backbone/interface.py`，可在本地快速验证 DualAE-QCC 模块（不依赖 mamba-ssm）。
3. **d_model 已更新为 512**：符合 TSL 标准，旧实验结果（d=128）作废。
4. **N=10 qubits**：特征空间维度 2^10=1024，旧实验结果（N=8）作废。
5. **METR_LA/PEMS_BAY 需要转换**：原始 .h5 文件需用 `data/dataset.py` 中的 `load_h5_to_csv()` 转换为统一 CSV 格式。

---

## 8. 引用

- Schuld, M. "Supervised quantum machine learning models are kernel methods." arXiv:2101.11020 (2021).
- Åsgrim, et al. "Quantum kernels are spectral tensor networks." arXiv:2606.20402 (2026).
- Shen, et al. "Quantum vs classical kernels for financial time series." arXiv:2607.20168 (2026).

---

## 9. 修改记录

详见 [`CHANGELOG.md`](./CHANGELOG.md)。
