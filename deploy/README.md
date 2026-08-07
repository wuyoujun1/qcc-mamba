# DualAE-QCC 服务器执行指南

> 本文档指导在 Linux GPU 服务器上配置环境、下载数据、运行 DualAE-QCC 实验。

---

## 1. 获取代码

### 方式 A：Git 克隆（推荐）
```bash
git clone <仓库URL> qcc_mamba
cd qcc_mamba
```

### 方式 B：SCP 上传
```bash
# 本地打包
tar czf qcc_mamba.tar.gz qcc_mamba/ --exclude='.git' --exclude='__pycache__' --exclude='*.pyc'

# 上传
scp qcc_mamba.tar.gz user@server:/path/to/workspace/

# 服务器解压
cd /path/to/workspace
tar xzf qcc_mamba.tar.gz
cd qcc_mamba
```

---

## 2. 环境配置

### 2.1 检查 CUDA
```bash
nvidia-smi | grep "CUDA Version"
```

### 2.2 创建虚拟环境
```bash
# 使用 uv（如未安装：curl -LsSf https://astral.sh/uv/install.sh | sh）
uv venv --python 3.11
source .venv/bin/activate

# 安装 PyTorch（根据 CUDA 版本选择，示例 CUDA 12.1）
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装 mamba-ssm（需要 CUDA 工具链，耗时 2-5 分钟）
uv pip install causal-conv1d mamba-ssm

# 安装其余依赖
uv pip install -r requirements.txt
```

### 2.3 验证安装
```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "from mamba_ssm import Mamba; print('mamba-ssm OK')"
python -c "from qcc import QCCBlock, SpectrumFeature; print('QCC modules OK')"
```

---

## 3. 下载数据集

```bash
cd deploy

# 下载全部标准数据集（electricity, etth1/h2, ettm1/m2, traffic, weather, solar, exchange）
python download_datasets.py --dir ../datasets

# 或只下载指定数据集
python download_datasets.py --dir ../datasets --datasets electricity etth1 etth2
```

**数据集目录结构**：
```
qcc_mamba/../ts_quantum/datasets/
├── electricity.csv
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── traffic.csv
├── weather.csv
├── solar.csv
└── exchange_rate.csv
```

**新增数据集**（chinaaqi / metr_la / pems_bay / ili）需手动放置到同一目录，格式要求：
- 第一列时间戳（`date` / `datetime` / `timestamp`）
- 其余列为变量数值
- CSV 格式

---

## 4. 冒烟测试

```bash
cd qcc_mamba
python tests/test_smoke.py
```

**预期输出**：
```
============================================================
DualAE-QCC Smoke Tests
============================================================

============================================================
Test 1: Spectrum Module
============================================================
Input: x = torch.Size([2, 96, 10])
Output: S = torch.Size([2, 10, 64])
Expected: (2, 10, 64)
✓ Spectrum module output dimension correct
✓ S is detached (no gradient)

============================================================
Test 2: Dual-Stage Feature Map
============================================================
Input: H = torch.Size([2, 10, 512]), S = torch.Size([2, 10, 64])
Output: ψ = torch.Size([2, 10, 1024])
Expected: (2, 10, 1024)
✓ Feature map output dimension correct
✓ Quantum states are normalized

============================================================
Test 3: QCC Block
============================================================
Input: H = torch.Size([2, 10, 512]), y_main = torch.Size([2, 96, 10]), S = torch.Size([2, 10, 64])
Output: y = torch.Size([2, 96, 10]), K = torch.Size([2, 10, 10]), correction = torch.Size([2, 96, 10])
✓ QCC block output dimensions correct
✓ K matrix diagonal ≈ 1
✓ K matrix is symmetric
✓ γ parameter is valid

============================================================
Test 4: Full Model (QCCMamba)
============================================================
Input: x = torch.Size([2, 96, 10])
Output: y = torch.Size([2, 96, 10]), y_main = torch.Size([2, 96, 10]), K = torch.Size([2, 10, 10])
✓ Full model output dimensions correct
✓ Gradient flows correctly

============================================================
Test 5: Ablation Modes
============================================================
Testing H-only mode (use_H=True, use_S=False)...
✓ H-only mode works
Testing S-only mode (use_H=False, use_S=True)...
✓ S-only mode works
Testing no-align mode (time_align=False, freq_align=False)...
✓ No-align mode works

============================================================
✅ All smoke tests passed!
============================================================
```

---

## 5. 运行实验

### 5.1 Phase 1：完整架构（H + S 双阶段）

```bash
# 单卡运行
python run_dual_ae.py --config configs/dual_phase1.yaml

# 指定 GPU
CUDA_VISIBLE_DEVICES=0 python run_dual_ae.py --config configs/dual_phase1.yaml
```

**输出目录**：`results/dual_phase1/`
- `dual_phase1_best.pt`：最佳模型 checkpoint
- `dual_phase1_history.npy`：训练历史（loss、metrics、α、γ）

**训练日志示例**：
```
Epoch 1/100  train_loss=0.123456  val_mse=0.234567  test_mse=0.345678  MSE_norm=0.123456  MAE_norm=0.234567  α=0.1000  γ=0.5000
  ✅ 保存最佳模型: results/dual_phase1/dual_phase1_best.pt (epoch 1, val_mse=0.2346)
...
Early stopping at epoch 15
```

### 5.2 消融实验

#### 消融 1：仅 H 编码（关闭 S 路）
```bash
python run_dual_ae.py --config configs/dual_ablation_h_only.yaml
```
**目的**：验证 H 路（backbone 语义特征）的独立贡献。

#### 消融 2：仅 S 编码（关闭 H 路）
```bash
python run_dual_ae.py --config configs/dual_ablation_s_only.yaml
```
**目的**：验证 S 路（频谱特征）的独立贡献。

#### 消融 3：无对齐（关闭时间轴 + 频率轴对齐）
```bash
python run_dual_ae.py --config configs/dual_ablation_no_align.yaml
```
**目的**：验证三层对齐（时间轴、频率轴、幅度归一化）的整体贡献。

#### 消融 4：仅时间轴对齐
```bash
python run_dual_ae.py --config configs/dual_ablation_time_only.yaml
```
**目的**：验证时间轴对齐的独立贡献。

#### 消融 5：仅频率轴对齐
```bash
python run_dual_ae.py --config configs/dual_ablation_freq_only.yaml
```
**目的**：验证频率轴对齐的独立贡献。

### 5.3 批量运行脚本

**串行运行所有实验**：
```bash
#!/bin/bash
# run_all_experiments.sh

CONFIGS=(
    "dual_phase1"
    "dual_ablation_h_only"
    "dual_ablation_s_only"
    "dual_ablation_no_align"
    "dual_ablation_time_only"
    "dual_ablation_freq_only"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Running: $cfg"
    echo "=========================================="
    python run_dual_ae.py --config "configs/${cfg}.yaml"
    if [ $? -ne 0 ]; then
        echo "ERROR: $cfg failed"
        exit 1
    fi
done

echo "All experiments completed!"
```

**并行运行**（多卡场景）：
```bash
# Terminal 1: Phase 1 完整架构
CUDA_VISIBLE_DEVICES=0 python run_dual_ae.py --config configs/dual_phase1.yaml

# Terminal 2: H-only 消融
CUDA_VISIBLE_DEVICES=1 python run_dual_ae.py --config configs/dual_ablation_h_only.yaml

# Terminal 3: S-only 消融
CUDA_VISIBLE_DEVICES=2 python run_dual_ae.py --config configs/dual_ablation_s_only.yaml

# ... 依此类推
```

---

## 6. 结果分析

### 6.1 查看训练历史
```python
import numpy as np
import matplotlib.pyplot as plt

# 加载训练历史
history = np.load('results/dual_phase1/dual_phase1_history.npy', allow_pickle=True).item()

# 绘制损失曲线
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(history['train_loss'], label='Train Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(history['val_mse'], label='Val MSE')
axes[0, 1].plot(history['test_mse'], label='Test MSE')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('MSE')
axes[0, 1].set_title('MSE Curves')
axes[0, 1].legend()
axes[0, 1].grid(True)

axes[1, 0].plot(history['val_mse_norm'], label='Val MSE (norm)')
axes[1, 0].plot(history['test_mse_norm'], label='Test MSE (norm)')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('MSE (normalized)')
axes[1, 0].set_title('Normalized MSE Curves')
axes[1, 0].legend()
axes[1, 0].grid(True)

axes[1, 1].plot(history['alpha'], label='α')
axes[1, 1].plot(history['gamma'], label='γ')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Value')
axes[1, 1].set_title('Learnable Parameters')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('results/dual_phase1/training_curves.png', dpi=150)
print("Saved: results/dual_phase1/training_curves.png")
```

### 6.2 对比消融结果

```python
import os
import numpy as np

EXPERIMENTS = {
    'Phase 1 (Full)': 'results/dual_phase1/dual_phase1_history.npy',
    'H-only': 'results/dual_ablation_h_only/dual_h_only_history.npy',
    'S-only': 'results/dual_ablation_s_only/dual_s_only_history.npy',
    'No Align': 'results/dual_ablation_no_align/dual_no_align_history.npy',
    'Time Only': 'results/dual_ablation_time_only/dual_time_only_history.npy',
    'Freq Only': 'results/dual_ablation_freq_only/dual_freq_only_history.npy',
}

print(f"{'Experiment':<20} {'Val MSE':<12} {'Test MSE':<12} {'MSE_norm':<12} {'MAE_norm':<12}")
print("-" * 70)

for name, path in EXPERIMENTS.items():
    if os.path.exists(path):
        history = np.load(path, allow_pickle=True).item()
        val_mse = min(history['val_mse'])
        test_mse = history['test_mse'][history['val_mse'].index(val_mse)]
        mse_norm = history['test_mse_norm'][history['val_mse'].index(val_mse)]
        mae_norm = history['test_mae_norm'][history['val_mse'].index(val_mse)]
        print(f"{name:<20} {val_mse:<12.6f} {test_mse:<12.6f} {mse_norm:<12.6f} {mae_norm:<12.6f}")
    else:
        print(f"{name:<20} {'NOT FOUND':<12}")
```

### 6.3 结果判据

| 对比 | 预期结果 | 结论 |
|------|----------|------|
| Phase 1 vs H-only | Phase 1 显著更优 | S 路（频谱特征）有效 |
| Phase 1 vs S-only | Phase 1 显著更优 | H 路（语义特征）有效 |
| Phase 1 vs No Align | Phase 1 显著更优 | 三层对齐有效 |
| No Align vs Time Only | Time Only 更优 | 时间轴对齐贡献更大 |
| No Align vs Freq Only | Freq Only 更优 | 频率轴对齐贡献更大 |

**统计显著性**：使用 `engine.evaluate.paired_t_test` 进行配对 t 检验（p < 0.05 视为显著）。

---

## 7. 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ImportError: No module named mamba_ssm` | `uv pip install causal-conv1d mamba-ssm`，需要 Linux + CUDA 工具链 |
| `CUDA out of memory` | 降低 `batch_size`（如 16 → 8），或增加 `accumulation_steps`（如 1 → 2） |
| `FileNotFoundError: electricity.csv` | 先运行 `python deploy/download_datasets.py --dir ../datasets` |
| `test_smoke.py 失败` | 检查 PyTorch 版本（≥2.0）、mamba-ssm 是否正确安装 |
| `mse_norm 始终为 inf` | 检查 `run_dual_ae.py` 是否正确调用 `set_global_stats`（已修复） |
| `use_S=True but use_spectrum=False` 警告 | 正常防护机制，自动降级为 H-only 模式 |

---

## 8. 实验配置说明

### 8.1 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_token` | 512 | Backbone token 维度（也是 QCC 旁路维度） |
| `n_qubits` | 10 | 量子比特数 N（特征空间 2^N = 1024） |
| `n_layers` | 2 | 数据重上传层数 D |
| `spectrum_M` | 32 | 频谱采样点数（S 维度 = 2M = 64） |
| `alpha0` | 0.1 | 旁路融合强度 α 初始值（可学习） |
| `theta_S_scale0` | 0.5 | S 路调制强度 γ 初始值（可学习，clamp [0.1, 2]） |
| `beta` | 0.1 | 辅助损失权重 |

### 8.2 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `batch_size` | 32 | 批大小（大 V 数据集建议 8-16） |
| `epochs` | 100 | 最大训练轮数 |
| `patience` | 10 | Early stopping 耐心值 |
| `lr` | 1e-4 | 学习率 |
| `use_amp` | true | 混合精度训练（节省显存） |
| `accumulation_steps` | 1 | 梯度累积步数（等效 batch_size = batch_size × accumulation_steps） |

### 8.3 消融开关

| 配置 | use_H | use_S | spectrum_time_align | spectrum_freq_align | 说明 |
|------|-------|-------|---------------------|---------------------|------|
| Phase 1 | ✓ | ✓ | ✓ | ✓ | 完整架构 |
| H-only | ✓ | ✗ | - | - | 仅 backbone 语义特征 |
| S-only | ✗ | ✓ | ✓ | ✓ | 仅频谱特征 |
| No Align | ✓ | ✓ | ✗ | ✗ | 关闭所有对齐 |
| Time Only | ✓ | ✓ | ✓ | ✗ | 仅时间轴对齐 |
| Freq Only | ✓ | ✓ | ✗ | ✓ | 仅频率轴对齐 |

---

## 9. 目录结构

```
qcc_mamba/
├── configs/                    # 实验配置
│   ├── dual_phase1.yaml       # Phase 1 完整架构
│   ├── dual_ablation_h_only.yaml
│   ├── dual_ablation_s_only.yaml
│   ├── dual_ablation_no_align.yaml
│   ├── dual_ablation_time_only.yaml
│   └── dual_ablation_freq_only.yaml
├── data/                       # 数据加载
│   ├── dataloader.py          # DataLoader 工厂
│   ├── dataset.py             # Dataset 实现
│   └── preprocess.py          # RevIN + 时间特征
├── backbone/                   # Backbone 实现
│   ├── interface.py           # BaseBackbone + MockBackbone
│   ├── smamba_backbone.py     # S-Mamba 官方实现
│   ├── smamba_embed.py        # DataEmbeddingInverted
│   └── smamba_encdec.py       # SMambaEncoder
├── qcc/                        # 量子核 QCC 旁路
│   ├── spectrum.py            # 频谱特征提取（三层对齐）
│   ├── feature_map.py         # 时频双阶段量子编码
│   ├── kernel.py              # 量子核（保真度）
│   ├── classical_kernels.py   # 经典核对照（RBF/Periodic/RFF）
│   ├── message_passing.py     # GAT 风格消息传递
│   ├── qcc_block.py           # QCCBlock
│   └── mps_kernel.py          # MPS 张量网络旁路
├── model/
│   └── qcc_mamba.py           # 端到端模型
├── engine/
│   ├── train.py               # 训练循环
│   └── evaluate.py            # 评估指标 + 统计检验
├── tests/
│   └── test_smoke.py          # 冒烟测试
├── deploy/
│   ├── README.md              # 本文档
│   └── download_datasets.py   # 数据集下载脚本
├── run_dual_ae.py              # 统一运行入口
└── requirements.txt
```

---

## 10. 快速开始清单

```bash
# 1. 克隆代码
git clone <仓库URL> && cd qcc_mamba

# 2. 配置环境
uv venv --python 3.11 && source .venv/bin/activate
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install causal-conv1d mamba-ssm
uv pip install -r requirements.txt

# 3. 下载数据
cd deploy && python download_datasets.py --dir ../datasets && cd ..

# 4. 冒烟测试
python tests/test_smoke.py

# 5. 运行 Phase 1
python run_dual_ae.py --config configs/dual_phase1.yaml

# 6. 运行消融实验
for cfg in dual_ablation_h_only dual_ablation_s_only dual_ablation_no_align dual_ablation_time_only dual_ablation_freq_only; do
    python run_dual_ae.py --config configs/${cfg}.yaml
done

# 7. 查看结果
python -c "import numpy as np; h = np.load('results/dual_phase1/dual_phase1_history.npy', allow_pickle=True).item(); print(f'Best val MSE: {min(h[\"val_mse\"]):.6f}')"
```

---

## 11. 联系与支持

- **代码问题**：检查 `tests/test_smoke.py` 是否通过
- **数据问题**：确认 `datasets/` 目录下有对应 CSV 文件
- **显存不足**：降低 `batch_size` 或增加 `accumulation_steps`
- **安装失败**：确认 CUDA 工具链可用（`nvcc --version`）

---

**最后更新**：2026-08-07  
**对应架构**：DualAE-QCC（时频双阶段对齐量子编码）
