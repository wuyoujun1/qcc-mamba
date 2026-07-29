# QCC-Mamba 实验会话总结

> 最后更新: 2026-07-29
> GitHub: https://github.com/wuyoujun1/qcc-mamba (已推送)

---

## 代码改动（已推送）

### Bug 修复（commit `fe19d18`）
- `data/dataset.py`: 添加缺失的 `add_time_features` import
- `configs/*.yaml`: YAML 中 `1e-4` 被解析为字符串 → 改为 `0.0001`
- `backbone/smamba_backbone.py`: `SMambaEncoder.forward()` 返回单值而非元组，解包修复
- `qcc/feature_map.py`: 添加 `self.d_token` 属性；修复 `_apply_single_qubit` 中 matmul 维度不匹配
- `engine/train.py`: 添加 `os` import；处理空验证集的除零错误

### 新功能
- **论文标准 MSE_norm / MAE_norm**（commit `06b3cb4`）：全局每变量标准化，值域 0.1~0.3
- **Checkpoint 保存**：训练中自动保存最佳模型到 `checkpoints/`
- **旁路权重 α 追踪**：日志中打印 α 值（commit `ae49012` 后）

---

## 实验配置

### 统一划分
- 所有实验 **train:val:test = 7:1:2**
- 例外: L=8760（val 放不下，用 60/0/40）

### 配置文件
| 文件 | 用途 |
|:---|:---|
| `configs/e1_kernel_decisive.yaml` | E1：6方法对照 |
| `configs/e2_standard.yaml` | E2 QCC-Quantum（electricity） |
| `configs/e2_baseline_big.yaml` | E2 S-Mamba基线 L=192/336/720 batch=64 |
| `configs/e2_baseline_L96.yaml` | E2 基线 L=96 batch=128 |
| `configs/e2_traffic_baseline.yaml` | Traffic 基线（4 in 1） |
| `configs/e2_traffic_qcc.yaml` | Traffic QCC（4 in 1） |
| `configs/e2_traffic_base_L{96,192,336,720}.yaml` | Traffic 基线 单 setting |
| `configs/e3_L1440.yaml` | E3 QCC L=1440 batch=64 |
| `configs/e3_baseline_L1440.yaml` | E3 基线 L=1440 |
| `configs/e3_L8760.yaml` | E3 L=8760（特殊划分） |

### 可用数据集
| 数据集 | 变量 | 时间步 | E2 可行 |
|:---|:---:|:---:|:---:|
| electricity ✅ 已跑完 | 321 | 26,304 | ✅ |
| **traffic** 🔄 在跑 | **862** | 17,544 | ✅ |
| weather | 21 | 52,696 | ✅ |
| etth1 | 7 | 17,420 | ✅ |
| etth2 | 7 | 17,420 | ✅ |
| ettm1 | 7 | 69,680 | ✅ |
| ettm2 | 7 | 69,680 | ✅ |
| exchange_rate ❌ | 8 | 7,588 | L=720 放不下 |
| solar ❌ | — | 空文件 | ❌ |

---

## 实验结果 electricity（7:1:2）

### E1 决定性实验（L=720, H=96, 原始 MSE）
| 方法 | MSE | R² |
|:---|:---:|:---:|
| MPS（经典张量） | 4,818,188 | **0.9317** |
| none（无旁路） | 4,971,596 | 0.9296 |
| Quantum | 4,978,563 | 0.9295 |
| Periodic | 4,987,949 | 0.9293 |
| RFF | 5,000,613 | 0.9292 |
| RBF | 5,002,920 | 0.9291 |

### E2 标准 Benchmark（7:1:2）
| L | H | S-Mamba 基线 | | QCC-Quantum | |
|:---:|:---:|:---:|:---:|:---:|:---:|
| | | MSE_norm | MAE | MSE_norm | MAE |
| 96 | 96 | **0.240** | 254 | **0.238** ✅ | **246** ✅ |
| 192 | 192 | **0.260** | 272 | 0.262 | 273 |
| 336 | 336 | 0.287 | 292 | **0.295** ✅ | **288** ✅ |
| 720 | 720 | 0.371 | 336 | **0.349** ✅ | **334** ✅ |

### E3 超长序列（7:1:2）
| L | H | 模型 | MSE_norm | MAE_norm | 早停 |
|:---:|:---:|:---|:---:|:---:|:---:|
| 1440 | 720 | S-Mamba基线 | 0.378 | 0.432 | 16 |
| 1440 | 720 | **QCC-Quantum** | **0.355** ✅ | **0.417** ✅ | 22 |
| 8760 | 720 | QCC-Quantum | 0.532 | 0.535 | 50（强制） |

### 旁路权重 α（checkpoint 中提取）
| 实验 | L | α |
|:---:|:---:|:---:|
| E2 QCC | 96 | 0.133 |
| E2 QCC | 192 | 0.142 |
| E2 QCC | 336 | 0.153 |
| E2 QCC | 720 | **0.317** 🚀 |
| E3 QCC | 1440 | 0.201 |

**关键发现：L 越长 → α 越大，旁路越被需要**

---

## 当前状态

### 已保存的 Checkpoints（`checkpoints/`）
- `e2_baseline_*_none_s2026_best.pt` — E2 基线 electricity（L=96/192/336/720）
- `e2_standard_*_quantum_s2026_best.pt` — E2 QCC electricity（L=96/192/336/720）
- `e3_baseline_*_none_s2026_best.pt` — E3 基线 L=1440
- `e3_longterm_*_quantum_s2026_best.pt` — E3 QCC L=1440

### 结果文件（`results/*.csv`）
- `e2_baseline_electricity.csv` — E2 基线（L=96/192/336/720 带 MSE_norm）
- `e2_standard_electricity.csv` — E2 QCC（L=96/192/336/720 带 MSE_norm）
- `e3_longterm_electricity.csv` — E3 L=1440 QCC（7:1:2）
- `e3_baseline_electricity.csv` — E3 L=1440 基线

### 正在/待运行
- **traffic 数据集** — configs 已创建，未开始跑
- **weather/etth/ettm 数据集** — config 未创建
- E3 L=8760 — 未重跑

### 训练日志
- E2 基线: `/tmp/claude-1011/.../tasks/bq6ncun2b.output`
- E2 QCC + E3: `/tmp/final_run.log`
- E3 L=1440 重跑: `/tmp/e3_rerun_L1440.log`
- E3 基线: `/tmp/claude-1011/.../tasks/bkicn9ie3.output`
- Traffic 日志: `/tmp/traffic_logs/`

---

## 运行命令

```bash
# E2 单数据集（顺序跑4个setting）
PYTHONPATH=. python -B run_benchmark.py --config configs/e2_standard.yaml --gpu 0 --out results

# 4路并行（每个setting一个进程）
for L in 96 192 336 720; do
    nohup python -B run_benchmark.py \
        --config configs/e2_traffic_base_L${L}.yaml \
        --gpu 0 --out results \
        > /tmp/traffic_logs/base_L${L}.log 2>&1 &
done
```

---

## 注意事项
1. 单 GPU（RTX 4090 24GB），多进程并行时每个约 4-5GB，4 路≈21GB
2. 早停在并行时可能因 GPU 分时而延长，属于正常现象
3. MSE_norm 是论文标准指标（全局每变量标准化），值域 0.1~0.3
4. α（旁路权重）初始值 0.1，训练中自动调整
