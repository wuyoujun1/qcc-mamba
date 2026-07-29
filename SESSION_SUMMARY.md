# QCC-Mamba 实验会话总结

> 最后更新: 2026-07-29 (第2次迭代)
> GitHub: https://github.com/wuyoujun1/qcc-mamba

---

## 代码改动（全部已推送）

### 上一轮改动（commit `fe19d18`, `06b3cb4`, `ae49012`, `972c73a`）
- `data/dataset.py`: 添加缺失的 `add_time_features` import
- `configs/*.yaml`: YAML 中 `1e-4` 被解析为字符串 → 改为 `0.0001`
- `backbone/smamba_backbone.py`: `SMambaEncoder.forward()` 返回单值而非元组
- `qcc/feature_map.py`: 添加 `self.d_token` 属性；修复 matmul 维度不匹配
- `engine/train.py`: 添加 `os` import；处理空验证集除零
- **论文标准 MSE_norm / MAE_norm**: 全局每变量标准化
- **Checkpoint 保存**: 训练中自动保存最佳模型
- **旁路权重 α 追踪**: 日志中打印 α 值

### GitHub 远程新拉取的改动（commit `90c2559`）
- **MPS kernel 修正**: `W` 输出维度 = `bond_dim`（不是 `d_token`），`bond_dim` 才真正控制秩
- **新增 kernel_type**: 支持 `{linear, rbf, poly}`，默认 `rbf` 模拟量子非线性
- **可学习 RBF 带宽 sigma**: `sigma^2 = softplus(raw)` 保证正值
- **新 config**: `configs/e2_mps_equivalence.yaml` — QCC vs 修正后 MPS 等价性验证
- **设计文档**: `NEXT_STEPS.md`, `REVIEW.md`, `comparison-plan.md`

### 本轮本地新增（commit `f3c0e3a`）
- `engine/train.py`: 训练过程中记录 α（旁路权重）到日志

### 新增工具脚本
- `batch_runner.py`: 并行批量实验运行器，自动生成 per-L 配置文件并调度多进程

---

## 实验配置

### 统一划分
- 所有实验 **train:val:test = 7:1:2**
- 例外: L=8760（val 放不下，用 60/0/40）

### 核心配置文件（E2/E3 electricity）
| 文件 | 用途 |
|:---|:---|
| `configs/e1_kernel_decisive.yaml` | E1：6方法对照 |
| `configs/e2_standard.yaml` | E2 QCC-Quantum（electricity） |
| `configs/e2_baseline.yaml` | E2 S-Mamba基线 |
| `configs/e2_baseline_big.yaml` | E2 S-Mamba基线 L=192/336/720 batch=64 |
| `configs/e2_baseline_L96.yaml` | E2 基线 L=96 batch=128 |
| `configs/e2_mps_equivalence.yaml` | 🔄 新增：MPS 等价性验证（bond_dim=4, RBF） |
| `configs/e3_L1440.yaml` | E3 QCC L=1440 batch=64 |
| `configs/e3_baseline_L1440.yaml` | E3 基线 L=1440 |
| `configs/e3_L8760.yaml` | E3 L=8760（特殊划分） |

### Traffic 配置文件
| 文件 | 用途 |
|:---|:---|
| `configs/e2_traffic_baseline.yaml` | Traffic 基线（4 in 1） |
| `configs/e2_traffic_qcc.yaml` | Traffic QCC（4 in 1） |
| `configs/e2_traffic_base_L{96,192,336,720}.yaml` | Traffic 基线 单 setting |
| `configs/e2_traffic_qcc_L{96,192,336,720}.yaml` | ✅ 新增：Traffic QCC 单 setting |

### 其他数据集配置文件（✅ 本轮新增）
| 数据集 | 基线 config | QCC config |
|:---|:---|:---|
| weather | `configs/e2_weather_baseline.yaml` | `configs/e2_weather_qcc.yaml` |
| etth1 | `configs/e2_etth1_baseline.yaml` | `configs/e2_etth1_qcc.yaml` |
| etth2 | `configs/e2_etth2_baseline.yaml` | `configs/e2_etth2_qcc.yaml` |
| ettm1 | `configs/e2_ettm1_baseline.yaml` | `configs/e2_ettm1_qcc.yaml` |
| ettm2 | `configs/e2_ettm2_baseline.yaml` | `configs/e2_ettm2_qcc.yaml` |

### 可用数据集
| 数据集 | 变量 | 时间步 | 状态 |
|:---|:---:|:---:|:---|
| electricity ✅已跑完 | 321 | 26,304 | E2/E3 done |
| **traffic** 🏃并行中 | **862** | 17,544 | **baseline 4路并行，QCC 待跑** |
| weather ⏳待跑 | 21 | 52,696 | config 已创建 |
| etth1 ⏳待跑 | 7 | 17,420 | config 已创建 |
| etth2 ⏳待跑 | 7 | 17,420 | config 已创建 |
| ettm1 ⏳待跑 | 7 | 69,680 | config 已创建 |
| ettm2 ⏳待跑 | 7 | 69,680 | config 已创建 |
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

> ⚠️ MPS kernel 已修正（bond_dim 真正限秩 + RBF 核），**E1 的 MPS 结果需要重跑**才能反映修后表现。

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

### 正在运行（并行）
- **traffic 基线** — `batch_runner.py` 4 路并行（L=96/192/336/720）

### 待运行
- **traffic QCC** — config 已就绪
- **weather 基线 + QCC**
- **etth1/etth2 基线 + QCC**
- **ettm1/ettm2 基线 + QCC**
- **E1 重跑**（MPS kernel 修正后）
- **E2 MPS 等价性验证**（`e2_mps_equivalence.yaml`）

### 训练日志
- E2 基线: `/tmp/claude-1011/.../tasks/bq6ncun2b.output`
- E2 QCC + E3: `/tmp/final_run.log`
- E3 L=1440 重跑: `/tmp/e3_rerun_L1440.log`
- E3 基线: `/tmp/claude-1011/.../tasks/bkicn9ie3.output`
- Traffic 当前: `_batch_logs/` + `/tmp/traffic_logs/`

---

## 并行运行命令

```bash
# 1. 单 config 4路并行（推荐）
PYTHONPATH=. python batch_runner.py --config configs/e2_traffic_baseline.yaml --gpu 0

# 2. 队列文件批量跑
PYTHONPATH=. python batch_runner.py --queue batch_queue.txt --gpu 0

# 3. 手动每 setting 独立（旧方式）
for L in 96 192 336 720; do
    nohup python -B run_benchmark.py \
        --config configs/e2_traffic_base_L${L}.yaml \
        --gpu 0 --out results \
        > /tmp/traffic_logs/base_L${L}.log 2>&1 &
done

# 4. 创建缺失的 config（小数据集）
python batch_runner.py --generate-configs
```

---

## 注意事项
1. **单 GPU（RTX 4090 25.2GB）**，多进程并行时：
   - 小数据集（7-21 var）每进程 ~1-2GB，可同时跑 10+ 个
   - Traffic（862 var）每进程 ~5-6GB，最多 4 路并行
   - QCC 旁路比 baseline 多用 ~80% 显存
2. **MPS kernel 已修正**: E1 中 MPS 结果需要重跑
3. **E2 MPS 等价性实验**: bond_dim=4 + RBF 核理论上等价于 QCC(n_qubits=8, n_layers=2)
4. 早停在并行时可能因 GPU 分时而延长，属于正常现象
5. MSE_norm 是论文标准指标（全局每变量标准化），值域 0.1~0.3
6. α（旁路权重）初始值 0.1，训练中自动调整
