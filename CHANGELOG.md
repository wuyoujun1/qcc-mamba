# DualAE-QCC 重构修改清单

> 日期：2026-08-07  
> 版本：v3.0 (QCC-Mamba → DualAE-QCC)

---

## 一、架构升级概述

从 **QCC-Mamba**（单阶段量子编码）升级到 **DualAE-QCC**（时频双阶段对齐量子编码）：

| 维度 | 旧 QCC-Mamba | 新 DualAE-QCC |
|------|-------------|---------------|
| 编码来源 | 只用 H（backbone 特征） | 首层 H（身份）+ 重上传 S（频谱调制） |
| 频谱模块 | 无 | rfft → 三层对齐 → M=32 采样 → S (B,V,64) |
| qubit 数 | N=8（256 维） | N=10（1024 维，2×512） |
| d_model | 128（不合规） | 512（TSL 标准） |
| L 依赖性 | 绑定 L | L 无关（M=32 固定采样） |

---

## 二、新增文件

### 核心模块
- **`qcc/spectrum.py`** - 频谱特征提取模块（rfft → 三层对齐 → 重采样）
  - 时间轴对齐：δ̂_v = 互相关(vs 共识时钟) → φ̃ = φ − 2πf·δ̂_v
  - 频率轴对齐：f̂_peak = argmax A → f̃ = f/f̂_peak → 重采样 M 点
  - 幅度归一化（消融档）：Ã = A/A_max
  - 全 detach，无梯度

### 配置文件
- **`configs/dual_phase1.yaml`** - 主线配置（H + S 双阶段）
- **`configs/dual_ablation_h_only.yaml`** - 仅 H 消融
- **`configs/dual_ablation_s_only.yaml`** - 仅 S 消融
- **`configs/dual_ablation_no_align.yaml`** - 无对齐消融
- **`configs/dual_ablation_time_only.yaml`** - 仅时间轴对齐
- **`configs/dual_ablation_freq_only.yaml`** - 仅频率轴对齐

### 运行脚本
- **`run_dual_ae.py`** - DualAE-QCC 统一运行入口

### 测试
- **`tests/test_smoke.py`** - 冒烟测试（验证维度流转、核矩阵对角线≈1、梯度流）

---

## 三、修改文件

### 核心模块

#### `qcc/feature_map.py`（大改）
- N 默认 8 → **10**（2^10=1024 维）
- 新增 `proj_H(d_token → 2N)` + `proj_S(2M → 2N)` 双投影（Xavier 初始化）
- `forward(h, s)` 接收第二个参数 `s`（频谱特征）
- 首层用 H 角度，重上传用 S 角度
- 新增 `use_H` / `use_S` 消融开关

#### `qcc/qcc_block.py`（中改）
- 新增 `S` 参数（可选）
- 新增可学习 γ（`theta_S_scale`，init=0.5, clamp [0.1, 2]）
- S 路 LayerNorm → proj_S → γ 缩放
- 新增 `use_H` / `use_S` 消融开关

#### `model/qcc_mamba.py`（中改）
- `d_token` 默认 128 → **512**
- `n_qubits` 默认 8 → **10**
- 集成 `SpectrumFeature`
- `forward` 中计算 S 并传给 QCCBlock
- 新增频谱模块参数（`spectrum_M`, `spectrum_range`, `spectrum_amp_normalize`, `spectrum_time_align`, `spectrum_freq_align`）

#### `backbone/smamba_backbone.py`（小改）
- `d_model` 默认 128 → **512**

#### `backbone/interface.py`（小改）
- `MockBackbone` 的 `d_model` 默认 128 → **512**，`d_token` 默认 64 → **256**

### 数据模块

#### `data/dataset.py`（中改）
- 新增 `ILIDataset`（周频，L=H=24/36/48/60 特殊配置）
- 新增 `load_h5_to_csv()`（METR_LA/PEMS_BAY .h5 → CSV 转换）
- `ETDataset` 默认使用 `12mo:4mo:4mo` 月份划分（TSL 标准）
- `SplitConfig` 支持月份模式（`use_months=True`）

#### `data/dataloader.py`（中改）
- `DATASET_FILES` 新增 `chinaaqi / metr_la / pems_bay / ili`
- 新增 `build_standard_loaders()` 统一接口（自动识别数据集类型和频率）
- ETT 系列自动使用 `12mo:4mo:4mo` split

### 训练引擎

#### `engine/train.py`（小改）
- `train_one_epoch()` 支持 AMP（混合精度）+ 梯度累积（`accumulation_steps`）
- `fit()` 支持 `use_amp` + 记录 γ（`theta_S_scale`）
- 新增 `build_optimizer()`（投影层权重衰减豁免）

---

## 四、删除文件

### 旧脚本
- `run_e1.py` - 旧 E1 实验脚本
- `run_benchmark.py` - 旧 E2/E3 实验脚本
- `batch_runner.py` - 批量运行脚本
- `resume_traffic.py` - Traffic 续训脚本
- `run_night.sh` - 夜间运行脚本
- `run_experiments.sh` - 实验运行脚本
- `run_overnight.sh` - 通宵运行脚本
- `run_overnight_final.sh` - 最终通宵运行脚本

### 旧配置
- `configs/e1_*.yaml` - 旧 E1 配置（2 个）
- `configs/e2_*.yaml` - 旧 E2 配置（28 个）
- `configs/e3_*.yaml` - 旧 E3 配置（5 个）

### 旧模块
- `baselines/` - 整个目录（包含 `mps_mamba.py` 占位实现）
- `hardware/` - 整个目录（包含 `ibmq_verify.py` 占位实现）

### 旧测试
- `tests/test_qcc_basic.py` - 旧单元测试（已被 `test_smoke.py` 替代）

### 旧文档
- `SESSION_SUMMARY.md` - 会话总结
- `NIGHT_PLAN.md` - 夜间计划
- `NEXT_STEPS.md` - 下一步计划
- `REVIEW.md` - 审查文档
- `comparison-plan.md` - 对比计划

### 临时文件
- `test_output.txt` - 测试输出
- `batch_queue.txt` - 批量队列

### 旧工具
- `engine/spectrum.py` - 旧核矩阵频谱分析工具（与新 `qcc/spectrum.py` 重名易混淆）

---

## 五、保留文件

### 核心文档
- `IDEA_DualAE_QCC.md` - 最终设计文档（v3.1）
- `PROJECT_IDEA.md` - 项目总体说明
- `README.md` - 项目说明
- `DATASET_SPEC_9.md` - 12 数据集 E2 规格
- `EXPERIMENT_PREMISE_VERIFICATION.md` - 前提验证 v3 结果
- `DATASET_CANDIDATES_LARGE.md` - 新数据集下载渠道
- `TSL_STANDARDS.md` - TSL 划分标准
- `deploy/README.md` - 服务器部署指南（需更新）

### S-Mamba 官方实现
- `s_mamba_official/` - 整个目录保留（backbone 参考实现）

---

## 六、关键技术决策

1. **δ̂ 估计**：样本内留一均值（per-sample），与前提验证口径统一
2. **S 路梯度**：S 全 detach，Linear_S 有梯度
3. **γ 参数**：主线可学习（init=0.5, clamp [0.1, 2]），消融固定值
4. **ETT 划分**：默认 12mo:4mo:4mo（TSL 标准）
5. **METR_LA/PEMS_BAY**：.h5 → CSV 转换（取 data[:,:,0] 速度通道）

---

## 七、下一步

1. **运行冒烟测试**：`python tests/test_smoke.py` 验证维度流转
2. **准备数据集**：确保服务器上 12 个数据集已转换为统一 CSV 格式
3. **运行主线实验**：`python run_dual_ae.py --config configs/dual_phase1.yaml`
4. **运行消融实验**：依次运行 6 个消融配置
5. **更新 deploy/README.md**：添加新架构说明和使用方法

---

## 八、兼容性说明

- **旧实验结果作废**：d=128→512、N=8→10、编码架构改为双阶段后，旧参数下的一切实验（含 E1/E2 数字）不再作为任何基线
- **仅保留一条 motivation 结论**："量子旁路在旧架构中验证过有效性"
- **本文所有配置**（仅H/仅S/双阶段、N=10、d=512）**全部用新代码重新跑**

---

*最后更新：2026-08-07*
