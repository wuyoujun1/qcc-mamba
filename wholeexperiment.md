# EXPERIMENT_RUN_36H：36 小时无人值守批量实验方案

> 目标：在服务器上连续运行约 36 小时，一口气跑完 DualAE-QCC 的关键实验，
> 覆盖：冒烟 → P1 定方向 → P2 生死线 → P3 消融 → E2 主档位。
> 配套文件：`run_experiment_batch.py`（批处理驱动）+ `summarize_results.py`（汇总/画图）。
> 最后更新：2026-08-07

---

## 0. 一句话说明

`run_experiment_batch.py` 内置了 4 个批次的实验矩阵，自动完成：生成临时 yaml → 调用
`run_dual_ae.py` → 超时控制 → 失败继续 → 日志 → 指标解析进 `results/summary.csv` →
断点续跑（done 标记）。36 小时后回来直接看 `EXPERIMENT_SUMMARY.md` + `hetero_gain.png`。

---

## 0.5 修订说明（2026-08-08 审查后修正，实际执行以此为准）

| 项 | 方案原文 | 修正 | 原因 |
|---|---|---|---|
| P1 baseline | `use_fmap=false, use_spectrum=false, use_H=false, use_S=false` | `use_bypass=false`（真·无旁路） | 四个开关关掉后 QCCBlock 仍会跑量子核旁路，不是"无旁路"；`use_bypass=false` 才真正禁用旁路（qcc=None） |
| 配套文件 | `run_experiment_batch.py` / `summarize_results.py` 标"✅ 本目录" | 已补写 | 审查时两文件不存在 |
| AMP | 各 config 默认 `use_amp=true` | 全矩阵 `use_amp=false` | AMP fp16 偶发 NaN（实测复现），关掉后稳定 |
| epoch 评估 | 每 epoch 评 train+val+test | `eval_test_every_epoch=false`（test 留结尾） | 省 ~25% 时间，最终 test 指标不变 |
| 训练档位 | configs 默认 epochs=100/patience=10 | 筛选档 epochs=50/patience=8 | 保证 36h 内 P1/P2/P3 跑完；同档对比公平 |
| 数据集名 | `ecl` / `pems_bay` | loader 用 `electricity` / `pems_bay`（驱动内置别名） | loader 注册名不符 |
| 重上传消融 | needs_code=True，跳过 | 代码已实现（reupload_source 开关 + 2 个 yaml），本次仍按方案跳过 | 可随时排入 |
| **数据加载死锁** | （原方案未涉及） | 全矩阵 `num_workers=0` | `pin_memory=True + num_workers=4 + persistent_workers=False` 时 DataLoader 在 CUDA 初始化后才 fork worker（每 epoch 边界重fork），实测 QCC 作业在 **epoch 8 处死锁**（main=futex_wait，workers=do_poll 空等，GPU 0%）。`num_workers=0` 消除 fork，加载开销 ~几秒/epoch 可忽略 |

**时间预算现实核算（2026-08-08 实测更新）**：QCC 作业 patience=8 早停于 ~10–11 epochs
（ECL dual 实测 25–28min/作业，baseline ~3min/作业）。
phase1 ≈ 2.8h，phase2 ≈ 3.7h，phase3 ≈ 2.7h，phase4（长 L）≈ 8h，**合计 ≈ 17h < 36h**。
剩余 ~18h 缓冲可投入 §2 缓冲项：phase3 补 seed、ChinaAQI/PEMS_BAY×720、E2 no_align 长档、重跑 failed。

---

## 1. 前置检查（跑之前必须确认，缺一项就可能白跑）

| # | 检查项 | 命令/说明 |
|---|---|---|
| 1 | **8 处 bug 修复已同步** | `engine/__init__.py`（无 spectrum 引用）、`run_dual_ae.py`（evaluate import）、`qcc/feature_map.py`（matmul 方向）、`qcc/qcc_block.py`（kernel_fn 字符串解析）、`model/qcc_mamba.py`（use_S 防护）、`tests/test_smoke.py`（MockBackbone）——**如果服务器代码是旧的，先 pull/同步** |
| 2 | **单测通过** | `python tests/test_smoke.py` 5 个测试全绿 |
| 3 | **6 个数据集 CSV 就绪** | `../ts_quantum/datasets/` 下：`electricity.csv / chinaaqi.csv / metr_la.csv / pems_bay.csv / ETTh1.csv / weather.csv`（`data/dataloader.py` 的 `DATASET_FILES` 已包含全部） |
| 4 | **mamba-ssm 可用** | `python -c "import mamba_ssm"` 无报错（Linux+CUDA） |
| 5 | **GPU 显存** | 24 GB 足够（全模型约 8–12 GB），无需分块 |
| 6 | **校准计时** | 跑一次 phase0 冒烟，看单实验实际耗时，用 `--calib` 修正时间预算 |

---

## 2. 时间预算与批次规划（默认校准系数 1.0）

| 批次 | 内容 | 实验数 | 预计耗时 |
|---|---|---|---|
| phase0_smoke | 冒烟（ECL×96×epochs=1） | 1 | ~10 min |
| phase1 | P1 定方向：ECL×96 编码消融 4 配置 × 2 seed | 8 | ~5.5 h |
| phase2 | P2 生死线：6 数据集 × {dual, no_align} × L=96 | 12 | ~6 h |
| phase3 | P3：对齐消融 2 + 参数消融 N/M 4（ECL×96） | 6 | ~4 h |
| phase4 | E2 主档位：长 L（336/720）双阶段 | 7 | ~6.5 h |
| **合计** | | **34** | **~22 h（剩 14 h 缓冲）** |

缓冲用途（优先级从高到低）：
1. phase3 补 1 个 seed（对齐消融 2 seed 平均更稳）；
2. phase4 补 ChinaAQI/PEMS_BAY × 720；
3. 重跑 failed/timeout 项；
4. 若中途某批次全绿且时间充裕，把 E2 的 no_align 长档补上（异质度图第二个 L 点）。

---

## 3. 实验矩阵全表（34 个实验）

### phase0_smoke（冒烟，1 个）
| id | ds | L | seed | 验证 |
|---|---|---|---|---|
| smoke_ecl_96_dual | ecl | 96 | 42 | epochs=1 不崩、K 对角≈1、梯度正常、指标可解析 |

### phase1：P1 定方向（ECL × L=96 × 2 seed，8 个）
| id | 配置 | 模型 overrides | 验证什么 |
|---|---|---|---|
| p1_ecl_baseline_{42,2024} | 纯 S-Mamba（无旁路） | use_fmap=false, use_spectrum=false, use_H=false, use_S=false | 旁路整体贡献 |
| p1_ecl_h_{42,2024} | 仅 H | use_spectrum=false, use_H=true, use_S=false | 语义路单独贡献 |
| p1_ecl_s_{42,2024} | 仅 S | use_spectrum=true, use_H=false, use_S=true | 频谱路单独贡献（预期最弱） |
| p1_ecl_dual_{42,2024} | 双阶段（完整） | 默认（=phase1 基座） | 融合 > 单域 |

### phase2：P2 生死线（6 数据集 × 2 配置 × L=96 × seed42，12 个）
| id | ds | 配置 | overrides |
|---|---|---|---|
| p2_{ds}_dual_42 | ecl / chinaaqi / metr_la / pems_bay / etth1 / weather | 双阶段 | 默认 |
| p2_{ds}_no_align_42 | 同上 | 无对齐 | spectrum_time_align=false, spectrum_freq_align=false |

> 产出：`hetero_gain.png`——x=异质强度（δ̂ std % 主导周期），y=ΔMSE（no_align − dual）。
> **预期单调上升**（异质越强、对齐越值钱）。这是 idea 存亡的核心证据。

### phase3：P3 对齐 + 参数消融（ECL × L=96 × seed42，6 个）
| id | 消融项 | overrides | 验证什么 |
|---|---|---|---|
| p3_ecl_time_only_42 | 仅时间轴对齐 | spectrum_time_align=true, spectrum_freq_align=false | 时间轴对齐独立贡献 |
| p3_ecl_freq_only_42 | 仅频率轴对齐 | spectrum_time_align=false, spectrum_freq_align=true | 频率轴对齐独立贡献 |
| p3_ecl_N8_42 | qubit=8 | n_qubits=8 | 量子空间大小 |
| p3_ecl_N12_42 | qubit=12 | n_qubits=12 | 同上 |
| p3_ecl_M16_42 | M=16 | spectrum_M=16 | 采样密度 |
| p3_ecl_M48_42 | M=48 | spectrum_M=48 | 同上 |

> 说明：no_align（ECL）已在 phase2 跑过，这里不重复；双轴 = p1_ecl_dual_42，直接引用。
> 幅度归一化 / 采样区间 [0,1] / γ 固定档：**等 phase1-3 结果出来后按需补跑**，避免一次性摊薄预算。

### phase4：E2 主档位（6 数据集 × 长 L × 双阶段 × seed42，7 个）
| id | ds | L=H | 说明 |
|---|---|---|---|
| p4_ecl_336_42 / p4_ecl_720_42 | ecl | 336 / 720 | 核心主战场全档 |
| p4_chinaaqi_336_42 | chinaaqi | 336 | 大变量主战场 |
| p4_pemsbay_336_42 | pems_bay | 336 | 大变量主战场 |
| p4_metrla_336_42 | metr_la | 336 | 大变量主战场 |
| p4_weather_720_42 | weather | 720 | 小变量长档 |
| p4_etth1_336_42 | etth1 | 336 | 少变量参照 |

---

## 4. 新增文件清单

| 文件 | 作用 | 状态 |
|---|---|---|
| `run_experiment_batch.py` | 批处理驱动：生成 yaml → 调 run_dual_ae.py → 日志/超时/断点续跑 → summary.csv | ✅ 本目录 |
| `summarize_results.py` | 读 summary.csv → EXPERIMENT_SUMMARY.md + hetero_gain.png | ✅ 本目录 |
| `configs/generated/*.yaml` | 每实验自动生成的临时配置（可删，脚本会重建） | 自动 |
| `logs/batch/*.log` | 每个实验的完整 stdout | 自动 |
| `results/.done/*.done` | 完成标记（断点续跑依据） | 自动 |
| `results/summary.csv` | 全部实验指标汇总 | 自动 |

**运行方式（服务器上，项目根目录）**：

```bash
# 第 0 步：单测 + 冒烟
python tests/test_smoke.py
python run_experiment_batch.py --only phase0_smoke

# 第 1 步：校准（看冒烟实际耗时，估算 34 个实验总时长）
# 若冒烟用了 3 分钟而预算 10 分钟，说明实际比预算快 → --calib 0.5 可缩短超时；
# 若明显更慢 → --calib 1.5 防止误杀长任务

# 第 2 步：全量跑（nohup 挂后台，断线不中断）
nohup python run_experiment_batch.py --calib 1.0 > logs/batch_runner.log 2>&1 &

# 中途掉电/中断后的恢复（自动跳过已完成项）
python run_experiment_batch.py --resume

# 第 3 步：汇总（可随时跑，报告反映当前进度）
python summarize_results.py
```

---

## 5. 判据（36h 后怎么读结果）

### 5.1 P1（ECL×96，2 seed 均值）
```
双阶段 ≤ 仅H ≤ 原版（baseline）    → 旁路有效，方向对
双阶段 < 仅S                        → 融合优于纯频域
仅S 明显差                          → 正常（频谱无语义），不代表方法失败
双阶段 ≈ 仅H                        → ⚠️ S 路没起作用，查 γ / 投影 / 编码
```

### 5.2 P2（生死线，最重要）
```
ΔMSE = MSE(no_align) − MSE(dual) 随异质强度单调上升 → ✅ idea 成立
所有数据集 ΔMSE ≈ 0               → ⚠️ 对齐无增益，回头查频谱模块实现
强异质数据集 ΔMSE < 0（反而变差）  → 🔴 严重：对齐在帮倒忙，先查 bug 再谈论文
```

### 5.3 P3（对齐/参数消融）
```
双轴 ≤ 仅时间轴 ≈ 仅频率轴 ≤ 无对齐  → 每层对齐都有独立贡献
N=10 附近 MSE 最低                   → qubit 数选择合理
M=32 附近最优                        → 采样密度选择合理
```

### 5.4 指标口径
- 主看 **MSE / MAE**；跨数据集比较看 **mse_norm / mae_norm**（已修 `set_global_stats`，前提是 run_dual_ae.py 里能取到 train data）；
- 每个实验日志里有 K 矩阵相关信息可提取（可解释性素材）。

---

## 6. 容错与恢复说明

1. **失败继续**：每个实验独立 try，崩了只记 `failed` 到 summary.csv，继续下一个；
2. **超时保护**：每实验超时 = 数据集预算 × 3 × calib（宽松上限），超时记 `timeout`，不会卡死整批；
3. **断点续跑**：`results/.done/{id}.done` 存在即跳过；`--resume` 后重跑只会补没完成的；
4. **日志完备**：每实验完整 stdout 在 `logs/batch/{id}.log`，失败项可以直接翻日志定位；
5. **中途查看进度**：随时 `python summarize_results.py`，报告反映当前已完成部分；
6. **GPU 监控**（可选，服务器侧）：`nvidia-smi` 每 30 分钟看一次，防止异常占用。

---

## 7. 36h 后需要带回的东西（回传清单）

1. `results/summary.csv`（全部实验指标）；
2. `EXPERIMENT_SUMMARY.md`（自动报告）；
3. `hetero_gain.png`（P2 生死线散点图）；
4. `logs/batch/` 中 failed/timeout 项的日志（用于排查）；
5. 一句话结论：P1 方向对不对、P2 散点图单不单调、哪些实验要重跑/补 seed。

---

## 8. 已知限制与后续补跑项（不阻塞本次 36h）

| 项 | 状态 | 说明 |
|---|---|---|
| 重上传消融（H/S/交替） | needs_code=True，本次跳过 | 需在 feature_map 加 `reupload_source` 开关，实现后改矩阵标记重跑 |
| γ 固定档（0.1/0.5/1.0） | needs_code=True，本次跳过 | 需在 qcc_block 加 `gamma_mode` 开关 |
| 幅度归一化消融 | 未排入 | spectrum_amp_normalize=true/false，等 P1 结果后补 |
| 采样区间 [0,1] vs [0,2] | 未排入 | spectrum_range="0_1"，等 P3 结果后补 |
| 新数据集异质度回填 | 待服务器填 | ChinaAQI/METR_LA/PEMS_BAY 的 δ̂ std % 填进 `summarize_results.py` 的 `HETEROGENEITY` 后散点图自动补全 |
| E2 对照组（Traffic/Exchange/ETTm） | 后续补全 | 12 数据集中其余 6 个，等初步 6 个结果稳定后跑 |
