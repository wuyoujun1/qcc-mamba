# 2026-08-16 状态快照 + 下一步计划（API 过期前留存）

> 本文件是会话中断前的完整状态记录：复现成果、量子移植、公平测试结论、服务器问题、P2-1 计划。
> 若新会话需要接手，先读本文件 + `ins.md`。

---

## 1. 已完成：官方 S-Mamba 完整复现 ✅（最大成果）

**位置**：`/home/wuyoujun/ts_quantum/S-D-Mamba`（官方仓库 wzhwzhwzh0921/S-D-Mamba，arXiv 2403.11144 官方代码，已双重验证）

**复现数字 vs 论文表（arXiv 2403.11144 v3，28 格几乎全部 ±2.5% 内）：**

| 数据集 | 96 | 192 | 336 | 720 |
|---|---|---|---|---|
| ETTh1 | 0.3878 | 0.4449 | 0.4902 | 0.5064 |
| ETTh2 | 0.2971 | 0.3784 | 0.4254 | 0.4321 |
| ETTm1 | 0.3313 | 0.3779 | 0.4096 | 0.4743 |
| ETTm2 | 0.1817 | 0.2524 | 0.3126 | 0.4134 |
| Weather | 0.1649 | 0.2154 | 0.2731 | 0.3531 |
| ECL | 0.1387 | 0.1610 | 0.1804 | 0.2037 |
| Exchange | 0.0863 | 0.1825 | 0.3305 | 0.8581 |
| Traffic | 0.3924 | 0.4154 | ⚠️未跑完 | ⚠️未跑完 |

- Traffic 336/720 因服务器风暴中断，**需补跑**（约 30 分钟，负载正常时）
- 结果文件：`/home/wuyoujun/ts_quantum/S-D-Mamba/result_long_term_forecast.txt`（官方格式：setting + mse/mae）
- 复现用官方脚本原样 + 唯一兼容补丁：`utils/tools.py` 的 `np.Inf → np.inf`（NumPy 2.0）
- 环境：`/home/wuyoujun/qcc-mamba/.venv/bin/python`（venv 内已装 sklearn/reformer-pytorch，用 `uv pip install --python ...` 装的；**注意 `pip` 是 miniconda 的，别用错**）

**指标口径（已三重验证与论文一致）**：全局 z-score 空间（TSL `--inverse` 默认 False + S-Mamba `use_norm=True` 内部 denorm 回全局 z 空间 + 我们的 mse_norm 同空间）。测试窗 = 官方 30 天月划分 `[11424, 14400)`，4 个月。

---

## 2. 已完成：量子核移植进官方仓库（git 分支 qmix-port，commit 8c80405）

**位置**：`/home/wuyoujun/ts_quantum/S-D-Mamba`（分支 `qmix-port`，官方 main 保持纯净）

**新增文件**：
- `qcc/`：kernel.py / classical_kernels.py / feature_map.py / spectrum.py / quantum_mix.py + `__init__.py`（从 qcc-mamba 逐字拷贝，5 个模块，仅依赖 torch）
- `model/Q_S_Mamba.py`：官方 Model 子类 + 量子混合（qmix_layers=0 时与官方**逐位一致**，state_dict 可互换，已离线验证 torch.equal）
- `scripts/.../ETT/Q_S_Mamba_ETTh1.sh` / `ETTh2.sh`（stage1：qmix=0）+ `_vd.sh`（stage2：vd 风格参数）

**修改**：
- `run.py`：新增量子参数（--qmix_layers/--n_qubits/--qmix_n_layers/--qmix_norm/--kernel_T/--offdiag/--topk/--entangle_topo/--kernel_fn/--angle_norm/--theta_S_scale0/--qmix_gate/--qmix_gate_init/--spectrum_M/--spectrum_time_align/--spectrum_freq_align/--spectrum_range/--spectrum_amp_normalize/--delay_in_s/--qmix_use_H/--qmix_fixed_s_scale）
- `experiments/exp_basic.py`：注册 Q_S_Mamba

**阶段 1 验证已通过**：Q_S_Mamba(qmix=0) 8 档 MSE 与官方 15 位有效数字一致（ETTh1/ETTh2 全档）。

---

## 3. 公平测试结果：add-on 式量子核三连败（重要结论）

**协议**：官方脚本原样（seq_len 96, d_model 256, d_state 2, lr 7e-5, epochs 10, seed 2023），数据集 ETTh1:96，对照官方 0.3878。

| 变体 | 机制 | 结果 | 诊断 |
|---|---|---|---|
| vd（K 读 H, δ̂ 入 S） | H 投影可学习 | 0.4108 (+5.9%) | pre_ln.weight 塌缩到 0.0016 → 路径被优化器关闭 |
| P1-1（K 只读 S） | S 调制可学习 | 0.3921 (+1.1%) | s_ln.weight 塌缩到 0.007 → 又一条关闭通道 |
| QS2（fixed_s_scale + 50ep） | S 固定尺度不可压制 | 0.4174 (+7.6%) | 强制开启 → 反而更伤 |

**核心规律**：只要量子路径存在可学习的"关闭通道"，优化器一定会压塌它（pre_ln / s_ln）；强制开启则有害。→ **量子核作为官方 S-Mamba 的附加混合层 = 冗余**（官方双向 Mamba 已做跨变量混合）。

**注**：50-epoch 变体也测过（vd@50ep = 0.4114，平台化无改善）—— 不是训练不足。

**旧仓库结论作废声明**：qcc-mamba 仓库此前所有"vd 赢 plain"的结论基于有缺陷的基线（d_model 512/lookback=horizon/50ep/RevIN/8 个月测试集 + split_by_months 的 test bug），不可作为论文依据。

---

## 4. 服务器问题（未解决，必须处理）

- **症状**：进程风暴，load 从 400 涨到 1459+ 持续膨胀
- **根源**：用户 `wangshuolei` 的 VS Code Server（`/home/wangshuolei/.vscode-server/`）循环执行 `ps -F -A -l | grep root` 类命令（Shell Integration 进程树轮询），负载高时 ps 变慢 → 轮询堆叠 → 正反馈
- **处理**：root 执行 `pkill -9 -f vscode-server`，或联系 wangshuolei 关 VS Code（设置 `terminal.integrated.shellIntegration.enabled: false` 防复发），或重启
- **教训**：8 个用户共用的机器上，避免在风暴期间跑重训练；训练与 VS Code 用户错峰
- 复现数据全部落盘，无丢失

---

## 5. 下一步计划：P2-1 双路径重构（量子核独占跨变量）

### 为什么
三连败诊断指向唯一剩余结构选项：**量子核必须取代主干的跨变量混合**。官方 S-Mamba 的双向 Mamba 在变量 token 上做混合 —— 量子核与它冗余。去掉主干的跨变量混合后，量子核成为唯一通道，优化器无法关闭它。

### 架构（ins.md P2-1，官方仓库内实现）
```
输入 x → 实例 norm → 频域双轴对齐 S（我们已有）
  ├→ 时间路径：主干单向化（去掉 token 维双向 Mamba 的跨变量混合，
  │             只保留时间建模 = per-token/通道独立处理 + FFN）
  ├→ 变量路径：量子核独占跨变量（K = 保真度核，输入对齐频谱 S）
  └→ 可学习加权融合（α·时间 + β·量子核）→ 投影头 → denorm
```

### 实现要点（官方仓库）
1. `model/Q_S_Mamba.py` 或新文件：改造 encoder —— EncoderLayer 去掉 `attention(x) + attention_r(x.flip(dims=[1])).flip(dims=[1])` 的跨 token 混合（Mamba 只做 per-variable 时间处理，或者把 token 维当 batch 维），保留 FFN/残差/LN
2. 量子变量路径：`H_var = K_n @ H @ W_q`，K 来自 S（proj_S 端到端，S detach）—— 复用现有 qcc
3. 融合：`H = α·H_time + β·H_var`（或 concat + 线性）
4. 参数：新增 `--dp_mode` 类 flag（时间路径模式 / 融合权重）
5. **对照基线 = 官方 S-Mamba**（双向混合）→ 论文叙事："量子核 vs Mamba 谁做跨变量混合更好"

### 筛选协议（沿用 ins.md，不能改）
- 筛选：单 seed（官方 run.py 硬编码 2023）+ L=96 + etth1 先砍人，门槛 **≥3% 优于官方**（etth1:96 需 ≤0.376）
- 过了再铺 weather/electricity；确认 3 seed + ins.md 五条判据
- 若 P2-1 也败 → 需要和用户重新讨论方向（不要再自行继续）

### 可选并行
- P0-3：rbf 经典核对照（论文必需，定位实验）—— 只需 `--kernel_fn rbf`

---

## 6. 接手速查

```bash
# 官方仓库 + 量子移植（分支）
cd /home/wuyoujun/ts_quantum/S-D-Mamba && git checkout qmix-port

# 跑官方脚本（串行，一个脚本跑完再下一个；脚本间 sleep 60）
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH
bash scripts/multivariate_forecasting/ETT/S_Mamba_ETTh1.sh > log 2>&1

# 跑量子筛选（vd 风格，ETTh1:96）
bash scripts/multivariate_forecasting/ETT/Q_S_Mamba_ETTh1_vd.sh   # 或用 /tmp/qs2.sh 类单档脚本

# 看结果
grep "mse:" repro_S_Mamba_ETTh1.log   # 或 result_long_term_forecast.txt
```

**重要提醒**：
- 训练必须**串行**（一个 python run.py 在跑时不要启动第二个）
- 修改 vd 脚本追加参数时，**注意 bash 续行反斜杠**（`--delay_in_s` 行后接参数必须补 `\`，空行会终结命令 —— 已踩过两次坑）
- 生成新脚本后用 `sed 's|python -u run.py|echo PYTHON_CALL|' script.sh | bash` 验证参数链
- 服务器风暴处理前，别跑重训练
