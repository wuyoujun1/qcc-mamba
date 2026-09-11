# QCCK-M: Quantum Cross-Variable Coupling Kernel for Multivariate Time-Series Forecasting

在 S-Mamba 主干上插入量子跨变量耦合模块（QCCM / QCCK-M）的长时序预测代码与论文材料。
唯一主线分支：**main**（旧的 `paper-sync` / `qmix-port` 已废弃）。

## 目录
| 路径 | 说明 |
|---|---|
| `qcc/` | 量子核与混合层（`quantum_mix.py`=QCCM；`kernel.py`/`feature_map.py` 量子核） |
| `model/` | `Q_S_Mamba.py`（QCCK-M）、`S_Mamba.py`（主干基线） |
| `layers/` `utils/` `data_provider/` `experiments/` | 主干、数据、训练框架 |
| `run.py` | 训练/测试入口（`--is_training 1` 训练，0 测试） |
| `scripts/` | 上游 S-Mamba 官方实验脚本（增加回看长度/换编码器等） |
| `dataset/` | 9 个数据集（ETTh1/2、ETTm1/2、Weather、ECL、Traffic、Exchange、Beijing-AQI，gz 压缩存放） |
| `paper/` | 论文材料：`ch3/` `ch4/` `ch5/`（正文 PDF、生成器、全部图、记录）、`outline/`（大纲/风格/批注） |

## 环境
**新服务器从零搭建请看 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md)（~15 分钟，含全部已知坑）。**

```bash
# 本机环境在 /tmp 易失；重建/校验见 技能 rebuild-qcc-env（推荐）
# 或手工：
pip install -r requirements.txt          # 含 mamba_ssm / causal_conv1d / torch
export OMP_NUM_THREADS=8                 # 本机必须，否则线程爆炸
```

## ⚠️ 代码口径差异（读复现之前必看）
**本仓库的代码只是参考快照，不是跑出论文结果的那一份，两者不一致。** 论文结果由工作副本 `/home/youjun/dataops_ws/` 跑出。后果：**照本仓库 clone 下来，跑不出论文的任何一格。**

仓库 `main` 相对工作副本缺这些功能：

| 缺失项 | 工作副本有 | 本仓库 |
|---|---|---|
| `--qmix_norm raw_k`（**论文最新理论**：去对角原始保真度核 K 直接做消息传播，不做 softmax / 温度 / 行归一化） | ✅ 已实现 | ❌ 未实现，传了直接抛 `ValueError`（本仓库只认 `avg/softmax/l1`） |
| `--qmix_norm rawk_row`（去对角后行和归一化） | ✅ | ❌ |
| `--qmix_msg H\|S\|HS`（消息来源，杠杆1） | ✅ | ❌ run.py 无此参数 |
| `--qmix_ln_hp`（Hp 是否做 LN，杠杆2） | ✅ | ❌ run.py 无此参数 |
| `intervention_mask`（推理期按比例屏蔽耦合 → 第五章**表 7**） | ✅ | ❌ |

涉及文件：`qcc/quantum_mix.py`、`run.py`、`model/Q_S_Mamba.py`、`data_provider/data_loader.py`（后者是 pandas 新旧 API 写法差异）。

另外，**论文正文的若干描述与代码（两份都是）对不上**，写/改正文时要留意：

| 正文写法 | 代码实际 |
|---|---|
| 编码态经**两层相同的 U**演化，`\|ψ_v⟩ = U_v·U_v·\|χ_v⟩`（两层各含一次 CNOT） | 首层只做旋转、不纠缠；纠缠层数为 `D−1`。即 `D=2` 时只有 **1 个** CNOT 层，且两层角度来源不同（首层取 H、重上传取 S） |
| 振幅通道 `p(h_v) = softmax(W_r·h_v) ∈ ℝ^32`（H→概率振幅、S→基态相位） | **没有 softmax 编码**。`proj_H`/`proj_S` 都是 `Linear(·→2N)`，输出直接当**每比特的 (θ,φ) 旋转角**，首层构造乘积态。不存在"H→32 维概率 / S→32 个自由相位"的两通道分工 |
| 投影矩阵与线路旋转角**对每个变量独立可学习** | `proj_H`/`proj_S`/`W_q` 都是全变量**共享**的单个 `nn.Linear`；变量间差异只来自输入 |
| `H′ = LN(H + γ·LN(H_p))` | gate 开时是 `H + γ·LN(H_p)`，**没有外层 LN** |
| 去对角与温度缩放 `K ← (K−I)/T` 后 `softmax(K)` | offdiag 时把对角 logit 直接置为 `row.min()−1e6`，使**对角权重严格为 0**（2026-09-07 改） |
| 3.2 主频 `argmax_{f≥1}` 只忽略直流 | 实际跳过 k=0 **和 k=1**（防去趋势后残余低频伪峰） |
| `S_v` 维度恒为 2M | 带 `--delay_in_s` 时为 **2M+1**（脚本里普遍开着该开关） |
| 3.2 的两级对齐是必备环节 | `--spectrum_time_align` / `--spectrum_freq_align` 都是 `store_true`，**默认关**；不开则不做相位校正、频率轴不归一 |

**还有两个未决口径**（正文与实验哪边为准尚未拍板）：
- `kernel_fn`：正文写保真度核 `K=|⟨ψ|ψ⟩|²`，但早期架构文档说最终用 `quantum_exp`（测地核＋可学习 κ）；主表逐格命令行已不可考。
- 主表 QCCK-M 列（`qf2_*` 那批）当年是用 **softmax + T=0.1** 跑的，而正文现已改按 raw K 写；换口径后主表是否重跑未定。

## 复现（示例：ETTh1-96 训练 + 测试）
先按 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) 第 7 节把 `dataset/gz/*.csv.gz` 解压到 `/tmp` 并软链回 `dataset/`——仓库里只有 gz，`--root_path ./dataset/ETT-small/` 需要的是解压后的目录。

**A. 论文最新口径**（`raw K` 那套）——**本仓库跑不了**，需要在工作副本 `/home/youjun/dataops_ws/` 下执行：
```bash
export OMP_NUM_THREADS=8
python run.py --is_training 1 --model_id demo_ETTh1_96 --model Q_S_Mamba --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --target OT --freq 15min \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00007 \
  --qmix_layers 2 --n_qubits 5 --qmix_norm raw_k --offdiag --qmix_msg H --qmix_ln_hp 1 \
  --qmix_gate --qmix_gate_init 0.1 --batch_size 32 --train_epochs 10 --patience 3
```

**B. 本仓库能跑通的示例**（旧口径：softmax 行归一化 + 温度 T）：
```bash
export OMP_NUM_THREADS=8                # 必须
python run.py --is_training 1 --model_id demo_ETTh1_96 --model Q_S_Mamba --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --target OT --freq 15min \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00007 \
  --qmix_layers 2 --n_qubits 5 --qmix_norm softmax --kernel_T 0.1 --offdiag \
  --qmix_gate --qmix_gate_init 0.1 --batch_size 32 --train_epochs 10 --patience 3
```
关键开关：

| 开关 | 论文最新口径 | 本仓库 | 说明 |
|---|---|---|---|
| `--qmix_norm` | `raw_k` | 只支持 `avg` / `softmax` / `l1` | 传其它值在建 `QuantumMixLayer` 时抛 `ValueError` |
| `--kernel_T` | —（raw K 不用温度） | `0.1` | 仅 softmax 分支使用 |
| `--offdiag` | 开 | 开 | 去对角。本仓库里**只在 softmax 分支生效**，`avg`/`l1` 分支完全不用它 |
| `--qmix_msg` / `--qmix_ln_hp` | `H` / `1` | ❌ 无此参数 | 杠杆 1 / 杠杆 2，只在工作副本里 |
| `--n_qubits` | `5` | `5` | `scripts/**/Q_S_Mamba_*_vd.sh` 里写的是 `2`，不是本论文口径 |
| `--qmix_gate` / `--qmix_gate_init` | 开 / 按盘定 | 开 / `0.1` | 耦合放大实验用 `0.5~1.0`（跨变量盘）或 `0.1`（通道独立盘）；旧主表用 `0.1` |

> `--offdiag` 与 `--qmix_gate` 都是**无参数开关**，写成 `--offdiag True` 会被 argparse 当成多余参数直接报错。
> `scripts/` 下 `S_Mamba_*.sh` 是上游 S-Mamba 官方脚本，`Q_S_Mamba_*.sh` 是我们自己的脚本；两者超参均与论文口径不完全一致。

## 渲染第五章 PDF
```bash
cd paper/ch5
python ch5cn_pdf.py [可选:输出路径.pdf]     # 默认输出 第五章初稿_中文_20260905.pdf
```
> **第五章的现行定版就是这个生成器**（小节为 5.1 实验设置 / 5.2 主结果 / 5.3 机制验证 / 5.4 消融 / 5.5 可解释性）。
> 文件名是历史遗留：默认输出的 `..._20260905.pdf` 与 `..._20260909.pdf` 内容一致（仅 PDF 内部 ID 不同），改完正文重跑即可。
- 生成器 `paper/ch5/ch5cn_pdf.py`（reportlab）：正文/表格/图注都在脚本里；
- 图在 `paper/ch5/figs/`。仓库里的造图脚本只有三个，**且它们的输出路径目前仍写死 `/home/youjun/paper/figs/`，换机要改**：
  - `paper/ch5/ch5_winheat.py` → 耦合度-增益热力图；
  - `paper/ch5/ch5_supp_20260909/gen_figs_stats.py` → 耦合核热力图 / ECDF；
  - `paper/ch5/ch5_supp_20260909/gen_pred_adv.py` → 预测案例图。
- 中文字体随仓库：`paper/ch5/cjkfont/wqy-zenhei.ttf`（可用 `QCC_CJK_FONT` 覆盖）；
- 主表数据 md 在 `paper/ch5/data/`（可用 `QCC_DATA_DIR` 覆盖）；
- 依赖：`reportlab`、`pillow`（已加入 requirements）。

## 实验与论文材料
- **主表/消融/干预（表 8）**：见 `paper/ch5/第五章初稿_中文_20260909.pdf`（= 生成器 `paper/ch5/ch5cn_pdf.py` 的当前输出），**这是第五章定版**。
- **补充实验（K 可视化/ECDF/斯皮尔曼/干预脚本与结果）**：`paper/ch5/ch5_supp_20260909/`。
- **第三/四章**：`paper/ch3/`、`paper/ch4/`；**大纲与风格**：`paper/outline/`。

## 交接
新服务器继续工作请读 `HANDOFF.md`。
