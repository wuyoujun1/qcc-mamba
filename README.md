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

## 复现（示例：ETTh1-96 训练 + 测试）
先按 [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) 第 7 节把 `dataset/gz/*.csv.gz` 解压到 `/tmp` 并软链回 `dataset/`——仓库里只有 gz，`--root_path ./dataset/ETT-small/` 需要的是解压后的目录。

```bash
export OMP_NUM_THREADS=8                # 必须
python run.py --is_training 1 --model_id demo_ETTh1_96 --model Q_S_Mamba --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --target OT --freq 15min \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --e_layers 2 --d_model 256 --d_ff 256 --d_state 2 --learning_rate 0.00007 \
  --qmix_layers 2 --n_qubits 5 --qmix_norm softmax --kernel_T 0.1 --offdiag \
  --qmix_gate --qmix_gate_init 0.1 --batch_size 32 --train_epochs 10 --patience 3
```
关键开关（**以论文主表实际口径为准**）：

| 开关 | 主表/论文口径 | 说明 |
|---|---|---|
| `--qmix_norm` | `softmax` | 消息权重归一化方式，只实现 `avg` / `softmax` / `l1`，传其它值会在建模时直接抛 `ValueError` |
| `--kernel_T` | `0.1` | 保真度核温度（softmax 分支） |
| `--offdiag` | 开 | 去对角，在 `K - I` 上做 softmax，跨变量权重占主导 |
| `--n_qubits` | `5` | 主表口径。注意 `scripts/**/Q_S_Mamba_*.sh` 里写的是 `2`，不是本论文口径 |
| `--qmix_gate` / `--qmix_gate_init` | 开 / `0.1` | 混合门控与初值。门控尺度在验证集上选定，测试结果对其取值不敏感（变化小于 1%）；`0.5~1.0` 是耦合放大实验用的配置，**未用于主表** |

> `--offdiag` 与 `--qmix_gate` 都是**无参数开关**，写成 `--offdiag True` 会被 argparse 当成多余参数直接报错。
> 早期文档里出现过的 `--qmix_norm raw_k` 与 `--qmix_msg` / `--qmix_ln_hp` **在当前 main 的代码里不存在**：前者的取值不在 `avg/softmax/l1` 内会抛错，后两个参数 run.py 根本没有定义。
> `scripts/` 下 `S_Mamba_*.sh` 是上游 S-Mamba 官方脚本，`Q_S_Mamba_*.sh` 是我们自己的脚本；两者超参均与主表口径不完全一致，复现主表请以上表为准。

## 渲染第五章 PDF
```bash
cd paper/ch5
python ch5cn_pdf.py [可选:输出路径.pdf]     # 默认输出 第五章初稿_中文_20260905.pdf
```
> **第五章的现行定版就是这个生成器**（小节为 5.1 实验设置 / 5.2 主结果 / 5.3 机制验证 / 5.4 消融 / 5.5 可解释性）。
> 文件名是历史遗留：默认输出的 `..._20260905.pdf` 与 `..._20260909.pdf` 内容一致（仅 PDF 内部 ID 不同），改完正文重跑即可。
> `第五章_完整修正版_95保留.tex` / `.pdf` 是**旧版**，小节编号为 5.6/5.7 且含**已删除的敏感性节**，勿据此写作。
- 生成器 `paper/ch5/ch5cn_pdf.py`（reportlab）：正文/表格/图注都在脚本里；
- 图在 `paper/ch5/figs/`。仓库里的造图脚本只有三个，**且它们的输出路径目前仍写死 `/home/youjun/paper/figs/`，换机要改**：
  - `paper/ch5/ch5_winheat.py` → 耦合度-增益热力图；
  - `paper/ch5/ch5_supp_20260909/gen_figs_stats.py` → 耦合核热力图 / ECDF；
  - `paper/ch5/ch5_supp_20260909/gen_pred_adv.py` → 预测案例图。
  - ⚠️ `figs/ch5_sens*.png` 是**已删除的敏感性节**的遗留图，定版正文不再引用（造图脚本已随该节一并移除）。
- 中文字体随仓库：`paper/ch5/cjkfont/wqy-zenhei.ttf`（可用 `QCC_CJK_FONT` 覆盖）；
- 主表数据 md 在 `paper/ch5/data/`（可用 `QCC_DATA_DIR` 覆盖）；
- 依赖：`reportlab`、`pillow`（已加入 requirements）。

## 实验与论文材料
- **主表/消融/干预（表 8）**：见 `paper/ch5/第五章初稿_中文_20260909.pdf`（= 生成器 `paper/ch5/ch5cn_pdf.py` 的当前输出），**这是第五章定版**。
- **补充实验（K 可视化/ECDF/斯皮尔曼/干预脚本与结果）**：`paper/ch5/ch5_supp_20260909/`。
- **第三/四章**：`paper/ch3/`、`paper/ch4/`；**大纲与风格**：`paper/outline/`。

## 交接
新服务器继续工作请读 `HANDOFF.md`。
