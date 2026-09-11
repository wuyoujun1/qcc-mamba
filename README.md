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
```bash
python run.py --is_training 1 --model_id demo_ETTh1_96 --model Q_S_Mamba --data ETTh1 \
  --root_path ./dataset/ETT-small/ --data_path ETTh1.csv --features M --target OT --freq 15min \
  --seq_len 96 --label_len 48 --pred_len 96 --enc_in 7 --dec_in 7 --c_out 7 \
  --qmix_layers 2 --n_qubits 5 --qmix_norm raw_k --offdiag True --qmix_msg H --qmix_ln_hp 1 \
  --qmix_gate True --qmix_gate_init 0.5 --batch_size 32 --train_epochs 10 --patience 3
```
关键开关（**以论文主表实际口径为准**）：

| 开关 | 主表/论文口径 | 说明 |
|---|---|---|
| `--qmix_norm` | `softmax` | 消息权重归一化方式；`raw_k` = 去对角保真度核直传（新理论口径，可选） |
| `--kernel_T` | `0.1` | softmax 分支的温度（`scripts/` 上游脚本同为 0.1；0.05 出现在个别长步长） |
| `--offdiag` | `True` | 去对角，跨变量权重占主导 |
| `--n_qubits` | `5` | 主表口径。注意：`scripts/` 下是**上游 S-Mamba 官方脚本**，其中多用 `--n_qubits 2`，不是本论文口径 |
| `--qmix_gate` / `--qmix_gate_init` | `True` / `0.1` | 混合门控与初值（实验研究中发现 0.5~1.0 会使耦合被模型依赖、MSE 基本不变；该配置**未用于主表**） |
| `--qmix_msg` / `--qmix_ln_hp` | `H` / `1` | 消息来源与是否对 Hp 做 LN（实验表明 H+保留 LN 最优） |

> 复现主表请以上表为准；`scripts/*.sh` 是上游 S-Mamba 官方实验脚本（回看长度/编码器替换等），其超参与本论文主表并不相同。

## 渲染第五章 PDF
```bash
cd paper/ch5
python ch5cn_pdf.py [可选:输出路径.pdf]     # 默认输出 第五章初稿_中文_20260905.pdf
```
- 生成器 `paper/ch5/ch5cn_pdf.py`（reportlab）：正文/表格/图注都在脚本里；
- 图在 `paper/ch5/figs/`（由 `ch5_winheat.py`、`ch5_sens_full.py` 等生成）；
- 中文字体随仓库：`paper/ch5/cjkfont/wqy-zenhei.ttf`（可用 `QCC_CJK_FONT` 覆盖）；
- 主表数据 md 在 `paper/ch5/data/`（可用 `QCC_DATA_DIR` 覆盖）；
- 依赖：`reportlab`、`pillow`（已加入 requirements）。

## 实验与论文材料
- **主表/消融/敏感性/干预（表 8）**：见 `paper/ch5/第五章初稿_中文_20260905.pdf`（生成器 `paper/ch5/ch5cn_pdf.py`）。
- **补充实验（K 可视化/ECDF/斯皮尔曼/干预脚本与结果）**：`paper/ch5/ch5_supp_20260909/`。
- **第三/四章**：`paper/ch3/`、`paper/ch4/`；**大纲与风格**：`paper/outline/`。

## 交接
新服务器继续工作请读 `HANDOFF.md`。
