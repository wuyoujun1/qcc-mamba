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
关键开关（2026-09 为"让耦合真正起作用"引入）：
- `--qmix_norm {softmax|raw_k|rawk_row}`：消息权重归一化；`raw_k`=直接用去对角保真度核（新理论口径）。
- `--qmix_gate_init`：混合门控初值。**这是让耦合从"可有可无"变为"被模型依赖"的关键杠杆**（0.1→0.5/1.0；详见 `paper/ch5/` 记录）。
- `--qmix_msg {H|S}`、`--qmix_ln_hp {0|1}`：消息来源与是否对 Hp 做 LN（实验表明 H + 保留 LN 最优）。
- `--kernel_T`、`--topk`、`--offdiag`：softmax 分支的温度/稀疏化。

## 实验与论文材料
- **主表/消融/敏感性/干预（表 8）**：见 `paper/ch5/第五章初稿_中文_20260905.pdf`（生成器 `paper/ch5/ch5cn_pdf.py`）。
- **补充实验（K 可视化/ECDF/斯皮尔曼/干预脚本与结果）**：`paper/ch5/ch5_supp_20260909/`。
- **第三/四章**：`paper/ch3/`、`paper/ch4/`；**大纲与风格**：`paper/outline/`。

## 交接
新服务器继续工作请读 `HANDOFF.md`。
