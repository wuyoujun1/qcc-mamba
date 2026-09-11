# 交接说明（换服务器继续工作）

## 1. 拉代码
```bash
git clone git@github.com:wuyoujun1/qcc-mamba.git
cd qcc-mamba && git checkout main     # 只有 main 一条主线
```

> ⚠️ **注意：仓库里的代码只是参考快照，不是跑出论文结果的那一份。** 论文结果由工作副本 `/home/youjun/dataops_ws/` 跑出，两者不一致——仓库缺 `--qmix_norm raw_k`（论文最新理论）、`--qmix_msg`、`--qmix_ln_hp`、以及表 7 屏蔽实验要用的 `intervention_mask`，照本仓库跑不出论文主表。完整清单见 `README.md` 的「⚠️ 代码口径差异」。

## 2. 数据
仓库自带 `dataset/`（gz 压缩）：ETTh1、ETTh2、ETTm1、ETTm2、Weather、ECL、Traffic、Exchange、Beijing-AQI。
训练时按各盘传 `--root_path/--data_path`（ETT 家族 `./dataset/ETT-small/`，`--data custom` 的盘见各自日志模板）。
⚠️ 仓库里只有 `dataset/gz/*.csv.gz`，**训练前要先解压**：照 `docs/ENVIRONMENT.md` 第 7 节解压到 `/tmp` 再软链回 `dataset/`，否则 `--root_path ./dataset/ETT-small/` 找不到文件。

## 3. 环境
- 本机原环境在 `/tmp/qcc-env`（易失）。重建走技能 **`/rebuild-qcc-env`**（含全部坑位）。
- 依赖：`requirements.txt`（torch 2.0.1+cu118、mamba_ssm、causal_conv1d）。
- **必须** `export OMP_NUM_THREADS=8`。

## 4. 现在做到哪了
- 主表/消融/可解释性（含**表 8 按比例屏蔽耦合**）已写进 `paper/ch5/第五章初稿_中文_20260909.pdf`（= 生成器 `ch5cn_pdf.py` 的当前输出，**第五章定版**；改完重跑即可）。
  - 敏感性一节**已删除**（结论不支持主张且与门控调参矛盾），现行小节只有 5.1 实验设置 / 5.2 主结果 / 5.3 机制验证 / 5.4 消融 / 5.5 可解释性。
- 第五章补充实验（K 热力图、ECDF、斯皮尔曼、干预/屏蔽脚本与结果 json）在 `paper/ch5/ch5_supp_20260909/`。
- 关键结论（2026-09）：**耦合的作用由混合门控初值决定**——`qmix_gate_init` 0.1→0.5/1.0 可让"屏蔽耦合"的误差上升从 ~1% 升到 20%~150%，而测试 MSE 与主表持平（Weather/Beijing 甚至更好）；通道独立盘（ECL/Traffic/Exchange）无此效应，需保持小门控。

## 4.5 渲染论文 PDF
```bash
cd paper/ch5 && python ch5cn_pdf.py [输出.pdf]
```
字体/图片/数据都已随仓库（`cjkfont/` `figs/` `data/`），路径相对脚本，换机即用。

## 5. 待办（按优先级）
1. **按盘定 `qmix_gate_init`**（跨变量盘 0.5~1.0；通道独立盘 0.1），重训主表相关格并核对 MSE 不劣于现主表。
2. 在新配置上重跑 **消融**（去掉量子核/耦合应显著变差）与 **表 8 的屏蔽实验**，确认结论一致。
3. 论文：把"门控尺度"按验证集选择的口径写进实验节，或按组内决定不写（当前 PDF 未提）。
4. 结果/ckpt 归档：ckpt 在旧机 `/tmp/dataops_checkpoints`（软链），换机需重训或从旧机拷贝。

## 6. 常用命令
```bash
# 训练（示例见 README） / 测试：
python run.py --is_training 0 --model_id <已训模型id> --model Q_S_Mamba --data ETTh1 ...
# 屏蔽实验（推理期，按比例）：
python paper/ch5/ch5_supp_20260909/partial_removal.py <model_id> 0.5 0
# 训练后评估（MSE + 去掉全部耦合的 Δ%）：
python paper/ch5/ch5_supp_20260909/eval_coupling.py <model_id> 0
```
