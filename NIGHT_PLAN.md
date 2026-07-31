# 🔥 今晚实验计划 2026-07-30 → 07-31

> **改进版 feature map**：`use_full_bloch=True`（可学习降维 + RZ·RY 全 Bloch 球编码）
> 自动串行执行，GPU 冲突自动等待，异常自动跳过

---

## 执行顺序

```
 状态  | 实验                        | 预计
:----:|-----------------------------|:----:
 🟢 进行中 | Traffic QCC (V=862) L=96     | ~4h 总
 ⏳      | Traffic Baseline (V=862)    | ~3h
 ⏳      | Weather Baseline (V=21)     | ~30min
 ⏳      | Weather QCC (V=21)          | ~30min
 ⏳      | ETTh1 Baseline + QCC (V=7)  | ~1h
 ⏳      | ETTh2 Baseline + QCC (V=7)  | ~1h
 ⏳      | ETTm1 Baseline + QCC (V=7)  | ~1h
 ⏳      | ETTm2 Baseline + QCC (V=7)  | ~1h
```

## 自动监控

| 检查 | 方式 |
|------|------|
| GPU 利用率 | 每小时自动检查，< 50% 则等待 |
| 温度 | nvidia-smi 监控，> 90°C 报 |
| 实验崩溃 | logs/ 检查 exit code，自动跳下一项 |
| 断连恢复 | 检查 logs/ 最新进度，从中断处续跑 |

## 结果输出

- 日志: `logs/{exp_name}.log`
- CSV: `results/{exp_name}_{dataset}.csv`
- Checkpoints: `checkpoints/{exp_name}_L*H*.pt`

## 改动说明

本次运行使用改进版 `EntanglingFeatureMap`：
- `input_proj`: `Linear(128 → 16)` 可学习降维，消除信息瓶颈
- `use_full_bloch=True`: 每 qubit RZ·RY 覆盖完整 Bloch 球
- 对比旧版 R_Y-only 编码，表达能力翻倍
