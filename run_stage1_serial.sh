#!/bin/bash
# 阶段1验证：Q_S_Mamba (qmix_layers=0) 串行跑 ETTh1/ETTh2，与阶段0 逐位对比
set -u
cd /home/wuyoujun/ts_quantum/S-D-Mamba
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

for ds in ETTh1 ETTh2; do
  s="scripts/multivariate_forecasting/ETT/Q_S_Mamba_${ds}.sh"
  log="stage1_${ds}.log"
  echo "[$(date '+%H:%M:%S')] 开始: $s -> $log"
  bash "$s" > "$log" 2>&1
  echo "[$(date '+%H:%M:%S')] 完成: $s rc=$?"
  sleep 30
done
echo "[$(date '+%H:%M:%S')] 阶段1全部完成"
