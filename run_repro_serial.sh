#!/bin/bash
# 官方 S-Mamba 复现 —— 串行调度器（一个脚本跑完才跑下一个）
# 用法: bash run_repro_serial.sh
set -u
cd /home/wuyoujun/ts_quantum/S-D-Mamba
# 让脚本里的 `python` 解析到 venv
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

SCRIPTS=(
  scripts/multivariate_forecasting/ETT/S_Mamba_ETTh1.sh
  scripts/multivariate_forecasting/ETT/S_Mamba_ETTh2.sh
  scripts/multivariate_forecasting/ETT/S_Mamba_ETTm1.sh
  scripts/multivariate_forecasting/ETT/S_Mamba_ETTm2.sh
  scripts/multivariate_forecasting/Weather/S_Mamba.sh
  scripts/multivariate_forecasting/ECL/S_Mamba.sh
  scripts/multivariate_forecasting/Exchange/S_Mamba.sh
  scripts/multivariate_forecasting/Traffic/S_Mamba.sh
)

for i in "${!SCRIPTS[@]}"; do
  s="${SCRIPTS[$i]}"
  log="repro_$(basename "$s" .sh).log"
  echo "[$(date '+%H:%M:%S')] ($((i+1))/${#SCRIPTS[@]}) 开始: $s -> $log"
  bash "$s" > "$log" 2>&1
  rc=$?
  echo "[$(date '+%H:%M:%S')] 完成: $s rc=$rc"
  # 慢节奏：脚本之间休息 60 秒
  sleep 60
done
echo "[$(date '+%H:%M:%S')] 全部完成"
