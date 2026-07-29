#!/bin/bash
export PYTHONPATH=/home/wuyoujun/qcc-mamba
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
cd /home/wuyoujun/qcc-mamba

# 清缓存确保跑新代码
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "=========================================="
echo "夜间实验启动: $(date)"
echo "=========================================="

echo "===== [1/3] E2 S-Mamba 基线 ====="
python -B run_benchmark.py --config configs/e2_baseline_big.yaml --gpu 0 --out results
echo "Done E2: $(date)"

echo "===== [2/3] E3 L=1440 ====="
python -B run_benchmark.py --config configs/e3_L1440.yaml --gpu 0 --out results
echo "Done L=1440: $(date)"

echo "===== [3/3] E3 L=8760 ====="
python -B run_benchmark.py --config configs/e3_L8760.yaml --gpu 0 --out results
echo "Done L=8760: $(date)"

echo "全部完成: $(date)"
ls -la results/*.csv
