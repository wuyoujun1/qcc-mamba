#!/bin/bash
# 夜间全自动实验脚本 — 充分利用 24GB 显存
# 并行跑 E2基线 + E3 L=1440，完成后跑 E3 L=8760

cd /home/wuyoujun/qcc-mamba
mkdir -p results
export PYTHONPATH=.
export PYTHONUNBUFFERED=1

DATE=$(date +%Y%m%d_%H%M)
echo "=========================================="
echo "夜间实验启动: $(date)"
echo "主机: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "显存: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)"
echo "=========================================="
echo ""

# ==================== Phase 1: 并行 ====================
echo "[Phase 1] 并行启动 E2 基线 + E3 L=1440"
echo "时间: $(date)"
echo ""

# 启动 E2 基线（3个setting，batch=64，~1小时）
nohup python run_benchmark.py --config configs/e2_baseline_big.yaml --gpu 0 > /tmp/e2_baseline_phase1.log 2>&1 &
E2_PID=$!
echo "  E2 基线 (L=192,336,720) PID=$E2_PID"

# 启动 E3 L=1440（batch=32，~1小时）
nohup python run_benchmark.py --config configs/e3_L1440.yaml --gpu 0 > /tmp/e3_L1440_phase1.log 2>&1 &
E3_1440_PID=$!
echo "  E3 L=1440            PID=$E3_1440_PID"

echo ""
echo "等待 Phase 1 完成..."
echo "时间: $(date)"
echo ""

# 等待两个进程都完成
FAIL=0
wait $E2_PID || { echo "  ⚠️ E2 基线异常退出 (code $?)"; FAIL=1; }
echo "  ✅ E2 基线完成: $(date)"
wait $E3_1440_PID || { echo "  ⚠️ E3 L=1440 异常退出 (code $?)"; FAIL=1; }
echo "  ✅ E3 L=1440 完成: $(date)"

echo ""
nvidia-smi --query-gpu=memory.used --format=csv,noheader | head -1

# ==================== Phase 2: E3 L=8760 ====================
echo ""
echo "[Phase 2] 启动 E3 L=8760（长跑，预计 9-12 小时）"
echo "时间: $(date)"
echo ""

nohup python run_benchmark.py --config configs/e3_L8760.yaml --gpu 0 > /tmp/e3_L8760_phase2.log 2>&1 &
E3_8760_PID=$!
echo "  E3 L=8760 PID=$E3_8760_PID"
echo ""

# 等 L=8760 跑完
wait $E3_8760_PID
echo "  ✅ E3 L=8760 完成: $(date)"

# ==================== 汇总结果 ====================
echo ""
echo "=========================================="
echo "全部实验完成! $(date)"
echo "=========================================="
echo ""
echo "--- E2 基线结果 ---"
cat results/e2_baseline_electricity.csv 2>/dev/null || echo "无结果文件"
echo ""
echo "--- E3 长期预测结果 ---"
cat results/e3_longterm_electricity.csv 2>/dev/null || echo "无结果文件"
echo ""
echo "=========================================="
echo "日志文件:"
echo "  E2 基线:  /tmp/e2_baseline_phase1.log"
echo "  E3 L=1440: /tmp/e3_L1440_phase1.log"
echo "  E3 L=8760: /tmp/e3_L8760_phase2.log"
echo "=========================================="
