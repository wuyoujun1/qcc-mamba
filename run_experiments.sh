#!/usr/bin/env bash
# QCC-Mamba 实验启动脚本（单进程串行，不并行）
# 用法: bash run_experiments.sh [step_name]
#       不传参则按优先级顺序跑全部
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONPATH="$PWD:$PYTHONPATH"
RESULTS="results"
LOGS="logs"
mkdir -p "$RESULTS" "$LOGS"

# 确认 GPU 空闲
gpu_check() {
    local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader | tr -d ' %')
    if [ "$util" -gt 50 ]; then
        echo "⚠️  GPU 利用率 ${util}% > 50%，等待空闲..."
        return 1
    fi
    echo "✅ GPU 利用率 ${util}%"
    return 0
}

wait_gpu() {
    while ! gpu_check; do
        sleep 60
    done
}

run_exp() {
    local name="$1"
    local config="$2"
    local log="$LOGS/${name}.log"
    local extra="${3:-}"

    wait_gpu
    echo ""
    echo "========================================"
    echo "🚀 $(date): 开始 $name"
    echo "   config: $config"
    echo "========================================"

    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

    python -u -B run_benchmark.py \
        --config "$config" \
        --gpu 0 --out "$RESULTS" \
        $extra 2>&1 | tee "$log"

    local ret=$?
    echo "📝 $(date): $name 结束 (exit=$ret)"
    echo "   日志: $log"
    return $ret
}

echo "========================================"
echo "🔥 QCC-Mamba 实验批量启动"
echo "    GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "    $(date)"
echo "========================================"

# ──────────────────────────────────────────────
# Step 1: E2 MPS bd=4（纠缠能力等价对照）
# ──────────────────────────────────────────────
run_exp "e2_mps_bd4" "configs/e2_mps_sweep_bd4.yaml"

# ──────────────────────────────────────────────
# Step 3: Traffic QCC（大变量数据集）
# ──────────────────────────────────────────────
run_exp "e2_traffic_qcc" "configs/e2_traffic_qcc.yaml"

# ──────────────────────────────────────────────
# Step 2: E2 3 seeds（补 H=192/336/720）
# ──────────────────────────────────────────────
# run_exp "e2_3seeds_baseline" "configs/e2_3seeds_baseline.yaml"
# run_exp "e2_3seeds_qcc" "configs/e2_3seeds_qcc.yaml"

# ──────────────────────────────────────────────
# Step 3: Weather baseline + QCC
# ──────────────────────────────────────────────
# run_exp "e2_weather_baseline" "configs/e2_weather_baseline.yaml"
# run_exp "e2_weather_qcc" "configs/e2_weather_qcc.yaml"

echo ""
echo "========================================"
echo "✅ $(date): 全部实验完成！"
echo "========================================"
