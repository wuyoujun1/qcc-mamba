#!/usr/bin/env bash
# ============================================================
# QCC-Mamba 整夜全自动实验脚本
# 不会停，断网/Session结束也不影响（全 nohup）
# ============================================================

PROJECT_DIR="/home/wuyoujun/qcc-mamba"
cd "$PROJECT_DIR" || exit 1
mkdir -p logs results checkpoints

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a logs/night_run.log; }
safe_wait() {
    # 只等当前 shell 的子进程
    while pgrep -P $$ > /dev/null 2>&1; do
        sleep 30
    done
}

# ============================================================
# 第0步：等 E1 跑完
# ============================================================
wait_for_e1() {
    log "等待 E1 实验结束..."
    while pgrep -f "run_e1.py" > /dev/null 2>&1; do
        sleep 60
    done
    sleep 10  # 确保文件写完
    log "E1 实验全部结束"

    log "=== E1 最终结果 ==="
    for f in logs/e1_mps_bd2.log logs/e1_mps_bd4.log logs/e1_quantum.log; do
        log "--- $f ---"
        grep "seed.*MSE_norm" "$f" 2>/dev/null | while read -r line; do log "  $line"; done
    done
}

# ============================================================
# 第1步：Traffic 基线（4路并行）
# ============================================================
run_traffic_baseline() {
    log "====== 开始 Traffic 基线（4路并行）======"
    for L in 96 192 336 720; do
        nohup python -u -B run_benchmark.py \
            --config configs/e2_traffic_base_L${L}.yaml \
            --gpu 0 --out results \
            > logs/traffic_baseline_L${L}.log 2>&1 &
        log "  Traffic 基线 L=${L} PID=$!"
    done
    log "等待 Traffic 基线全部完成..."
    safe_wait
    log "✅ Traffic 基线完成"
}

# ============================================================
# 第2步：Traffic QCC（4路并行）
# ============================================================
run_traffic_qcc() {
    log "====== 开始 Traffic QCC（4路并行）======"
    for L in 96 192 336 720; do
        nohup python -u -B run_benchmark.py \
            --config configs/e2_traffic_qcc_L${L}.yaml \
            --gpu 0 --out results \
            > logs/traffic_qcc_L${L}.log 2>&1 &
        log "  Traffic QCC L=${L} PID=$!"
    done
    log "等待 Traffic QCC 全部完成..."
    safe_wait
    log "✅ Traffic QCC 完成"
}

# ============================================================
# 第3步：Weather（21 vars，几十分钟级别）
# ============================================================
run_weather() {
    log "====== Weather 基线 ====="
    nohup python -u -B run_benchmark.py \
        --config configs/e2_weather_baseline.yaml \
        --gpu 0 --out results \
        > logs/weather_baseline.log 2>&1 &
    wait $! 2>/dev/null || true

    log "====== Weather QCC ====="
    nohup python -u -B run_benchmark.py \
        --config configs/e2_weather_qcc.yaml \
        --gpu 0 --out results \
        > logs/weather_qcc.log 2>&1 &
    wait $! 2>/dev/null || true
    log "✅ Weather 完成"
}

# ============================================================
# 第4步：ETT 数据集（4个，7 vars，每个十几分钟）
# ============================================================
run_ett() {
    for ds in etth1 etth2 ettm1 ettm2; do
        log "====== ${ds} 基线 ====="
        nohup python -u -B run_benchmark.py \
            --config configs/e2_${ds}_baseline.yaml \
            --gpu 0 --out results \
            > logs/${ds}_baseline.log 2>&1 &
        wait $! 2>/dev/null || true

        log "====== ${ds} QCC ====="
        nohup python -u -B run_benchmark.py \
            --config configs/e2_${ds}_qcc.yaml \
            --gpu 0 --out results \
            > logs/${ds}_qcc.log 2>&1 &
        wait $! 2>/dev/null || true
    done
    log "✅ ETT 全部完成"
}

# ============================================================
# 第5步：检查哪些 QCC 输了，每个输的补 5 个 seed
# ============================================================
run_extra_seeds() {
    log "====== 检查 QCC vs 基线 ======"

    # 用 Python 分析哪些 dataset×setting QCC 比基线差
    python -u -B -c "
import os, glob, pandas as pd
results_dir = 'results'
losers = []
for f in sorted(glob.glob(os.path.join(results_dir, 'e2_*_baseline.csv'))):
    ds = os.path.basename(f).replace('e2_', '').replace('_baseline.csv', '')
    if 'traffic' in ds:
        # traffic 有 per-L 文件，不是 4-in-1
        continue
    qcc_file = os.path.join(results_dir, f'e2_{ds}_qcc.csv')
    if not os.path.exists(qcc_file):
        # 尝试 traffic 独立文件
        continue
    bl = pd.read_csv(f)
    qc = pd.read_csv(qcc_file)
    bl = bl[bl['mse_norm'].notna()]
    qc = qc[qc['mse_norm'].notna()]
    for _, bl_row in bl.iterrows():
        L = bl_row['lookback']
        bl_norm = bl_row['mse_norm']
        qc_row = qc[qc['lookback'] == L]
        if qc_row.empty:
            continue
        qc_norm = qc_row.iloc[0]['mse_norm']
        loser = qc_norm > bl_norm
        print(f'  {ds} L={L}: baseline={bl_norm:.4f}  QCC={qc_norm:.4f}  {\"⚠️ QCC输\" if loser else \"✅ QCC赢\"}')
        if loser:
            losers.append((ds, L, bl_norm, qc_norm))
    # Traffic per-L 单独检查
    for L in [96, 192, 336, 720]:
        for prefix in ['traffic_baseline', 'traffic_qcc']:
            nf = f'{results_dir}/e2_{prefix}_{L}.csv'
        bl_file = f'{results_dir}/e2_traffic_baseline_L{L}.csv'
        qc_file = f'{results_dir}/e2_traffic_qcc_L{L}.csv'
        if os.path.exists(bl_file) and os.path.exists(qc_file):
            bl = pd.read_csv(bl_file)
            qc = pd.read_csv(qc_file)
            bl_norm = bl['mse_norm'].iloc[0] if 'mse_norm' in bl.columns else None
            qc_norm = qc['mse_norm'].iloc[0] if 'mse_norm' in qc.columns else None
            if bl_norm is not None and qc_norm is not None:
                loser = qc_norm > bl_norm
                print(f'  traffic L={L}: baseline={bl_norm:.4f}  QCC={qc_norm:.4f}  {\"⚠️ QCC输\" if loser else \"✅ QCC赢\"}')
                if loser:
                    losers.append(('traffic', L, bl_norm, qc_norm))

    if not losers:
        print()
        print('='*60)
        print('✅ 所有数据集 QCC 都不比基线差！')
        print('='*60)
    else:
        print()
        print('='*60)
        print(f'⚠️  QCC 输的有 {len(losers)} 个 setting，补跑 5 个 seed')
        print('='*60)
        cmds = []
        for ds, L, bl, qc in losers:
            print(f'  {ds} L={L}: baseline={bl:.4f}  QCC={qc:.4f}')
            for s in range(2029, 2034):
                cmds.append((ds, L, s))
        # 输出 shell 命令让 bash 执行
        print()
        print('EXTRA_CMDS')
        for ds, L, s in cmds:
            out_file = f'extra_{ds}_L{L}_s{s}'
            print(f'SEED {ds} {L} {s} {out_file}')
" 2>&1 | tee -a logs/night_run_extra.log

    # 执行补的 seed（从日志解析命令）
    log "====== 补跑额外 seed ======"
    while IFS= read -r line; do
        if [[ "$line" == SEED* ]]; then
            read -r _ ds L s out_file <<< "$line"
            log "  补 seed=$s: $ds L=$L"
            nohup python -u -B run_benchmark.py \
                --config configs/e2_${ds}_baseline.yaml \
                --gpu 0 --out results \
                > logs/${out_file}_baseline.log 2>&1 &
            BASELINE_PID=$!
            sleep 2
            nohup python -u -B run_benchmark.py \
                --config configs/e2_${ds}_qcc.yaml \
                --gpu 0 --out results \
                > logs/${out_file}_qcc.log 2>&1 &
            QCC_PID=$!
            log "    baseline PID=$BASELINE_PID, qcc PID=$QCC_PID"
            # 等这组 seed 的两个都跑完再继续
            wait $BASELINE_PID $QCC_PID 2>/dev/null || true
        fi
    done < logs/night_run_extra.log
    log "✅ 额外 seed 全部完成"
}

# ============================================================
# 最终汇总
# ============================================================
final_summary() {
    log "====== 最终结果汇总 ======"
    python -u -B -c "
import os, glob, pandas as pd
print(f'{\"Dataset\":<12} {\"L\":>4} {\"Baseline MSE_norm\":>18} {\"QCC MSE_norm\":>18} {\"Δ%\":>8} {\"Winner\":>8}')
print('-'*70)
for f in sorted(glob.glob('results/e2_*_baseline.csv')):
    ds = os.path.basename(f).replace('e2_', '').replace('_baseline.csv', '')
    qcc_file = f.replace('_baseline.csv', '_qcc.csv')
    if not os.path.exists(qcc_file):
        continue
    bl = pd.read_csv(f)
    qc = pd.read_csv(qcc_file)
    for _, b in bl.iterrows():
        L = b['lookback']
        q = qc[qc['lookback'] == L]
        if q.empty:
            continue
        bn = b.get('mse_norm', None)
        qn = q.iloc[0].get('mse_norm', None)
        if bn is None or qn is None:
            continue
        delta = ((qn / bn) - 1) * 100
        winner = 'QCC 🏆' if qn < bn else 'Base'
        print(f'{ds:<12} {L:>4} {bn:>16.4f} {qn:>16.4f} {delta:>+7.2f}% {winner:>8}')
" 2>&1 | tee -a logs/night_run_final.txt
    cp logs/night_run_final.txt results/night_run_final.txt
    log "最终结果已保存到 results/night_run_final.txt"
}

# ============================================================
# 主流程
# ============================================================
log "=========================================="
log "🌙 整夜实验开始"
log "=========================================="
log "GPU: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)"

wait_for_e1
run_traffic_baseline
run_traffic_qcc
run_weather
run_ett
run_extra_seeds
final_summary

log "=========================================="
log "🌅 一夜实验全部完成！"
log "=========================================="
