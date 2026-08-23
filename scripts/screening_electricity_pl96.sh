#!/bin/bash
# Electricity 数据集筛选 pl96（321变量）
# 4 个变体 × pred_len=96
# 基线：0.138655
# num_workers=0 避免 DataLoader worker 卡死

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/wuyoujun/qcc-mamba/.venv/bin/python
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

model_name=Q_S_Mamba
DATASET=ECL
PRED_LEN=96
VARS=321

echo "=========================================="
echo "Electricity 筛选 pl96（基线 0.138655）"
echo "=========================================="

run_one() {
  local tag=$1; shift
  echo ""
  echo ">>> 变体 $tag (pl=$PRED_LEN)"
  $PYTHON -u run.py \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ${DATASET}_96_${PRED_LEN}_${tag} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $PRED_LEN \
    --e_layers 3 \
    --enc_in $VARS \
    --dec_in $VARS \
    --c_out $VARS \
    --des "${tag}_pl96" \
    --d_model 512 \
    --d_state 16 \
    --d_ff 512 \
    --itr 1 \
    --batch_size 16 \
    --num_workers 0 \
    "$@" \
    2>&1 | tee screening_ecl_${tag}_pl96.log
}

# ang1: ring 纠缠
run_one ang1 \
  --learning_rate 0.001 \
  --train_epochs 5 \
  --qmix_layers 2 \
  --n_qubits 2 \
  --qmix_n_layers 2 \
  --qmix_norm softmax \
  --kernel_T 0.1 \
  --offdiag \
  --entangle_topo ring \
  --kernel_fn quantum \
  --angle_norm clamp \
  --spectrum_M 32 \
  --spectrum_time_align \
  --spectrum_freq_align \
  --spectrum_range 0_2 \
  --delay_in_s \
  --hp_scale_v \
  --qmix_use_S_only

# ang2: 4层重上传
run_one ang2 \
  --learning_rate 0.001 \
  --train_epochs 5 \
  --qmix_layers 2 \
  --n_qubits 2 \
  --qmix_n_layers 4 \
  --qmix_norm softmax \
  --kernel_T 0.1 \
  --offdiag \
  --entangle_topo linear \
  --kernel_fn quantum \
  --angle_norm clamp \
  --spectrum_M 32 \
  --spectrum_time_align \
  --spectrum_freq_align \
  --spectrum_range 0_2 \
  --delay_in_s \
  --hp_scale_v \
  --qmix_use_S_only

# ang7: 30 epochs
run_one ang7 \
  --learning_rate 0.001 \
  --train_epochs 30 \
  --patience 5 \
  --qmix_layers 2 \
  --n_qubits 2 \
  --qmix_n_layers 2 \
  --qmix_norm softmax \
  --kernel_T 0.1 \
  --offdiag \
  --entangle_topo linear \
  --kernel_fn quantum \
  --angle_norm clamp \
  --spectrum_M 32 \
  --spectrum_time_align \
  --spectrum_freq_align \
  --spectrum_range 0_2 \
  --delay_in_s \
  --hp_scale_v \
  --qmix_use_S_only

# ang8: 组合优化
run_one ang8 \
  --learning_rate 0.001 \
  --train_epochs 5 \
  --qmix_layers 2 \
  --n_qubits 2 \
  --qmix_n_layers 2 \
  --qmix_norm softmax \
  --kernel_T 0.05 \
  --offdiag \
  --entangle_topo ring \
  --kernel_fn quantum \
  --angle_norm clamp \
  --spectrum_M 32 \
  --spectrum_time_align \
  --spectrum_freq_align \
  --spectrum_range 0_2 \
  --delay_in_s \
  --hp_scale_v \
  --qmix_use_S_only \
  --qmix_gate \
  --qmix_gate_init 0.5

echo ""
echo "=========================================="
echo "Electricity pl96 筛选完成"
echo "=========================================="
for tag in ang1 ang2 ang7 ang8; do
  echo "变体 $tag:"
  grep "mse:" screening_ecl_${tag}_pl96.log | tail -1
done
echo "基线：0.138655"
