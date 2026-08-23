#!/bin/bash
# P2-1 双路径重构 ETTh1:96 验证
# 官方基线：0.3878（d_model=256, d_state=2, lr=7e-5, 10ep）
# 变体：dp_time（新 plain）/ dp（S消息）/ dp_h（H消息）/ dp_both
# num_workers=0 避免 DataLoader worker 卡死

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/wuyoujun/qcc-mamba/.venv/bin/python
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

model_name=Q_S_Mamba_dp
DATASET=ETTh1
PRED_LEN=96

COMMON=(
  --is_training 1
  --root_path ./dataset/ETT-small/
  --data_path ETTh1.csv
  --data ETTh1
  --features M
  --seq_len 96
  --pred_len $PRED_LEN
  --e_layers 2
  --enc_in 7
  --dec_in 7
  --c_out 7
  --d_model 256
  --d_state 2
  --d_ff 256
  --itr 1
  --num_workers 0
  --learning_rate 0.00007
  --train_epochs 10
  --patience 3
  --dp_time_layers 2
  --dp_time_dim 256
  --dp_time_pool mean
  --dp_var_embed 1
  --use_dp_feats
  --n_qubits 2
  --qmix_n_layers 2
  --kernel_T 0.1
  --offdiag
  --entangle_topo linear
  --kernel_fn quantum
  --angle_norm clamp
  --spectrum_M 32
  --spectrum_time_align
  --spectrum_freq_align
  --spectrum_range 0_2
  --delay_in_s
  --theta_S_scale0 0.5
)

echo "=========================================="
echo "P2-1 双路径 ETTh1:96（基线 0.3878）"
echo "=========================================="

run_one() {
  local tag=$1; shift
  echo ""
  echo ">>> 变体 $tag"
  $PYTHON -u run.py "${COMMON[@]}" \
    --model $model_name \
    --model_id ${DATASET}_${PRED_LEN}_${tag} \
    --des "p21_${tag}" \
    "$@" \
    2>&1 | tee p21_etth1_${tag}.log
}

# dp_time：新 plain 基线（无量子核变量路径）
run_one dp_time --dp_fusion time_only --dp_gate_init 0.0

# dp：量子核变量路径，S 消息
run_one dp --dp_fusion add --dp_msg S --dp_gate_init 0.05

# dp_h：量子核变量路径，H 消息
run_one dp_h --dp_fusion add --dp_msg H --dp_gate_init 0.05

# dp_both：量子核变量路径，S+H 消息
run_one dp_both --dp_fusion add --dp_msg both --dp_gate_init 0.05

echo ""
echo "=========================================="
echo "P2-1 ETTh1:96 完成"
echo "=========================================="
for tag in dp_time dp dp_h dp_both; do
  echo "变体 $tag:"
  grep "mse:" p21_etth1_${tag}.log | tail -1
done
echo "官方基线：0.3878"
