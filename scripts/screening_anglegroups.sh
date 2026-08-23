#!/bin/bash
# 第五轮：多组角度桥接筛选（PLAN_ANGLE_BRIDGE 方向1）
# 对照官方 S-Mamba 基线 ETTh1:96 = 0.3878（当前仓复现，用户认可的正确基线）
# 变体：dp_time（对照）/ qmix_g2 / qmix_g4 / qmix_g4m
# 协议：单 seed 2023 + L=96 + ETTh1，门槛 ≥3% 优于官方（≤0.376）
# 注意：num_workers=0，串行跑，一个跑完再下一个

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
  --n_qubits 4
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
echo "多组角度桥接 ETTh1:96（官方基线 0.3878）"
echo "=========================================="

run_one() {
  local tag=$1; shift
  echo ""
  echo ">>> 变体 $tag"
  $PYTHON -u run.py "${COMMON[@]}" \
    --model $model_name \
    --model_id ${DATASET}_${PRED_LEN}_${tag} \
    --des "anglebridge_${tag}" \
    "$@" \
    2>&1 | tee anglebridge_${tag}.log
}

# 对照：新 plain 基线（无量子核变量路径）
run_one dp_time --dp_fusion time_only --dp_gate_init 0.0

# qmix_g2：G=2 组角度，product 聚合
run_one qmix_g2 --dp_fusion add --dp_msg S --dp_gate_init 0.05 \
  --angle_groups 2 --kernel_group_agg product

# qmix_g4：G=4 组角度，product 聚合
run_one qmix_g4 --dp_fusion add --dp_msg S --dp_gate_init 0.05 \
  --angle_groups 4 --kernel_group_agg product

# qmix_g4m：G=4 组角度，mean 聚合
run_one qmix_g4m --dp_fusion add --dp_msg S --dp_gate_init 0.05 \
  --angle_groups 4 --kernel_group_agg mean

echo ""
echo "=========================================="
echo "多组角度桥接 ETTh1:96 完成"
echo "=========================================="
for tag in dp_time qmix_g2 qmix_g4 qmix_g4m; do
  echo "变体 $tag:"
  grep "mse:" anglebridge_${tag}.log | tail -1
done
echo "官方基线：0.3878（门槛 ≤0.376）"
