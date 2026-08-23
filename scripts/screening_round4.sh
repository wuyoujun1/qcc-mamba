#!/bin/bash
# 第四轮筛选：振幅编码量子核（零信息损失）
# 3 个变体 × ETTh1:96，单 seed=2023
# 基线：官方 S-Mamba ETTh1:96 = 0.3878
# 门槛：≥3% 优于基线（≤0.376）才进下一轮

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/wuyoujun/qcc-mamba/.venv/bin/python
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

model_name=Q_S_Mamba
DATASET=ETTh1
PRED_LEN=96

echo "=========================================="
echo "第四轮筛选：振幅编码量子核"
echo "$DATASET:$PRED_LEN"
echo "基线：0.3878"
echo "=========================================="

# 变体 1: 振幅编码 + N=6（64维态空间，零信息损失）
echo ""
echo ">>> 变体 1: 振幅编码 + N=6"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_amp1 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'amp1_N6' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
  --qmix_layers 2 \
  --n_qubits 6 \
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
  --qmix_use_S_only \
  --qmix_amplitude_encoding \
  2>&1 | tee screening_amp1.log

# 变体 2: 振幅编码 + N=6 + gate
echo ""
echo ">>> 变体 2: 振幅编码 + N=6 + gate"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_amp2 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'amp2_N6_gate' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
  --qmix_layers 2 \
  --n_qubits 6 \
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
  --qmix_use_S_only \
  --qmix_amplitude_encoding \
  --qmix_gate \
  --qmix_gate_init 0.5 \
  2>&1 | tee screening_amp2.log

# 变体 3: 振幅编码 + N=8（256维态空间，更大容量）
echo ""
echo ">>> 变体 3: 振幅编码 + N=8"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_amp3 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'amp3_N8' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
  --qmix_layers 2 \
  --n_qubits 8 \
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
  --qmix_use_S_only \
  --qmix_amplitude_encoding \
  2>&1 | tee screening_amp3.log

echo ""
echo "=========================================="
echo "第四轮筛选完成"
echo "=========================================="
echo "结果汇总："
for i in 1 2 3; do
  echo "变体 amp$i:"
  grep "test mse:" screening_amp${i}.log | tail -1
done
echo "基线：0.3878"
echo "门槛：≤0.376（≥3% 优于基线）"
