#!/bin/bash
# 第二轮筛选：P1-1 量子核直接读频谱 S（不读主干 H）
# 4 个变体 × ETTh1:96，单 seed=2023
# 基线：官方 S-Mamba ETTh1:96 = 0.3878
# 门槛：≥3% 优于基线（≤0.376）才进下一轮

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/wuyoujun/qcc-mamba/.venv/bin/python
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

model_name=Q_S_Mamba
DATASET=ETTh1
PRED_LEN=96

echo "=========================================="
echo "第二轮筛选：P1-1 量子核直接读频谱 S"
echo "$DATASET:$PRED_LEN"
echo "基线：0.3878"
echo "=========================================="

# 变体 1: P1-1 use_S_only + hp_scale_v（量子核独占频域通道）
echo ""
echo ">>> 变体 1: P1-1 use_S_only + hp_scale_v"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_r1 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'r1_S_only_hpv' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
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
  --qmix_use_S_only \
  2>&1 | tee screening_r1.log

# 变体 2: P1-1 use_S_only + hp_scale_v + gate（门控保底）
echo ""
echo ">>> 变体 2: P1-1 use_S_only + hp_scale_v + gate"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_r2 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'r2_S_only_hpv_gate' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
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
  --qmix_use_S_only \
  --qmix_gate \
  --qmix_gate_init 0.5 \
  2>&1 | tee screening_r2.log

# 变体 3: P1-1 use_S_only + hp_scale_v + rbf（经典核对照）
echo ""
echo ">>> 变体 3: P1-1 use_S_only + hp_scale_v + rbf"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_r3 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'r3_S_only_hpv_rbf' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
  --qmix_layers 2 \
  --n_qubits 2 \
  --qmix_n_layers 2 \
  --qmix_norm softmax \
  --kernel_T 0.1 \
  --offdiag \
  --entangle_topo linear \
  --kernel_fn rbf \
  --angle_norm clamp \
  --spectrum_M 32 \
  --spectrum_time_align \
  --spectrum_freq_align \
  --spectrum_range 0_2 \
  --delay_in_s \
  --hp_scale_v \
  --qmix_use_S_only \
  2>&1 | tee screening_r3.log

# 变体 4: P1-1 use_S_only + hp_scale_v + 50 epochs（排除训练不足）
echo ""
echo ">>> 变体 4: P1-1 use_S_only + hp_scale_v + 50 epochs"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_r4 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'r4_S_only_hpv_50ep' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 50 \
  --patience 10 \
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
  --qmix_use_S_only \
  2>&1 | tee screening_r4.log

echo ""
echo "=========================================="
echo "第二轮筛选完成"
echo "=========================================="
echo "结果汇总："
for i in 1 2 3 4; do
  echo "变体 r$i:"
  grep "test mse:" screening_r${i}.log | tail -1
done
echo "基线：0.3878"
echo "门槛：≤0.376（≥3% 优于基线）"
