#!/bin/bash
# 第五轮筛选：角度编码量子核深度探索
# 8 个变体 × ETTh1:96，单 seed=2023
# 基线：官方 S-Mamba ETTh1:96 = 0.3878

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/wuyoujun/qcc-mamba/.venv/bin/python
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

model_name=Q_S_Mamba
DATASET=ETTh1
PRED_LEN=96

echo "=========================================="
echo "第五轮筛选：角度编码量子核深度探索"
echo "$DATASET:$PRED_LEN"
echo "基线：0.3878"
echo "=========================================="

# 变体 1: 角度编码 + ring 纠缠拓扑
echo ""
echo ">>> 变体 1: 角度编码 + ring 纠缠"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang1 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang1_ring' \
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
  2>&1 | tee screening_ang1.log

# 变体 2: 角度编码 + 更多重上传层数 (n_layers=4)
echo ""
echo ">>> 变体 2: 角度编码 + n_layers=4"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang2 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang2_4layers' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
  --train_epochs 10 \
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
  --qmix_use_S_only \
  2>&1 | tee screening_ang2.log

# 变体 3: 角度编码 + 更小的核温度 (T=0.05)
echo ""
echo ">>> 变体 3: 角度编码 + T=0.05"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang3 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang3_T005' \
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
  --kernel_T 0.05 \
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
  2>&1 | tee screening_ang3.log

# 变体 4: 角度编码 + topk=3 变量选择
echo ""
echo ">>> 变体 4: 角度编码 + topk=3"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang4 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang4_topk3' \
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
  --topk 3 \
  2>&1 | tee screening_ang4.log

# 变体 5: 角度编码 + 更大的频谱 M=64
echo ""
echo ">>> 变体 5: 角度编码 + M=64"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang5 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang5_M64' \
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
  --spectrum_M 64 \
  --spectrum_time_align \
  --spectrum_freq_align \
  --spectrum_range 0_2 \
  --delay_in_s \
  --hp_scale_v \
  --qmix_use_S_only \
  2>&1 | tee screening_ang5.log

# 变体 6: 角度编码 + 更小的学习率
echo ""
echo ">>> 变体 6: 角度编码 + lr=3e-5"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang6 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang6_lr3e5' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00003 \
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
  2>&1 | tee screening_ang6.log

# 变体 7: 角度编码 + 更长的训练 (30 epochs)
echo ""
echo ">>> 变体 7: 角度编码 + 30 epochs"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang7 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang7_30ep' \
  --d_model 256 \
  --d_state 2 \
  --d_ff 256 \
  --itr 1 \
  --learning_rate 0.00007 \
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
  --qmix_use_S_only \
  2>&1 | tee screening_ang7.log

# 变体 8: 角度编码 + 组合优化 (ring + T=0.05 + gate)
echo ""
echo ">>> 变体 8: 角度编码 + 组合优化"
$PYTHON -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ${DATASET}.csv \
  --model_id ${DATASET}_${PRED_LEN}_ang8 \
  --model $model_name \
  --data $DATASET \
  --features M \
  --seq_len 96 \
  --pred_len $PRED_LEN \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'ang8_combo' \
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
  --qmix_gate_init 0.5 \
  2>&1 | tee screening_ang8.log

echo ""
echo "=========================================="
echo "第五轮筛选完成"
echo "=========================================="
echo "结果汇总："
for i in 1 2 3 4 5 6 7 8; do
  echo "变体 ang$i:"
  grep "test mse:" screening_ang${i}.log | tail -1
done
echo "基线：0.3878"
