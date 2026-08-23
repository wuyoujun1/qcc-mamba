#!/bin/bash
# Weather 数据集筛选（21变量）
# 预测长度：96/720（筛选标准）
# 种子：42
# 四个变体：ang1（ring）、ang2（4层）、ang7（30ep）、ang8（组合）

export CUDA_VISIBLE_DEVICES=0
PYTHON=/home/wuyoujun/qcc-mamba/.venv/bin/python
export PATH=/home/wuyoujun/qcc-mamba/.venv/bin:$PATH

model_name=Q_S_Mamba
DATASET=weather
DATA_PATH=./dataset/weather/weather.csv
PRED_LENS=(96 720)
VARS=21

echo "=========================================="
echo "Weather 数据集筛选（21变量）"
echo "预测长度：96/720"
echo "=========================================="

for PRED_LEN in "${PRED_LENS[@]}"; do
  echo ""
  echo ">>> 预测长度: $PRED_LEN"
  
  # 变体 1: ring 纠缠拓扑
  echo "  变体 1: ring"
  $PYTHON -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id ${DATASET}_96_${PRED_LEN}_ang1 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $PRED_LEN \
    --e_layers 3 \
    --enc_in $VARS \
    --dec_in $VARS \
    --c_out $VARS \
    --des 'ang1_ring' \
    --d_model 512 \
    --d_state 2 \
    --d_ff 512 \
    --itr 1 \
    --learning_rate 0.00005 \
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
    --qmix_use_S_only \
    2>&1 | tee screening_weather_ang1_pl${PRED_LEN}.log

  # 变体 2: 4层重上传
  echo "  变体 2: 4层"
  $PYTHON -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id ${DATASET}_96_${PRED_LEN}_ang2 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $PRED_LEN \
    --e_layers 3 \
    --enc_in $VARS \
    --dec_in $VARS \
    --c_out $VARS \
    --des 'ang2_4layers' \
    --d_model 512 \
    --d_state 2 \
    --d_ff 512 \
    --itr 1 \
    --learning_rate 0.00005 \
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
    --qmix_use_S_only \
    2>&1 | tee screening_weather_ang2_pl${PRED_LEN}.log

  # 变体 7: 30 epochs
  echo "  变体 7: 30ep"
  $PYTHON -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id ${DATASET}_96_${PRED_LEN}_ang7 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $PRED_LEN \
    --e_layers 3 \
    --enc_in $VARS \
    --dec_in $VARS \
    --c_out $VARS \
    --des 'ang7_30ep' \
    --d_model 512 \
    --d_state 2 \
    --d_ff 512 \
    --itr 1 \
    --learning_rate 0.00005 \
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
    2>&1 | tee screening_weather_ang7_pl${PRED_LEN}.log

  # 变体 8: 组合优化
  echo "  变体 8: 组合"
  $PYTHON -u run.py \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id ${DATASET}_96_${PRED_LEN}_ang8 \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --pred_len $PRED_LEN \
    --e_layers 3 \
    --enc_in $VARS \
    --dec_in $VARS \
    --c_out $VARS \
    --des 'ang8_combo' \
    --d_model 512 \
    --d_state 2 \
    --d_ff 512 \
    --itr 1 \
    --learning_rate 0.00005 \
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
    --qmix_gate_init 0.5 \
    2>&1 | tee screening_weather_ang8_pl${PRED_LEN}.log

done

echo ""
echo "=========================================="
echo "Weather 筛选完成"
echo "=========================================="
echo "结果汇总："
for PRED_LEN in "${PRED_LENS[@]}"; do
  echo "预测长度 $PRED_LEN:"
  for v in 1 2 7 8; do
    echo "  变体 ang$v:"
    grep "test mse:" screening_weather_ang${v}_pl${PRED_LEN}.log | tail -1
  done
done
