model_name=SAMBA

for pred_len in 96 192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96\
  --model $model_name \
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 1 \
  --d_layers 1\
  --enc_in 137 \
  --dec_in 137 \
  --c_out 137 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --learning_rate 0.0005 \
  --itr 1
done