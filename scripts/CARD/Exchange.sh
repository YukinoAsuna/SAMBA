
model_name=CARD
for pred_len in 96 192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 1\
  --d_layers 0\
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --des 'Exp' \
  --patience 5\
  --d_model 128 \
  --d_ff 128 \
  --itr 1
done