
model=GPT4TS
seq_len=96
percent=100
for pred_len in 96 192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --model_id solar_96_96 \
  --model $model \
  --batch_size 8\
  --data Solar \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --decay_fac 0.75\
  --d_model 768\
  --d_ff 768\
  --n_heads 4\
  --enc_in 137 \
  --dec_in 137 \
  --percent 100\
  --is_gpt 1\
  --gpt_layer 6\
  --c_out 137 \
  --des 'Exp' \
  --learning_rate 0.0005 \
  --itr 1
done