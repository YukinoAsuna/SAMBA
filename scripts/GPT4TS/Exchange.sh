model=GPT4TS
seq_len=96
percent=100
for pred_len in 96 
do

python run.py \
    --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
   --model_id Exchange_96_96 \
   --data custom \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --batch_size 16 \
    --train_epochs 10 \
    --decay_fac 0.5 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
    --patch_len 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --tmax 20 \
    --cos 1 \
    --is_gpt 1

done