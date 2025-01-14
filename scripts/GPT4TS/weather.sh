seq_len=96
model=GPT4TS

for percent in 100
do
for pred_len in 96 
do

python run.py \
    --is_training 1\
    --root_path ./dataset/\
    --data_path weather.csv \
    --model_id weather_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 32 \
    --learning_rate 0.0001 \
     --data custom \
  --features M \
    --train_epochs 10 \
    --decay_fac 0.9 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
     --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
     --factor 3 \
    --lradj type3 \
    --patch_len 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --is_gpt 1
    
done
done