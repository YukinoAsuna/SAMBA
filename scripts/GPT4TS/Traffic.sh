seq_len=96
model=GPT4TS

for percent in 100
do
for pred_len in 96 
do

python run.py \
    --is_training 1\
    --root_path ./dataset/\
    --data_path traffic.csv \
    --model_id traffic_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 2048 \
    --learning_rate 0.001 \
    --train_epochs 10 \
    --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --patch_len 16 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 3 \
    --model $model \
    --patience 3 \
    --cos 1 \
    --tmax 10 \
    --batch_size 4\
    --is_gpt 1

done
done