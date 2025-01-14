

seq_len=96
model=GPT4TS

for pred_len in 96 
do
for percent in 100
do

python run.py \
    --is_training 1\
    --root_path ./dataset/\
    --data_path electricity.csv \
    --model_id ECL_$model'_'$gpt_layer'_'$seq_len'_'$pred_len'_'$percent \
    --data custom \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --batch_size 8 \
    --learning_rate 0.0001 \
    --train_epochs 10 \
    --decay_fac 0.75 \
    --d_model 768 \
    --n_heads 4 \
    --d_ff 768 \
    --dropout 0.3 \
     --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
    --patch_len 6 \
    --stride 8 \
    --percent $percent \
    --gpt_layer 6 \
    --itr 1 \
    --model $model \
    --cos 1 \
    --tmax 10 \
    --is_gpt 1
done
done