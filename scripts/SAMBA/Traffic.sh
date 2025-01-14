model_name=SAMBA
root_path_name=./dataset/
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
seq_len=96

for pred_len in  96 
do
  python -u run.py \
    --is_training 1 \
    --num_workers 0 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --enc_in 862 \
    --dec_in 862 \
    --c_out 862 \
    --model_id $model_id_name'_'$seq_len'_'$pred_len \
    --model $model_name \
    --data $data_name \
    --features M \
    --batch_size 128\
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 1 \
    --d_layers 3 \
    --factor 1 \
    --des 'Exp' \
    --itr 1 \
    --n_heads 16 \
    --d_model 512 \
    --d_ff 512 \
    --dropout 0. \
    --patch_len $seq_len --stride $seq_len \
    --train_epochs 10 --patience 5 --batch_size 8 --learning_rate 0.0005 
done