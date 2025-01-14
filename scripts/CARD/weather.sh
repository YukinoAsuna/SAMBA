model_name=CARD
seq_len=96
pred_len=96
    python -u run.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path weather.csv \
    --model_id weather_96_$pred_len \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --des 'Exp' \
    --itr 1 \
    --e_layers 2 \
    --n_heads 16 \
    --d_model 128 \
    --d_ff 256 \
    --dropout 0.2\
    --fc_dropout 0.2\
    --head_dropout 0\
    --patch_len 16\
    --stride 8\
    --patience 5\
    --train_epochs 10 --lradj CARD \
    --itr 1 --batch_size 128 --learning_rate 0.0001 \
    --dp_rank 8 --top_k 5   --mask_rate 0 --warmup_epochs 0 \
