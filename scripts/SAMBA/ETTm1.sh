model_name=SAMBA

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96\
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --e_layers 1 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128  \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 5 \
  --lradj 'TST' \
  --pct_start 0.4 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --d_state2 8 \
  --d_state1 8 \
  --d_ff 0

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_192\
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --e_layers 1 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128  \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 5 \
  --lradj 'TST' \
  --pct_start 0.4 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --d_state2 8 \
  --d_state1 8 

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_336\
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --e_layers 1 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128  \
  --dropout 0.2 \
  --des 'Exp' \
  --train_epochs 10 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --d_state2 8 \
  --d_state1 8 \

python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_720\
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --e_layers 1 \
  --d_layers 1 \
  --n_heads 16 \
  --d_model 128  \
  --dropout 0.2 \
  --fc_dropout 0.2 \
  --head_dropout 0 \
  --patch_len 16 \
  --stride 8 \
  --des 'Exp' \
  --train_epochs 10 \
  --patience 5 \
  --lradj 'TST' \
  --pct_start 0.4 \
  --itr 1 \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --d_state2 8 \
  --d_state1 8 
