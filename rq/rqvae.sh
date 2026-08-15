python rqvae.py \
      --data_path ../data/Amazon/index/Industrial_and_Scientific.emb-qwen-td.npy \
      --ckpt_dir ./output/Industrial_and_Scientific \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480 \
      --balance_loss_weight 0.01 \
      --balance_temperature 1.0 \
      --usage_ema_decay 0.99 \
      --dead_code_threshold 1e-4 \
      --dead_code_patience 100

