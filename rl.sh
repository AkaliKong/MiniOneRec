#!/bin/bash

export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE

for category in "Industrial_and_Scientific"; do
    ALPHA_CTR=1.0
    ALPHA_RANK=0.5
    ALPHA_ACC=0.5
    USE_HISTORY_COMPRESSION=true
    HISTORY_THRESHOLD=100
    COMPRESSION_TYPE=attention

    train_file=$(ls -f ./data/Amazon/train/${category}*.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)

    HF_ENDPOINT=https://hf-mirror.com accelerate launch \
                                    --config_file ./config/zero2_opt.yaml \
                                    --num_processes 8 --main_process_port 29503 \
                                    rl.py \
                        --model_path path_to_model \
                        --train_batch_size 64 \
                        --eval_batch_size 128 \
                        --num_train_epochs 2 \
                        --gradient_accumulation_steps 2 \
                        --train_file ${train_file} \
                        --eval_file ${eval_file} \
                        --info_file ${info_file} \
                        --category ${category} \
                        --sample_train False \
                        --eval_step 0.0999 \
                        --reward_type hybrid \
                        --num_generations 16 \
                        --mask_all_zero False \
                        --dynamic_sampling False \
                        --sync_ref_model True \
                        --beam_search True \
                        --test_during_training False \
                        --temperature 1.0 \
                        --learning_rate 1e-5 \
                        --add_gt False \
                        --beta 1e-3 \
                        --dapo False \
                        --output_dir output_dir \
                        --wandb_run_name wandb_name \
                        --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
                        --item_meta_path ./data/Amazon/index/Industrial_and_Scientific.item.json \
                        --alpha_ctr ${ALPHA_CTR} \
                        --alpha_rank ${ALPHA_RANK} \
                        --alpha_acc ${ALPHA_ACC} \
                        --use_history_compression ${USE_HISTORY_COMPRESSION} \
                        --history_threshold ${HISTORY_THRESHOLD} \
                        --compression_type ${COMPRESSION_TYPE}
done
