export NCCL_IB_DISABLE=1        # 完全禁用 IB/RoCE
# Office_Products, Industrial_and_Scientific
for category in "Industrial_and_Scientific"; do
    USE_CTR_HEAD=true
    LAMBDA_CTR=0.5
    USE_HISTORY_COMPRESSION=true
    HISTORY_THRESHOLD=100
    COMPRESSION_TYPE=attention

    train_file=$(ls -f ./data/Amazon/train/${category}*11.csv)
    eval_file=$(ls -f ./data/Amazon/valid/${category}*11.csv)
    test_file=$(ls -f ./data/Amazon/test/${category}*11.csv)
    info_file=$(ls -f ./data/Amazon/info/${category}*.txt)
    echo ${train_file} ${eval_file} ${info_file} ${test_file}
    
    torchrun --nproc_per_node 8 \
            sft.py \
            --base_model your_model_path \
            --batch_size 1024 \
            --micro_batch_size 16 \
            --train_file ${train_file} \
            --eval_file ${eval_file} \
            --output_dir output_dir/xxx \
            --wandb_project wandb_proj \
            --wandb_run_name wandb_name \
            --category ${category} \
            --train_from_scratch False \
            --seed 42 \
            --sid_index_path ./data/Amazon/index/Industrial_and_Scientific.index.json \
            --item_meta_path ./data/Amazon/index//Industrial_and_Scientific.item.json \
            --freeze_LLM False \
            --use_ctr_head ${USE_CTR_HEAD} \
            --lambda_ctr ${LAMBDA_CTR} \
            --use_history_compression ${USE_HISTORY_COMPRESSION} \
            --history_threshold ${HISTORY_THRESHOLD} \
            --compression_type ${COMPRESSION_TYPE}
done
