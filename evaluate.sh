# Industrial_and_Scientific
# Office_Products
for category in "Industrial_and_Scientific"
do
    # your model path
    exp_name="xxx"
    USE_CTR_HEAD=true
    USE_HISTORY_COMPRESSION=true
    HISTORY_THRESHOLD=100
    COMPRESSION_TYPE=attention

    exp_name_clean=$(basename "$exp_name")
    echo "Processing category: $category with model: $exp_name_clean (STANDARD MODE)"
    
    train_file=$(ls ./data/Amazon/train/${category}*.csv 2>/dev/null | head -1)
    test_file=$(ls ./data/Amazon/test/${category}*11.csv 2>/dev/null | head -1)
    info_file=$(ls ./data/Amazon/info/${category}*.txt 2>/dev/null | head -1)
    
    if [[ ! -f "$test_file" ]]; then
        echo "Error: Test file not found for category $category"
        continue
    fi
    if [[ ! -f "$info_file" ]]; then
        echo "Error: Info file not found for category $category"
        continue
    fi
    
    temp_dir="./temp/${category}-${exp_name_clean}"
    echo "Creating temp directory: $temp_dir"
    mkdir -p "$temp_dir"
    
    echo "Splitting test data..."
    python ./split.py --input_path "$test_file" --output_path "$temp_dir" --cuda_list "0,1,2,3,4,5,6,7"
    
    if [[ ! -f "$temp_dir/0.csv" ]]; then
        echo "Error: Data splitting failed for category $category"
        continue
    fi
    
    cudalist="0 1 2 3 4 5 6 7"  
    echo "Starting parallel evaluation (STANDARD MODE)..."
    for i in ${cudalist}
    do
        if [[ -f "$temp_dir/${i}.csv" ]]; then
            echo "Starting evaluation on GPU $i for category ${category}"
            CUDA_VISIBLE_DEVICES=$i python -u ./evaluate.py \
                --base_model "$exp_name" \
                --info_file "$info_file" \
                --category ${category} \
                --test_data_path "$temp_dir/${i}.csv" \
                --result_json_data "$temp_dir/${i}.json" \
                --batch_size 8 \
                --num_beams 50 \
                --max_new_tokens 256 \
                --temperature 1.0 \
                --guidance_scale 1.0 \
                --length_penalty 0.0 \
                --use_ctr_head ${USE_CTR_HEAD} \
                --use_history_compression ${USE_HISTORY_COMPRESSION} \
                --history_threshold ${HISTORY_THRESHOLD} \
                --compression_type ${COMPRESSION_TYPE} &
        else
            echo "Warning: Split file $temp_dir/${i}.csv not found, skipping GPU $i"
        fi
    done
    echo "Waiting for all evaluation processes to complete..."
    wait
    
    result_files=$(ls "$temp_dir"/*.json 2>/dev/null | wc -l)
    if [[ $result_files -eq 0 ]]; then
        echo "Error: No result files generated for category $category"
        continue
    fi
    
    output_dir="./results/${exp_name_clean}"
    echo "Creating output directory: $output_dir"
    mkdir -p "$output_dir"

    actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
    echo "Merging results from GPUs: $actual_cuda_list"
    
    python ./merge.py \
        --input_path "$temp_dir" \
        --output_path "$output_dir/final_result_${category}.json" \
        --cuda_list "$actual_cuda_list"
    
    if [[ ! -f "$output_dir/final_result_${category}.json" ]]; then
        echo "Error: Result merging failed for category $category"
        continue
    fi
    
    echo "Calculating metrics..."
    python ./calc.py \
        --path "$output_dir/final_result_${category}.json" \
        --item_path "$info_file"
    
    echo "Completed processing for category: $category"
    echo "Results saved to: $output_dir/final_result_${category}.json"
    echo "----------------------------------------" 
done

echo "All categories processed!"
