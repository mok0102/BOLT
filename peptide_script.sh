#!/usr/bin/env bash

cd /home/mok/module/steering/BOLT/optimization/query_plans
source .venv/bin/activate
cd /home/mok/module/steering/BOLT

BASE_CKPT_DIR="/home/mok/module/steering/BOLT/fine-tuning/query_plans/ckpt/Qwen2.5-3B-Instruct"

list="0 1 2"  #3 4 5
for var in $list
do
    echo "conducting $var -th task now"

    ##### SAMPLE AT FINE-TUNED LLM
    cd /home/mok/module/steering/BOLT/fine-tuning/peptides
    if [ "$var" -eq 0 ]; then
        #### 첫 pi_0는 여기서 준거쓰기. pi_0가 튜닝 안했다고 생각하면 진짜 잘 안됨
        cp /home/mok/module/steering/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_init.txt /home/mok/module/steering/BOLT/fine-tuning/peptides/sampled_output_from_ft/task_${var}_init.txt
        cp /home/mok/module/steering/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_scores.csv /home/mok/module/steering/BOLT/fine-tuning/peptides/sampled_output_from_ft/task_${var}_scores.csv
    else
        PREV_VAR=$((var - 1))
        SAMPLE_MODEL_PATH="/home/mok/module/steering/BOLT/fine-tuning/peptides/output/qwen_2_5_3B_lora_output_${PREV_VAR}/epoch_4"
        if [ ! -d "$SAMPLE_MODEL_PATH" ]; then
            echo "Missing sampling model checkpoint: $SAMPLE_MODEL_PATH" >&2
            exit 1
        fi
        CUDA_VISIBLE_DEVICES=1 python3 sampling_transformers.py --model-path "$SAMPLE_MODEL_PATH" --start-index $((var)) --num-peptides 1 --output-file ./sampled_output_from_ft/sample_output_transformers_$var.jsonl
        
        ##### & MAKE D_i (initialization data)
        cd sampled_output_from_ft
        python3 make_initialization_data.py \
            --input-jsonl sample_output_transformers_${var}.jsonl \
            --output-init task_${var}_init.txt \
            --output-scores task_${var}_scores.csv
    fi
    
    ##### CONDUCT BAYESIAN OPTIMIZATION AT I-TH TASK 
    cd /home/mok/module/steering/BOLT/optimization/peptides/lolbo_scripts
    CUDA_VISIBLE_DEVICES="1" python info_transformer_vae_optimization.py \
        --task_id "apex" \
        --max_n_oracle_calls 50000 \
        --bsz 50 \
        --constraint_function_ids "[similarity]" \
        --constraint_thresholds "[0.75]" \
        --constraint_types "[$var]" \
        --track_with_wandb False \
        --wandb_entity xxx \
        --wandb_run_tags "[seed_${var}_single_run]" \
        --wandb_run_name "seed_${var}_single_run" \
        --num_initialization_points 1000 \
        --max_string_length 30 \
        --init_n_update_epochs 20 \
        --task_specific_args "[bacteria_0]" \
        --init_offset_helper "$var" \
        --init_data_path /home/mok/module/steering/BOLT/fine-tuning/peptides/sampled_output_from_ft/task_${var}_init.txt \
        --init_scores_path /home/mok/module/steering/BOLT/fine-tuning/peptides/sampled_output_from_ft/task_${var}_scores.csv \
        run_lolbo

    ##### MAKE FINE-TUNING DATASET
    cd /home/mok/module/steering/BOLT/fine-tuning/peptides
    input_csvs=()
    reference_indices=()
    for task_idx in $(seq 0 "$var")
    do
        input_csvs+=("/home/mok/module/steering/BOLT/optimization/peptides/lolbo_scripts/optimization_all_collected_data/BOLT-apex_seed_${task_idx}_single_run_all-data-collected.csv")
        reference_indices+=("$task_idx")
    done
    python3 make_train_data_csv.py \
        --input-csv "${input_csvs[@]}" \
        --reference-index "${reference_indices[@]}" \
        --output-csv ./train_data/train_data_${var}.csv
    TRAIN_JSONL="/home/mok/module/steering/BOLT/fine-tuning/peptides/train_data/train_data_${var}.jsonl"
    python3 generate_openai_ft_data.py \
        --data-path ./train_data/train_data_${var}.csv \
        --save-path "$TRAIN_JSONL"

    ##### TRAIN ON FINE-TUNING DATASET
    PEPTIDE_CKPT_DIR="/home/mok/module/steering/BOLT/fine-tuning/peptides/output/qwen_2_5_3B_lora_output_${var}"
    
    ##### SFT
    CUDA_VISIBLE_DEVICES=1 tune run --nnodes 1 --nproc_per_node 1 lora_finetune_distributed --config torchtune_config/qwen_2_5_3B_lora.yaml output_dir="$PEPTIDE_CKPT_DIR" dataset.data_files="$TRAIN_JSONL" ### 학습할때도 모든 task 다 쓰고
done
