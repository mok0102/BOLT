#!/usr/bin/env bash

cd /home/mok/module/steering/BOLT/optimization/query_plans
source .venv/bin/activate
cd /home/mok/module/steering/BOLT

BASE_CKPT_DIR="/home/mok/module/steering/BOLT/fine-tuning/query_plans/ckpt/Qwen2.5-3B-Instruct"
OUTPUT_ROOT="/home/mok/module/steering/BOLT/fine-tuning/peptides/output"
RUN_ID="$(date +%Y%m%d_%H%M%S)_only_sft"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
SAMPLE_DIR="$RUN_DIR/sampled_output_from_ft"
TRAIN_DIR="$RUN_DIR/train_data"
CKPT_DIR="$RUN_DIR/checkpoints"
COLLECTED_DATA_DIR="$RUN_DIR/optimization_all_collected_data"
CKPT_REGISTRY="$OUTPUT_ROOT/ckpt_registry.tsv"
SFT_CKPT_REGISTRY="$RUN_DIR/ckpt_registry_only_sft.tsv"
mkdir -p "$SAMPLE_DIR" "$TRAIN_DIR" "$CKPT_DIR" "$COLLECTED_DATA_DIR"
if [ ! -f "$CKPT_REGISTRY" ]; then
    printf "run_id\tmethod\ttask\tfinal_ckpt\toutput_dir\ttrain_jsonl\n" > "$CKPT_REGISTRY"
fi
printf "run_id\tmethod\ttask\tfinal_ckpt\toutput_dir\ttrain_jsonl\n" > "$SFT_CKPT_REGISTRY"
echo "Writing this run under: $RUN_DIR"

find_latest_epoch() {
    local ckpt_root="$1"
    local latest_epoch
    latest_epoch="$(find "$ckpt_root" -maxdepth 1 -type d -name 'epoch_*' -printf '%f\n' | sort -V | tail -n 1)"
    if [ -z "$latest_epoch" ]; then
        echo "No epoch checkpoint found under: $ckpt_root" >&2
        return 1
    fi
    printf "%s/%s\n" "$ckpt_root" "$latest_epoch"
}

list="0 1 2" # 3 4 5
for var in $list
do
    echo "conducting $var -th task now"

    ##### SAMPLE AT FINE-TUNED LLM
    cd /home/mok/module/steering/BOLT/fine-tuning/peptides
    if [ "$var" -eq 0 ]; then
        #### 첫 pi_0는 여기서 준거쓰기. pi_0가 튜닝 안했다고 생각하면 진짜 잘 안됨
        cp /home/mok/module/steering/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_init.txt "$SAMPLE_DIR/task_${var}_init_only_sft.txt"
        cp /home/mok/module/steering/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_scores.csv "$SAMPLE_DIR/task_${var}_scores_only_sft.csv"
    else
        PREV_VAR=$((var - 1))
        SAMPLE_MODEL_PATH="$(find_latest_epoch "$CKPT_DIR/qwen_2_5_3B_lora_output_${PREV_VAR}_only_sft")" || exit 1
        if [ ! -d "$SAMPLE_MODEL_PATH" ]; then
            echo "Missing sampling model checkpoint: $SAMPLE_MODEL_PATH" >&2
            exit 1
        fi
        CUDA_VISIBLE_DEVICES=1 python3 sampling_transformers.py --model-path "$SAMPLE_MODEL_PATH" --start-index $((var)) --num-peptides 1 --output-file "$SAMPLE_DIR/sample_output_transformers_${var}_only_sft.jsonl"
        
        ##### & MAKE D_i (initialization data)
        python3 sampled_output_from_ft/make_initialization_data.py \
            --input-jsonl "$SAMPLE_DIR/sample_output_transformers_${var}_only_sft.jsonl" \
            --output-init "$SAMPLE_DIR/task_${var}_init_only_sft.txt" \
            --output-scores "$SAMPLE_DIR/task_${var}_scores_only_sft.csv"
    fi
    
    ##### CONDUCT BAYESIAN OPTIMIZATION AT I-TH TASK 
    cd /home/mok/module/steering/BOLT/optimization/peptides/lolbo_scripts
    CUDA_VISIBLE_DEVICES="1" python info_transformer_vae_optimization.py \
        --task_id "apex" \
        --max_n_oracle_calls 10000 \
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
        --init_data_path "$SAMPLE_DIR/task_${var}_init_only_sft.txt" \
        --init_scores_path "$SAMPLE_DIR/task_${var}_scores_only_sft.csv" \
        run_lolbo

    ##### MAKE FINE-TUNING DATASET
    cd /home/mok/module/steering/BOLT/fine-tuning/peptides
    input_csvs=()
    reference_indices=()
    for task_idx in $(seq 0 "$var")
    do
        input_csv="/home/mok/module/steering/BOLT/optimization/peptides/lolbo_scripts/optimization_all_collected_data/BOLT-apex_seed_${task_idx}_single_run_all-data-collected.csv"
        if [ ! -f "$input_csv" ]; then
            echo "Missing collected data CSV: $input_csv" >&2
            exit 1
        fi
        echo "Adding task ${task_idx} to SFT train data: $input_csv"
        cp "$input_csv" "$COLLECTED_DATA_DIR/"
        input_csvs+=("$input_csv")
        reference_indices+=("$task_idx")
    done
    python3 make_train_data_csv.py \
        --input-csv "${input_csvs[@]}" \
        --reference-index "${reference_indices[@]}" \
        --output-csv "$TRAIN_DIR/train_data_${var}_only_sft.csv"
    TRAIN_JSONL="$TRAIN_DIR/train_data_${var}_only_sft.jsonl"
    python3 generate_openai_ft_data.py \
        --data-path "$TRAIN_DIR/train_data_${var}_only_sft.csv" \
        --save-path "$TRAIN_JSONL"

    ##### TRAIN ON FINE-TUNING DATASET
    PEPTIDE_CKPT_DIR="$CKPT_DIR/qwen_2_5_3B_lora_output_${var}_only_sft"
    
    ##### SFT
    CUDA_VISIBLE_DEVICES=1 tune run --nnodes 1 --nproc_per_node 1 lora_finetune_distributed --config torchtune_config/qwen_2_5_3B_lora.yaml output_dir="$PEPTIDE_CKPT_DIR" dataset.data_files="$TRAIN_JSONL" ### 학습할때도 모든 task 다 쓰고
    FINAL_SFT_CKPT="$(find_latest_epoch "$PEPTIDE_CKPT_DIR")" || exit 1
    printf "%s\tsft_only\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$var" "$FINAL_SFT_CKPT" "$PEPTIDE_CKPT_DIR" "$TRAIN_JSONL" >> "$SFT_CKPT_REGISTRY"
    printf "%s\tsft_only\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$var" "$FINAL_SFT_CKPT" "$PEPTIDE_CKPT_DIR" "$TRAIN_JSONL" >> "$CKPT_REGISTRY"
    echo "Registered SFT-only checkpoint: $FINAL_SFT_CKPT"
done
