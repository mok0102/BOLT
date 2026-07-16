#!/usr/bin/env bash

cd /home/mok/module/steering/BOLT/optimization/query_plans
source .venv/bin/activate
cd /home/mok/module/steering/BOLT

BASE_CKPT_DIR="/home/mok/module/steering/BOLT/fine-tuning/query_plans/ckpt/Qwen2.5-3B-Instruct"
OUTPUT_ROOT="/home/mok/module/steering/BOLT/fine-tuning/peptides/output"
RUN_ID="$(date +%Y%m%d_%H%M%S)_sft_dpo"
# RUN_ID="$(date +%Y%m%d)_sft_dpo"
RUN_DIR="$OUTPUT_ROOT/$RUN_ID"
SAMPLE_DIR="$RUN_DIR/sampled_output_from_ft"
TRAIN_DIR="$RUN_DIR/train_data"
CKPT_DIR="$RUN_DIR/checkpoints"
COLLECTED_DATA_DIR="$RUN_DIR/optimization_all_collected_data"
CKPT_REGISTRY="$OUTPUT_ROOT/ckpt_registry.tsv"
DPO_CKPT_REGISTRY="$RUN_DIR/ckpt_registry_sft_dpo.tsv"
mkdir -p "$SAMPLE_DIR" "$TRAIN_DIR" "$CKPT_DIR" "$COLLECTED_DATA_DIR"
if [ ! -f "$CKPT_REGISTRY" ]; then
    printf "run_id\tmethod\ttask\tfinal_ckpt\toutput_dir\ttrain_jsonl\n" > "$CKPT_REGISTRY"
fi
printf "run_id\tmethod\ttask\tfinal_ckpt\toutput_dir\ttrain_jsonl\n" > "$DPO_CKPT_REGISTRY"
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

list="0 1 2 3 4 5" #3 4 5
for var in $list
do
    echo "conducting $var -th task now"

    ##### SAMPLE AT FINE-TUNED LLM
    cd /home/mok/module/steering/BOLT/fine-tuning/peptides
    if [ "$var" -eq 0 ]; then
        #### 첫 pi 0는 여기서 준거쓰기. pi 0가 튜닝 안했다고 생각하면 진짜 잘 안됨
        cp /home/mok/module/steering/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_init.txt "$SAMPLE_DIR/task_${var}_init_sft_dpo.txt"
        cp /home/mok/module/steering/BOLT/optimization/peptides/apex_oracle/init_data/seed_0_scores.csv "$SAMPLE_DIR/task_${var}_scores_sft_dpo.csv"
    else
        PREV_VAR=$((var - 1))
        SAMPLE_MODEL_DIR="$CKPT_DIR/qwen_2_5_3B_lora_sft_dpo_output_${PREV_VAR}"
        SAMPLE_MODEL_PATH="$(find_latest_epoch "$SAMPLE_MODEL_DIR")" || exit 1
        if [ ! -d "$SAMPLE_MODEL_PATH" ]; then
            echo "Missing sampling model checkpoint: $SAMPLE_MODEL_PATH" >&2
            exit 1
        fi
        CUDA_VISIBLE_DEVICES=1 python3 sampling_transformers.py --model-path "$SAMPLE_MODEL_PATH" --start-index "$var" --num-peptides 1 --max-new-tokens 30 --output-file "$SAMPLE_DIR/sample_output_transformers_${var}_sft_dpo.jsonl"
        
        ##### & MAKE D_i (initialization data)
        python3 sampled_output_from_ft/make_initialization_data.py \
            --input-jsonl "$SAMPLE_DIR/sample_output_transformers_${var}_sft_dpo.jsonl" \
            --output-init "$SAMPLE_DIR/task_${var}_init_sft_dpo.txt" \
            --output-scores "$SAMPLE_DIR/task_${var}_scores_sft_dpo.csv"
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
        --init_data_path "$SAMPLE_DIR/task_${var}_init_sft_dpo.txt" \
        --init_scores_path "$SAMPLE_DIR/task_${var}_scores_sft_dpo.csv" \
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
        echo "Adding task ${task_idx} to SFT/DPO train data: $input_csv"
        cp "$input_csv" "$COLLECTED_DATA_DIR/"
        input_csvs+=("$input_csv")
        reference_indices+=("$task_idx")
    done

    SFT_CSV="$TRAIN_DIR/train_data_${var}_sft_dpo.csv"
    SFT_JSONL="$TRAIN_DIR/train_data_${var}_sft_dpo.jsonl"
    python3 make_train_data_csv.py \
        --input-csv "${input_csvs[@]}" \
        --reference-index "${reference_indices[@]}" \
        --top-n 10000 \
        --train-n 1000 \
        --seed "$var" \
        --output-csv "$SFT_CSV"
    python3 generate_openai_ft_data.py \
        --data-path "$SFT_CSV" \
        --save-path "$SFT_JSONL"

    ##### TRAIN ON FINE-TUNING DATASET
    SFT_CKPT_DIR="$CKPT_DIR/qwen_2_5_3B_lora_sft_output_${var}"
    DPO_CKPT_DIR="$CKPT_DIR/qwen_2_5_3B_lora_sft_dpo_output_${var}"
    
    ##### SFT
    CUDA_VISIBLE_DEVICES=1 tune run --nnodes 1 --nproc_per_node 1 lora_finetune_distributed --config torchtune_config/qwen_2_5_3B_lora.yaml output_dir="$SFT_CKPT_DIR" dataset.data_files="$SFT_JSONL"
    FINAL_SFT_CKPT="$(find_latest_epoch "$SFT_CKPT_DIR")" || exit 1
    printf "%s\tsft_before_dpo\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$var" "$FINAL_SFT_CKPT" "$SFT_CKPT_DIR" "$SFT_JSONL" >> "$DPO_CKPT_REGISTRY"
    printf "%s\tsft_before_dpo\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$var" "$FINAL_SFT_CKPT" "$SFT_CKPT_DIR" "$SFT_JSONL" >> "$CKPT_REGISTRY"
    echo "Registered SFT-before-DPO checkpoint: $FINAL_SFT_CKPT"

    DPO_CSV="$TRAIN_DIR/dpo_train_data_${var}_sft_dpo.csv"
    DPO_JSONL="$TRAIN_DIR/dpo_train_data_${var}_sft_dpo.jsonl"
    python3 make_dpo_train_data_csv.py \
        --input-csv "${input_csvs[@]}" \
        --reference-index "${reference_indices[@]}" \
        --min-score-gap 50.0 \
        --output-csv "$DPO_CSV" \
        --output-jsonl "$DPO_JSONL"

    ##### DPO
    CUDA_VISIBLE_DEVICES=1 tune run --nnodes 1 --nproc_per_node 1 lora_dpo_distributed --config torchtune_config/qwen_2_5_3B_lora_dpo.yaml output_dir="$DPO_CKPT_DIR" checkpointer.checkpoint_dir="$FINAL_SFT_CKPT" dataset.data_files="$DPO_JSONL" epochs=1
    FINAL_DPO_CKPT="$(find_latest_epoch "$DPO_CKPT_DIR")" || exit 1
    printf "%s\tdpo\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$var" "$FINAL_DPO_CKPT" "$DPO_CKPT_DIR" "$DPO_JSONL" >> "$DPO_CKPT_REGISTRY"
    printf "%s\tdpo\t%s\t%s\t%s\t%s\n" "$RUN_ID" "$var" "$FINAL_DPO_CKPT" "$DPO_CKPT_DIR" "$DPO_JSONL" >> "$CKPT_REGISTRY"
    echo "Registered DPO checkpoint: $FINAL_DPO_CKPT"
done
