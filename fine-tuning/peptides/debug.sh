# #### DPO TRAINING
var="1"
CKPT_DIR="/home/mok/module/steering/BOLT/fine-tuning/peptides/output/20260716_132046_sft_dpo/checkpoints"
SFT_CKPT_DIR="$CKPT_DIR/qwen_2_5_3B_lora_sft_output_${var}/epoch_4"
DPO_CKPT_DIR="$CKPT_DIR/qwen_2_5_3B_lora_sft_dpo_output_${var}/epoch_0"
# DPO_JSONL="/home/mok/module/steering/BOLT/fine-tuning/peptides/output/20260715_091045_sft_dpo/train_data/dpo_train_data_0_sft_dpo.jsonl"
# DPO_JSONL="/home/mok/module/steering/BOLT/fine-tuning/peptides/output/20260715_080107_sft_dpo/train_data/dpo_train_data_0_sft_dpo.jsonl"

# find_latest_epoch() {
#     local ckpt_root="$1"
#     local latest_epoch
#     latest_epoch="$(find "$ckpt_root" -maxdepth 1 -type d -name 'epoch_*' -printf '%f\n' | sort -V | tail -n 1)"
#     if [ -z "$latest_epoch" ]; then
#         echo "No epoch checkpoint found under: $ckpt_root" >&2
#         return 1
#     fi
#     printf "%s/%s\n" "$ckpt_root" "$latest_epoch"
# }

# SFT_CHECKPOINT="$(find_latest_epoch "$SFT_CKPT_DIR")" || exit 1
# CUDA_VISIBLE_DEVICES=2 tune run --nnodes 1 --nproc_per_node 1 lora_dpo_distributed --config torchtune_config/qwen_2_5_3B_lora_dpo.yaml output_dir="$DPO_CKPT_DIR" checkpointer.checkpoint_dir="$SFT_CHECKPOINT" dataset.data_files="$DPO_JSONL" epochs=5
# FINAL_DPO_CKPT="$(find_latest_epoch "$DPO_CKPT_DIR")" || exit 1

# echo $FINAL_DPO_CKPT

#### INFERENCE
CUDA_VISIBLE_DEVICES=1 python3 sampling_transformers.py \
    --model-path "$SFT_CKPT_DIR" \
    --start-index 0 \
    --num-peptides 5 \
    --max-new-tokens 30 \
    --output-file /home/mok/module/steering/BOLT/fine-tuning/peptides/output/20260716_132046_sft_dpo/sampled_output_from_ft/sft_task${var}_pep0to5.jsonl

CUDA_VISIBLE_DEVICES=1 python3 sampling_transformers.py \
    --model-path "$DPO_CKPT_DIR" \
    --start-index 0 \
    --num-peptides 5 \
    --max-new-tokens 30 \
    --output-file /home/mok/module/steering/BOLT/fine-tuning/peptides/output/20260716_132046_sft_dpo/sampled_output_from_ft/dpo_task${var}_pep0to5.jsonl
