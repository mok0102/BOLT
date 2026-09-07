#!/usr/bin/env bash
# Training runbook for the paper's own ORPT-family arms (main experiment +
# ablation study) -- NOT baselines, see run_main_baselines_train.sh for
# those (run after this script, since MTBO/OptFormer read BOLT's *completed*
# cumulative trajectory data at each milestone -- racing them concurrently
# with BOLT's own still-in-progress chain would be wrong).
#
# Stage 1 (concurrent, shared 8-GPU pool): BOLT (peptide_main_bolt_v2.yaml)
# and ORPT-H1 (peptide_main_orpt_h1_v2.yaml) trajectory chains, milestones
# [10,20,50,250,400,500,600], lora_rank=16
# (main_qwen_2_5_3B_lora_{bsz32,dpo_bsz32}_rank16.yaml -- bumped from rank=4,
# which produced the old [126..900] run; see that config's own comment for
# why). Both are idempotent/resumable (trajectory_chain.py's own
# per-milestone skip-if-exists checks) -- a dropped connection is safe to
# just re-run.
#
# Stage 2 (after stage 1 frees the 8-GPU pool): the ablation study's ORPT-H0
# arm (peptide_ablation_orpt_h0.yaml, mi_bo_steps=0) -- structurally the same
# matched-intervention trajectory chain as ORPT-H1, just the zero-step
# ablation, not a literature baseline, so it lives here rather than in the
# baselines script.
#
# COST: each trajectory chain interleaves BO sampling with SFT/DPO
# fine-tuning at every one of the 1426-task pool's tasks, training a fresh
# milestone checkpoint at each of the 7 milestones -- expect multi-day
# wall-clock time per chain even with mi_parallel_gpus using all 8 GPUs
# during ORPT's one-step pair-construction stage. Check nvidia-smi fresh
# before launching -- do not assume any GPU id is free (shared machine).
#
# Run detached so a dropped connection doesn't kill it:
#   nohup bash experiments/eval/run_main_bolt_vs_orpt_train.sh \
#       > experiments/eval/logs/main_bolt_vs_orpt_train_full.log 2>&1 &

set -euo pipefail
cd /workspace/BOLT

BOLT_CONFIG=${BOLT_CONFIG:-peptide_experiment/configs/peptide_main_bolt_v2.yaml}
ORPT_CONFIG=${ORPT_CONFIG:-peptide_experiment/configs/peptide_main_orpt_h1_v2.yaml}
ABLATION_CONFIG=${ABLATION_CONFIG:-peptide_experiment/configs/peptide_ablation_orpt_h0.yaml}
mkdir -p experiments/eval/logs

wait_all() {
    # Waits on every pid passed in; exits the script if any failed.
    local fail=0
    for pid in "$@"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "[run_main_bolt_vs_orpt_train] one or more jobs failed -- check experiments/eval/logs/" >&2
        exit 1
    fi
}

echo "[run_main_bolt_vs_orpt_train] stage 1: BOLT + ORPT-H1 trajectory chains (concurrent, shared 8-GPU pool)"
pids=()
nohup python -m peptide_experiment.cli trajectory_chain --config "$BOLT_CONFIG" \
    > experiments/eval/logs/main_bolt_vs_orpt_train_stage1_bolt.log 2>&1 &
pids+=("$!")
nohup python -m peptide_experiment.cli trajectory_chain --config "$ORPT_CONFIG" \
    > experiments/eval/logs/main_bolt_vs_orpt_train_stage1_orpt_h1.log 2>&1 &
pids+=("$!")
wait_all "${pids[@]}"

echo "[run_main_bolt_vs_orpt_train] stage 2: ORPT-H0 ablation arm (after stage 1 frees the 8-GPU pool)"
python -m peptide_experiment.cli trajectory_chain --config "$ABLATION_CONFIG" \
    > experiments/eval/logs/main_bolt_vs_orpt_train_stage2_orpt_h0.log 2>&1

echo "[run_main_bolt_vs_orpt_train] done -- next: experiments/eval/run_main_baselines_train.sh"
