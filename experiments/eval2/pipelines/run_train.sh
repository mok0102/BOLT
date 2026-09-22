#!/usr/bin/env bash
# Training runbook for the paper's own ORPT-family arms (main experiment +
# ablation study) -- NOT baselines, see run_baselines_train.sh for
# those (run after this script, since MTBO/OptFormer read BOLT's *completed*
# cumulative trajectory data at each milestone -- racing them concurrently
# with BOLT's own still-in-progress chain would be wrong).
#
# Two disjoint 4-GPU pipelines run concurrently (2026-09 restructure -- the
# previous "BOLT+ORPT-H1 share all 8 GPUs, then ORPT-H0 alone after" layout
# let BOLT_CONFIG (no cuda_visible_devices/mi_parallel_gpus of its own) and
# ORPT_CONFIG (explicit 8-GPU claim) silently contend for the same physical
# GPUs when actually run concurrently, and made ORPT-H0 wait for ORPT-H1's
# entire multi-day chain to finish even though the two share no data
# dependency):
#
#   Pipeline A (GPUs 0-3, peptide_main_bolt.yaml's own mi_parallel_gpus/
#   cuda_visible_devices): BOLT_CONFIG's chain, then -- once it frees this
#   4-GPU slice -- ABLATION_CONFIG's chain (peptide_ablation_orpt_h0.yaml,
#   mi_bo_steps=0, the zero-step ablation). These are independent
#   experiment_ids/trajectory chains (BOLT_CONFIG's own BOLT-<m> checkpoints
#   are NOT what ABLATION_CONFIG's chain trains against -- each arm builds
#   its own from task 0), sequenced here purely to avoid GPU contention.
#
#   Pipeline B (GPUs 4-7, peptide_main_orpt_h1.yaml's own mi_parallel_gpus/
#   cuda_visible_devices): ORPT_CONFIG's chain alone -- the paper's actual
#   method (H0-AND-H1: a preference pair requires BOTH the one-step-BO
#   winner (H1, mi_bo_steps=1) AND the raw pre-rollout y winner (H0) to
#   agree, via mi_require_h0, default true -- see
#   peptide_experiment/mi_orpt/pair_construction.py).
#
# Both pipelines are idempotent/resumable (trajectory_chain.py's own
# per-milestone skip-if-exists checks) -- a dropped connection is safe to
# just re-run. See each config's own comment for its reduced-budget cost
# knobs (mi_num_backgrounds=4, mi_max_candidates_per_task=16,
# mi_max_tasks_per_milestone=50 -- all three must match between
# ORPT_CONFIG/ABLATION_CONFIG for a controlled H0-vs-H1 comparison).
#
# COST: each trajectory chain interleaves BO sampling with SFT/DPO
# fine-tuning at every training task, training a fresh milestone checkpoint
# at each of the 7 milestones. init_size=100/oracle_budget=500 (2026-09,
# down from init_size=1000/oracle_budget=20000 -- see peptide_main_bolt.yaml's
# comment) keeps a single BO run cheap, but wall-clock time hasn't been
# measured end-to-end at full scale in this container yet -- time
# peptide_smoke.yaml first (see peptide_experiment/README.md's Quick start)
# before launching this. Check nvidia-smi fresh before launching -- do not
# assume GPUs 0-7 are free (shared machine); if they aren't, override
# BOLT_CONFIG/ORPT_CONFIG/ABLATION_CONFIG's own mi_parallel_gpus/
# cuda_visible_devices fields to a free slice instead of assuming this
# script's hardcoded 0-3/4-7 split still applies.
#
# Run detached so a dropped connection doesn't kill it:
#   nohup bash experiments/eval2/pipelines/run_train.sh \
#       > experiments/eval2/logs/train_full.log 2>&1 &

set -euo pipefail
cd /workspace/BOLT

BOLT_CONFIG=${BOLT_CONFIG:-peptide_experiment/configs/peptide_main_bolt.yaml}
ORPT_CONFIG=${ORPT_CONFIG:-peptide_experiment/configs/peptide_main_orpt_h1.yaml}
ABLATION_CONFIG=${ABLATION_CONFIG:-peptide_experiment/configs/peptide_ablation_orpt_h0.yaml}
mkdir -p experiments/eval2/logs

wait_all() {
    # Waits on every pid passed in; exits the script if any failed.
    local fail=0
    for pid in "$@"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "[run_train] one or more jobs failed -- check experiments/eval2/logs/" >&2
        exit 1
    fi
}

echo "[run_train] Pipeline A (GPUs 0-3): BOLT -> ORPT-H0 ablation, sequential"
nohup bash -c "
    python -m peptide_experiment.cli trajectory_chain --config '$BOLT_CONFIG' \
        > experiments/eval2/logs/train_pipelineA_bolt.log 2>&1 &&
    python -m peptide_experiment.cli trajectory_chain --config '$ABLATION_CONFIG' \
        > experiments/eval2/logs/train_pipelineA_orpt_h0.log 2>&1
" &
pipeline_a_pid=$!

echo "[run_train] Pipeline B (GPUs 4-7): ORPT-H1 alone"
nohup python -m peptide_experiment.cli trajectory_chain --config "$ORPT_CONFIG" \
    > experiments/eval2/logs/train_pipelineB_orpt_h1.log 2>&1 &
pipeline_b_pid=$!

wait_all "$pipeline_a_pid" "$pipeline_b_pid"

echo "[run_train] done -- next: experiments/eval2/pipelines/run_baselines_train.sh"
