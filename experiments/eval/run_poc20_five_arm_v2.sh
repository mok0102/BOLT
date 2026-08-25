#!/usr/bin/env bash
# Runbook for the poc20_five_arm_v2 comparison (BOLT / MTBO / OptFormer /
# ORPT-MI / ORPT-MI-BPO, heldout task_set): raw proposal generation ->
# incumbent vs pool size (Experiment 1) -> constraint violation/duplication
# rate (Experiment 4) -> fixed-target pool BO (Experiment 2) -> fixed-budget
# rejection-sampled BO (Experiment 3, real LOLBO -- the expensive step).
#
# Literal, saved copy of the manual command sequence from
# experiments/eval/README.md -- no GPU-chunked parallelism (unlike
# run_bpo_eval.sh), single process per step. set -e stops the script if any
# step fails, so a bad cheap step doesn't waste time on the expensive ones.
#
# Run detached so a dropped connection doesn't kill the expensive steps:
#   nohup bash experiments/eval/run_poc20_five_arm_v2.sh \
#       > experiments/eval/logs/five_arm_v2_full.log 2>&1 &

set -euo pipefail
cd /workspace/BOLT

CONFIG=peptide_experiment/configs/peptide_poc20_bolt.yaml
MANIFEST=experiments/eval/manifests/poc20_five_arm_v2.yaml
RESULTS=experiments/eval/results/poc20_five_arm_v2
mkdir -p experiments/eval/logs

echo "[run_poc20_five_arm_v2] step 0: raw proposal generation (fills in OptFormer + ORPT-MI-BPO only; BOLT/ORPT-MI already have raw proposals)"
python experiments/eval/generate_raw_proposals.py \
    --config "$CONFIG" --manifest "$MANIFEST" \
    > experiments/eval/logs/five_arm_v2_step0.log 2>&1

echo "[run_poc20_five_arm_v2] step 1: incumbent vs pool size (Experiment 1, no BO)"
python experiments/eval/incumbent_vs_pool_size.py \
    --config "$CONFIG" --manifest "$MANIFEST" \
    > experiments/eval/logs/five_arm_v2_step1.log 2>&1
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir "$RESULTS" \
    > experiments/eval/logs/five_arm_v2_step1_plot.log 2>&1

echo "[run_poc20_five_arm_v2] step 4: violation/duplication rate (Experiment 4, no BO)"
python experiments/eval/violation_duplication_rate.py \
    --config "$CONFIG" --manifest "$MANIFEST" --n 500 \
    > experiments/eval/logs/five_arm_v2_step4.log 2>&1

echo "[run_poc20_five_arm_v2] step 2: fixed-target pool BO (Experiment 2, real LOLBO -- expensive)"
python experiments/eval/fixed_target_rejection_bo.py \
    --config "$CONFIG" --manifest "$MANIFEST" \
    > experiments/eval/logs/five_arm_v2_step2.log 2>&1
python experiments/eval/plot_fixed_target_rejection_bo.py \
    --results-dir "$RESULTS" \
    > experiments/eval/logs/five_arm_v2_step2_plot.log 2>&1

echo "[run_poc20_five_arm_v2] step 3: fixed-budget rejection-sampled BO (Experiment 3, real LOLBO -- expensive)"
python experiments/eval/fixed_budget_rejection_bo.py \
    --config "$CONFIG" --manifest "$MANIFEST" \
    > experiments/eval/logs/five_arm_v2_step3.log 2>&1
python experiments/eval/plot_fixed_budget_rejection_bo.py \
    --results-dir "$RESULTS" \
    > experiments/eval/logs/five_arm_v2_step3_plot.log 2>&1

echo "[run_poc20_five_arm_v2] done -- see ${RESULTS}/plots/"
