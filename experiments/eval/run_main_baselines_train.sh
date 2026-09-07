#!/usr/bin/env bash
# Training runbook for the main experiment's literature baselines (MTBO,
# OptFormer, GP-expert-transfer/POGPE/SGPE) -- run AFTER
# run_main_bolt_vs_orpt_train.sh's BOLT trajectory chain has completed all 7
# milestones: MTBO/OptFormer both train on BOLT's *completed* cumulative
# trajectory data at each milestone (train_mtbo_surrogate(cfg, m) etc.), so
# racing them concurrently with BOLT's own still-in-progress chain would try
# to train e.g. MTBO-600 before BOLT has even reached task 600.
#
# Sequential, cheap, single GPU -- no concurrency needed (unlike
# run_main_bolt_vs_orpt_train.sh's BOLT/ORPT-H1 stage). All three train into
# $BOLT_CONFIG's own run_dir (runs/peptide_main_bolt_v2), matching the old
# eval runbook's step 0 exactly (just moved here since training and eval are
# now separate concerns).
#
# Sanity-check before launching:
#   ls runs/peptide_main_bolt_v2/checkpoints/  # expect BOLT-10 .. BOLT-600 (all 7)
#
# Run detached:
#   nohup bash experiments/eval/run_main_baselines_train.sh \
#       > experiments/eval/logs/main_baselines_train_full.log 2>&1 &

set -euo pipefail
cd /workspace/BOLT

BOLT_CONFIG=${BOLT_CONFIG:-peptide_experiment/configs/peptide_main_bolt_v2.yaml}
mkdir -p experiments/eval/logs

echo "[run_main_baselines_train] train MTBO (shared surrogate)"
python -m peptide_experiment.cli train_mtbo --config "$BOLT_CONFIG" \
    > experiments/eval/logs/main_baselines_train_mtbo.log 2>&1

echo "[run_main_baselines_train] train GP-expert-transfer (POGPE/SGPE, n_experts=5/10/20)"
python -m peptide_experiment.cli train_gp_expert_transfer --config "$BOLT_CONFIG" \
    > experiments/eval/logs/main_baselines_train_gp_expert_transfer.log 2>&1

echo "[run_main_baselines_train] train OptFormer"
python -m peptide_experiment.cli train_optformer --config "$BOLT_CONFIG" \
    > experiments/eval/logs/main_baselines_train_optformer.log 2>&1

echo "[run_main_baselines_train] done -- next: experiments/eval/run_main_bolt_vs_orpt_mi_eval.sh"
