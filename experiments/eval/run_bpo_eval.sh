#!/usr/bin/env bash
# Runbook for the poc20_orpt_mi_bpo_vs_baselines 3-arm comparison
# (BOLT-v2 / ORPT-MI / ORPT-MI-BPO, heldout task_set only): raw proposal
# generation -> incumbent vs pool size (Experiment 1) -> constraint
# violation/duplication ("rejection rate", Experiment 4) -> fixed-target
# pool BO (Experiment 2, real LOLBO -- the expensive step).
#
# This is a literal, saved copy of the manual command sequence -- no new
# dispatch/orchestration logic beyond tracking each background job's exit
# status so a failed step doesn't silently let the next (more expensive)
# step run on bad data.
#
# Run detached so a dropped connection doesn't kill step 4:
#   nohup bash experiments/eval/run_bpo_eval.sh > experiments/eval/logs/bpo_full.log 2>&1 &

set -euo pipefail
cd /workspace/BOLT

MANIFEST=experiments/eval/manifests/poc20_orpt_mi_bpo_vs_baselines.yaml
RESULTS=experiments/eval/results/poc20_orpt_mi_bpo_vs_baselines
RESULTS_GPUS=${RESULTS}__gpu0,${RESULTS}__gpu1,${RESULTS}__gpu2,${RESULTS}__gpu3,${RESULTS}__gpu4,${RESULTS}__gpu5,${RESULTS}__gpu6,${RESULTS}__gpu7
mkdir -p experiments/eval/logs

wait_all() {
    # Waits on every pid passed in; exits the script if any failed.
    local fail=0
    for pid in "$@"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "[run_bpo_eval] one or more jobs failed -- check experiments/eval/logs/" >&2
        exit 1
    fi
}

echo "[run_bpo_eval] step 1: raw proposal generation (8-way GPU parallel)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    nohup python experiments/eval/generate_raw_proposals.py \
        --config peptide_experiment/configs/peptide_poc20_eval_bpo_gpu${gpu}.yaml \
        --manifest experiments/eval/manifests/poc20_orpt_mi_bpo_vs_baselines__gpu${gpu}.yaml \
        > experiments/eval/logs/bpo_step1_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

echo "[run_bpo_eval] step 2: incumbent vs pool size (Experiment 1, no BO, 8-way parallel)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    nohup python experiments/eval/incumbent_vs_pool_size.py \
        --config peptide_experiment/configs/peptide_poc20_eval_bpo_gpu${gpu}.yaml \
        --manifest experiments/eval/manifests/poc20_orpt_mi_bpo_vs_baselines__gpu${gpu}.yaml \
        > experiments/eval/logs/bpo_step2_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir "$RESULTS_GPUS" --out-dir "$RESULTS" \
    > experiments/eval/logs/bpo_step2_combined.log 2>&1

echo "[run_bpo_eval] step 3: held-out rejection rate (Experiment 4, no BO, single process)"
nohup python experiments/eval/violation_duplication_rate.py \
    --config peptide_experiment/configs/peptide_poc20_bolt_v2.yaml \
    --manifest "$MANIFEST" --task-sets heldout \
    > experiments/eval/logs/bpo_step3.log 2>&1 &
wait_all "$!"

echo "[run_bpo_eval] step 4: fixed-target pool BO (Experiment 2, real LOLBO -- expensive, 8-way parallel)"
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    nohup python experiments/eval/fixed_target_rejection_bo.py \
        --config peptide_experiment/configs/peptide_poc20_eval_bpo_gpu${gpu}.yaml \
        --manifest experiments/eval/manifests/poc20_orpt_mi_bpo_vs_baselines__gpu${gpu}.yaml \
        > experiments/eval/logs/bpo_step4_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

python experiments/eval/plot_fixed_target_rejection_bo.py \
    --results-dir "$RESULTS_GPUS" --out-dir "$RESULTS" \
    > experiments/eval/logs/bpo_step4_combined.log 2>&1

echo "[run_bpo_eval] done -- see ${RESULTS}/plots/"
