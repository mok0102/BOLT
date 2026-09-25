#!/usr/bin/env bash
# Evaluation runbook for the main experiment (BOLT vs ORPT-H1 vs baselines)
# plus, optionally, the H0-vs-H1 ablation:
#   step 1  raw proposal generation      (LLM sampling, 7-way GPU parallel)
#   step 2  incumbent vs pool size       (no BO, cheap -- fig:fewshot data)
#   step 3  fixed-target pool BO         (real LOLBO, expensive -- fig:main-bo data)
#   step 3b baseline text reports        (POGPE/SGPE by n_experts, LLAMBO truncation)
#   step 4  paper figures                (fig:main-bo, fig:fewshot, fig:scaling)
#   step 5  ablation                     (tab:ablation, only when RUN_ABLATION=1)
#
# PREREQUISITE: training is NOT this script's job -- run, in order:
#   1. experiments/eval2/pipelines/run_train.sh            (BOLT + ORPT-H1 + ORPT-H0)
#   2. experiments/eval2/pipelines/run_baselines_train.sh  (MTBO/OptFormer/GP experts)
# Sanity-check before launching:
#   ls runs/peptide_main_bolt/checkpoints/        # expect BOLT-10 .. BOLT-600
#   ls runs/peptide_main_orpt_h1/checkpoints/     # expect ORPT-10 .. ORPT-600
#   ls runs/peptide_ablation_orpt_h0/checkpoints/ # expect ORPT-10 .. ORPT-600
#
# COST WARNING: step 3 is (BOLT+ORPT-H1+OptFormer+MTBO) x 7 milestones x N
# tasks + (POGPE+SGPE) x 3 expert-counts x N tasks + (STBO+LLAMBO) x N tasks,
# each one a real BO run to the full oracle budget -- likely multiple days of
# wall clock even 8-way parallel. Sanity-check on a slice FIRST (see the
# commented-out smoke command under step 3), and check step 2's cheap
# coverage_rate_at_n_proposals output before committing: whether every arm
# can even reach TARGET_POOL_SIZES feasible candidates is an open empirical
# question for the LLM arms. (The self-seeding baselines always reach it by
# construction, so coverage isn't a question for them.)
#
# Sharding: all 8 GPUs. gpu{0..6} each take one milestone of the LLM/
# milestone-indexed arms; gpu7 takes the milestone-independent baselines
# (POGPE/SGPE overload the milestone field to mean n_experts, so they cannot
# share the real milestone axis -- see arm_specs/main.yaml).
#
# Run detached so a dropped connection doesn't kill step 3:
#   nohup bash experiments/eval2/pipelines/run_eval.sh \
#       > experiments/eval2/logs/main_eval_full.log 2>&1 &

set -euo pipefail
cd /workspace/BOLT

CLI="python -m experiments.eval2.cli"

ARMS_FILE=${ARMS_FILE:-experiments/eval2/arm_specs/main.yaml}
CONFIG=${CONFIG:-peptide_experiment/configs/peptide_main_bolt.yaml}
RUN_TAG=${RUN_TAG:-main}
TASK_SET=${TASK_SET:-heldout50}

# Arms that carry the real milestone axis, sharded one milestone per GPU.
MILESTONE_ARMS=${MILESTONE_ARMS:-BOLT,ORPT-H1,OptFormer,MTBO}
MILESTONES=(10 20 50 250 400 500 600)
# Arms with no real milestone axis, all on the last GPU.
BASELINE_ARMS=${BASELINE_ARMS:-STBO,POGPE,SGPE,LLAMBO}
BASELINE_GPU=${BASELINE_GPU:-7}

# == cfg.init_size / cfg.oracle_budget for CONFIG above. Kept to a single
# target: step 3's cost is dominated by the oracle budget (fixed regardless
# of target), so extra targets multiply runtime for a sensitivity panel the
# paper doesn't use.
TARGET_POOL_SIZES=${TARGET_POOL_SIZES:-100}
BO_CALLS=${BO_CALLS:-500}

RUN_ABLATION=${RUN_ABLATION:-0}
ABLATION_ARMS_FILE=${ABLATION_ARMS_FILE:-experiments/eval2/arm_specs/ablation.yaml}
ABLATION_TAG=${ABLATION_TAG:-ablation_h0_vs_h1}

RESULTS=experiments/eval2/results/${RUN_TAG}
FIGURES=experiments/eval2/results/${RUN_TAG}/paper_figures
LOGS=experiments/eval2/logs
mkdir -p "$LOGS"

# Comma-joined shard dirs: figures read all of them at once.
RESULTS_SHARDS=""
for gpu in 0 1 2 3 4 5 6; do
    RESULTS_SHARDS="${RESULTS_SHARDS:+${RESULTS_SHARDS},}${RESULTS}__gpu${gpu}"
done
RESULTS_SHARDS_ALL="${RESULTS_SHARDS},${RESULTS}__gpu${BASELINE_GPU}"

wait_all() {
    # Waits on every pid passed in; exits the script if any failed.
    local fail=0
    for pid in "$@"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "[run_eval] one or more jobs failed -- check ${LOGS}/" >&2
        exit 1
    fi
}

echo "[run_eval] step 1: raw proposal generation (7-way GPU parallel, task_set=${TASK_SET}) -- only arms with a real LLM checkpoint; self-seeding baselines have nothing to generate"
pids=()
for gpu in 0 1 2 3 4 5 6; do
    nohup $CLI generate_raw \
        --config "$CONFIG" --arms-file "$ARMS_FILE" \
        --arms "$MILESTONE_ARMS" --milestones "${MILESTONES[$gpu]}" \
        --cuda-visible-devices "$gpu" --task-sets "$TASK_SET" \
        > "${LOGS}/${RUN_TAG}_step1_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

echo "[run_eval] step 2: incumbent vs pool size (no BO, 7-way parallel)"
pids=()
for gpu in 0 1 2 3 4 5 6; do
    nohup $CLI incumbent \
        --config "$CONFIG" --arms-file "$ARMS_FILE" \
        --arms "$MILESTONE_ARMS" --milestones "${MILESTONES[$gpu]}" \
        --cuda-visible-devices "$gpu" --task-sets "$TASK_SET" \
        --out-dir "${RESULTS}__gpu${gpu}" \
        > "${LOGS}/${RUN_TAG}_step2_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

echo "[run_eval] step 3: fixed-target pool BO (real LOLBO -- expensive, 8-way parallel)"
# Sanity check first, on one shard, before the full sweep (uncomment):
#   $CLI fixed_target_bo --config "$CONFIG" --arms-file "$ARMS_FILE" \
#       --arms BOLT --milestones 10 --cuda-visible-devices 0 \
#       --task-sets "$TASK_SET" --target-pool-sizes 10 --limit-tasks 1 \
#       --out-dir "${RESULTS}__smoke"
pids=()
for gpu in 0 1 2 3 4 5 6; do
    nohup $CLI fixed_target_bo \
        --config "$CONFIG" --arms-file "$ARMS_FILE" \
        --arms "$MILESTONE_ARMS" --milestones "${MILESTONES[$gpu]}" \
        --cuda-visible-devices "$gpu" --task-sets "$TASK_SET" \
        --target-pool-sizes "$TARGET_POOL_SIZES" \
        --out-dir "${RESULTS}__gpu${gpu}" \
        > "${LOGS}/${RUN_TAG}_step3_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
nohup $CLI fixed_target_bo \
    --config "$CONFIG" --arms-file "$ARMS_FILE" \
    --arms "$BASELINE_ARMS" \
    --cuda-visible-devices "$BASELINE_GPU" --task-sets "$TASK_SET" \
    --target-pool-sizes "$TARGET_POOL_SIZES" \
    --out-dir "${RESULTS}__gpu${BASELINE_GPU}" \
    > "${LOGS}/${RUN_TAG}_step3_gpu${BASELINE_GPU}.log" 2>&1 &
pids+=("$!")
wait_all "${pids[@]}"

echo "[run_eval] step 3b: POGPE/SGPE report (own axis is n_experts, not milestone) + LLAMBO token-budget-truncation report"
$CLI baselines_report \
    --config "$CONFIG" --results-dir "$RESULTS_SHARDS_ALL" --bo-calls "$BO_CALLS" \
    | tee "${LOGS}/${RUN_TAG}_step3b_baselines_report.log"

echo "[run_eval] step 4: paper figures (fig:main-bo, fig:fewshot, fig:scaling)"
FINAL_MILESTONE=${MILESTONES[${#MILESTONES[@]}-1]}

RUN_DIR_ARGS=()
for pair in \
    "BOLT=runs/peptide_main_bolt" \
    "ORPT-H1=runs/peptide_main_orpt_h1" \
    "STBO=runs/peptide_main_bolt" \
    "MTBO=runs/peptide_main_bolt" \
    "POGPE=runs/peptide_main_bolt" \
    "SGPE=runs/peptide_main_bolt" \
    "OptFormer=runs/peptide_main_bolt" \
    "LLAMBO=runs/peptide_main_bolt"
do
    RUN_DIR_ARGS+=(--run-dir "$pair")
done

$CLI main_bo \
    --config "$CONFIG" --milestone "$FINAL_MILESTONE" "${RUN_DIR_ARGS[@]}" \
    --task-set "$TASK_SET" --target-pool-size "$TARGET_POOL_SIZES" --out-dir "$FIGURES" \
    > "${LOGS}/${RUN_TAG}_step4_main_bo.log" 2>&1

$CLI fewshot \
    --milestone "$FINAL_MILESTONE" --results-dir "$RESULTS_SHARDS_ALL" \
    --task-set "$TASK_SET" --out-dir "$FIGURES" \
    > "${LOGS}/${RUN_TAG}_step4_fewshot.log" 2>&1

$CLI scaling \
    --results-dir "$RESULTS_SHARDS_ALL" --task-set "$TASK_SET" --out-dir "$FIGURES" \
    > "${LOGS}/${RUN_TAG}_step4_scaling.log" 2>&1

if [ "$RUN_ABLATION" = "1" ]; then
    echo "[run_eval] step 5: ablation (tab:ablation) -- idempotent; BOLT/ORPT-H1 rows reuse steps 1-3's output above (same run_dir/checkpoints), only ORPT-H0 does real new work here"
    ABLATION_RESULTS=experiments/eval2/results/${ABLATION_TAG}
    $CLI generate_raw \
        --config "$CONFIG" --arms-file "$ABLATION_ARMS_FILE" --task-sets "$TASK_SET" \
        > "${LOGS}/${ABLATION_TAG}_step1.log" 2>&1
    $CLI incumbent \
        --config "$CONFIG" --arms-file "$ABLATION_ARMS_FILE" --task-sets "$TASK_SET" \
        --out-dir "$ABLATION_RESULTS" \
        > "${LOGS}/${ABLATION_TAG}_step2.log" 2>&1
    $CLI fixed_target_bo \
        --config "$CONFIG" --arms-file "$ABLATION_ARMS_FILE" --task-sets "$TASK_SET" \
        --target-pool-sizes "$TARGET_POOL_SIZES" --out-dir "$ABLATION_RESULTS" \
        > "${LOGS}/${ABLATION_TAG}_step3.log" 2>&1
    $CLI ablation \
        --milestone "$FINAL_MILESTONE" --results-dir "$ABLATION_RESULTS" \
        --task-set "$TASK_SET" --out-dir "$FIGURES" \
        > "${LOGS}/${ABLATION_TAG}_step4.log" 2>&1
else
    echo "[run_eval] step 5: skipped (RUN_ABLATION=0)"
fi

echo "[run_eval] done -- figures in ${FIGURES}, summary CSVs in ${RESULTS}__gpu*"
