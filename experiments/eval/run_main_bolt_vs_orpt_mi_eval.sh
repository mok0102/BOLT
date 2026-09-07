#!/usr/bin/env bash
# Runbook for the main-experiment (paper-fidelity scale) BOLT-vs-ORPT-MI-vs-
# baselines comparison (runs/peptide_main_bolt vs runs/peptide_main_orpt_mi,
# plus OptFormer/MTBO/POGPE/SGPE/LLAMBO baselines trained into
# runs/peptide_main_bolt, 100-task held-out set): baseline training ->
# raw proposal generation -> incumbent vs pool size (the "few-shot"/
# init-only-style table, BOLT/ORPT-MI/OptFormer only) -> fixed-target pool BO
# (the scaling curve at a fixed init-pool size -- real LOLBO, the expensive
# step, all 6 arms).
#
# PREREQUISITE: only run this once BOLT training (runs/peptide_main_bolt) has
# reached milestone 900 (all 7 milestones: [126,252,378,504,630,756,900]).
# ORPT-MI (runs/peptide_main_orpt_mi) already has all 7 milestones as of this
# writing. Sanity-check before launching:
#   ls runs/peptide_main_bolt/checkpoints/    # expect BOLT-126 .. BOLT-900
#   ls runs/peptide_main_orpt_mi/checkpoints/ # expect ORPT-126 .. ORPT-900
#
# COST WARNING: with the 100-task held-out set, step 3 (fixed-target BO) is
# now (BOLT+ORPT-MI+OptFormer+MTBO) x 7 milestones x 100 tasks
# + (POGPE+SGPE) x 3 expert-counts x 100 tasks + LLAMBO x 100 tasks, all at
# 1 target pool size (1000 == this config's cfg.init_size, matching the
# paper's own Table 11/Figure 1-2 scale) = up to ~3500 real full-budget
# (oracle_budget=20000) LOLBO/BO runs (up from ~1400 for BOLT+ORPT-MI alone).
# Based on this run's own observed per-task BO timing (~4-10 min/task at
# this budget), that's roughly 234-584 GPU-hours of aggregate compute --
# split across 8 parallel GPU shards below (gpu7 newly added, previously
# idle), still likely multiple days of wall-clock time. LLAMBO in practice
# likely uses less than its share of that budget -- see llambo_max_input_tokens
# note below. Sanity-check on a small slice FIRST (see the commented-out
# smoke command under step 3) before launching the full sweep -- and note
# that whether target=1000 is even achievable (coverage_rate) for every arm
# is an open empirical question; step 2's own coverage_rate_at_n_proposals
# output (cheap, no BO) is the place to check this for BOLT/ORPT-MI/OptFormer
# before committing to step 3 (MTBO/POGPE/SGPE/LLAMBO self-seed their init
# pool directly and always achieve target=1000 by construction, so coverage
# isn't a question for them).
#
# This is a literal, saved copy of the manual command sequence -- no new
# dispatch/orchestration logic beyond tracking each background job's exit
# status so a failed step doesn't silently let the next (more expensive)
# step run on bad data. Mirrors experiments/eval/run_bpo_eval.sh's own
# structure/conventions.
#
# All 8 GPUs are used: gpu{0-6} for the 7 real milestones (BOLT/ORPT-MI/
# OptFormer/MTBO), gpu7 for the milestone-independent baselines (POGPE/SGPE/
# LLAMBO -- see manifests/main_bolt_vs_orpt_mi__gpu7.yaml's own comment for
# why these can't share the real milestone axis).
#
# Run detached so a dropped connection doesn't kill step 3:
#   nohup bash experiments/eval/run_main_bolt_vs_orpt_mi_eval.sh \
#       > experiments/eval/logs/main_bolt_vs_orpt_mi_full.log 2>&1 &
#
# Drives the [126..900] schedule above by default. To drive the new v2
# schedule ([10,20,50,250,400,500,600], once peptide_main_bolt_v2/
# peptide_main_orpt_h1_v2 have real checkpoints -- verify first, neither is
# trained as of this writing) instead, without a second script file:
#   MANIFEST_PREFIX=main_v2_orpt_vs_bolt \
#   CONFIG_PREFIX=peptide_main_v2_eval_gpu \
#   STEP0_CONFIG=peptide_experiment/configs/peptide_main_bolt_v2.yaml \
#   PLOT_ARMS=BOLT,ORPT-H1 \
#   nohup bash experiments/eval/run_main_bolt_vs_orpt_mi_eval.sh \
#       > experiments/eval/logs/main_v2_orpt_vs_bolt_full.log 2>&1 &
# (v2's manifest has no MTBO/OptFormer/POGPE/SGPE/LLAMBO rows uncommented yet
# -- set RUN_STEP0=0 to skip step 0 entirely until those baselines are
# retrained at the new schedule, and gpu7 in that manifest is STBO only, not
# POGPE/SGPE/LLAMBO, so step 3b's report below will find no rows there yet.)

set -euo pipefail
cd /workspace/BOLT

MANIFEST_PREFIX=${MANIFEST_PREFIX:-main_bolt_vs_orpt_mi}
CONFIG_PREFIX=${CONFIG_PREFIX:-peptide_main_eval_gpu}  # per-GPU pointer config basename, before the gpu index
STEP0_CONFIG=${STEP0_CONFIG:-peptide_experiment/configs/peptide_main_bolt.yaml}
PLOT_ARMS=${PLOT_ARMS:-BOLT,ORPT-MI,OptFormer,MTBO,LLAMBO}
RUN_STEP0=${RUN_STEP0:-1}

MANIFEST=experiments/eval/manifests/${MANIFEST_PREFIX}.yaml
CONFIG=peptide_experiment/configs/${CONFIG_PREFIX}0.yaml  # any shard's pointer config works; see README's "0. Prerequisite"
RESULTS=experiments/eval/results/${MANIFEST_PREFIX}
RESULTS_GPUS=${RESULTS}__gpu0,${RESULTS}__gpu1,${RESULTS}__gpu2,${RESULTS}__gpu3,${RESULTS}__gpu4,${RESULTS}__gpu5,${RESULTS}__gpu6
RESULTS_GPUS_ALL=${RESULTS_GPUS},${RESULTS}__gpu7
TASK_SET=heldout100
TARGET_POOL_SIZES=1000  # == cfg.init_size, matches the paper's own Table 11/Figure 1-2 scale.
# Trimmed from [10,50,200,1000]: step 3's cost is ~flat in target_pool_size
# (dominated by bo_calls=oracle_budget=20000, fixed regardless of target --
# target only changes how many raw draws are rejection-sampled into the
# init pool, a cheap step relative to 20000 real oracle calls), so 1 target
# instead of 4 cuts step 3 to a quarter of what it would otherwise be.
# Trade-off: loses fixedtarget_mic_bytarget_bo*.png (the target-size
# sensitivity/robustness small-multiples) and the 10/50/200 coverage/
# rejection curves -- backfill those later only if reviewers ask.
BO_CALLS=20000  # this config's cfg.oracle_budget -- must match for plot_fixed_target_rejection_bo.py's --bo-calls filter to find any rows at all
mkdir -p experiments/eval/logs

wait_all() {
    # Waits on every pid passed in; exits the script if any failed.
    local fail=0
    for pid in "$@"; do
        wait "$pid" || fail=1
    done
    if [ "$fail" -ne 0 ]; then
        echo "[run_main_bolt_vs_orpt_mi_eval] one or more jobs failed -- check experiments/eval/logs/" >&2
        exit 1
    fi
}

if [ "$RUN_STEP0" = "1" ]; then
    echo "[run_main_bolt_vs_orpt_mi_eval] step 0: train MTBO/GP-expert-transfer/OptFormer baselines into \$STEP0_CONFIG's run_dir (cheap, idempotent, single GPU, serial -- no training step for LLAMBO)"
    python -m peptide_experiment.cli train_mtbo --config "$STEP0_CONFIG" \
        > experiments/eval/logs/${MANIFEST_PREFIX}_step0_mtbo.log 2>&1
    python -m peptide_experiment.cli train_gp_expert_transfer --config "$STEP0_CONFIG" \
        > experiments/eval/logs/${MANIFEST_PREFIX}_step0_gp_expert_transfer.log 2>&1
    python -m peptide_experiment.cli train_optformer --config "$STEP0_CONFIG" \
        > experiments/eval/logs/${MANIFEST_PREFIX}_step0_optformer.log 2>&1
else
    echo "[run_main_bolt_vs_orpt_mi_eval] step 0: skipped (RUN_STEP0=0)"
fi

echo "[run_main_bolt_vs_orpt_mi_eval] step 1: raw proposal generation (7-way GPU parallel, task_set=${TASK_SET} -- only for arms with a real LLM checkpoint; self-seeding baselines have no raw proposals to generate)"
pids=()
for gpu in 0 1 2 3 4 5 6; do
    nohup python experiments/eval/generate_raw_proposals.py \
        --config peptide_experiment/configs/${CONFIG_PREFIX}${gpu}.yaml \
        --manifest experiments/eval/manifests/${MANIFEST_PREFIX}__gpu${gpu}.yaml \
        --task-sets ${TASK_SET} \
        > experiments/eval/logs/${MANIFEST_PREFIX}_step1_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

echo "[run_main_bolt_vs_orpt_mi_eval] step 2: incumbent vs pool size (no BO, 7-way parallel)"
pids=()
for gpu in 0 1 2 3 4 5 6; do
    nohup python experiments/eval/incumbent_vs_pool_size.py \
        --config peptide_experiment/configs/${CONFIG_PREFIX}${gpu}.yaml \
        --manifest experiments/eval/manifests/${MANIFEST_PREFIX}__gpu${gpu}.yaml \
        --task-sets ${TASK_SET} \
        > experiments/eval/logs/${MANIFEST_PREFIX}_step2_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir "$RESULTS_GPUS" --out-dir "$RESULTS" \
    > experiments/eval/logs/${MANIFEST_PREFIX}_step2_combined.log 2>&1

echo "[run_main_bolt_vs_orpt_mi_eval] step 3: fixed-target pool BO (real LOLBO -- expensive, 8-way parallel, every arm in the manifest)"
# Sanity check first, on one GPU shard, before the full sweep (uncomment):
#   python experiments/eval/fixed_target_rejection_bo.py \
#       --config peptide_experiment/configs/${CONFIG_PREFIX}0.yaml \
#       --manifest experiments/eval/manifests/${MANIFEST_PREFIX}__gpu0.yaml \
#       --task-sets ${TASK_SET} --target-pool-sizes 10 --limit-tasks 1
# Or for the milestone-independent baselines specifically (gpu7's manifest):
#   python experiments/eval/fixed_target_rejection_bo.py \
#       --config peptide_experiment/configs/${CONFIG_PREFIX}7.yaml \
#       --manifest experiments/eval/manifests/${MANIFEST_PREFIX}__gpu7.yaml \
#       --task-sets ${TASK_SET} --target-pool-sizes 10 --limit-tasks 1
pids=()
for gpu in 0 1 2 3 4 5 6 7; do
    nohup python experiments/eval/fixed_target_rejection_bo.py \
        --config peptide_experiment/configs/${CONFIG_PREFIX}${gpu}.yaml \
        --manifest experiments/eval/manifests/${MANIFEST_PREFIX}__gpu${gpu}.yaml \
        --task-sets ${TASK_SET} --target-pool-sizes ${TARGET_POOL_SIZES} \
        > experiments/eval/logs/${MANIFEST_PREFIX}_step3_gpu${gpu}.log 2>&1 &
    pids+=("$!")
done
wait_all "${pids[@]}"

# PLOT_ARMS excludes POGPE/SGPE by default: their milestone field means
# n_experts (5/10/20), not a real training milestone -- plotting them against
# the real milestone axis would render as misleading clutter, not a real
# line (see gpu7 manifest's comment). LLAMBO (when present in PLOT_ARMS) is
# milestone-independent but safe to include (plot_arm_line already renders
# single-milestone arms as a flat dashed reference line).
python experiments/eval/plot_fixed_target_rejection_bo.py \
    --results-dir "$RESULTS_GPUS_ALL" --out-dir "$RESULTS" --bo-calls ${BO_CALLS} \
    --arms ${PLOT_ARMS} \
    > experiments/eval/logs/${MANIFEST_PREFIX}_step3_combined.log 2>&1

echo "[run_main_bolt_vs_orpt_mi_eval] step 3b: POGPE/SGPE report (own axis is n_experts, not milestone -- not on the main plots) + LLAMBO token-budget-truncation report (per the user's explicit request to surface this, not hide it)"
python - "$RESULTS_GPUS_ALL" "$BO_CALLS" <<'PYEOF'
import sys
import pandas as pd

results_dirs = sys.argv[1].split(",")
bo_calls = int(sys.argv[2])

summary = pd.concat([pd.read_csv(f"{d}/summary_fixed_target_bo.csv") for d in results_dirs], ignore_index=True)
summary = summary[summary["bo_calls"] == bo_calls]

print("\n=== POGPE/SGPE: mean_best_mic by n_experts (own axis, not milestone) ===")
gpe = summary[summary["arm"].isin(["POGPE", "SGPE"])].sort_values(["arm", "milestone"])
print(gpe[["arm", "milestone", "n_tasks_ran_bo", "mean_best_mic"]].rename(columns={"milestone": "n_experts"}).to_string(index=False) or "(no rows)")

per_task = pd.concat([pd.read_csv(f"{d}/per_task_fixed_target_bo.csv") for d in results_dirs], ignore_index=True)
llambo = per_task[per_task["arm"] == "LLAMBO"].dropna(subset=["llambo_terminated_early"])
print("\n=== LLAMBO: token-budget early-termination report ===")
if llambo.empty:
    print("(no LLAMBO rows with recorded llambo_terminated_early)")
else:
    n = len(llambo)
    n_early = int(llambo["llambo_terminated_early"].sum())
    print(f"{n_early}/{n} tasks ({100 * n_early / n:.1f}%) terminated early due to llambo_max_input_tokens budget, before reaching oracle_budget={bo_calls} real BO calls")
    print(llambo["llambo_input_tokens_used"].describe().to_string())
PYEOF

echo "[run_main_bolt_vs_orpt_mi_eval] step 4: paper-figure scripts (fig:main-bo, fig:fewshot, fig:scaling, tab:main-bo-summary -- see README.md's \"Paper-figure scripts\" section; tab:ablation is a separate manifest/config, not part of this runbook, see tab_ablation.py's own docstring)"
python experiments/eval/fig_main_bo.py \
    --peptide-config "$CONFIG" --peptide-manifest "$MANIFEST" --peptide-task-set ${TASK_SET} \
    --out-dir "${RESULTS}/paper_figures/fig_main_bo" \
    > experiments/eval/logs/${MANIFEST_PREFIX}_step4_fig_main_bo.log 2>&1

python experiments/eval/fig_fewshot.py \
    --peptide-config "$CONFIG" --peptide-results-dir "$RESULTS_GPUS" --peptide-task-set ${TASK_SET} \
    --out-dir "${RESULTS}/paper_figures/fig_fewshot" \
    > experiments/eval/logs/${MANIFEST_PREFIX}_step4_fig_fewshot.log 2>&1

python experiments/eval/fig_scaling.py \
    --peptide-config "$CONFIG" --peptide-results-dir "$RESULTS_GPUS" --peptide-task-set ${TASK_SET} \
    --out-dir "${RESULTS}/paper_figures/fig_scaling" \
    > experiments/eval/logs/${MANIFEST_PREFIX}_step4_fig_scaling.log 2>&1

python experiments/eval/tab_main_bo_summary.py \
    --peptide-config "$CONFIG" --peptide-results-dir "$RESULTS_GPUS" --peptide-task-set ${TASK_SET} \
    --out-dir "${RESULTS}/paper_figures/tab_main_bo_summary" \
    > experiments/eval/logs/${MANIFEST_PREFIX}_step4_tab_main_bo_summary.log 2>&1

echo "[run_main_bolt_vs_orpt_mi_eval] done -- see ${RESULTS}/plots/ and ${RESULTS}/paper_figures/"
