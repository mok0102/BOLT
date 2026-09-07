#!/usr/bin/env bash
# Runbook for the main-experiment (paper-fidelity scale) BOLT-vs-ORPT-vs-
# baselines comparison, v2 milestone schedule [10,20,50,250,400,500,600],
# lora_rank=16 (runs/peptide_main_bolt_v2 vs runs/peptide_main_orpt_h1_v2,
# plus OptFormer/MTBO/POGPE/SGPE/LLAMBO baselines trained into
# runs/peptide_main_bolt_v2, 100-task held-out set), plus the ablation study
# (runs/peptide_ablation_orpt_h0): raw proposal generation -> incumbent vs
# pool size (the "few-shot"/init-only-style table) -> fixed-target pool BO
# (the scaling curve at a fixed init-pool size -- real LOLBO, the expensive
# step) -> paper-figure scripts -> ablation table.
#
# PREREQUISITE: training is NOT this script's job -- run, in order:
#   1. experiments/eval/run_main_bolt_vs_orpt_train.sh   (BOLT + ORPT-H1 +
#      the ORPT-H0 ablation arm)
#   2. experiments/eval/run_main_baselines_train.sh      (MTBO/OptFormer/
#      GP-expert-transfer, depends on BOLT's completed trajectory data)
# before this script. Sanity-check before launching:
#   ls runs/peptide_main_bolt_v2/checkpoints/      # expect BOLT-10 .. BOLT-600
#   ls runs/peptide_main_orpt_h1_v2/checkpoints/   # expect ORPT-10 .. ORPT-600
#   ls runs/peptide_ablation_orpt_h0/checkpoints/  # expect ORPT-10 .. ORPT-600
#
# COST WARNING: with the 100-task held-out set, step 3 (fixed-target BO) is
# (BOLT+ORPT-H1+OptFormer+MTBO) x 7 milestones x 100 tasks + (POGPE+SGPE) x 3
# expert-counts x 100 tasks + LLAMBO x 100 tasks, all at 1 target pool size
# (1000 == this config's cfg.init_size, matching the paper's own Table
# 11/Figure 1-2 scale) -- likely multiple days of wall-clock time even
# 8-way GPU parallel. Sanity-check on a small slice FIRST (see the
# commented-out smoke command under step 3) before launching the full sweep
# -- and note that whether target=1000 is even achievable (coverage_rate) for
# every arm is an open empirical question; step 2's own
# coverage_rate_at_n_proposals output (cheap, no BO) is the place to check
# this for BOLT/ORPT-H1/OptFormer before committing to step 3 (MTBO/POGPE/
# SGPE/LLAMBO self-seed their init pool directly and always achieve
# target=1000 by construction, so coverage isn't a question for them).
#
# This is a literal, saved copy of the manual command sequence -- no new
# dispatch/orchestration logic beyond tracking each background job's exit
# status so a failed step doesn't silently let the next (more expensive)
# step run on bad data.
#
# All 8 GPUs are used: gpu{0-6} for the 7 real milestones (BOLT/ORPT-H1/
# OptFormer/MTBO), gpu7 for the milestone-independent baselines (POGPE/SGPE/
# LLAMBO -- see manifests/main_v2_orpt_vs_bolt__gpu7.yaml's own comment for
# why these can't share the real milestone axis).
#
# Run detached so a dropped connection doesn't kill step 3:
#   nohup bash experiments/eval/run_main_bolt_vs_orpt_mi_eval.sh \
#       > experiments/eval/logs/main_v2_orpt_vs_bolt_full.log 2>&1 &
#
# Drives the v2/rank=16 schedule above by default (MANIFEST_PREFIX/
# CONFIG_PREFIX/PLOT_ARMS are still env-var overridable for a future
# schedule). The old, already-completed [126..900]/rank=4 schedule's
# manifest/results are kept as a historical record
# (manifests/results main_bolt_vs_orpt_mi*) but its configs
# (peptide_main_bolt.yaml/peptide_main_orpt_mi.yaml/peptide_main_eval_gpu*)
# were removed -- that schedule won't be driven again.

set -euo pipefail
cd /workspace/BOLT

MANIFEST_PREFIX=${MANIFEST_PREFIX:-main_v2_orpt_vs_bolt}
CONFIG_PREFIX=${CONFIG_PREFIX:-peptide_main_v2_eval_gpu}  # per-GPU pointer config basename, before the gpu index
PLOT_ARMS=${PLOT_ARMS:-BOLT,ORPT-H1,OptFormer,MTBO,LLAMBO}
RUN_ABLATION=${RUN_ABLATION:-1}
ABLATION_MANIFEST=${ABLATION_MANIFEST:-experiments/eval/manifests/ablation_h0_vs_h1.yaml}
ABLATION_RESULTS=${ABLATION_RESULTS:-experiments/eval/results/ablation_h0_vs_h1}

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

echo "[run_main_bolt_vs_orpt_mi_eval] step 4: paper-figure scripts (fig:main-bo, fig:fewshot, fig:scaling, tab:main-bo-summary -- see README.md's \"Paper-figure scripts\" section)"
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

if [ "$RUN_ABLATION" = "1" ]; then
    echo "[run_main_bolt_vs_orpt_mi_eval] step 5: ablation study (tab:ablation) -- idempotent, BOLT/ORPT-H1 rows reuse steps 1-3's output above (same run_dir/checkpoint_dir), only ORPT-H0 does real new work here"
    python experiments/eval/generate_raw_proposals.py \
        --config "$CONFIG" --manifest "$ABLATION_MANIFEST" --task-sets ${TASK_SET} \
        > experiments/eval/logs/ablation_step1.log 2>&1
    python experiments/eval/incumbent_vs_pool_size.py \
        --config "$CONFIG" --manifest "$ABLATION_MANIFEST" --task-sets ${TASK_SET} \
        > experiments/eval/logs/ablation_step2.log 2>&1
    python experiments/eval/fixed_target_rejection_bo.py \
        --config "$CONFIG" --manifest "$ABLATION_MANIFEST" \
        --task-sets ${TASK_SET} --target-pool-sizes ${TARGET_POOL_SIZES} \
        > experiments/eval/logs/ablation_step3.log 2>&1
    python experiments/eval/tab_ablation.py \
        --config "$CONFIG" --results-dir "$ABLATION_RESULTS" --task-set ${TASK_SET} \
        > experiments/eval/logs/ablation_step4_tab_ablation.log 2>&1
else
    echo "[run_main_bolt_vs_orpt_mi_eval] step 5: skipped (RUN_ABLATION=0)"
fi

echo "[run_main_bolt_vs_orpt_mi_eval] done -- see ${RESULTS}/plots/, ${RESULTS}/paper_figures/, and ${ABLATION_RESULTS}/"
