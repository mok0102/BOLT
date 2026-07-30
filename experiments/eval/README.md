# experiments/eval

Pipeline for comparing checkpoints trained under `peptide_experiment` (BOLT/ORPT/ORPT-FA/ORPT-LEX,
etc., at arbitrary milestones).

All commands below assume they're run from the **BOLT repo root**.

## TL;DR

Prerequisite: one manifest YAML listing the (arm, milestone, run_dir, checkpoint_dir) combinations
you want to evaluate. Example: [`manifests/poc20_four_arm.yaml`](manifests/poc20_four_arm.yaml)
(BOLT/ORPT/ORPT-FA/ORPT-LEX × milestone 5/10/20/30/40/50/60/70/80/90/100).

Copy-paste the block below to run raw proposal generation → experiment 1-3 compute → plot, start
to finish. (Only `CONFIG`/`MANIFEST` need to change. `CONFIG` doesn't select a model — it's only
read for constants shared by every model in the manifest — `similarity_threshold`,
`oracle_budget`, `table_k_checkpoints`, task universe (trainset/heldout definitions) — so any
config belonging to a model in the manifest works, as long as those fields agree. See "0.
Prerequisite" and the note under Experiment 1 for why.)

```bash
CONFIG=peptide_experiment/configs/peptide_poc20_bolt.yaml
MANIFEST=experiments/eval/manifests/poc20_four_arm.yaml
RESULTS=experiments/eval/results/poc20_four_arm

# 0) generate raw proposals (covers every arm/milestone in the manifest, run once)
python experiments/eval/generate_raw_proposals.py --config $CONFIG --manifest $MANIFEST

# 1) incumbent vs pool size (no BO, cheap -- start here)
python experiments/eval/incumbent_vs_pool_size.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_incumbent_vs_pool_size.py --results-dir $RESULTS

# 2) fixed-target pool BO (launches real LOLBO runs -- expensive)
# sanity check: python experiments/eval/fixed_target_rejection_bo.py --config $CONFIG --manifest $MANIFEST --limit-tasks 1 --target-pool-sizes 10
python experiments/eval/fixed_target_rejection_bo.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_fixed_target_rejection_bo.py --results-dir $RESULTS

# 3) fixed-budget rejection-sampled BO (launches real LOLBO runs -- expensive)
# sanity check: python experiments/eval/fixed_budget_rejection_bo.py --config $CONFIG --manifest $MANIFEST --limit-tasks 1
python experiments/eval/fixed_budget_rejection_bo.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_fixed_budget_rejection_bo.py --results-dir $RESULTS
```

Resulting PNGs land in `$RESULTS/plots/`. See the per-experiment sections below for exactly what
each step does and its input/output.

## 0. Prerequisite: manifest

You need one YAML file listing the (arm, milestone, run_dir) combinations you want to evaluate.
Example: [`manifests/poc20_four_arm.yaml`](manifests/poc20_four_arm.yaml)

```yaml
models:
  - arm: BOLT
    milestone: 5
    run_dir: runs/peptide_poc20_bolt
    checkpoint_dir: runs/peptide_poc20_bolt/checkpoints/BOLT-5/epoch_4  # only read by generate_raw_proposals.py
  - arm: ORPT
    milestone: 5
    run_dir: runs/peptide_poc20_orpt
    checkpoint_dir: runs/peptide_poc20_orpt/checkpoints/ORPT-5/epoch_0
  # ...
```

- Paths are relative to the repo root.
- `checkpoint_dir` can be omitted once raw proposals already exist for that (arm, milestone).
- Models trained under different `experiment_id`s/configs can freely be mixed into one manifest.

Every analysis script also takes one `--config <experiment yaml>`. This config doesn't select a
model — it's purely for constants shared across the whole comparison (`similarity_threshold`,
`oracle_budget`, `table_k_checkpoints`, task universe), so any one of the manifest's underlying
configs works, as long as they agree on those fields.

Task universes are fixed, not disk-discovered: `trainset` = the tasks every compared milestone has
already been trained on (`range(min(cfg.milestones))`); `heldout` = `cfg.heldout_tasks("heldout20")`.

## 1. Common prerequisite step: generate raw proposals

Must be run once before any of experiments 1-3, regardless of which one you run.

| | |
|---|---|
| Script | [`generate_raw_proposals.py`](generate_raw_proposals.py) |
| What it does | For **every (arm, milestone) entry in `--manifest`**, loads the checkpoint from its own `checkpoint_dir` and samples raw candidate peptides against the fixed task sets (`trainset`/`heldout`) |
| Input | `--config`, `--manifest` |
| Output | `<run_dir>/eval_raw/<task_set>/<arm>-<milestone>/task_<idx>_sampled_attempt*.jsonl` |

```bash
python experiments/eval/generate_raw_proposals.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

Note that `--config` and `--manifest` play different roles:
- **Only `--manifest` decides which models get evaluated (whose raw proposals get generated).**
  The command above generates raw proposals for all 44 entries listed in `poc20_four_arm.yaml`
  (BOLT/ORPT/ORPT-FA/ORPT-LEX × milestone 5/10/20/30/40/50/60/70/80/90/100) — not just BOLT.
- **`--config` does not select a model.** It's only read for constants shared by every entry in
  the manifest (`similarity_threshold`, task universe, etc.), so passing
  `peptide_poc20_bolt.yaml` as in the example still processes the ORPT/ORPT-FA/ORPT-LEX
  checkpoints correctly, too. (As long as those field values agree across configs, which they do
  by design for this 4-arm PoC — see "0. Prerequisite" above.)

If raw proposals already exist for every (arm, milestone, task_set) combination in the manifest,
you don't need to re-run this. Experiments 1-3 below are all pure post-hoc analysis over this
output, and never touch `checkpoint_dir` or re-invoke LLM sampling.

---

## Experiment 1: Proposal-level incumbent vs. pool size (no BO)

A cheap way to see "how much does the best score improve as the model draws more feasible
proposals". Doesn't run BO, so start here.

| | |
|---|---|
| Compute | [`incumbent_vs_pool_size.py`](incumbent_vs_pool_size.py) |
| Plot | [`plot_incumbent_vs_pool_size.py`](plot_incumbent_vs_pool_size.py) |

**Compute**
- Input: raw proposals from step 0 (`<run_dir>/eval_raw/...`), `--config`, `--manifest`
- What it does: for each (arm, milestone, task_set, task_idx), reads the raw jsonls to build a
  feasible/unique proposal pool and rescores it via `apex_wrapper`. At each `n_proposals`
  checkpoint (default `[1, 5, 10, 20, 50]`), records (a) the incumbent (lowest MIC) among the
  first `n_proposals`, and (b) the raw draw count/rejection rate needed to accumulate that many
- Output: `results/<manifest stem>/per_task_incumbent_vs_pool_size.csv`,
  `results/<manifest stem>/summary_incumbent_vs_pool_size.csv`

```bash
python experiments/eval/incumbent_vs_pool_size.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- What it does: draws three kinds of figures per task_set (trainset/heldout)
  - `incumbent_mic_bymilestone_n<n_proposals>_<task_set>.png`: headline chart — arm-by-arm trend
    across milestones at a fixed `n_proposals` (default 10)
  - `incumbent_mic_byNProposals_<task_set>.png`: small multiples, one subplot per `n_proposals`
    checkpoint
  - `incumbent_coverage_bymilestone_n<n_proposals>_<task_set>.png`: `coverage_rate_at_n_proposals`
    (share of tasks with enough feasible proposals to even measure an incumbent) vs. milestone, at
    the same reference `n_proposals` as the headline chart
- Output: `results/<manifest stem>/plots/*.png`

```bash
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir experiments/eval/results/poc20_four_arm
```

---

## Experiment 2: Fixed-target pool BO

Forces every task onto the **same size** feasible pool (target_pool_size), then runs real BO —
comparing models on an equal initialization footing. Launches real LOLBO runs, so it's
expensive — sanity-check with `--limit-tasks` first.

| | |
|---|---|
| Compute | [`fixed_target_rejection_bo.py`](fixed_target_rejection_bo.py) |
| Plot | [`plot_fixed_target_rejection_bo.py`](plot_fixed_target_rejection_bo.py) |

**Compute**
- Input: raw proposals, `--config`, `--manifest`, `--target-pool-sizes` (default `10,20,50`),
  `--limit-tasks` (optional)
- What it does: for each task, if rejection sampling over its raw proposals can build a pool of
  exactly `target` feasible/unique sequences (no extra top-up sampling), runs real BO (LOLBO) on
  that pool. Tasks that can't reach the target are skipped and counted separately via
  `coverage_rate` (not silently folded into the average)
- Output: `results/<manifest stem>/fixed_target_bo_coverage.csv`,
  `results/<manifest stem>/per_task_fixed_target_bo.csv`,
  `results/<manifest stem>/summary_fixed_target_bo.csv`
  (heavy per-task artifacts from the BO run are written under `<run_dir>/eval_fixed_target_bo/`)

```bash
# sanity check: 1 task, 1 target size
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml \
    --limit-tasks 1 --target-pool-sizes 10

# full run
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- What it does: four figures
  - `fixedtarget_mic_bymilestone_target<T>_bo<bo_calls>_<task_set>.png`: headline chart at a
    reference target_pool_size (default: the largest present) and bo_calls (default 5000)
  - `fixedtarget_mic_bytarget_bo<bo_calls>_<task_set>.png`: small multiples, one subplot per
    target_pool_size
  - `fixedtarget_rejection_bymilestone_target<T>_<task_set>.png`: rejection rate (share of raw
    draws thrown away) vs. milestone, at the reference target
  - `fixedtarget_coverage_bymilestone_target<T>_<task_set>.png`: `coverage_rate` (share of tasks
    that could reach `target_pool_size` at all) vs. milestone, at the same reference target
- Output: `results/<manifest stem>/plots/*.png`

```bash
python experiments/eval/plot_fixed_target_rejection_bo.py \
    --results-dir experiments/eval/results/poc20_four_arm
```

---

## Experiment 3: Fixed-budget (real) rejection-sampled BO

Doesn't force a target pool size — runs BO on whatever feasible pool survives a fixed sampling
budget (whatever `generate_raw_proposals.py` already produced). Pool size varying task-to-task
and arm-to-arm is itself part of what's being measured (an arm with more constraint violations
ends up with a smaller pool).

| | |
|---|---|
| Compute | [`fixed_budget_rejection_bo.py`](fixed_budget_rejection_bo.py) |
| Plot | [`plot_fixed_budget_rejection_bo.py`](plot_fixed_budget_rejection_bo.py) |

**Compute**
- Input: raw proposals, `--config`, `--manifest`, `--limit-tasks` (optional), `--min-feasible`
  (default 5 — minimum pool size floor to dodge the LOLBO trust-region hang)
- What it does: runs BO on whatever feasible pool (variable size) each task's raw proposals
  yield. Tasks that don't clear the floor are skipped and counted separately via `coverage_rate`/
  pool-size distribution
- Output: `results/<manifest stem>/fixed_budget_bo_coverage.csv`,
  `results/<manifest stem>/per_task_fixed_budget_bo.csv`,
  `results/<manifest stem>/summary_fixed_budget_bo.csv`
  (heavy per-task artifacts are written under `<run_dir>/eval_fixed_budget_bo/`)

```bash
# sanity check
python experiments/eval/fixed_budget_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml \
    --limit-tasks 1

# full run
python experiments/eval/fixed_budget_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- What it does: four figures
  - `fixedbudget_mic_bymilestone_bo<bo_calls>_<task_set>.png`: headline chart at a reference
    bo_calls (default 5000 — i.e. the full oracle budget spent)
  - `fixedbudget_mic_byboCalls_<task_set>.png`: small multiples, one subplot per bo_calls
    checkpoint
  - `fixedbudget_rejection_bymilestone_<task_set>.png`: rejection rate vs. milestone
  - `fixedbudget_coverage_bymilestone_<task_set>.png`: `coverage_rate` (share of tasks clearing
    the `min_feasible` floor) vs. milestone
- Output: `results/<manifest stem>/plots/*.png`

```bash
python experiments/eval/plot_fixed_budget_rejection_bo.py \
    --results-dir experiments/eval/results/poc20_four_arm
```

---

## Comparing multiple result sets in one chart

Every `plot_*.py` accepts a comma-separated `--results-dir` list and concatenates them before
plotting — useful if two manifests were run separately and need to appear together.

```bash
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir experiments/eval/results/poc20_four_arm,experiments/eval/results/other_manifest
```

## Notes

- `results/` is gitignored (already covered by the `experiments/*/results/` glob).
- File layout:

```
experiments/eval/
├── manifests/                        # (arm, milestone, run_dir) definitions to compare
├── generate_raw_proposals.py         # step 0: generate raw proposals (only script that reads checkpoints)
├── incumbent_vs_pool_size.py         # experiment 1 compute
├── plot_incumbent_vs_pool_size.py    # experiment 1 plot
├── fixed_target_rejection_bo.py      # experiment 2 compute
├── plot_fixed_target_rejection_bo.py # experiment 2 plot
├── fixed_budget_rejection_bo.py      # experiment 3 compute
├── plot_fixed_budget_rejection_bo.py # experiment 3 plot
├── common.py / plot_common.py        # shared utilities (manifest loading, pool construction, styling)
└── results/                          # output CSVs/PNGs, one subdir per manifest stem (gitignored)
```
