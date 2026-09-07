# experiments/eval

Pipeline for comparing checkpoints trained under `peptide_experiment` or `query_plan_experiment`
(BOLT/ORPT-H1/ORPT-H0/OptFormer/MTBO/POGPE/SGPE/LLAMBO/STBO, at arbitrary milestones), and for
producing the exact figures/tables `paper/experiments.tex` needs (see "Paper-figure scripts" below).

All commands below assume they're run from the **BOLT repo root**.

## TL;DR

Prerequisite: one manifest YAML listing the (arm, milestone, run_dir, checkpoint_dir) combinations
you want to evaluate. Example: [`manifests/main_bolt_vs_orpt_mi__gpu0.yaml`](manifests/main_bolt_vs_orpt_mi__gpu0.yaml)
(one real milestone's slice of the main BOLT-vs-ORPT-MI-vs-baselines comparison).

Copy-paste the block below to run raw proposal generation → compute → plot, start to finish.
(Only `DOMAIN`/`CONFIG`/`MANIFEST` need to change. `CONFIG` doesn't select a model — it's only
read for constants shared by every model in the manifest — `similarity_threshold`,
`oracle_budget`, `table_k_checkpoints`, task universe (trainset/heldout definitions) — so any
config belonging to a model in the manifest works, as long as those fields agree. See "0.
Prerequisite" and "1. Domains" below for why.)

```bash
DOMAIN=peptide  # or query_plan
CONFIG=peptide_experiment/configs/peptide_main_bolt.yaml
MANIFEST=experiments/eval/manifests/main_bolt_vs_orpt_mi__gpu0.yaml
RESULTS=experiments/eval/results/main_bolt_vs_orpt_mi__gpu0

# 0) generate raw proposals (covers every arm/milestone in the manifest, run once)
python experiments/eval/generate_raw_proposals.py --domain $DOMAIN --config $CONFIG --manifest $MANIFEST

# 1) incumbent vs pool size (no BO, cheap -- start here)
python experiments/eval/incumbent_vs_pool_size.py --domain $DOMAIN --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_incumbent_vs_pool_size.py --results-dir $RESULTS

# 2) fixed-target pool BO (launches real LOLBO runs -- expensive)
# sanity check: python experiments/eval/fixed_target_rejection_bo.py --domain $DOMAIN --config $CONFIG --manifest $MANIFEST --limit-tasks 1 --target-pool-sizes 10
python experiments/eval/fixed_target_rejection_bo.py --domain $DOMAIN --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_fixed_target_rejection_bo.py --results-dir $RESULTS
```

Resulting PNGs land in `$RESULTS/plots/`. See the per-experiment sections below for exactly what
each step does and its input/output, and "Paper-figure scripts" for turning these CSVs into
`paper/experiments.tex`-ready figures/tables.

## 0. Prerequisite: manifest

You need one YAML file listing the (arm, milestone, run_dir) combinations you want to evaluate.
Example: [`manifests/main_bolt_vs_orpt_mi.yaml`](manifests/main_bolt_vs_orpt_mi.yaml)

```yaml
models:
  - arm: BOLT
    milestone: 126
    run_dir: runs/peptide_main_bolt
    checkpoint_dir: runs/peptide_main_bolt/checkpoints/BOLT-126/epoch_4  # only read by generate_raw_proposals.py
  - arm: ORPT-MI
    milestone: 126
    run_dir: runs/peptide_main_orpt_mi
    checkpoint_dir: runs/peptide_main_orpt_mi/checkpoints/ORPT-126/epoch_0
  # ...
```

- Paths are relative to the repo root.
- `checkpoint_dir` can be omitted once raw proposals already exist for that (arm, milestone), or
  for arms that self-seed their own init pool (STBO/MTBO/OptFormer/POGPE/SGPE/LLAMBO -- see
  "Self-seeding baselines" below).
- Models trained under different `experiment_id`s/configs can freely be mixed into one manifest.
- Internal arm names don't have to match the paper's own terminology (e.g. `ORPT-MI`/`ORPT-H1` are
  both internally-named variants of the paper's "ORPT" method) -- see `paper_labels.py`.

Every analysis script also takes one `--config <experiment yaml>`. This config doesn't select a
model — it's purely for constants shared across the whole comparison (`similarity_threshold`,
`oracle_budget`, `table_k_checkpoints`, task universe), so any one of the manifest's underlying
configs works, as long as they agree on those fields.

Task universes are fixed, not disk-discovered. Peptide: `trainset` = the tasks every compared
milestone has already been trained on (`range(min(cfg.milestones))`); `heldout` =
`cfg.heldout_tasks("heldout20")`; `heldout100` = `cfg.heldout_tasks("heldout100")` (the 100-task
universe the main-scale comparison uses for both experiments below). Query-plan: `trainset` =
`task_splits.train_workloads()[:min(cfg.milestones)]`; `heldout` = `cfg.heldout_tasks`.

## 1. Domains

Every compute script (`generate_raw_proposals.py`, `incumbent_vs_pool_size.py`,
`fixed_target_rejection_bo.py`) takes `--domain {peptide,query_plan}` (default `peptide`). The two
domains' actual differences (task identity, whether a pre-oracle feasibility constraint exists,
init-pool file format, whether a candidate's usability is knowable before scoring) are abstracted
behind `domains.py`'s `Domain` object -- see that file's own docstring for the full mapping. Every
existing peptide-only invocation keeps working unchanged (`--domain` defaults to `peptide`).

Baselines other than BOLT/ORPT (STBO, MTBO, OptFormer, POGPE, SGPE, LLAMBO) are **peptide-only** --
`query_plan_experiment` has no analogous training pipeline for any of them yet (nor for ORPT/DPO
itself -- see `query_plan_experiment/README.md`'s own "## Scope" note). Pointing a query-plan
manifest at one of these arm names raises immediately with a clear message rather than failing
deep in an import.

## 2. Common prerequisite step: generate raw proposals

Must be run once before Experiment 1 or 2, for every (arm, milestone) that has a real LLM
checkpoint (self-seeding baselines don't need this -- see below).

| | |
|---|---|
| Script | [`generate_raw_proposals.py`](generate_raw_proposals.py) |
| What it does | For **every (arm, milestone) entry in `--manifest`** with a `checkpoint_dir`, samples raw candidates against the fixed task sets (`trainset`/`heldout`/`heldout100`) |
| Input | `--domain`, `--config`, `--manifest` |
| Output | `<run_dir>/eval_raw/<task_set>/<arm>-<milestone>/...` (raw jsonls; exact filename convention is domain-specific, see `domains.py`) |

```bash
python experiments/eval/generate_raw_proposals.py \
    --config peptide_experiment/configs/peptide_main_bolt.yaml \
    --manifest experiments/eval/manifests/main_bolt_vs_orpt_mi__gpu0.yaml
```

Note that `--config` and `--manifest` play different roles:
- **Only `--manifest` decides which models get evaluated (whose raw proposals get generated).**
- **`--config` does not select a model.** It's only read for constants shared by every entry in
  the manifest (`similarity_threshold`, task universe, etc.).

If raw proposals already exist for every (arm, milestone, task_set) combination in the manifest,
you don't need to re-run this. Experiments 1-2 below are pure post-hoc analysis over this output
(plus, for self-seeding baselines, their own in-process init pool), and never touch
`checkpoint_dir` or re-invoke LLM sampling except here.

### Self-seeding baselines

STBO, MTBO, OptFormer, POGPE, SGPE, LLAMBO don't go through this step at all -- they build their
own init pool directly inside `fixed_target_rejection_bo.py` (random mutations of the reference
task, or their own history-conditioned propose/score loop). Their manifest rows need no
`checkpoint_dir` except OptFormer (which does have a real per-milestone LLM checkpoint). POGPE/SGPE
overload `milestone` to mean expert count (5/10/20, not a training milestone); LLAMBO and STBO are
milestone-independent (milestone is an unused placeholder) -- see
`fixed_target_rejection_bo.py`'s own per-arm branches for exactly how each one is seeded.

---

## Experiment 1: Proposal-level incumbent vs. pool size (no BO)

A cheap way to see "how much does the best score improve as the model draws more feasible
proposals". Doesn't run BO, so start here. This is the compute engine behind `fig:fewshot`.

| | |
|---|---|
| Compute | [`incumbent_vs_pool_size.py`](incumbent_vs_pool_size.py) |
| Plot | [`plot_incumbent_vs_pool_size.py`](plot_incumbent_vs_pool_size.py) |

**Compute**
- Input: raw proposals from step 0 (`<run_dir>/eval_raw/...`), `--domain`, `--config`, `--manifest`
- What it does: for each (arm, milestone, task_set, task_id), reads the raw jsonls to build a
  feasible/unique proposal pool and rescores it via the domain's own oracle. At each `n_proposals`
  checkpoint (default: `cfg.table_k_checkpoints`, falling back to `[1, 5, 10, 20, 50]` if that's
  unset), records (a) the incumbent (best objective) among the first `n_proposals`, and (b) the
  raw draw count/rejection rate needed to accumulate that many
- Output: `results/<manifest stem>/per_task_incumbent_vs_pool_size.csv`,
  `results/<manifest stem>/summary_incumbent_vs_pool_size.csv`

```bash
python experiments/eval/incumbent_vs_pool_size.py \
    --config peptide_experiment/configs/peptide_main_bolt.yaml \
    --manifest experiments/eval/manifests/main_bolt_vs_orpt_mi__gpu0.yaml
```

**Plot**
- Input: `--results-dir results/<manifest stem>`
- What it does: draws three kinds of figures per task_set (trainset/heldout/heldout100)
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
    --results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu0
```

---

## Experiment 2: Fixed-target pool BO

Forces every task onto the **same size** feasible pool (target_pool_size), then runs real BO —
comparing models on an equal initialization footing. Launches real LOLBO runs, so it's
expensive — sanity-check with `--limit-tasks` first. This is the compute engine behind
`fig:main-bo` and `fig:scaling`'s final-BO panel.

| | |
|---|---|
| Compute | [`fixed_target_rejection_bo.py`](fixed_target_rejection_bo.py) |
| Plot | [`plot_fixed_target_rejection_bo.py`](plot_fixed_target_rejection_bo.py) |

**Compute**
- Input: raw proposals, `--domain`, `--config`, `--manifest`, `--target-pool-sizes` (default
  `10,20,50`), `--limit-tasks` (optional)
- What it does: for each task, if rejection sampling over its raw proposals can build a pool of
  exactly `target` feasible/unique candidates (no extra top-up sampling), runs real BO on that
  pool (self-seeding baselines build their own `target`-sized pool directly instead). Tasks that
  can't reach the target are skipped and counted separately via `coverage_rate` (not silently
  folded into the average)
- Output: `results/<manifest stem>/fixed_target_bo_coverage.csv`,
  `results/<manifest stem>/per_task_fixed_target_bo.csv`,
  `results/<manifest stem>/summary_fixed_target_bo.csv`
  (heavy per-task artifacts from the BO run are written under `<run_dir>/eval_fixed_target_bo/`)

```bash
# sanity check: 1 task, 1 target size
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_main_bolt.yaml \
    --manifest experiments/eval/manifests/main_bolt_vs_orpt_mi__gpu0.yaml \
    --limit-tasks 1 --target-pool-sizes 10

# full run
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_main_bolt.yaml \
    --manifest experiments/eval/manifests/main_bolt_vs_orpt_mi__gpu0.yaml
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
    --results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu0
```

---

## Paper-figure scripts

Five scripts, each mapping 1:1 to one `paper/experiments.tex` result-bearing subsection, each
consuming Experiment 1/2's CSVs above (no new compute) and emitting paper-terminology-consistent
output (see [`paper_labels.py`](paper_labels.py) for the internal-arm-name → paper-name mapping,
e.g. `ORPT-MI`/`ORPT-H1` → "ORPT"). All accept comma-separated `--*-results-dir` lists (or, for
`fig_main_bo.py`, a `--*-manifest`, since it needs to resolve each spec's own checkpoint/run_dir)
the same way `plot_*.py` above does, and either domain may be omitted (that panel/row is skipped
with a printed note, or rendered as `--` for the table scripts).

| Script | `experiments.tex` target | Source |
|---|---|---|
| [`fig_main_bo.py`](fig_main_bo.py) | `fig:main-bo` (sec:main-results, "Full-budget optimization") | dense per-task trajectory CSVs under `eval_fixed_target_bo/` |
| [`fig_fewshot.py`](fig_fewshot.py) | `fig:fewshot` (sec:main-results, "Initialization and few-shot proposal quality") | `summary_incumbent_vs_pool_size.csv` |
| [`fig_scaling.py`](fig_scaling.py) | `fig:scaling` (sec:scaling) | both summary CSVs |
| [`tab_main_bo_summary.py`](tab_main_bo_summary.py) | `tab:main-bo-summary` (new label; sec:main-results' optional table) | both summary CSVs |
| [`tab_ablation.py`](tab_ablation.py) | `tab:ablation` (new label; sec:ablations) | both summary CSVs, from `manifests/ablation_h0_vs_h1.yaml`'s results |

```bash
python experiments/eval/fig_main_bo.py \
    --peptide-config peptide_experiment/configs/peptide_main_bolt.yaml \
    --peptide-manifest experiments/eval/manifests/main_bolt_vs_orpt_mi.yaml

python experiments/eval/fig_fewshot.py \
    --peptide-config peptide_experiment/configs/peptide_main_bolt.yaml \
    --peptide-results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu6

python experiments/eval/fig_scaling.py \
    --peptide-config peptide_experiment/configs/peptide_main_bolt.yaml \
    --peptide-results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu0,...,experiments/eval/results/main_bolt_vs_orpt_mi__gpu6

python experiments/eval/tab_main_bo_summary.py \
    --peptide-config peptide_experiment/configs/peptide_main_bolt.yaml \
    --peptide-results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu6

# tab_ablation.py: see its own docstring for the real prerequisite (ORPT-H0 has
# never been trained at any scale yet) before trusting its output.
python experiments/eval/tab_ablation.py \
    --config peptide_experiment/configs/peptide_ablation_orpt_h1.yaml \
    --results-dir experiments/eval/results/ablation_h0_vs_h1
```

Compute Cost (`sec:compute-overhead`) has no script -- no GPU-hour-tracking infra exists in this
repo; that table is filled in manually.

---

## Comparing multiple result sets in one chart

Every `plot_*.py`/`fig_*.py`/`tab_*.py` accepts a comma-separated `--results-dir` (or
`--*-results-dir`) list and concatenates them before plotting — useful since a full milestone sweep
is typically split across per-GPU shards (`results/<manifest stem>__gpu<N>/`).

```bash
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu0,experiments/eval/results/main_bolt_vs_orpt_mi__gpu1
```

## Restricting to a subset of arms

Every `plot_*.py` accepts `--arms` (comma-separated, e.g. `--arms BOLT,ORPT-MI`) to draw a subset
of arms instead of everything present in the results/manifest. Combine with `--out-dir` to keep the
subset's figures separate from the full comparison instead of overwriting it:

```bash
python experiments/eval/plot_fixed_target_rejection_bo.py \
    --results-dir experiments/eval/results/main_bolt_vs_orpt_mi__gpu0 \
    --arms BOLT,ORPT-MI \
    --out-dir experiments/eval/results/comparison_bolt_orpt
```

## Notes

- `results/` is gitignored (already covered by the `experiments/*/results/` glob).
- File layout:

```
experiments/eval/
├── manifests/                        # (arm, milestone, run_dir) definitions to compare
├── domains.py                        # peptide/query_plan abstraction (Domain, DOMAINS registry)
├── paper_labels.py                   # internal arm name -> paper/experiments.tex terminology
├── generate_raw_proposals.py         # step 0: generate raw proposals (only script that reads checkpoints)
├── incumbent_vs_pool_size.py         # experiment 1 compute
├── plot_incumbent_vs_pool_size.py    # experiment 1 plot
├── fixed_target_rejection_bo.py      # experiment 2 compute
├── plot_fixed_target_rejection_bo.py # experiment 2 plot
├── fig_main_bo.py                    # fig:main-bo
├── fig_fewshot.py                    # fig:fewshot
├── fig_scaling.py                    # fig:scaling
├── tab_main_bo_summary.py            # tab:main-bo-summary
├── tab_ablation.py                   # tab:ablation
├── common.py / plot_common.py        # shared utilities (manifest loading, pool construction, styling)
└── results/                          # output CSVs/PNGs, one subdir per manifest stem (gitignored)
```
