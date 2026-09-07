# experiments/eval

Pipeline for comparing checkpoints trained under `peptide_experiment` or `query_plan_experiment`
(BOLT/ORPT-H1/ORPT-H0/OptFormer/MTBO/POGPE/SGPE/LLAMBO/STBO, at arbitrary milestones), and for
producing the exact figures/tables `paper/experiments.tex` needs (see "Paper-figure scripts" below).
All commands assume they're run from the **BOLT repo root**.

## Quick start

To reproduce everything for the main experiment, run these three runbooks in order:

1. [`run_main_bolt_vs_orpt_train.sh`](run_main_bolt_vs_orpt_train.sh) — trains BOLT + ORPT-H1
   (concurrent) + the ORPT-H0 ablation arm.
2. [`run_main_baselines_train.sh`](run_main_baselines_train.sh) — trains MTBO/OptFormer/
   GP-expert-transfer, after step 1's BOLT chain has completed.
3. [`run_main_bolt_vs_orpt_mi_eval.sh`](run_main_bolt_vs_orpt_mi_eval.sh) — runs every eval step
   below plus the ablation table, producing every `paper/experiments.tex` figure/table.

Each is env-var overridable (see its own header comment) to target a different config/manifest.
The sections below document the individual scripts each runbook calls, for running pieces by hand.

### Default configs at a glance

All defaults are the v2/rank=16 schedule (`milestones: [10,20,50,250,400,500,600]`):

| Runbook | Default config(s) | Default manifest |
|---|---|---|
| `run_main_bolt_vs_orpt_train.sh` | `peptide_main_bolt_v2.yaml` (BOLT), `peptide_main_orpt_h1_v2.yaml` (ORPT-H1), `peptide_ablation_orpt_h0.yaml` (ORPT-H0) | — (training only) |
| `run_main_baselines_train.sh` | `peptide_main_bolt_v2.yaml` | — (training only) |
| `run_main_bolt_vs_orpt_mi_eval.sh` | `peptide_main_v2_eval_gpu{0-7}.yaml` (per-GPU pointer configs) | `main_v2_orpt_vs_bolt.yaml` (+ ablation step: `ablation_h0_vs_h1.yaml`) |

The old rank=4 `[126..900]` schedule's configs were removed (its manifest/results are kept as a
frozen historical record under `manifests/main_bolt_vs_orpt_mi*`/`results/main_bolt_vs_orpt_mi*`,
but there's no config left to re-run it). `peptide_smoke.yaml`/`peptide_smoke_mi_orpt.yaml` are for
quick pipeline debugging only — not used by any runbook.

## Manifests, configs, and domains

Every script takes a **manifest** (`manifests/*.yaml`, an `(arm, milestone, run_dir,
checkpoint_dir)` list — see [`manifests/main_v2_orpt_vs_bolt.yaml`](manifests/main_v2_orpt_vs_bolt.yaml)
for an example) and a **`--config`**. The manifest decides which models get evaluated;
`--config` never selects a model, it's only read for constants shared by the whole comparison
(`similarity_threshold`, `oracle_budget`, `table_k_checkpoints`, task universe) — any one of the
manifest's underlying configs works, as long as they agree on those fields. `checkpoint_dir` can be
omitted once raw proposals already exist, or for self-seeding baselines (see below).

`--domain {peptide,query_plan}` (default `peptide`) selects which oracle/task-identity convention
applies (`domains.py`'s `Domain` object). STBO/MTBO/OptFormer/POGPE/SGPE/LLAMBO are **peptide-only**
— `query_plan_experiment` has no training pipeline for them yet (or for ORPT/DPO itself).

Task universes are fixed, not disk-discovered: peptide `trainset`/`heldout`/`heldout100` =
`range(min(cfg.milestones))` / `cfg.heldout_tasks("heldout20")` / `cfg.heldout_tasks("heldout100")`;
query-plan `trainset`/`heldout` = `task_splits.train_workloads()[:min(cfg.milestones)]` /
`cfg.heldout_tasks`.

Internal arm names don't have to match the paper's own terminology (e.g. `ORPT-MI`/`ORPT-H1` are
both internal names for the paper's "ORPT" method) — see [`paper_labels.py`](paper_labels.py).

### Self-seeding baselines

STBO, MTBO, OptFormer, POGPE, SGPE, LLAMBO don't need step 0 (raw proposal generation) — they build
their own init pool directly inside `fixed_target_rejection_bo.py` (mutations of the reference
task, or their own history-conditioned propose/score loop). POGPE/SGPE overload `milestone` to mean
expert count (5/10/20); LLAMBO and STBO are milestone-independent (an unused placeholder).

---

## Experiment 1: Proposal-level incumbent vs. pool size (no BO)

Cheap: "how much does the best score improve as the model draws more feasible proposals?" Compute
engine behind `fig:fewshot`.

| | |
|---|---|
| Compute | [`incumbent_vs_pool_size.py`](incumbent_vs_pool_size.py) — reads step-0 raw proposals, rescores via the domain's oracle, records the best-of-`n_proposals` incumbent + rejection rate at each `cfg.table_k_checkpoints` value |
| Plot | [`plot_incumbent_vs_pool_size.py`](plot_incumbent_vs_pool_size.py) — incumbent-vs-milestone, incumbent-vs-`n_proposals`, and coverage-rate charts |
| Output | `results/<manifest stem>/{per_task,summary}_incumbent_vs_pool_size.csv`, `.../plots/*.png` |

```bash
python experiments/eval/generate_raw_proposals.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/incumbent_vs_pool_size.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_incumbent_vs_pool_size.py --results-dir $RESULTS
```

---

## Experiment 2: Fixed-target pool BO

Forces every task onto the **same size** feasible pool, then runs real BO — comparing models on an
equal initialization footing. Launches real LOLBO runs (expensive — sanity-check with
`--limit-tasks`/`--target-pool-sizes 10` first). Compute engine behind `fig:main-bo` and
`fig:scaling`'s final-BO panel.

| | |
|---|---|
| Compute | [`fixed_target_rejection_bo.py`](fixed_target_rejection_bo.py) — `--target-pool-sizes` (default `10,20,50`); tasks that can't reach the target are skipped and counted via `coverage_rate`, not silently averaged away |
| Plot | [`plot_fixed_target_rejection_bo.py`](plot_fixed_target_rejection_bo.py) — BO-objective-vs-milestone, vs-target-size, rejection-rate, and coverage-rate charts |
| Output | `results/<manifest stem>/{fixed_target_bo_coverage,per_task_fixed_target_bo,summary_fixed_target_bo}.csv`, `.../plots/*.png` (heavy per-task BO artifacts under `<run_dir>/eval_fixed_target_bo/`) |

```bash
python experiments/eval/fixed_target_rejection_bo.py --config $CONFIG --manifest $MANIFEST
python experiments/eval/plot_fixed_target_rejection_bo.py --results-dir $RESULTS
```

---

## Paper-figure scripts

Five scripts, each mapping 1:1 to one `paper/experiments.tex` result-bearing subsection, consuming
Experiment 1/2's CSVs above (no new compute) and emitting paper-terminology-consistent output (see
`paper_labels.py`). All accept comma-separated `--*-results-dir` lists (or, for `fig_main_bo.py`,
`--*-manifest`, since it needs each spec's own checkpoint/run_dir), and either domain may be omitted
(that panel/row is skipped with a printed note, or rendered as `--`).

| Script | `experiments.tex` target | Source |
|---|---|---|
| [`fig_main_bo.py`](fig_main_bo.py) | `fig:main-bo` ("Full-budget optimization") | dense per-task trajectory CSVs under `eval_fixed_target_bo/` |
| [`fig_fewshot.py`](fig_fewshot.py) | `fig:fewshot` ("Initialization and few-shot proposal quality") | `summary_incumbent_vs_pool_size.csv` |
| [`fig_scaling.py`](fig_scaling.py) | `fig:scaling` | both summary CSVs |
| [`tab_main_bo_summary.py`](tab_main_bo_summary.py) | `tab:main-bo-summary` (new label) | both summary CSVs |
| [`tab_ablation.py`](tab_ablation.py) | `tab:ablation` (new label) | both summary CSVs, from `manifests/ablation_h0_vs_h1.yaml` |

```bash
python experiments/eval/fig_main_bo.py --peptide-config $CONFIG --peptide-manifest $MANIFEST
python experiments/eval/fig_fewshot.py --peptide-config $CONFIG --peptide-results-dir $RESULTS
python experiments/eval/fig_scaling.py --peptide-config $CONFIG --peptide-results-dir $RESULTS
python experiments/eval/tab_main_bo_summary.py --peptide-config $CONFIG --peptide-results-dir $RESULTS
```

`tab_ablation.py` has its own manifest/config (`ablation_h0_vs_h1.yaml`) — see its docstring for the
real prerequisite (ORPT-H0 training) before trusting its output; BOLT/ORPT-H1 rows are shared with
the main experiment (`paper_labels.ABLATION_SCALE_NOTE`).

Compute Cost (`sec:compute-overhead`) has no script — no GPU-hour-tracking infra exists in this
repo; that table is filled in manually.

---

## Tips

- **Multiple result sets in one chart**: every `plot_*.py`/`fig_*.py`/`tab_*.py` accepts a
  comma-separated `--results-dir` (or `--*-results-dir`) list — useful since a full milestone sweep
  is typically split across per-GPU shards (`results/<manifest stem>__gpu<N>/`).
- **Restricting to a subset of arms**: every `plot_*.py` accepts `--arms` (e.g. `--arms
  BOLT,ORPT-H1`), combine with `--out-dir` to keep the subset separate from the full comparison.
- `results/` is gitignored (`experiments/*/results/` glob).

## File layout

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
├── fig_main_bo.py / fig_fewshot.py / fig_scaling.py       # paper figures
├── tab_main_bo_summary.py / tab_ablation.py               # paper tables
├── common.py / plot_common.py        # shared utilities (manifest loading, pool construction, styling)
├── run_main_bolt_vs_orpt_train.sh    # training runbook: BOLT + ORPT-H1 + ORPT-H0
├── run_main_baselines_train.sh       # training runbook: MTBO/OptFormer/GP-expert-transfer
├── run_main_bolt_vs_orpt_mi_eval.sh  # eval runbook: every step above + paper-figure scripts
└── results/                          # output CSVs/PNGs, one subdir per manifest stem (gitignored)
```
