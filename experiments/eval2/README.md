# experiments/eval2/

Everything evaluation, for the peptide domain: the compute engines, the
train/eval pipeline runbooks, and the `paper/experiments.tex` figures and
tables. Nothing about evaluation lives in `peptide_experiment/` — that
package is training only, and the dependency runs one way (`eval2` imports
`peptide_experiment`, never the reverse).

Run everything as a module from the BOLT repo root:

```bash
python -m experiments.eval2.cli <subcommand> ...
```

## Layout

```
cli.py          single entry point, 8 subcommands
core/           plumbing: arm specs, pool building, the CSV contract, paper constants
domains/        peptide science (feasibility, scoring, pool format, trajectory reduction)
compute/        the expensive steps: LLM sampling, best-of-k, real BO
figures/        figure + table generation (pure post-hoc, no GPU)
pipelines/      shell runbooks
arm_specs/      which (arm, milestone) pairs each comparison covers
results/        generated CSVs/PNGs (gitignored)
logs/           runbook logs (gitignored)
```

Imports flow one way: `cli` → `{compute, figures}` → `{domains, core}` →
`peptide_experiment`. `core/` never imports `domains/`, and `figures/` never
imports `compute/` — the summary CSVs are the contract between them, read
through `core/results_io.py`. That is what keeps the figure path runnable
with no GPU, no oracle and no checkpoints.

**Adding a second domain** means adding a module to `domains/` with the same
function surface as `peptide.py` and selecting between them in `cli.py`.
There is deliberately no callback/dataclass indirection in place for that
yet — `peptide.py` is the seam, and one domain does not need a framework.

## Pipeline

Training first, then evaluation. Each is a detachable runbook:

```bash
# 1. BOLT + ORPT-H1 trajectory chains, then the ORPT-H0 ablation arm
nohup bash experiments/eval2/pipelines/run_train.sh \
    > experiments/eval2/logs/train_full.log 2>&1 &

# 2. MTBO / OptFormer / GP-expert baselines (needs BOLT's chain complete)
nohup bash experiments/eval2/pipelines/run_baselines_train.sh \
    > experiments/eval2/logs/baselines_train_full.log 2>&1 &

# 3. Evaluation + figures (RUN_ABLATION=1 to also produce tab:ablation)
nohup bash experiments/eval2/pipelines/run_eval.sh \
    > experiments/eval2/logs/main_eval_full.log 2>&1 &
```

`run_eval.sh` shards across 8 GPUs: `gpu{0..6}` take one milestone each of
the milestone-indexed arms, `gpu7` takes the milestone-independent baselines
(POGPE/SGPE overload the milestone field to mean `n_experts`, so they cannot
share the real milestone axis). Sharding is expressed as `--arms` /
`--milestones` / `--cuda-visible-devices` filters over one arm-spec file, not
as separate manifest files.

**Before launching step 3**: it is the expensive one (one real BO run per
arm × milestone × task). Check step 2's cheap `coverage_rate_at_n_proposals`
output first — whether the LLM arms can even reach `TARGET_POOL_SIZES`
feasible candidates is an open empirical question. Smoke it on one slice:

```bash
python -m experiments.eval2.cli fixed_target_bo \
    --config peptide_experiment/configs/peptide_main_bolt.yaml \
    --arms-file experiments/eval2/arm_specs/main.yaml \
    --arms BOLT --milestones 10 --cuda-visible-devices 0 \
    --task-sets heldout100 --target-pool-sizes 10 --limit-tasks 1 \
    --out-dir experiments/eval2/results/smoke
```

## Subcommands

Compute (needs a GPU and the APEX oracle):

| Subcommand | Produces |
|---|---|
| `generate_raw` | `<run_dir>/eval_raw/<task_set>/<arm>-<milestone>/*.jsonl` |
| `incumbent` | `summary_incumbent_vs_pool_size.csv` (+ per-task) |
| `fixed_target_bo` | `summary_fixed_target_bo.csv`, `fixed_target_bo_coverage.csv`, dense per-task BO trajectories |
| `baselines_report` | stdout: POGPE/SGPE by `n_experts`, LLAMBO token truncation |

Figures (post-hoc, no GPU — just point at results dirs):

| Subcommand | Produces |
|---|---|
| `main_bo` | `main_bo_peptide.png` (fig:main-bo) |
| `fewshot` | `fewshot_peptide.png` (fig:fewshot) |
| `scaling` | `scaling_peptide_init.png`, `scaling_peptide_finalbo.png` (fig:scaling) |
| `ablation` | `ablation_peptide.tex` + `.csv` (tab:ablation) |

Example, regenerating a figure from an existing run:

```bash
python -m experiments.eval2.cli fewshot --milestone 600 \
    --results-dir experiments/eval2/results/main__gpu0,experiments/eval2/results/main__gpu1 \
    --out-dir experiments/eval2/results/main/paper_figures
```

## Arm specs

`arm_specs/*.yaml` replaces the old per-GPU manifest files. One file per
comparison; `default_milestones` is crossed with each arm, and an arm may
override `milestones`, `run_dir` or supply a `checkpoint_template` rendered
with `{run_dir}` and `{milestone}`. An arm with no template is a
self-seeding baseline that builds its own init pool at BO time.

| File | Comparison |
|---|---|
| `main.yaml` | BOLT vs ORPT-H1 vs 6 baselines, 36 (arm, milestone) pairs |
| `ablation.yaml` | BOLT vs ORPT-H0 vs ORPT-H1 (tab:ablation) |
| `smoke.yaml` | `peptide_smoke.yaml`-scale end-to-end check |

Inspect what a file resolves to before running against it:

```bash
python -c "from experiments.eval2.core.arms import load_arms; \
  [print(s.arm, s.milestone, s.checkpoint_dir) for s in load_arms('experiments/eval2/arm_specs/main.yaml')]"
```
