# experiments/eval

A clean, **self-contained** evaluation pipeline for comparing BOLT/ORPT/ORPT-FA/
ORPT-LEX (or any other arm) at arbitrary milestones -- independent of
`experiments/constraint_violation/` (no imports from that directory; the only
dependency is the core `peptide_experiment`/`apex_oracle` pipeline). Every
script here takes a **manifest** describing which (arm, milestone, run_dir)
combinations to compare, so a comparison can freely span models trained under
different `experiment_id`s/configs in one invocation -- no per-experiment
re-running + CSV-merging required.

## Manifest format

One YAML file per comparison, e.g. `manifests/poc20_four_arm.yaml`:
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
Relative paths resolve against the repo root. `checkpoint_dir` is optional
once raw proposals already exist for that (arm, milestone).

Every analysis script also needs one `--config <experiment yaml>`, purely for
constants shared across the whole comparison (`similarity_threshold`,
`oracle_budget`, `table_k_checkpoints`, task universe) -- any one of the
manifest's underlying configs works, as long as they agree on those fields
(true by design for the 4-arm PoC configs).

Task universes are fixed, not disk-discovered: `trainset` = the tasks every
compared milestone has already been trained on (`range(min(cfg.milestones))`);
`heldout` = `cfg.heldout_tasks("heldout20")` (respects `heldout_tasks_override`).

## Pipeline

`eval_raw/` (this package's own raw-generation directory, parallel to but
independent of `constraint_violation/`'s `trainset_eval/`/`heldout20/init_only/`
convention) must exist before any analysis script can run:

```bash
python experiments/eval/generate_raw_proposals.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
```

Then, compute script -> plot script, same convention throughout: compute
writes CSVs to `results/<manifest stem>/`, plot reads those CSVs and writes
PNGs to `results/<manifest stem>/plots/`.

| Question | Compute | Plot |
|---|---|---|
| Proposal-level incumbent (best score among first k feasible proposals) and rejection cost to reach k, no BO | `incumbent_vs_pool_size.py` | `plot_incumbent_vs_pool_size.py` |
| BO performance when every task is forced to the *same* fixed-size feasible pool (several target sizes) | `fixed_target_rejection_bo.py` | `plot_fixed_target_rejection_bo.py` |
| BO performance under a real, variable-size reject-and-discard pool from a fixed sampling budget | `fixed_budget_rejection_bo.py` | `plot_fixed_budget_rejection_bo.py` |

```bash
# cheap, no BO -- start here
python experiments/eval/incumbent_vs_pool_size.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml
python experiments/eval/plot_incumbent_vs_pool_size.py \
    --results-dir experiments/eval/results/poc20_four_arm

# expensive -- launches real LOLBO runs; sanity-check with --limit-tasks first
python experiments/eval/fixed_target_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml \
    --limit-tasks 1 --target-pool-sizes 10
python experiments/eval/fixed_budget_rejection_bo.py \
    --config peptide_experiment/configs/peptide_poc20_bolt.yaml \
    --manifest experiments/eval/manifests/poc20_four_arm.yaml \
    --limit-tasks 1
```

Both BO scripts write their heavy per-task artifacts under
`<run_dir>/eval_fixed_target_bo/` / `<run_dir>/eval_fixed_target_bo/` respectively
(next to `eval_raw/`, inside whichever manifest entry's own `run_dir` it came
from) -- only the aggregated CSVs land in `experiments/eval/results/`.

`results/` is gitignored (covered by the repo's existing `experiments/*/results/`
glob).

## Comparing multiple result sets in one chart

Every `plot_*.py` accepts a comma-separated `--results-dir` list and
concatenates them before plotting -- useful if two manifests were run
separately and need to appear together.
