# Synthetic Branin evaluation figures

Run these commands from the `BOLT` directory. The scripts read the existing
`runs/<experiment_id>/heldout/<arm>-<milestone>/task_XXXX.csv` trajectories
(or `heldout/STBO` and `heldout/LLAMBO` for milestone-independent arms).
Generate those with `heldout_eval` first; no peptide evaluation CSVs or extra
oracle calls are needed here. The example plotting command below uses the
currently available v2 BOLT and v3 ORPT runs, with SGPE from the H=1 run. The config supplies shared
task values and budgets; its `experiment_id` does not select a plotted run. All plotted values are `train_y`, the **negative
Branin objective**, so higher is better. The initial pool contains `init_size`
points; BO call 0 is its best score.

```bash
python -m synthetic_experiment heldout_eval \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_bolt_inline_example_smaller.yaml \
  --arm BOLT-50
python -m synthetic_experiment heldout_eval \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller.yaml \
  --arm ORPT-50
```

Generate all figures and the summary table for the 50-task milestone, including
STBO, MTBO, POGPE, SGPE, OptFormer, and LLAMBO:

```bash
python synthetic_experiment/eval2/cli.py all \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller.yaml \
  --milestone 50 \
  --run-dir BOLT=runs/synthetic_branin_50train_20heldout_bolt_v2_smaller \
  --run-dir ORPT=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h1 \
  --run-dir STBO=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h2 \
  --run-dir MTBO=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h1 \
  --run-dir POGPE=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h2 \
  --run-dir SGPE=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h1 \
  --run-dir OptFormer=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h1 \
  --run-dir LLAMBO=runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h1 \
  --out-dir synthetic_experiment/eval2/results/synthetic
```

Use `main_bo`, `fewshot`, `scaling`, or `ablation` in place of `all` for one
output type. `--milestone` defaults to the last configured milestone.
For the inset layout, run `main_bo` with `--main-bo-inset`; it writes
`main_bo_synthetic_new.png` and preserves the standard figure.

Outputs:

- `main_bo_synthetic.png`: mean heldout running best score from BO call 0 to
  `oracle_budget` at the selected milestone. The right panel shows all methods
  on a fixed score range of -4.783 to -4.752. Each BO iteration is marked
  with a method-specific symbol every five calls. Shading shows pointwise
  bands of +/-1 standard error, calculated after removing each task's known
  optimum (-10t); the mean score remains on the original scale.
- `fewshot_synthetic.png`: mean best score among the first 1 through
  `init_size` initial proposals at the selected milestone.
- `scaling_synthetic_init.png` and `scaling_synthetic_finalbo.png`: mean
  heldout score across training milestones, at BO calls 0 and `oracle_budget`.
- `ablation_synthetic.csv` and `.tex`: initialization and final BO means for
  every supplied arm. The table reports only arms with evaluated trajectories.

Missing individual task trajectories are skipped, but an arm with no heldout
trajectories at the selected milestone raises an error. Figures use the available heldout tasks;
`ablation_synthetic.csv` records each arm's task count. For a fair comparison,
use runs with the same heldout task values and verify that task counts match.

OptFormer and SGPE currently have heldout trajectories only at milestone 50, so their
scaling figures show one point. The other supplied arms have trajectories at
all configured milestones. MTBO uses pooled top training observations and a
shared GP with target-task residual updates; its run is configured in
`synthetic_50train_20heldout_inline_example_smaller.yaml`.

## STBO/BOLT/ORPT paper tables

The current completed heldout trajectories support a BOLT-v2 / ORPT-v3
($H=1$) / STBO comparison. The H=2 ORPT run has no heldout ORPT trajectories
yet, so the generated tables label the available ORPT row as $H=1$.

```bash
python synthetic_experiment/eval2/table_baselines.py \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller.yaml \
  --stbo-run runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h2 \
  --bolt-run runs/synthetic_branin_50train_20heldout_bolt_v2_smaller \
  --orpt-run runs/synthetic_branin_50train_20heldout_mi30_composite_v3_smaller_h1 \
  --orpt-horizon 1 \
  --out-dir synthetic_experiment/eval2/results/synthetic
```

This writes `stbo_bolt_orpt_by_milestone.tex` (final simple regret at every
training milestone), `stbo_bolt_orpt_m50.tex` (initial and final simple
regret at milestone 50), and `stbo_bolt_orpt_regret.csv` with unrounded
values and source run paths. Both tables use mean $\\pm$ standard error over
20 heldout tasks and require `booktabs` in the paper preamble. The script
checks that each task has all 55 observations and the expected task value.

## STBO/BOLT/ORPT graphs

After generating the tables, plot the same CSV values with:

```bash
python synthetic_experiment/eval2/plot_baselines.py \
  --csv synthetic_experiment/eval2/results/synthetic/stbo_bolt_orpt_regret.csv \
  --out-dir synthetic_experiment/eval2/results/synthetic \
  --orpt-horizon 1
```

The command saves `stbo_bolt_orpt_by_milestone` and `stbo_bolt_orpt_m50`
as both PNG and PDF. Error bars are standard errors over 20 heldout tasks.
The milestone-50 initial/final graph uses a logarithmic y-axis because the
initial STBO regret is much larger than the other values.

`stbo_bolt_orpt_final_score` and `stbo_bolt_orpt_initialization_best` plot
mean raw `train_y` (higher is better) across milestones. They show mean
lines without error bars: task-to-task score offsets dominate the raw-score
standard error. The CSV retains both raw-score means and standard errors.

The initialization-best figure uses a visibly broken y-axis: the upper
range shows BOLT/ORPT near $-4.8$, and the lower range shows STBO near
$-11.6$. The score values are unchanged; diagonal marks indicate the gap.
