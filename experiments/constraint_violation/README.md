# experiments/constraint_violation

Post-hoc analysis of the peptide BOLT-vs-ORPT comparison: does ORPT's DPO training make
it propose sequences that violate the domain's similarity>=0.75 constraint more often
than BOLT-SFT, and if so, what does that actually cost in downstream BO performance?
This is *not* part of the paper reproduction itself (that's `../../peptide_experiment/`)
— it's a separate eval/comparison suite built on top of that pipeline's output.

All scripts are run from the BOLT repo root and take `--config
peptide_experiment/configs/<experiment_id>.yaml` (the same config used to train the
checkpoints being analyzed). Generated CSVs/PNGs go under `results/<experiment_id>/`
(gitignored — regenerate from `runs/` rather than expecting them to be tracked).

## Prerequisites

These scripts read raw model output that `peptide_experiment` doesn't produce by
default — run these first for whichever (arm, milestone) you want to analyze:

```bash
# Held-out proposals (Table-11-style init pool, already part of the main pipeline)
python -m peptide_experiment.cli init_only_eval \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \
    --arm all --tasks heldout20

# Proposals on the model's *own* training tasks (this directory's own eval path —
# nothing in peptide_experiment evaluates a milestone against its training set)
python experiments/constraint_violation/run_trainset_eval.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml
```

## Pipeline: compute script -> plot script

Each analysis is a compute script (writes CSVs to `results/<experiment_id>/`) plus a
matching `plot_*.py` (reads those CSVs, writes PNGs to `results/<experiment_id>/plots/`).
Palette/axis-style/multi-experiment-merge helpers used by every `plot_*.py` live in
`plot_common.py` — edit that file, not each plot script, to change the shared look.

| Question | Compute | Plot |
|---|---|---|
| How often do raw proposals violate the similarity constraint? | `measure_violation_rate.py` | `plot_violation_rate.py` |
| Table-11-style MIC, broken out per task/taggable per variant | `table11_variant.py` | `plot_table11.py` |
| What happens if you *actually* reject infeasible proposals and run real BO on the survivors? | `run_rejection_sampled_bo.py` (expensive: launches real BO) -> `summarize_rejection_sampled_bo.py` | `plot_rejection_sampled_bo.py` |
| Fine-grained BO convergence curve (not just a few k checkpoints) | `incumbent_curve.py` (post-processes `run_rejection_sampled_bo.py`'s output, no new BO) | `plot_incumbent_curve.py` |
| Does a proposal's sampling frequency track its score? | `peptide_frequency_alignment.py` | `plot_peptide_frequency_alignment.py` |
| For a lexicographic-pairing ORPT run: what fraction of DPO training pairs are genuine (both sides feasible) vs. feasibility-gated? | `pair_validity_rate.py` | *(none — prints summary directly)* |

### Example: violation rate

```bash
python experiments/constraint_violation/measure_violation_rate.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

python experiments/constraint_violation/plot_violation_rate.py \
    --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25
```

### Example: Table-11 variant (per-task, taggable)

```bash
python experiments/constraint_violation/table11_variant.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

python experiments/constraint_violation/plot_table11.py \
    --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25
```

### Example: real rejection-sampled BO (expensive — one full BO run per covered task)

```bash
# Sanity-check on one milestone first
python experiments/constraint_violation/run_rejection_sampled_bo.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \
    --milestones 14

# Full sweep once that looks right
python experiments/constraint_violation/run_rejection_sampled_bo.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

python experiments/constraint_violation/summarize_rejection_sampled_bo.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml
python experiments/constraint_violation/incumbent_curve.py \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

python experiments/constraint_violation/plot_rejection_sampled_bo.py \
    --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25
python experiments/constraint_violation/plot_incumbent_curve.py \
    --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25
```

### Comparing two experiments (e.g. two ORPT variants) in one chart

Every `plot_*.py` (except `plot_peptide_frequency_alignment.py`, which is single-task
by design) accepts a comma-separated `--results-dir` list and concatenates them before
plotting — use `--variant-label` on the matching compute script first so the two runs'
`ORPT` rows don't collide:

```bash
python experiments/constraint_violation/measure_violation_rate.py \
    --config peptide_experiment/configs/peptide_100task_orpt_lexicographic.yaml \
    --arms ORPT --variant-label ORPT-LEX

python experiments/constraint_violation/plot_violation_rate.py \
    --results-dir experiments/constraint_violation/results/peptide_100task_orpt_beta0.25,experiments/constraint_violation/results/peptide_100task_orpt_lexicographic \
    --out-dir experiments/constraint_violation/results/comparison_beta0.25_vs_lexicographic
```

### Example: pair validity rate (lexicographic ORPT only, no plot)

```bash
python experiments/constraint_violation/pair_validity_rate.py \
    --config peptide_experiment/configs/peptide_100task_orpt_lexicographic.yaml
```
