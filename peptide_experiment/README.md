# peptide_experiment

Reproduces the BOLT paper's peptide experiment (Table 11 + Figure 1/2) and extends it
with ORPT (a DPO-style fine-tuning stage on top of BOLT-SFT). One config file per
experiment, one CLI entry point — no more `peptide_script.sh` / `peptide_script_dpo.sh`.

Background/design rationale lives in `../imp_plan/` (gitignored, local planning docs);
this file is only the "how do I run it" reference.

## Concepts

- **Milestone**: a #tasks-trained checkpoint of the LLM proposal policy (e.g. `BOLT-10`
  = fine-tuned after 10 tasks' worth of BO trajectories). Paper milestones are
  `10/20/50/600`.
- **Arm**: `BOLT-<milestone>`, `ORPT-<milestone>` (only if the config sets
  `build_orpt: true`), or `STBO` (a random-mutation baseline, no LLM at all).
- **Task set**: `heldout20` (Table 11, peptide indices 900-919), `heldout100`
  (Figure 1/2, indices 900-999), or the training tasks themselves (0..899).
- Everything for one run lives under `runs/<experiment_id>/` (gitignored — generated
  output, not source).

## Quick start (smoke test — cheap, run this first)

```bash
# 1. Build the shared trajectory chain + BOLT-<m> (+ORPT-<m>) checkpoints
python -m peptide_experiment.cli trajectory_chain \
    --config peptide_experiment/configs/peptide_smoke.yaml

# 2. Table-11-style eval (init pool only, no BO) on every arm
python -m peptide_experiment.cli init_only_eval \
    --config peptide_experiment/configs/peptide_smoke.yaml \
    --arm all --tasks heldout20

# 3. Figure-1/2-style eval (full BO run per task) on every arm
python -m peptide_experiment.cli heldout_eval \
    --config peptide_experiment/configs/peptide_smoke.yaml \
    --arm all --tasks heldout100

# 4. Aggregate both into table11.csv / figure1_2.csv
python -m peptide_experiment.cli aggregate \
    --config peptide_experiment/configs/peptide_smoke.yaml --tasks both
```

Outputs: `runs/peptide_smoke_v1/aggregate/{table11,figure1_2}.csv`. This config uses a
tiny milestone schedule (`[1,2,3]`) and budget, so it finishes in minutes — use it to
confirm the pipeline works end to end before touching a real config.

For the ORPT (DPO) stage, use `peptide_smoke_orpt.yaml` instead (`build_orpt: true`) —
same 4 commands, just a different `--config`. Once any milestone is reached, task
sampling switches from `BOLT-<m>` to `ORPT-<m>` automatically; see
`configs/peptide_smoke_orpt.yaml`'s comments.

## Real experiment

Paper-fidelity settings (`oracle_budget: 20000`, `init_size: 1000`,
`milestones: [10, 20, 50, 600]`) are expensive — a full chain takes many days on one
GPU. Run only what you mean to:

```bash
# One shared BOLT-SFT chain (tasks 0..599), producing BOLT-<m> checkpoints
python -m peptide_experiment.cli trajectory_chain \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml

# Table 11 (cheap: init-pool-only, no BO)
python -m peptide_experiment.cli init_only_eval \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \
    --arm all --tasks heldout20
python -m peptide_experiment.cli aggregate \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml --tasks heldout20

# Figure 1/2 (expensive: one full BO run per task per milestone)
python -m peptide_experiment.cli heldout_eval \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \
    --arm all --tasks heldout100
python -m peptide_experiment.cli aggregate \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml --tasks heldout100
```

`trajectory_chain` and both eval commands are idempotent/resumable (skip any step whose
output file already exists), so a killed/interrupted run can just be re-launched with
the same command. Pin a GPU via the config's `cuda_visible_devices` field rather than
`CUDA_VISIBLE_DEVICES=...` on the command line (it also has to reach in-process CUDA
calls, not just subprocesses) — check GPU availability with the other users of this
machine before picking one.

## Configs (`configs/`)

| Config | Purpose |
|---|---|
| `peptide_smoke.yaml` | Tiny BOLT-only smoke test (own `experiment_id`, never collides with real runs) |
| `peptide_smoke_orpt.yaml` | Same, with `build_orpt: true` |
| `peptide_main.yaml` | Paper-fidelity BOLT-only config (`milestones: [10,20,50,600]`) |
| `peptide_100task_bolt_v1` / `peptide_100task_orpt_beta0.25.yaml` | The real 100-task, 7-milestone sweep this repo's results are based on; `orpt_beta=0.25`/`orpt_lr=2e-5` is the grid-search-confirmed winning ORPT hyperparameter set |
| `peptide_100task_orpt_lexicographic.yaml` | ORPT trained with `orpt_pairing_mode: lexicographic` instead of `feasible_only` (see `orpt.py`/`config.py`) |
| `configs/gridsearch/*.yaml` | Single-milestone ORPT hyperparameter sweep cells (beta/lr grid) |

Write a new config rather than editing one of the above in place — each `experiment_id`
owns its own `runs/<experiment_id>/` output dir, so a stale config edit can silently mix
results from two different runs.

## Where results land

```
runs/<experiment_id>/
  trajectories/, trajectories_csv/   # shared BO trajectory data (all train tasks)
  checkpoints/BOLT-<m>/, ORPT-<m>/   # fine-tuned LoRA checkpoints per milestone
  heldout20/, heldout100/            # per-task held-out eval output, per arm
  trainset_eval/                     # (see ../experiments/constraint_violation/README.md)
  aggregate/table11.csv              # Table 11 replica
  aggregate/figure1_2.csv            # Figure 1/2-style scaling curve
```

Downstream constraint-violation-rate / rejection-sampling analysis (not part of the
paper reproduction itself) lives in `../experiments/constraint_violation/` — see its own
README for that pipeline.
