# query_plan_experiment

Reproduces the paper's query-plan-domain BOLT-milestone comparison (Table 1:
BOLT-50/893/1138/1426 vs. the 99-query held-out set), mirroring
`peptide_experiment/`'s package shape. See
`imp_plan/02_query_plan_reimplementation_plan.md` for the full design,
confirmed paper facts, and known assumptions/deviations (no feasibility
constraint for this domain, VAE cold-starts with random weights by default).

## Prerequisites (manual, one-time)

1. IMDB Postgres container running (see
   `optimization/query_plans/query_plan_optimization/imdb_postgres/README.md`);
   export `DB_HOST`/`DB_USER`/`DB_PASSWORD`.
2. `optimization/query_plans/query_plan_optimization/workload/ceb-3k.zip`
   extracted in place.
3. `optimization/query_plans/query_plan_optimization/workload/job/schema.sql`
   (+ the JOB benchmark's own `.sql` query files) present -- confirmed
   missing in this environment; `workload/workloads.py` needs it just to
   import (a module-level side effect), even though this stage never uses
   the JOB workload set. Without it, `DatabaseObjective` can't be imported
   at all, blocking every real-oracle code path.
4. Confirm which Python environment runs `optimization/query_plans/`
   subprocess calls (its own poetry-managed deps, distinct from
   `fine-tuning/query_plans`'s) -- set `lolbo_python` in your config if it
   needs a dedicated interpreter.

## Usage (run from the BOLT repo root)

```bash
# Smoke test first -- tiny milestones/budget, own experiment_id.
python -m query_plan_experiment.cli trajectory_chain \
    --config query_plan_experiment/configs/query_plan_smoke.yaml

python -m query_plan_experiment.cli heldout_eval \
    --config query_plan_experiment/configs/query_plan_smoke.yaml --arm all

python -m query_plan_experiment.cli aggregate \
    --config query_plan_experiment/configs/query_plan_smoke.yaml

# Only after the smoke run is clean, the real paper-scale run:
python -m query_plan_experiment.cli trajectory_chain \
    --config query_plan_experiment/configs/query_plan_main.yaml
```

## Output layout

```
runs/<experiment_id>/
  trajectories/            # LLM-sampled init CSVs (post-first-milestone tasks)
  trajectories_csv/        # per-workload BO trajectory CSVs (train_x, train_y, censoring)
  checkpoints/BOLT-<m>/    # SFT checkpoints
  milestones/              # per-milestone SFT training CSV/JSONL
  heldout/BOLT-<m>/        # held-out eval trajectory CSVs, one per arm
  aggregate/table1.csv     # final Table-1-style output
```

## Config keys

See `query_plan_experiment/config.py`'s `ExperimentConfig` docstrings/inline
comments for every field; `configs/query_plan_main.yaml` spells out all of
them as a reference. `configs/query_plan_smoke.yaml` only overrides what
differs from default.

## Scope

BOLT-milestone reproduction only (Table 1). ORPT/DPO and the baseline arms
`peptide_experiment` grew (MTBO/OptFormer/GP-expert-transfer/LLAMBO) are out
of scope for this pass -- see `imp_plan/00_main_experiment_goal.md`.
