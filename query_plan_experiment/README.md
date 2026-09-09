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

### MI-ORPT one-step (query plans)

Run from the repository root with the DB environment variables and training
virtual environment configured:

```bash
python -m query_plan_experiment.cli trajectory_chain \
  --config query_plan_experiment/configs/query_plan_smoke_mi_orpt.yaml
```

This uses a separate experiment ID. At each milestone it trains BOLT SFT,
constructs matched-intervention preference pairs, then trains a new LoRA DPO
adapter on that milestone's merged BOLT model. The frozen DPO reference is
that same BOLT model (adapter disabled). Subsequent tasks sample ORPT rather
than BOLT. The new config preserves the current query-plan smoke's 100-call
BO budget, batch size 1, init size 5, SFT 50 epochs, and milestones 1–4.

Candidate banks contain unique integer plans from completed (`censoring=0`)
trajectory observations only. Scores remain negative seconds. Each of three
reserved candidates is inserted into the same eight backgrounds of four
plans; all reserved candidates are excluded from backgrounds. Backgrounds
are sampled without replacement using the SFT reference's assistant-token
log likelihood (including chat termination) and temperature `mi_tau_q`.
Each background has one shared seed across candidates. Evaluations run
serially against the DB, using existing scores for initialization and exactly
one acquisition round (`max_bo_steps=1`), not one cache-miss count.

Utility is the best completed score in the initial pool plus new acquisition
batch. Censored new observations remain in BO data but do not count as exact
completed runtimes. This completed-only policy is specific to this SQL port;
it excludes censored plans from the candidate bank instead of assigning them
an exact utility. At init size 5 and three reserved candidates, at least
seven unique completed plans are required. Completed raw encodings are
deduplicated; different encodings that decode to an equivalent SQL plan are
not canonicalized.

Each eligible task costs 3 x 8 = 24 additional BO rounds per milestone,
independent of the number of retained pairs. Each round proposes `bsz`
candidates; cache hits can reduce new DB evaluations. The paired mean/SE
filter uses `mi_z_min=1.96` and `mi_delta_t=0`; it never forces a label when
no reliable difference exists. A zero-pair milestone stops with an explicit
error rather than saving an untrained ORPT checkpoint.

DPO uses rank 16, alpha 32, batch size 1 and zero warmup, so even one retained
pair gives a training step. SFT/DPO reject datasets too small for one optimizer
step. The ordinary BO budget boundary also now stops before an extra round.

Artifacts live under `runs/query_plan_smoke_mi_orpt_onestep_v1/`:
`orpt_pairs/` holds chosen/rejected chat JSONL and pair statistics;
`mi_evaluations/milestone_<m>/<workload>/` holds each matched pool, trajectory,
and evaluation metadata; `checkpoints/ORPT-<m>/` holds merged DPO outputs.
Completed outputs are reused on rerun. Use a new experiment ID when changing
training data/model/method settings to avoid reusing old artifacts.

```bash
python -m query_plan_experiment.cli heldout_eval \
  --config query_plan_experiment/configs/query_plan_smoke_mi_orpt.yaml --arm all
python -m query_plan_experiment.cli aggregate \
  --config query_plan_experiment/configs/query_plan_smoke_mi_orpt.yaml
python -m unittest discover -s query_plan_experiment/tests -v
```

Evaluation/aggregation include both BOLT and ORPT arms when enabled. The
existing first-BOLT-arm initial-plan baseline convention is retained.

MI one-step evaluations initialize a fresh random VAE using the matched seed;
they do not load the trajectory VAE checkpoint. The VAE architecture is still
required for latent-space BO. Initial plans and scores come from the explicit
pool CSV, so no BAO initialization is read in these evaluations. This differs
from deployment when deployment uses a pretrained VAE. Evaluation cache keys
include this random-VAE policy. Existing preference pairs/ORPT checkpoints
are still reused; use a new experiment ID to rebuild an already completed
MI/DPO run under this policy.

Main MI-ORPT uses `configs/query_plan_main_mi_orpt.yaml` with its own run ID.
It keeps query-plan main scale (1426 training workloads, milestones
50/893/1138/1426, 4000 BO evaluations/task, init size 50, SFT 1 epoch).
MI/DPO settings are transferred from peptide main: 60 candidates, 8
backgrounds, target 10 pairs/task, z_min=1, beta=0.25, DPO lr=2e-5.
These settings are not validated as optimal for SQL. Each eligible task
requires 109 unique completed plans and costs 480 extra BO rounds at each
cumulative milestone; evaluation is serial against the database.
Trajectory initialization uses the available VAE checkpoint; MI still
uses a random VAE. Run with the same CLI and the main MI config path.

### Parallel trajectory intervals

```bash
python -m query_plan_experiment.cli trajectory_chain_batch \
  --config query_plan_experiment/configs/query_plan_main_mi_orpt.yaml \
  --workers 2 --gpu_ids '0,1'
```

`run_trajectory_chain_batch` uses spawned processes for independent workloads
between milestone boundaries. A stage shares a checkpoint, not an init CSV:
each workload keeps its own generated plans and measured scores. The full
stage must finish successfully before SFT and then ORPT train in the parent
process. The next stage uses that new checkpoint. MI pair construction stays
serial; this command parallelizes trajectory tasks only.

Each worker lane has one fixed GPU and runs its assigned workloads sequentially.
With explicit `gpu_ids`, provide at least one distinct visible-device ID per
worker. With no IDs, workers inherit the config/shell GPU visibility and may
share a GPU. SFT/ORPT training retains the config/shell GPU setting; `gpu_ids`
controls trajectory workers only. Python 3.11+ is required for isolated worker
lifetime management. Completed trajectory files are skipped before sampling.
If a worker fails, training does not proceed; other already-running lanes are
allowed to finish and save their outputs before the error returns. Resume with
the same command. Do not run a sequential chain concurrently with the batch
chain using the same experiment ID.

Batch workers now share an interprocess DB gate. Model work remains parallel,
but initial-plan scoring, BO and MI SQL evaluations acquire the gate before
opening a DB connection. The gate is released after connection cleanup;
waiting time is outside the query timer and is not included in y. This is a
mutual-exclusion gate, not a FIFO scheduler. Only one evaluation in the batch
runs at a time; external database clients are not controlled by it.

The gate is enabled only during `trajectory_chain_batch` through the scoped
`BOLT_BATCH_DB_LOCK` environment variable, inherited by spawned workers and
BO/sampling subprocesses. The ordinary `trajectory_chain` does not enable it
and retains its original execution behavior. The environment is restored even
on failure. Lock files are deliberately retained so every worker uses the same
inode; locks themselves are released automatically when a process exits.
Do not delete a lock file while a batch is running. Batch runs targeting the
same DB environment share the default lock path on this host.

If milestones are consecutive (1,2,3,4), each interval has one task and there
is no within-stage parallelism. More workers overlap GPU/model work with DB
execution but do not reduce the sum of SQL execution times. Existing artifacts
from earlier concurrent-DB runs are reused; use a new experiment ID if all y
measurements must be regenerated under the serial-DB policy.

### Timing JSON reports

New executions automatically write reports beneath `runs/<experiment_id>/timings/`:

- `tasks/<workload>/sampling_and_init_<id>.json`: LLM process time,
  model/tokenizer loading, generation, and initial-plan DB scoring.
- `tasks/<workload>/bo_<id>.json`: BO process time, initial-data/VAE setup,
  surrogate updates, candidate generation, timeout selection and DB scoring.
  `run_id` distinguishes ordinary chain, held-out, and MI evaluations.
- `milestones/<m>/sft_<id>.json`: data preparation and SFT process time.
- `milestones/<m>/dpo_<id>.json`: DPO process time and linked pair construction.
- `milestones/<m>/mi_pair_construction_<id>.json`: reference model loading,
  per-task likelihood scoring and linked one-step BO reports.
- `traces/<id>/*.jsonl`: per-process events underlying each report, including
  individual DB evaluation durations and BO iteration indices.

Each JSON contains `wall_seconds`, `phase_seconds`, `phase_counts`, UTC start/end,
status (`completed`, `failed`, or `reused`), and raw trace paths. A fresh ID is used
on every invocation, so retries and parallel workers never overwrite prior reports.
SFT/DPO are shared milestone training, not independent training per task:
`shared_training_tasks` lists all participating tasks. Nested-operation trace
events link child report paths rather than duplicating their times in the parent.

`db_query_execution` is measured elapsed time around SQL execution, including
actual time to cancellation/failure; it is not the timeout threshold or failure
penalty stored as y. `db_queue_wait` measures batch lock waiting separately.
`db_call_including_queue` also includes connection setup/cleanup and decoding.
GPU-related phase times are host wall-clock spans, not CUDA-kernel profiling.
Phase spans nest (e.g. DB inside BO acquisition inside BO process), so summing
all phase totals would double count. Use `wall_seconds` for that invocation's
end-to-end time. Concurrent workers' wall times are not whole-run elapsed time.

Reports are finalized on normal completion or Python exceptions. Raw events
are flushed during execution, but a SIGKILL/power loss may leave no finalized
summary. Historical runs cannot be reconstructed from these new timers. Cached
operations report reuse time, not the original training/execution cost; tasks
skipped entirely by the batch scheduler are not executed or newly timed.

MI small-bank fallback: the earlier strict minimum `init_size - 1 +
mi_max_candidates_per_task` is now the ideal size, not a hard requirement.
With at least 3 unique completed plans, reserve up to min(candidate cap,
bank size - 1) distinct comparison candidates; form backgrounds from the
remaining plans and pad missing slots by weighted resampling. Reserved
candidates never appear in the backgrounds, and existing scores are reused.
This is a multiset approximation, not the original unique-background method;
repeated rows add no new evidence. Pair diagnostics record background_padding,
unique_bank_size and reserved_candidates. Reliability filtering remains active;
zero reliable pairs still stops DPO, and fewer than 3 distinct completed plans
cannot support this construction.
