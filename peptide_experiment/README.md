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
| `peptide_100task_orpt_fa.yaml` | ORPT trained with `orpt_loss_type: fa_orpt` (feasibility-aware loss, see `fine-tuning/peptides/fa_orpt/`) instead of the DPOLoss-based `dpo` |
| `configs/gridsearch/*.yaml` | Single-milestone ORPT hyperparameter sweep cells (beta/lr grid) |

Write a new config rather than editing one of the above in place — each `experiment_id`
owns its own `runs/<experiment_id>/` output dir, so a stale config edit can silently mix
results from two different runs.

## Config keys (`ExperimentConfig`, see `config.py`)

`peptide_main.yaml` has every key spelled out (including ones left at their default) as
a reference; other configs only override what differs from the default.

Required (no default):

| Key | Meaning |
|---|---|
| `experiment_id` | Names `runs/<experiment_id>/` — must be unique per run |
| `milestones` | #tasks-trained checkpoints to produce (e.g. `[10, 20, 50, 600]`) |
| `oracle_budget` | Oracle-call budget per BO run (`--max_n_oracle_calls`) |
| `init_size` | Initial pool size per BO run (`--num_initialization_points`) |
| `bsz` | LLM generation batch size |
| `sft_epochs` | Epochs per BOLT-SFT milestone training stage |

Optional (default shown):

| Key | Default | Meaning |
|---|---|---|
| `similarity_threshold` | `0.75` | Min edit-distance similarity to the reference peptide for a candidate to be constraint-feasible |
| `task_specific_args` | `bacteria_0` | Objective-function score version; **only supported value** — `your_objective_functions.py` asserts 0 on anything else |
| `torchtune_config` | `qwen_2_5_3B_lora.yaml` | BOLT-SFT torchtune config, from `fine-tuning/peptides/torchtune_config/`; alternatives there: `qwen_2_5_3B_full.yaml`, `qwen_2_5_7B_full.yaml` |
| `torchtune_recipe` | `lora_finetune_distributed` | Must match `torchtune_config`'s tuning style — use `full_finetune_distributed` with the `_full.yaml` configs above |
| `base_checkpoint_dir` | `fine-tuning/peptides/ckpt/Qwen2.5-3B-Instruct` | Base model checkpoint (relative paths resolve against `bolt_root`) |
| `max_train_tasks` | `null` | Cap on train-task range; `null` -> `max(milestones)` |
| `cuda_visible_devices` | `null` | Pins subprocess + in-process CUDA to one GPU; `null` -> don't set it. Check GPU availability with other users of the machine before setting |
| `heldout_tasks_override` | `null` | Smoke-test escape hatch — replaces both `heldout20`/`heldout100` with a tiny custom task-index list; `null` -> use the real paper splits |
| `table_k_checkpoints` | `[1, 100, 200, 500, 1000]` | Oracle-call checkpoints for Table 11 / Figure 1-2 aggregation |
| `build_orpt` | `false` | Train an ORPT-`<m>` DPO stage on top of every BOLT-`<m>` |
| `orpt_epochs` | `1` | Epochs per ORPT milestone training stage |
| `orpt_pairs_per_task` | `1000` | DPO preference pairs sampled per task |
| `orpt_beta` | `0.1` | DPO loss beta (reference-relative logit scale); `0.25` is the grid-search-confirmed winner (see `peptide_100task_orpt_beta0.25.yaml`) |
| `orpt_lr` | `3.0e-4` | DPO optimizer learning rate; `2e-5` is the grid-search-confirmed winner — **write scientific notation with a decimal point** (`2.0e-5`, not `2e-5`), otherwise PyYAML parses it as a string, not a float |
| `orpt_pairing_mode` | `feasible_only` | Preference-pair ranking: `feasible_only` ranks by objective score among constraint-feasible candidates only; `lexicographic` keeps infeasible candidates and always ranks them behind any feasible one, skipping both-infeasible pairs (fixes naive DPO proposing constraint-violating sequences — see `experiments/constraint_violation/`); `feasibility_aware` is the same as `lexicographic` but also keeps both-infeasible pairs — **required** when `orpt_loss_type: fa_orpt` |
| `orpt_torchtune_config` | `qwen_2_5_3B_lora_dpo.yaml` | ORPT torchtune config; alternatives: `qwen_2_5_3B_dpo.yaml` (full DPO, not LoRA), `qwen_2_5_3B_lora_fa_orpt.yaml` (pair with `orpt_loss_type: fa_orpt`) |
| `orpt_torchtune_recipe` | `lora_dpo_distributed` | Must match `orpt_torchtune_config` — use `full_dpo_distributed` with `qwen_2_5_3B_dpo.yaml`, or `fa_orpt/recipe.py` with `qwen_2_5_3B_lora_fa_orpt.yaml` |
| `orpt_loss_type` | `dpo` | ORPT loss: `dpo` is the stock `torchtune.rlhf.loss.DPOLoss` DPO-style ranking loss; `fa_orpt` is the feasibility-aware loss in `fine-tuning/peptides/fa_orpt/loss.py`, which separates the feasible-vs-feasible objective-ranking signal from the feasible/infeasible-vs-infeasible feasibility signal (see its docstring). Requires `orpt_pairing_mode: feasibility_aware` (enforced in `config.py`'s `__post_init__`) |
| `fa_orpt_gamma_obj` | `0.0` | fa_orpt only: feasible-vs-feasible ranking margin |
| `fa_orpt_gamma_plus` | `0.0` | fa_orpt only: target the winning feasible candidate's reference-relative score should rise above |
| `fa_orpt_gamma_keep` | `-0.1` | fa_orpt only: floor the losing feasible candidate's reference-relative score shouldn't fall below |
| `fa_orpt_lambda_up` | `0.2` | fa_orpt only: weight on the winning-feasible-candidate-should-rise term |
| `fa_orpt_lambda_keep` | `0.2` | fa_orpt only: weight on the losing-feasible-candidate-shouldn't-fall-too-far term |
| `fa_orpt_gamma_f` | `0.0` | fa_orpt only: target the feasible side should rise above, in a feasible-vs-infeasible pair |
| `fa_orpt_gamma_i` | `0.5` | fa_orpt only: target below which an infeasible candidate's score should fall (feasible-vs-infeasible and infeasible-vs-infeasible pairs) |
| `fa_orpt_lambda_inf` | `1.0` | fa_orpt only: weight on infeasible-candidate-suppression terms |

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
