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
- **Task set**: `heldout20` (no_bo_milestone_eval / paper's Table 11, peptide indices
  900-919), `heldout100` (bo_scaling_curve / paper's Figure 1/2, indices 900-999), or the
  training tasks themselves (0..899).
- Everything for one run lives under `runs/<experiment_id>/` (gitignored — generated
  output, not source).

## Quick start (smoke test — cheap, run this first)

```bash
# 1. Build the shared trajectory chain + BOLT-<m> (+ORPT-<m>) checkpoints
python -m peptide_experiment.cli trajectory_chain \
    --config peptide_experiment/configs/peptide_smoke.yaml

# 2. no_bo_milestone_eval (init pool only, no BO) on every arm
python -m peptide_experiment.cli init_only_eval \
    --config peptide_experiment/configs/peptide_smoke.yaml \
    --arm all --tasks heldout20

# 3. bo_scaling_curve eval (full BO run per task) on every arm
python -m peptide_experiment.cli heldout_eval \
    --config peptide_experiment/configs/peptide_smoke.yaml \
    --arm all --tasks heldout100

# 4. Aggregate both into no_bo_milestone_eval.csv / bo_scaling_curve.csv
python -m peptide_experiment.cli aggregate \
    --config peptide_experiment/configs/peptide_smoke.yaml --tasks both
```

Outputs: `runs/peptide_smoke_v1/aggregate/{no_bo_milestone_eval,bo_scaling_curve}.csv`. This config uses a
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

# no_bo_milestone_eval / Table 11 (cheap: init-pool-only, no BO)
python -m peptide_experiment.cli init_only_eval \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml \
    --arm all --tasks heldout20
python -m peptide_experiment.cli aggregate \
    --config peptide_experiment/configs/peptide_100task_orpt_beta0.25.yaml --tasks heldout20

# bo_scaling_curve / Figure 1/2 (expensive: one full BO run per task per milestone)
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
| `configs/gridsearch/*.yaml` | Single-milestone ORPT hyperparameter sweep cells (beta/lr grid) |

Superseded candidate-level configs (`peptide_100task_orpt_lexicographic.yaml`,
`peptide_100task_orpt_fa.yaml`, `peptide_poc20_orpt_lex.yaml`, `peptide_poc20_orpt_fa.yaml`,
`peptide_100task_bolt_orpt_fa_bsz32.yaml`) were removed along with the ORPT-LEX/ORPT-FA
implementations — see `imp_plan/05_pool_orpt_phase1_plan.md`. Still present on
`main-swhur-fa-orpt`/other branches if needed.

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
| `table_k_checkpoints` | `[1, 100, 200, 500, 1000]` | Oracle-call checkpoints for no_bo_milestone_eval / bo_scaling_curve aggregation (paper's Table 11 / Figure 1-2) |
| `build_orpt` | `false` | Train an ORPT-`<m>` DPO stage on top of every BOLT-`<m>` |
| `orpt_epochs` | `1` | Epochs per ORPT milestone training stage |
| `orpt_pairs_per_task` | `1000` | `objective_ranked` only: DPO preference pairs sampled per task (`sample_pairs()` always reaches exactly this count, or raises). `matched_intervention` uses `mi_target_pairs_per_task`/`mi_max_candidates_per_task` instead, since its reliability filter can reject any given candidate |
| `orpt_infeasible_singles_per_task` | `1000` | `dpo_ii_penalty` loss_type only: individually-sampled (not paired) infeasible completions per task, for `InfeasibleSuppressionLoss` |
| `orpt_beta` | `0.1` | DPO loss beta (reference-relative logit scale); `0.25` is the grid-search-confirmed winner (see `peptide_100task_orpt_beta0.25.yaml`) |
| `orpt_lr` | `3.0e-4` | DPO optimizer learning rate; `2e-5` is the grid-search-confirmed winner — **write scientific notation with a decimal point** (`2.0e-5`, not `2e-5`), otherwise PyYAML parses it as a string, not a float |
| `orpt_pairing_mode` | `feasible_only` | Preference-pair ranking; `feasible_only` (the only supported mode) ranks by objective score among constraint-feasible candidates only. Superseded `lexicographic`/`feasibility_aware` modes were removed along with ORPT-LEX/ORPT-FA — see `imp_plan/05_pool_orpt_phase1_plan.md` |
| `orpt_torchtune_config` | `qwen_2_5_3B_lora_dpo.yaml` | ORPT torchtune config; alternatives: `qwen_2_5_3B_dpo.yaml` (full DPO, not LoRA), `poc_qwen_2_5_3B_lora_dpo_ii_penalty.yaml` (pair with `orpt_loss_type: dpo_ii_penalty`) |
| `orpt_torchtune_recipe` | `lora_dpo_distributed` | Must match `orpt_torchtune_config` — use `full_dpo_distributed` with `qwen_2_5_3B_dpo.yaml`, or `dpo_ii/recipe.py` with the dpo_ii config |
| `orpt_loss_type` | `dpo` | ORPT loss: `dpo` is the stock `torchtune.rlhf.loss.DPOLoss` DPO-style ranking loss; `dpo_ii_penalty` is an ablation that keeps `dpo`'s pairing/loss untouched and additively suppresses individually-sampled infeasible completions (`fine-tuning/peptides/dpo_ii/loss.py`'s `InfeasibleSuppressionLoss`) — see its docstring. (The feasibility-aware `fa_orpt` loss type this ablation was built against was removed along with ORPT-FA.) |
| `fa_orpt_gamma_i` | `0.5` | dpo_ii_penalty only: target below which an infeasible candidate's score should fall |
| `fa_orpt_lambda_inf` | `1.0` | dpo_ii_penalty only: weight on the infeasible-candidate-suppression term |
| `orpt_pair_source` | `objective_ranked` | Preference-pair *labeling mechanism* (distinct from `orpt_pairing_mode`, which only controls feasibility handling within the `objective_ranked` path): `objective_ranked` ranks by raw score via `make_dpo_train_data_csv.py`; `matched_intervention` labels pairs via a matched one-candidate-intervention evaluated with actual one-step BO instead (`peptide_experiment/mi_orpt/`) — see `paper/method.tex` |
| `mi_num_backgrounds` | `8` | `matched_intervention` only: M, shared backgrounds sampled once per task and reused across every candidate evaluated for it |
| `mi_tau_q` | `1.0` | `matched_intervention` only: reference-aligned candidate distribution's softmax temperature |
| `mi_z_min` | `1.96` | `matched_intervention` only: SNR reliability threshold a pair's paired-difference estimate must clear to be kept |
| `mi_delta_t` | `0.0` | `matched_intervention` only: minimum \|Delta_1\| (numerical tolerance only, not a tunable effect-size floor) |
| `mi_target_pairs_per_task` | `5` | `matched_intervention` only: stop evaluating new candidates once this many reliable pairs are found for a task |
| `mi_max_candidates_per_task` | `20` | `matched_intervention` only: safety cap — give up after evaluating this many candidates, even short of `mi_target_pairs_per_task` (each candidate costs exactly `mi_num_backgrounds` real-BO calls; bounds real-BO-call cost when the reliability filter rarely passes) |
| `mi_bo_steps` | `1` | `matched_intervention` only: `1` runs the paper's actual one-step BO evaluator (real oracle calls during pair construction); `0` is the zero-step ablation (rank by the pool's own best already-known value, no BO round, no additional oracle cost) |
| `mi_parallel_gpus` | `null` | `matched_intervention` only: CUDA device ids to round-robin across for concurrent one-step BO calls (each of a candidate's M background evaluations is independent) — cuts wall-clock cost by up to `len(mi_parallel_gpus)`x. `null` -> fully serial on `cuda_visible_devices` (today's behavior). Independent of `cuda_visible_devices` (which still governs trajectory sampling/SFT/DPO training) — check `nvidia-smi` fresh before setting, this doesn't have to match `cuda_visible_devices` |

## Where results land

```
runs/<experiment_id>/
  trajectories/, trajectories_csv/   # shared BO trajectory data (all train tasks)
  checkpoints/BOLT-<m>/, ORPT-<m>/   # fine-tuned LoRA checkpoints per milestone
  heldout20/, heldout100/            # per-task held-out eval output, per arm
  aggregate/no_bo_milestone_eval.csv # Table 11 replica
  aggregate/bo_scaling_curve.csv     # Figure 1/2-style scaling curve
```

Downstream constraint-violation-rate / rejection-sampling analysis (not part of the
paper reproduction itself) lives in `../experiments/eval/` — see its own README for that
pipeline. (The older `../experiments/constraint_violation/` framework it superseded was
removed — see `imp_plan/05_pool_orpt_phase1_plan.md`.)
