# peptide_experiment

Reproduces the BOLT paper's peptide experiment (Table 11 + Figure 1/2) and extends it
with ORPT (a DPO-style fine-tuning stage on top of BOLT-SFT). One config file per
experiment, one CLI entry point — no more `peptide_script.sh` / `peptide_script_dpo.sh`.

Background/design rationale lives in `../imp_plan/` (gitignored, local planning docs);
this file is only the "how do I run it" reference.

## Concepts

- **Milestone**: a #tasks-trained checkpoint of the LLM proposal policy (e.g. `BOLT-10`
  = fine-tuned after 10 tasks' worth of BO trajectories). Main-experiment milestones are
  `[10, 20, 50, 250, 400, 500, 600]`.
- **Arm**: `BOLT-<milestone>`, `ORPT-<milestone>` (only if the config sets
  `build_orpt: true`), or `STBO` (a random-mutation baseline, no LLM at all).
- **H0 / H1**: `mi_bo_steps` selects the preference-pair labeling horizon —
  `1` (H1) evaluates one real BO acquisition round per matched pool; `0` (H0) ranks
  purely by the two candidates' own pre-rollout objective value, no BO round, no
  additional oracle cost. `mi_require_h0` (default `true`) additionally requires H1's
  winner to *also* win on H0 before a pair is kept — see
  `mi_orpt/pair_construction.py::construct_pairs_for_task`'s `require_h0` docstring.
  Vacuous at `mi_bo_steps: 0` (H0 always agrees with itself).
- **Task set**: `heldout20` (no_bo_milestone_eval / paper's Table 11, peptide indices
  900-919), `heldout100` (bo_scaling_curve / paper's Figure 1/2, indices 900-999), or the
  training tasks themselves (`0..max(milestones)-1`).
- Everything for one run lives under `runs/<experiment_id>/` (gitignored — generated
  output, not source).

## Quick start (smoke test — cheap, run this first)

```bash
# 1. Build the shared trajectory chain + BOLT-<m> + ORPT-<m> checkpoints
python -m peptide_experiment.cli trajectory_chain \
    --config peptide_experiment/configs/peptide_smoke.yaml

# 2. Evaluate it (sampling -> best-of-k -> real BO), see experiments/eval2/
python -m experiments.eval2.cli generate_raw \
    --config peptide_experiment/configs/peptide_smoke.yaml \
    --arms-file experiments/eval2/arm_specs/smoke.yaml --task-sets heldout
```

This config uses a tiny milestone schedule (`[1,2,3]`) and `heldout_tasks_override`
(2 tasks instead of the real 20/100), so it finishes in minutes — use it to confirm the
pipeline works end to end, and to time a single BO run / one-step pair construction call,
before touching a real config. It already has `build_orpt: true`/`mi_bo_steps: 1` set, so
it exercises both the BOLT and the ORPT (H0-AND-H1) checkpoints. See
`experiments/eval2/README.md` for the rest of the eval chain.

## Real experiment

`peptide_main_bolt.yaml` / `peptide_main_orpt_h1.yaml` / `peptide_ablation_orpt_h0.yaml`
are the paper-scale configs (`milestones: [10, 20, 50, 250, 400, 500, 600]`,
`oracle_budget: 500`, `init_size: 100`). Run only what you mean to — even at this reduced
budget, a full 600-task trajectory chain (BO sampling interleaved with SFT/DPO at every
milestone) is real, multi-hour-to-multi-day work; time `peptide_smoke.yaml` first (Quick
start above) to get a grounded estimate before launching:

```bash
# BOLT + ORPT-H1 (H0-AND-H1) trajectory chains, then the ORPT-H0 ablation arm
bash experiments/eval2/pipelines/run_train.sh

# Then the baselines (MTBO/OptFormer/GP experts -- needs BOLT's chain done first)
bash experiments/eval2/pipelines/run_baselines_train.sh

# Or drive one arm by hand, e.g.:
python -m peptide_experiment.cli trajectory_chain \
    --config peptide_experiment/configs/peptide_main_bolt.yaml

# Evaluation is a separate runbook -- see experiments/eval2/README.md
bash experiments/eval2/pipelines/run_eval.sh
```

`trajectory_chain` and the eval runbooks are idempotent/resumable (skip any step whose
output file already exists), so a killed/interrupted run can just be re-launched with
the same command. Pin a GPU via the config's `cuda_visible_devices` field rather than
`CUDA_VISIBLE_DEVICES=...` on the command line (it also has to reach in-process CUDA
calls, not just subprocesses) — check GPU availability with the other users of this
machine before picking one.

Producing the actual `paper/experiments.tex` figures/tables from these runs is
`experiments/eval2/`'s job — see its own README.

## Configs (`configs/`)

| Config | Purpose |
|---|---|
| `peptide_smoke.yaml` | Tiny end-to-end sanity check, BOLT + ORPT (H0-AND-H1) both exercised (own `experiment_id`, never collides with a real run) |
| `peptide_main_bolt.yaml` | Main experiment, BOLT-only arm |
| `peptide_main_orpt_h1.yaml` | Main experiment, ORPT arm — `mi_bo_steps: 1` AND `mi_require_h0` (default `true`), i.e. H0-AND-H1 |
| `peptide_ablation_orpt_h0.yaml` | Ablation study's "Zero-step outcome ranking" arm — `mi_bo_steps: 0`, `mi_require_h0` vacuous |

Only these four peptide configs exist in this repo today; every superseded config
(previous `_v2`/`_v1`/PoC-scale/gridsearch/lexicographic/feasibility-aware variants) has
been removed rather than kept "just in case" — all recoverable from git history if
needed. Write a new config rather than editing one of the above in place — each
`experiment_id` owns its own `runs/<experiment_id>/` output dir, so a stale config edit
can silently mix results from two different runs.

## Config keys (`ExperimentConfig`, see `config.py`)

`peptide_main_orpt_h1.yaml` has the most keys spelled out (including several left at
their default, for documentation); other configs only override what differs from the
default.

Required (no default):

| Key | Meaning |
|---|---|
| `experiment_id` | Names `runs/<experiment_id>/` — must be unique per run |
| `milestones` | #tasks-trained checkpoints to produce (e.g. `[10, 20, 50, 250, 400, 500, 600]`) |
| `oracle_budget` | Oracle-call budget per BO run, counted *after* the init pool (`--max_n_oracle_calls`) — a run costs `init_size + oracle_budget` total oracle calls |
| `init_size` | Initial pool size per BO run (`--num_initialization_points`) |
| `bsz` | LLM generation batch size |
| `sft_epochs` | Epochs per BOLT-SFT milestone training stage |

Optional (default shown):

| Key | Default | Meaning |
|---|---|---|
| `similarity_threshold` | `0.75` | Min edit-distance similarity to the reference peptide for a candidate to be constraint-feasible |
| `task_specific_args` | `bacteria_0` | Objective-function score version; **only supported value** — `your_objective_functions.py` asserts 0 on anything else |
| `torchtune_config` | `qwen_2_5_3B_lora.yaml` | BOLT-SFT torchtune config, from `fine-tuning/peptides/torchtune_config/`; main-experiment configs use `main_qwen_2_5_3B_lora_bsz32_rank16.yaml` |
| `torchtune_recipe` | `lora_finetune_distributed` | Must match `torchtune_config`'s tuning style — use `full_finetune_distributed` with a `_full.yaml` config |
| `base_checkpoint_dir` | `fine-tuning/peptides/ckpt/Qwen2.5-3B-Instruct` | Base model checkpoint (relative paths resolve against `bolt_root`) |
| `max_train_tasks` | `null` | Cap on train-task range; `null` -> `max(milestones)` |
| `cuda_visible_devices` | `null` | Pins subprocess + in-process CUDA to one GPU; `null` -> don't set it. Check GPU availability with other users of the machine before setting |
| `heldout_tasks_override` | `null` | Smoke-test escape hatch — replaces both `heldout20`/`heldout100` with a tiny custom task-index list; `null` -> use the real paper splits |
| `table_k_checkpoints` | `[1, 100, 200, 500, 1000]` | Oracle-call checkpoints for no_bo_milestone_eval / bo_scaling_curve aggregation (paper's Table 11 / Figure 1-2) — keep every value `<= oracle_budget` or it's a meaningless flat duplicate point |
| `build_orpt` | `false` | Train an ORPT-`<m>` DPO stage on top of every BOLT-`<m>` |
| `orpt_epochs` | `1` | Epochs per ORPT milestone training stage |
| `orpt_pairs_per_task` | `1000` | `objective_ranked` only: DPO preference pairs sampled per task (`sample_pairs()` always reaches exactly this count, or raises). `matched_intervention` uses `mi_target_pairs_per_task`/`mi_max_candidates_per_task` instead, since its reliability filter can reject any given candidate |
| `orpt_infeasible_singles_per_task` | `1000` | `dpo_ii_penalty` loss_type only: individually-sampled (not paired) infeasible completions per task, for `InfeasibleSuppressionLoss` |
| `orpt_beta` | `0.1` | DPO loss beta (reference-relative logit scale); `0.25` is the grid-search-confirmed winner (see `peptide_main_orpt_h1.yaml`) |
| `orpt_lr` | `3.0e-4` | DPO optimizer learning rate; `2e-5` is the grid-search-confirmed winner — **write scientific notation with a decimal point** (`2.0e-5`, not `2e-5`), otherwise PyYAML parses it as a string, not a float |
| `orpt_pairing_mode` | `feasible_only` | Preference-pair ranking; `feasible_only` (the only supported mode) ranks by objective score among constraint-feasible candidates only. Superseded `lexicographic`/`feasibility_aware` modes were removed along with ORPT-LEX/ORPT-FA — see `imp_plan/05_pool_orpt_phase1_plan.md` |
| `orpt_torchtune_config` | `qwen_2_5_3B_lora_dpo.yaml` | ORPT torchtune config; alternatives: `qwen_2_5_3B_dpo.yaml` (full DPO, not LoRA), `poc_qwen_2_5_3B_lora_dpo_ii_penalty.yaml` (pair with `orpt_loss_type: dpo_ii_penalty`), main-experiment configs use `main_qwen_2_5_3B_lora_dpo_bsz32_rank16.yaml` |
| `orpt_torchtune_recipe` | `lora_dpo_distributed` | Must match `orpt_torchtune_config` — use `full_dpo_distributed` with `qwen_2_5_3B_dpo.yaml`, or `dpo_ii/recipe.py` with the dpo_ii config |
| `orpt_loss_type` | `dpo` | ORPT loss: `dpo` is the stock `torchtune.rlhf.loss.DPOLoss` DPO-style ranking loss; `dpo_ii_penalty` is an ablation that keeps `dpo`'s pairing/loss untouched and additively suppresses individually-sampled infeasible completions (`fine-tuning/peptides/dpo_ii/loss.py`'s `InfeasibleSuppressionLoss`) — see its docstring. (The feasibility-aware `fa_orpt` loss type this ablation was built against was removed along with ORPT-FA.) |
| `fa_orpt_gamma_i` | `0.5` | dpo_ii_penalty only: target below which an infeasible candidate's score should fall |
| `fa_orpt_lambda_inf` | `1.0` | dpo_ii_penalty only: weight on the infeasible-candidate-suppression term |
| `orpt_pair_source` | `objective_ranked` | Preference-pair *labeling mechanism* (distinct from `orpt_pairing_mode`, which only controls feasibility handling within the `objective_ranked` path): `objective_ranked` ranks by raw score via `make_dpo_train_data_csv.py`; `matched_intervention` labels pairs via a matched one-candidate-intervention evaluated with actual one-step BO instead (`peptide_experiment/mi_orpt/`) — see `paper/method.tex` |
| `mi_num_backgrounds` | `8` | `matched_intervention` only: M, shared backgrounds sampled once per task and reused across every candidate evaluated for it. Do not cut below 8 without checking pair yield — the reliability filter's power is ~1/sqrt(M), and M<8 has been observed to yield ~0 trainable pairs |
| `mi_tau_q` | `1.0` | `matched_intervention` only: reference-aligned candidate distribution's softmax temperature |
| `mi_z_min` | `1.96` | `matched_intervention` only: SNR reliability threshold a pair's paired-difference estimate must clear to be kept |
| `mi_delta_t` | `0.0` | `matched_intervention` only: minimum \|Delta_H\| (numerical tolerance only, not a tunable effect-size floor) |
| `mi_target_pairs_per_task` | `5` | `matched_intervention` only: stop evaluating new candidates once this many reliable pairs are found for a task |
| `mi_max_candidates_per_task` | `20` | `matched_intervention` only: safety cap — give up after evaluating this many candidates, even short of `mi_target_pairs_per_task` (each candidate costs exactly `mi_num_backgrounds` real-BO calls under `mi_bo_steps: 1`; bounds real-BO-call cost when the reliability filter rarely passes) |
| `mi_bo_steps` | `1` | `matched_intervention` only: `1` (H1) runs the paper's actual one-step BO evaluator (real oracle calls during pair construction); `0` (H0) ranks by the two candidates' own pre-rollout objective value, no BO round, no additional oracle cost |
| `mi_require_h0` | `true` | `matched_intervention` only: cumulative H>=0 rule — a pair is kept only if the `mi_bo_steps`-horizon winner is ALSO the H0 (raw pre-rollout `y`) winner. `false` reproduces the pre-2026-09 behavior (the horizon comparison alone decides the label). Vacuous when `mi_bo_steps: 0` |
| `mi_max_tasks_per_milestone` | `null` | `matched_intervention` only: caps pair construction to a fixed-seed uniform subsample of this many tasks from `[0, milestone)` per milestone (pair-construction cost is linear in task count); `null` -> every task, same as BOLT-`<m>`'s own SFT dataset. Only the DPO stage's pair pool is subsampled — BOLT's SFT is unaffected |
| `mi_parallel_gpus` | `null` | `matched_intervention` only: CUDA device ids to round-robin across for concurrent one-step BO calls (each of a candidate's M background evaluations is independent) — cuts wall-clock cost by up to `len(mi_parallel_gpus)`x. `null` -> fully serial on `cuda_visible_devices` (today's behavior). Independent of `cuda_visible_devices` (which still governs trajectory sampling/SFT/DPO training) — check `nvidia-smi` fresh before setting, this doesn't have to match `cuda_visible_devices` |

## Where results land

```
runs/<experiment_id>/
  trajectories/, trajectories_csv/   # shared BO trajectory data (all train tasks)
  checkpoints/BOLT-<m>/, ORPT-<m>/   # fine-tuned LoRA checkpoints per milestone
  orpt_pairs/                        # matched_intervention preference pairs + one-step-BO work dirs, per milestone
  eval_raw/                          # raw LLM proposals per (task_set, arm, milestone)
  eval_fixed_target_bo/              # per-task BO trajectories at a fixed init-pool size
```

The `eval_*` trees are written by `../experiments/eval2/`, which owns every evaluation,
rejection-sampling and figure/table step — see its own README.
