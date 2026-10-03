# Branin motivational experiments (A/B/C) — implementation plan

Source spec: `impl_plan/motivational_exp.txt`. This file is the doc's own
required step 3 ("write a short implementation plan") + step 5 ("check the
plan for fairness / confounds"), kept here rather than only in chat history.

## 0. Where everything lives

- **Compute/training backend**: reused as-is from `synthetic_experiment/`
  (branch `main-orpt-branin`) — `branin.py`, `steps.py`, `trajectory_chain.py`,
  `orpt.py`, `mtbo.py`, `gp_expert_transfer.py`, `optformer.py`,
  `llambo_optimization.py`, `mi_orpt/`. Per the doc's instruction to reuse
  existing task construction/BO/splits/baselines, **nothing in this list is
  reimplemented** — only extended where the doc requires something that does
  not exist yet (global-optimum verification, Experiment B's horizon sweep,
  Experiment C's two new baselines).
- **This package**: `experiments/eval2_branin_motivation/` — new, isolated,
  gitignored-compatible (`results/`, `logs/` mirror the real `experiments/
  eval2/` convention's own gitignore pattern). Does not touch
  `synthetic_experiment/configs/*.yaml` or `synthetic_experiment/eval2/`.
- **Figure format**: ported from `experiments/eval2/figures/style.py` as it
  exists on `main-orpt-mol`/`main-orpt-peptide` — **not** from
  `synthetic_experiment/eval2/style.py`, which is a visibly stale copy of it
  (missing `ABLATION_LINESTYLE`, the POGPE color fix, `save_csv`, PDF output).
  `main-orpt-branin` has no `experiments/eval2` source at all (confirmed via
  `git ls-tree`; the directory on this branch holds only leftover
  `__pycache__`). Ported by value (`git show main-orpt-mol:... > file`), not
  by cross-branch import.

## 1. Common setup — confirmed against the repository

| Spec requirement | Existing match |
|---|---|
| 50 train / 20 held-out, fixed | `synthetic_50train_20heldout_inline_example_smaller*.yaml`: `train_task_t_override`/`heldout_task_t_override` |
| milestones {2,5,10,20,30,40,50} | same file family, `milestones:` |
| init_size=5, oracle_budget=50 | same |
| M=3 shared matched base sets | `mi_num_backgrounds: 3` |
| m_pref=5 | **not a literal variable name anywhere in the repo.** Read as the per-comparison pool size each matched one-step-BO evaluation embeds a candidate into (`init_size`=5: `mi_num_backgrounds-1`=4 background points + 1 candidate). If this interpretation is wrong, the actual pair count knob is `mi_target_pairs_per_task` (30 in the main config) — flagged, not guessed silently into the final report. |
| H=1 default | `..._h1.yaml`, `mi_bo_steps: 1` |

**Known stale artifact, not followed**: the committed `..._h1.yaml` carries
the inline comment `mi_bo_steps: 1 # H=2, matching this run's cached MI
evaluations.`. The h0/h3 sibling configs establish `mi_bo_steps` *is* H
(`H=0: score the intervention without subsequent BO acquisitions`, `H=3: ...
k BO acquisition steps`), so `mi_bo_steps=1` is H=1, matching the
experiment_id's own `_h1` suffix and the filename. The comment is treated as
stale, not as evidence of an undocumented H=2 config. **No H=2 checkpoint
exists anywhere** (only H=0, H=1, H=3) — the doc's "if existing H=2 or H=3
already exist" secondary-variant clause therefore applies to H=3 only.

**Global optimum — VERIFIED, not assumed** (`core/global_optimum.py`):
`differential_evolution` (8 independent seeds) + L-BFGS-B multi-start polish
(20 restarts) + Nelder-Mead final polish, run against `BraninTask.raw` itself
(never a re-derivation of the formula), on all 50 train + 20 held-out task_t
values. Result: **70/70 match the repo-wide `f_t^* = 10·t` shortcut to
1.44e-15** (`results/verified_optima_{train,heldout}.json`, with `x_t^*`
saved per task — three tied global minima at `x1 ∈ {-π, π, 3π}`, each with a
matching `x2` inside bounds, independent of `t`, since `t` only scales the
`cos(x1)` term). The existing codebase's three independent occurrences of
`optimum = 10.0 * task_t` (`synthetic_experiment/aggregate.py`,
`eval2/table_baselines.py`, `benchmark_branin_bo_init.py`) are therefore
numerically confirmed correct, but every regret computation this package
produces reads `f_t^*` from the saved verification file, never re-hardcodes
`10*t` directly — so the provenance is "verified, saved, then used," matching
the spec's letter, not just its conclusion.

## 2. Environment fixes applied before any run (apply to all of
   `synthetic_experiment/`, not experiment-specific)

- `base_checkpoint_dir` (`fine-tuning/query_plans/ckpt/Qwen2.5-3B-Instruct`)
  did not exist on this machine. Symlinked to the same read-only, domain-
  neutral Qwen2.5-3B-Instruct checkpoint under `fine-tuning/peptides/ckpt/`
  that `mol_experiment` also references (same pattern, same justification:
  base weights are domain-independent, never written to).
- `steps.py`/`trajectory_chain.py`/`orpt.py`/`optformer.py` resolved
  torchtune's `tune` CLI via a bare `"tune"` in argv, i.e. via `PATH`. This
  machine has two venvs that each ship a `tune` binary with a different
  torch build (same hazard already documented in `mol_experiment/
  ENVIRONMENT.md`); confirmed live by checking `which tune` while the mol
  venv was the one meant to run — it resolved to `/opt/bolt-venv/bin/tune`,
  the wrong environment. Added `steps.py::tune_executable()` (resolves from
  `sys.prefix`, not `Path(sys.executable).resolve()` — the latter follows a
  uv venv's symlink out of the venv) and routed all three call sites through
  it. Confirmed fixed: the live BOLT chain's first SFT burst launched via
  `/bolt-storage/home/mol-venv/bin/tune`.

## 3. Fairness decision: what trajectories back MTBO/POGPE/SGPE (and STBO)

`gp_expert_transfer.py::train_gp_expert_pool` and `mtbo.py::
train_mtbo_surrogate` both read `cfg.trajectories_dir` literally — i.e.
whichever run produced the `cfg` object they're called with. The committed
`..._h1.yaml` sets `build_orpt: true` *and* `build_stbo/mtbo/pogpe/sgpe:
true` in the same file, which means if its own `trajectory_chain` were run,
`checkpoint_to_sample_from` would hand every post-milestone-2 task's
*initialization* sampling to the **ORPT** checkpoint
(`cfg.build_orpt` ⟹ ORPT branch), and MTBO/POGPE/SGPE would then be trained
on ORPT-policy-flavored trajectories — not neutral "cross-task experience,"
but experience already shaped by the very rollout-aware method Experiment A
means to compare against. That is exactly the kind of confound the doc's
workflow step 5 asks to catch before implementing.

**Decision: MTBO, POGPE, SGPE, and STBO are all built/evaluated against the
BOLT-only chain's run_dir** (`synthetic_branin_50train_20heldout_bolt_v3_
smaller`), via a new config
(`experiments/eval2_branin_motivation/configs/baselines_on_bolt_chain.yaml`)
that shares that `experiment_id`/`run_dir` and only turns on
`build_stbo/mtbo/pogpe/sgpe: true` — not by editing the committed BOLT-only
or H1 yaml files. Every comparison in Figure A is then transfer-interface-only:
same underlying per-task trajectories (BOLT's own SFT-feeding data), six
different ways of turning that into faster optimization on a new task.
ORPT's own trajectories remain separately rooted in the H1 (or H0/H3) run_dir,
since ORPT's rollout-aware *trajectories themselves* (not just its DPO
training signal) are part of what's being evaluated for that arm specifically.

## 4. Experiment A — task scaling

Reuses `heldout_eval` end-to-end (STBO/MTBO/POGPE/SGPE/BOLT/ORPT all already
implemented). New: a regret-based aggregator (`core/regret.py`, not yet
written) that joins `heldout/<arm>/task_XXXX.csv` against the verified
`f_t^*` file instead of `aggregate.py`'s inline `10*t`, at `b ∈ {0, 5, 10,
50}` plus the full `b=0..50` trajectory at T=50. OptFormer/LLAMBO: evaluated
and retained as supplementary per the doc ("may be evaluated... but the main
purpose... is to contrast transfer interfaces") — not in the primary 6-method
panel.

## 5. Experiment B — standalone vs. rollout ranking

**No existing script computes this.** Built from existing primitives:
`mi_orpt/candidate_bank.py` (bank), `mi_orpt/background_sampler.py`
(reference-aligned matched backgrounds), `mi_orpt/one_step_evaluator.py`
(`run_candidates_one_step`, parameterized by `mi_bo_steps` — this is exactly
the lever Experiment B needs to sweep `h`). New file:
`core/ranking_reversal.py`, sweeping `h ∈ {1,2,3,5,10,20,50}` per matched
pair, computing `reversal_rate(h)` and `agreement_H1(h)` with near-tie
handling (reuse `pair_construction.py`'s own `z = |delta|/se` reliability
criterion — report both filtered and unfiltered, per the doc).

## 6. Experiment C — initializer training signal

Random and Top-K SFT (BOLT) and H=0/H=1(/H=3) all exist. **Two baselines do
not exist and need new, deliberately-simple implementations**, per the doc's
own instruction ("keep implementation deliberately simple and document it"):
- Prior-best reuse (task-independent rule, e.g. the best point(s) seen across
  all completed training tasks so far, reused verbatim as the next task's
  init pool — exact rule to be documented in `motivation_results.md` when
  implemented, not decided here).
- Context→optimum regression (`task_t` → best `x`, a small non-LLM regressor
  — e.g. nearest-neighbor or tiny MLP over the 50 training tasks' own
  best-observed points; "keep it simple" per the doc).

## 7. Status at time of writing (scalar-t family, now superseded — see §8)

- [x] `core/global_optimum.py` written and run: 70/70 verified, saved.
- [x] Environment fixes (checkpoint symlink, `tune_executable`) applied and
      confirmed live.
- [x] BOLT-only trajectory chain, baselines, ORPT H=0/H=1/H=3 chains,
      `core/regret.py`, `core/ranking_reversal.py`, Experiment C's two new
      baselines, all figures — all completed on the scalar-t family. Fully
      documented in `archive/scalar_t_20261002/motivation_results.md`
      (superseded, kept as the historical record of the finding in §8).

## 8. Migration: scalar-t → affine-transform task family (2026-10-02/03)

**Why**: the scalar-t family's global optimum x-location was proven
independent of `t` (§1's task construction only varies the additive
`cos(x1)` term's weight, never the quadratic term that pins the
minimizer). This let Experiment C's context-free baselines
(Prior-best-reuse, Context-regression) score deceptively well — Prior-best-
reuse reached 0.016 initialization regret, next to zero, simply by
replaying old points, because every task shared the same answer. That
undermined the entire comparison this package exists to run.

**What changed**: every task now applies its own random affine transform
(shift + rotation + isotropic scale) to canonical Branin's input space
before evaluation, instead of varying `t`. Full design, ranges, and
rejection-sampling criteria: `synthetic_experiment/branin.py`,
`global_optimum.py`, `task_splits.py` docstrings. The fairness decision in
§3 (baselines read from the BOLT-only chain, not ORPT's own trajectories)
and the fixed-base-model/task-count setup in §1 both still apply unchanged
— only task construction changed, not the comparison's structure.

**Did the fix work?** Confirmed directly: on the new family, Prior-best-
reuse's initialization regret is 12.09 (now *worse* than Random's 10.14),
vs. 0.016 on the old family. The degenerate "free win" is gone — see
`motivation_results.md` (one level up; this file's own §1-§6 describe the
now-archived scalar-t implementation and are not updated for the new
family's task construction specifics).

Status: full re-run complete (4 training chains, all heldout evals,
Experiment B, Experiment C, all figures) — zero errors across every log.
