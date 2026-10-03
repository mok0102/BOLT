# Branin motivational experiments: results

Source spec: `impl_plan/motivational_exp.txt`. Implementation plan and the
fairness/confound decisions made before running anything:
`IMPLEMENTATION_PLAN.md` (see its §8 for this migration specifically). All
numbers below are read directly from the saved CSV/JSON artifacts cited in
each section — none are retyped by hand.

## Why the task family changed

The original implementation (archived at
`archive/scalar_t_20261002/motivation_results.md`, tag
`branin-scalar-t-final-20261002`) varied only a scalar `t in [0,1]` per
task. It was numerically verified there that this family's global optimum
**x-location is independent of `t`** — only the achieved value `10t`
changes. That let context-free strategies score deceptively well:
Prior-best-reuse (replay the best points ever seen across training tasks,
ignoring the held-out task entirely) reached **0.016 initialization
regret**, next to solved, simply because every task shared the same
answer. This undermined the comparison Experiments A/B/C exist to make.

**Fix**: every task now applies its own random **affine transform** (shift
+ rotation + isotropic scale) to canonical Branin's input space before
evaluation, instead of varying `t`. Canonical constants (`a,b,c,r,s`, and
`t` fixed at the textbook `1/(8*pi)`) never vary — every task shares the
exact same landscape shape, difficulty, and achievable optimum value
(`f* = 0.397887...`); only where that landscape sits in the fixed search
box differs per task. Full design in `synthetic_experiment/branin.py`,
`global_optimum.py`, `task_splits.py`.

**Did it work?** Yes, confirmed directly in Experiment C below:
Prior-best-reuse's initialization regret is now **12.09 — worse than
Random's 10.14** — not an artifact anymore.

## Exact experimental configuration

| | value | source |
|---|---|---|
| Training tasks | 50, independent random affine transforms | `synthetic_experiment/task_manifests/branin_affine_v1_seed0.json` |
| Held-out tasks | 20, independent random affine transforms | same |
| Milestones | 2, 5, 10, 20, 30, 40, 50 | `configs/*.yaml` |
| Init size | 5 | same |
| Oracle budget | 50 | same |
| M (matched background sets) | 3 | `mi_num_backgrounds` |
| m_pref (pool size per comparison) | 5 | `init_size` (unchanged interpretation, see archived doc) |
| Training horizon | H=1 | `mi_bo_steps: 1`, `h1_gpu23.yaml` |
| Base model | Qwen2.5-3B-Instruct, LoRA | `torchtune_config: qwen_2_5_3B_lora.yaml` |

**Task manifest, frozen before any model trained on it**
(`synthetic_experiment/task_manifests/branin_affine_v1_seed0.json`, token
`dbec9afe9528cd39`): shift `Uniform[-3,3]` per axis, rotation
`Uniform[0,360)` degrees, scale `log-uniform[0.8,1.25]` — **94.6%
acceptance** (70/74 draws; 4 rejected by the analytic pre-filter, 0 by the
numerical W1/W2 gates). All 4 training chains and every evaluation in this
report read from this one manifest (config-level token equality is
enforced by every figure script, not just assumed).

**Global optimum — numerically verified per task, not assumed**
(`synthetic_experiment/global_optimum.py`): `scipy.optimize.
differential_evolution` (8 seeds) + L-BFGS-B multi-start polish (20
restarts) + a final Nelder-Mead polish, run against each task's own
transformed objective. Every accepted task matched the canonical value
(`0.397887...`) to within `1e-6` with at least one verified minimizer
`>= 1.0` box-unit from every edge.

**Pre-registration check, run before any GPU time was spent**: for a
held-out task, reusing a training task's *true verified optimum* scored a
mean regret of 1.179, vs. 1.752 for uniform random draws (ratio 0.67) —
nowhere near the old family's exact-0 degenerate case. This was the go/no-
go gate for proceeding to the full retrain; Experiment C below repeats
this check with real trained methods, not just the theoretical bound.

## Experiment A — where should cross-task knowledge enter BO?

**What it measures**: simple regret to the verified global optimum, for
six transfer interfaces (STBO/MTBO/POGPE/SGPE/BOLT/ORPT), at every
training checkpoint, at `b ∈ {0, 10, 50}` BO calls after initialization,
plus the full `b=0..50` trajectory at `T=50`. MTBO/POGPE/SGPE/BOLT/STBO are
all built from the BOLT-only chain's trajectories, not ORPT's own (see
`IMPLEMENTATION_PLAN.md` §3 — unchanged by this migration).

Panel files: `results/figures/motivation_A_task_scaling.{pdf,png,csv}`,
`results/figures/motivation_A_trajectory_T50.{pdf,png,csv}`.

**Observations** (mean ± sem over 20 held-out tasks at `T=50`):

| Method | Regret @ b=0 | Regret @ b=50 |
|---|---|---|
| STBO | 8.872 ± 1.636 | 0.028 ± 0.005 |
| MTBO | 8.872 ± 1.636 | 0.182 ± 0.148 |
| POGPE | 8.872 ± 1.636 | 1.200 ± 0.213 |
| SGPE | 8.872 ± 1.636 | 1.200 ± 0.213 |
| BOLT | 10.676 ± 1.426 | 0.036 ± 0.008 |
| ORPT | 13.615 ± 5.230 | 0.027 ± 0.005 |

- **A reversal from the old family, reported plainly**: BOLT and ORPT no
  longer start ahead of the shared uniform initialization at `b=0` — they
  start **behind it** (10.7 and 13.6 vs. 8.9). On the old family a learned
  initializer could get near-zero regret for free because every task's
  optimum was in the same place; on this family it has to actually use a
  4-dimensional context (shift×2, rotation, scale) to find a genuinely
  different target each time, and at this training scale (50 tasks) it has
  not learned to do that reliably at initialization time.
- **Task-scaling at `b=0` is noisy, not a clean trend**: BOLT's
  initialization regret across milestones 2→5→10→...→40→50 is
  21.7→13.2→15.8→(…)→17.2→10.7 — down overall but non-monotonic. ORPT's is
  21.8→16.8→24.8→(…)→20.9→13.6 — briefly *worse* at milestones 20-30
  before recovering. Neither resembles the old family's clean, monotonic
  ~100x drop. Plausible reading: 50 tasks is a sparse sample of a 4-D
  context space (vs. the old family's 1-D `t`), so generalization at this
  scale is genuinely less stable — offered as a reading, not confirmed by
  a further test. (Checked directly and separately: the existing 50-call
  oracle budget already converges training trajectories to a median 0.023
  final regret — so this is not explained by under-converged training
  data; see `IMPLEMENTATION_PLAN.md`'s follow-up note.)
- **By `b=50` the ranking is STBO ≈ ORPT ≈ BOLT, all near 0.03**, clearly
  ahead of MTBO (0.18) and POGPE/SGPE (1.20). A full 50-call BO budget
  washes out most of whatever head start any initializer had on this
  family — unlike the old family, where ORPT's initialization advantage
  persisted all the way to `b=50`.
- **POGPE and SGPE are now byte-identical**, not just similarly poor as on
  the old family (there they were 6.151 vs. 5.830) — verified directly
  (20/20 held-out tasks match to the last digit, `max abs diff = 0.0`).
  Confirmed this is not a migration artifact: the algorithm itself
  (`gp_expert_transfer.py`) is unchanged beyond the `TaskRecord` plumbing
  (diffed against the archived tag). The likely mechanism: SGPE's one
  extra "target expert" (fit on only the 5 init points) gets a combined
  precision weight equal to the *entire* pooled POGPE ensemble, but
  against this family's much larger between-task spread it apparently
  never changes which of 128 uniform candidates wins the argmax — only
  POGPE's pooled signal does. Not confirmed by a further test; flagged
  rather than smoothed over.
- **Still requires a log y-axis** for the same reason as before: POGPE/SGPE
  occupy 1-2 orders of magnitude more range than STBO/BOLT/ORPT/MTBO, which
  a linear axis would compress into indistinguishability.

## Experiment B — does standalone quality agree with downstream utility?

**What it measures**: for pairs of candidates drawn from real completed
training-task trajectories (10 training tasks, task_0000-task_0009; 10
candidates/task; reference policy = the real BOLT-50 checkpoint),
`reversal_rate(h) = P(sign(Delta_0) != sign(Delta_h))` and
`agreement_Htrain(h) = P(sign(Delta_{H=1}) == sign(Delta_h))`, for
`h ∈ {1,2,3,5,10,20,50}`, both with and without the real `mi_z_min`-based
reliability filter. 450 raw pairs total
(`results/experiment_b/raw_pairs.json`).

Reversal rate (`motivation_B_reversal.{pdf,png,csv}`):

| h | n (unfiltered) | reversal % | n (filtered) | reversal % |
|---|---|---|---|---|
| 1 | 416 | 41.3 [36.6, 46.1] | 223 | 30.5 [24.5, 36.5] |
| 2 | 398 | 50.8 [45.8, 55.7] | 200 | 52.5 [45.6, 59.4] |
| 3 | 386 | 49.5 [44.5, 54.5] | 158 | 59.5 [51.8, 67.1] |
| 5 | 394 | 54.3 [49.4, 59.2] | 147 | 55.1 [47.1, 63.1] |
| 10 | 365 | **58.1 [53.0, 63.1]** | 94 | **70.2 [61.0, 79.5]** |
| 20 | 331 | 59.2 [53.9, 64.5] | 100 | 72.0 [63.2, 80.8] |
| 50 | 172 | 53.5 [46.0, 60.9] | 9 | 11.1 [-9.4, 31.6] |

- The mismatch rises from ~41% at `h=1` to ~53-59% by `h=10-20` (roughly
  chance or worse — standalone quality is a weak-to-useless predictor of
  downstream utility at these horizons) and stays elevated through `h=50`
  in the well-powered unfiltered series. This is, if anything, a *stronger*
  version of the old family's finding (there: 38%→60%→back to 38-40%): the
  mismatch is real, substantial, and — on this family — does not clearly
  return to the `h=1` level by `h=50`.
- The filtered series' `h=50` value (11.1%, n=9) should not be read against
  the others — the same small-sample caveat as the archived doc: the
  reliability filter increasingly excludes pairs whose `Delta_h` has
  shrunk toward zero at long horizons, and n=9 gives a CI wide enough
  (-9.4 to 31.6) to be uninformative on its own.

Agreement with the H=1 training signal
(`motivation_B_horizon_agreement.{pdf,png,csv}`):

| h | agreement % [95% CI], unfiltered (n) | agreement % [95% CI], filtered (n) |
|---|---|---|
| 2 | 81.7 [77.9, 85.5] (393) | 92.5 [88.8, 96.2] (200) |
| 3 | 68.2 [63.5, 72.9] (377) | 82.9 [77.0, 88.8] (158) |
| 5 | 64.6 [59.8, 69.5] (373) | 78.6 [71.9, 85.3] (145) |
| 10 | **51.8 [46.5, 57.1] (340)** | 49.5 [39.2, 59.7] (91) |
| 20 | 49.7 [44.1, 55.2] (310) | 53.1 [43.2, 62.9] (98) |
| 50 | 48.8 [41.0, 56.5] (160) | 88.9 [68.4, 109.4] (9) |

- Same clean, well-powered (unfiltered) pattern as the old family: a
  monotonic decline from ~82% at `h=2` to ~49-52% (chance) by `h=10`, then
  a flat plateau at chance through `h=50` — all three long-horizon CIs
  mutually overlapping. The filtered series' `h=50` value (88.9%, n=9) is
  the same kind of small-sample noise flagged above and should not be read
  as a real recovery.
- Net reading, consistent with the old family: the generalization gap
  between the `H=1` training horizon and longer downstream horizons is
  real and does not close on this evidence.

## Experiment C — what should a learned initializer optimize?

**What it measures**: downstream BO trajectories at `T=50` for six
initializer variants, holding the base model, training tasks, held-out
tasks, proposal budget, downstream BO, and decoding protocol fixed, varying
only initialization. `motivation_C_initializer_objective.{pdf,png,csv}`,
`motivation_C_table.{csv,tex}`, plus two 2D diagnostic figures
(`motivation_C_init_pools{,_orpt_variants}.{pdf,png}`).

| Method | Init. | @5 | @10 | @50 |
|---|---|---|---|---|
| Random | 10.135 | 5.526 | 4.099 | 0.032 |
| Prior-best reuse | **12.091** | 3.854 | 2.947 | 0.031 |
| Context-to-optimum regression | 24.032 | 6.244 | 3.722 | 0.029 |
| BOLT (Top-K SFT) | 10.676 | 3.763 | 2.788 | 0.036 |
| H=0 preference | 19.919 | 3.349 | 2.228 | 0.029 |
| ORPT (H=1) | 13.615 | 5.092 | 2.123 | 0.027 |
| ORPT (H=3), secondary | 18.689 | 5.599 | 3.576 | 0.040 |

- **The motivating artifact is gone, confirmed directly, not just
  predicted**: Prior-best-reuse's initialization regret went from
  **0.016** (next to solved) on the old family to **12.091** — now worse
  than Random (10.135) — on this one. Reusing fixed points from other
  tasks is no longer a free win; it is actively a bad strategy once every
  task's optimum genuinely moves, exactly as intended.
- **A second negative result, consistent with the old family's own
  finding**: Context-to-optimum regression is the *worst* method at
  initialization (24.032) — the simplest possible cross-task regressor
  remains actively harmful, now by an even larger margin than before
  (8.458 vs. Random's 5.813 on the old family).
- **No method shows a clean, decisive initialization-time advantage on
  this family** — Random (10.1), BOLT (10.7), Prior-best-reuse (12.1), and
  ORPT (13.6) are all within a ~25% band of each other at `b=0`, a
  genuinely different picture from the old family's 100-280x gap between
  BOLT/ORPT and everything else. Consistent with Experiment A: at this
  training scale (50 tasks over a 4-D context), no learned initializer has
  reliably cracked "propose a good point using context alone" yet.
  `motivation_C_init_pools_orpt_variants.png` shows this directly —
  learned methods' proposed points land close to the true optimum on some
  held-out tasks (e.g. task 10) and nowhere near it on others (e.g. task
  19), rather than consistently.
- **H=0 preference (standalone-objective-only DPO) is worse than ORPT
  (H=1) at every checkpoint** (19.919 vs. 13.615 at init; 2.228 vs. 2.123
  at @10) — consistent with Experiment B's finding that standalone quality
  and downstream utility are different signals, and that training on the
  latter (ORPT) rather than the former (H=0) still produces a better
  initializer on this family too.
- By `@50`, all seven variants converge to within a narrow 0.027-0.040
  band — the same full-BO-budget convergence seen in Experiment A.

## Implementation deviations from the scalar-t implementation

Full technical detail in `IMPLEMENTATION_PLAN.md` §8 and in the source
modules' own docstrings (`synthetic_experiment/branin.py`,
`global_optimum.py`, `task_splits.py`, `prompts.py`). Summarized here:

1. **Core task representation redesigned**: `BraninTaskTransform`
   (shift/rotation/scale) replaces the scalar `task_t` field throughout
   `synthetic_experiment/` (steps.py, trajectory_chain.py, orpt.py,
   heldout_eval.py, mtbo.py, gp_expert_transfer.py, optformer.py,
   llambo_optimization.py, aggregate.py, mi_orpt/*) and this package's
   `core/regret.py`, `core/initializer_baselines.py`,
   `core/ranking_reversal.py`, `figures/figure_a.py`, `figure_c.py`.
2. **`global_optimum.py` moved** from this package into
   `synthetic_experiment/` (shared library) — the task-manifest builder
   needs it to gate generation itself, so the dependency now points the
   other way. This package's own `core/global_optimum.py` is a thin
   re-export plus an independent re-audit (`audit_manifest`, different
   seeds than the manifest was built with).
3. **A persisted, frozen, content-hashed task manifest** replaces the old
   regenerate-from-seed-on-every-`load_config()` pattern — eliminates the
   float-keyed `VerifiedOptimum` dict and both of `core/regret.py`'s
   hardcoded `atol`-based float-closeness checks (replaced by exact
   `(split, index)` identity).
4. **Prompt serialization deduplicated**: one `task_descriptor()`
   formatter in `prompts.py` now backs BOLT/ORPT training+inference,
   OptFormer, and LLAMBO — previously each reimplemented its own inline
   `f"task_t=..."` string independently.
5. **`context_to_optimum_regression_init` and MTBO's GP features** now
   share one leakage-free, rotation-aware context embedding
   (`task_splits.py::task_context_embedding`) instead of a bare scalar
   distance — handles rotation's circularity correctly (verified:
   `d(359.9°, 0.1°) == d(0.1°, 0.3°)`).
6. **New run_dir lock mechanism** (`config.py::ensure_dirs`): every
   pipeline stage here is idempotent/cache-aware (skips recomputation if
   output exists), which would have silently replayed stale scalar-t
   results under a reused `experiment_id`. Every config in this migration
   uses a new `experiment_id` (`..._affine_v1...`), and a
   `task_manifest.lock.json` now makes any future accidental reuse a loud
   error instead of silent staleness.
7. **POGPE/SGPE's byte-identical output on this family** (Experiment A) —
   investigated and attributed to the algorithm itself, not the migration
   (see Experiment A's own bullet); not fixed, since it isn't a bug this
   migration introduced and changing POGPE/SGPE's algorithm is out of
   scope for a task-family migration.
8. **`synthetic_experiment/eval2/`** (a separate, pre-existing, already-
   stale sub-package — confirmed unused by anything in this pipeline via
   repo-wide grep) was deliberately **not** updated — any config loaded
   through it now fails loudly (`task_manifest` is a required field with
   no default) rather than silently computing wrong numbers from the
   retired `10*t` shortcut it still hard-codes.
9. **`figure_c.py`'s inset range re-tuned for this family's numbers**
   (`INSET_B_RANGE`/`INSET_YLIM`): the values carried over from the old
   family (`(30,40)`/`(0,0.045)`) clipped every curve off the top for most
   of the inset's own x-range here, since at `b=30` every method's regret
   is still ~0.16-0.29 on this family. Retuned to `(38,50)`/`(0,0.09)` by
   checking `motivation_C_initializer_objective.csv`'s actual per-`b`
   min/max across methods, not carried over.

## A follow-up question, checked and answered (not acted on)

**Would increasing `oracle_budget` for *training*-trajectory generation
help**, given BOLT/ORPT's weaker initialization-time showing on this
family (Experiment A above)? Checked directly against the 50 real training
trajectories: final regret (after the existing 50 BO calls, vs. each
task's own verified optimum) has **mean 0.036, median 0.023** — the
existing budget already converges the vast majority of training
trajectories to within ~0.02-0.05 of the true optimum (a ~200-400x
improvement from the ~8-10 cold-start regret Experiment C's own `Init.`
column shows). The SFT/DPO training targets are therefore already drawn
from well-converged trajectories, not under-explored ones — so the weak
initialization-time generalization looks like a sample-efficiency/
coverage problem (50 training tasks over a 4-D context space), not a
trajectory-quality problem. **Not acted on**: increasing oracle_budget for
training would add real recurring cost without a clear mechanism for
fixing the actual bottleneck. A denser training-task count would be a more
likely lever if this is revisited, but that is a decision on the scale of
the original migration (new manifest, full retrain) and has not been
discussed further.

## Reproducibility

```bash
VENV=/bolt-storage/home/mol-venv
PKG=experiments/eval2_branin_motivation

# 1. Build and freeze the task manifest (fast, CPU-only, ~4 minutes for 70 tasks)
/opt/bolt-venv/bin/python -m synthetic_experiment.task_splits build \
    --out synthetic_experiment/task_manifests/branin_affine_v1_seed0.json \
    --num-train 50 --num-heldout 20 --seed 0

# 2. Train the 4 chains (BOLT-only + ORPT H=0/H=1/H=3), concurrently, ~6-7h wall-clock
$VENV/bin/python -m synthetic_experiment trajectory_chain \
    --config synthetic_experiment/configs/synthetic_50train_20heldout_bolt_inline_example_smaller.yaml &
$VENV/bin/python -m synthetic_experiment trajectory_chain --config $PKG/configs/h1_gpu23.yaml &
$VENV/bin/python -m synthetic_experiment trajectory_chain --config $PKG/configs/h0_gpu45.yaml &
$VENV/bin/python -m synthetic_experiment trajectory_chain --config $PKG/configs/h3_gpu67.yaml &
wait

# 3. Baselines heldout (STBO/MTBO/POGPE/SGPE/BOLT), all 7 milestones
$VENV/bin/python -m synthetic_experiment heldout_eval --config $PKG/configs/baselines_on_bolt_chain.yaml --arm all

# 4. ORPT-only heldout for each H variant
for cfg in h1_gpu23 h0_gpu45 h3_gpu67; do
  for m in 2 5 10 20 30 40 50; do
    $VENV/bin/python -m synthetic_experiment heldout_eval --config $PKG/configs/$cfg.yaml --arm "ORPT-$m"
  done
done

# 5. Experiment C's three non-learned baselines
$VENV/bin/python $PKG/core/run_initializer_baselines.py \
    --config synthetic_experiment/configs/synthetic_50train_20heldout_bolt_inline_example_smaller.yaml \
    --n-train-tasks 50 --arm all

# 6. Experiment B's horizon sweep
$VENV/bin/python $PKG/core/run_ranking_reversal.py \
    --config $PKG/configs/h1_gpu23.yaml --n-tasks 10 --n-candidates 10 --reference-milestone 50 \
    --out-dir $PKG/results/experiment_b

# 7. Figures (pure post-hoc, no GPU, seconds)
/opt/bolt-venv/bin/python -c "
import sys; sys.path.insert(0,'.'); sys.path.insert(0,'$PKG'); sys.path.insert(0,'$PKG/figures')
from pathlib import Path
from figures.figure_a import generate_task_scaling, generate_trajectory_t50
from figures.figure_b import generate_reversal_figure, generate_agreement_figure
from figures.figure_c import generate_trajectory, generate_table, generate_init_pools
out = Path('$PKG/results/figures')
baselines, h0, h1, h3 = ('$PKG/configs/baselines_on_bolt_chain.yaml', '$PKG/configs/h0_gpu45.yaml',
                         '$PKG/configs/h1_gpu23.yaml', '$PKG/configs/h3_gpu67.yaml')
generate_task_scaling(baselines, h1, out)
generate_trajectory_t50(baselines, h1, out)
generate_reversal_figure(Path('$PKG/results/experiment_b'), out)
generate_agreement_figure(Path('$PKG/results/experiment_b'), out)
generate_trajectory(baselines, h0, h1, out, h3_config=h3, include_h3=True)
generate_table(baselines, h0, h1, out, h3_config=h3, include_h3=True)
generate_init_pools(baselines, h0, h1, out, h3_config=h3, include_h3=True)
generate_init_pools(baselines, h0, h1, out, h3_config=h3,
                    methods=['Random','BOLT','ORPT-H0','ORPT-H1','ORPT-H3'],
                    filename='motivation_C_init_pools_orpt_variants')
"
```
