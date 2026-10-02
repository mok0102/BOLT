# Branin motivational experiments: results

> **SUPERSEDED 2026-10-02.** This documents the *scalar-`t`* Branin task
> family, since replaced by a diverse, affine-transform-based family: it was
> numerically verified that this family's global optimum x-location is
> independent of `t`, which let naive context-free reuse baselines
> (Prior-best-reuse, Context-regression) score deceptively well — see
> Experiment C below. Kept as the historical record of that finding, not as
> current results. Source code at this state: git tag
> `branin-scalar-t-final-20261002`. Full `results/`/`logs/` (gitignored, not
> in git history) archived at `archive/branin_scalar_t_20261002.tgz`
> (repo root). Current results live in `motivation_results.md` one level up.

Source spec: `impl_plan/motivational_exp.txt`. Implementation plan and the
fairness/confound decisions made before running anything:
`IMPLEMENTATION_PLAN.md`. All numbers below are read directly from the saved
CSV/JSON artifacts cited in each section — none are retyped by hand.

## Exact experimental configuration

Reused verbatim from `synthetic_experiment/` (branch `main-orpt-branin`),
per the doc's instruction to reuse the existing task construction/BO/splits
unless there is a clear bug (two such bugs were found and fixed; see
"Implementation deviations" below).

| | value | source |
|---|---|---|
| Training tasks | 50, fixed `t` values | `configs/synthetic_50train_20heldout_inline_example_smaller.yaml` |
| Held-out tasks | 20, fixed `t` values | same |
| Milestones | 2, 5, 10, 20, 30, 40, 50 | same |
| Init size | 5 | same |
| Oracle budget | 50 | same |
| M (matched background sets) | 3 | `mi_num_backgrounds` |
| m_pref (pool size per comparison) | 5 | inferred as `init_size` — see note below |
| Training horizon | H=1 | `mi_bo_steps: 1`, `..._h1.yaml` |
| Base model | Qwen2.5-3B-Instruct, LoRA | `torchtune_config: qwen_2_5_3B_lora.yaml` |

**m_pref note**: `impl_plan/motivational_exp.txt` states "m_pref = 5" but no
field of that name exists anywhere in the repository. The real pair
construction (`mi_orpt/pair_construction.py::construct_pairs_for_task`) draws
backgrounds of size `cfg.init_size - 1` and adds one candidate, i.e. every
matched-intervention comparison embeds a candidate in a pool of exactly
`init_size` points. `init_size=5` in the config matches "m_pref=5" under that
reading; this is the interpretation used throughout Experiment B's
reimplementation (`core/ranking_reversal.py`). If "m_pref" meant something
else, that is a discrepancy to revisit, not silently resolved.

**Global optimum — numerically verified, not assumed** (`core/global_optimum.py`):
`scipy.optimize.differential_evolution` (8 independent seeds) + L-BFGS-B
multi-start polish (20 restarts) + a final Nelder-Mead polish, run directly
against `BraninTask.raw` for every one of the 50 train + 20 held-out task_t
values.

- **70/70 tasks matched the repository-wide `f_t^* = 10·t` shortcut**, to
  within **1.44e-15** (train set) and **0.0** (held-out set) —
  `results/verified_optima_{train,heldout}.json`, one `x_t^*`/`f_t^*` row
  per task.
- Why this holds exactly: the task family's quadratic term is independent of
  `t`; `t` only scales the additive `s·(1-t)·cos(x1)` term. The three global
  minima sit at `x1 ∈ {-π, π, 3π}` (with a matching `x2` inside bounds at
  each), none of which depend on `t` — only the achieved value does.
- Every regret number in this report is computed by `core/regret.py` reading
  these saved files, never by re-deriving or re-hardcoding `10·t` — the
  three existing call sites that do hardcode it
  (`synthetic_experiment/aggregate.py`, `eval2/table_baselines.py`,
  `benchmark_branin_bo_init.py`) are now independently confirmed correct,
  but are not what this report's numbers come from.

## Experiment A — where should cross-task knowledge enter BO?

**What it measures**: simple regret to the verified global optimum, for six
transfer interfaces (STBO/MTBO/POGPE/SGPE/BOLT/ORPT), at every training
checkpoint, at `b ∈ {0, 10, 50}` BO calls after initialization, plus the
full `b=0..50` trajectory at `T=50`. STBO and ORPT trajectories use their own
held-out BO runs; MTBO/POGPE/SGPE/BOLT/STBO are all built from the **same**
underlying 50-task training trajectories (the BOLT-only chain's), not from
ORPT's own chain — see "Implementation deviations" for why that separation
matters.

Panel files: `results/figures/motivation_A_task_scaling.{pdf,png,csv}`,
`results/figures/motivation_A_trajectory_T50.{pdf,png,csv}`.

**Observations** (mean over 20 held-out tasks at `T=50`):

| Method | Regret @ b=0 | Regret @ b=50 |
|---|---|---|
| STBO | 6.857 ± 0.464 | 0.015 ± 0.001 |
| MTBO | 6.857 ± 0.464 | 0.015 ± 0.001 |
| POGPE | 6.857 ± 0.464 | **6.151 ± 0.457** |
| SGPE | 6.857 ± 0.464 | **5.830 ± 0.474** |
| BOLT | 0.058 ± 0.006 | 0.010 ± 0.001 |
| ORPT | 0.025 ± 0.005 | 0.006 ± 0.001 |

- BOLT and ORPT are the only methods with non-trivial regret reduction
  *before any BO call* — their learned initializations start roughly
  100-280x closer to the optimum than the shared uniform initialization
  every other method uses.
- MTBO starts at the same uniform-init regret as STBO/POGPE/SGPE (its
  transfer is in the surrogate/acquisition, not initialization) but
  converges markedly faster across the `b=0..50` trajectory, scaling
  visibly with more completed training tasks in the task-scaling panel.
- **POGPE and SGPE do not meaningfully improve over 50 BO calls on this task
  family** — their final regret (~5.8-6.2) is barely below their own
  starting regret (6.857), i.e. close to the "do nothing" floor. This held
  consistently across every milestone, not just T=50.
- **The task-scaling panel requires a log y-axis to show this** (applied
  after an initial linear-scale version made it visually impossible to tell
  BOLT/ORPT/MTBO/STBO apart, all compressed near zero by POGPE/SGPE's much
  larger values occupying most of the linear range — the same problem
  `synthetic_experiment/eval2/plot_baselines.py` already uses a log y-axis
  to avoid). On the log axis, two things that were invisible on a linear
  one become clear: **ORPT's own initialization regret falls by roughly
  two orders of magnitude as training tasks accumulate** (~0.13 at 2
  training tasks to ~0.003 at 50), a real task-scaling effect specifically
  for the learned-initialization arms, and ORPT stays measurably below BOLT
  at every checkpoint rather than the two being indistinguishable.

## Experiment B — does standalone quality agree with downstream utility?

**What it measures**: for pairs of candidates drawn from real completed
training-task trajectories (10 training tasks, task_0000-task_0009 — fixed,
sequential, not cherry-picked; 10 candidates/task; reference policy = the
real BOLT-50 checkpoint), `reversal_rate(h) = P(sign(Delta_0) != sign(Delta_h))`
and `agreement_Htrain(h) = P(sign(Delta_{H=1}) == sign(Delta_h))`, for
`h ∈ {1,2,3,5,10,20,50}`, both with and without the real
`mi_z_min`-based reliability filter. 450 raw pairs total
(`results/experiment_b/raw_pairs.json`); `n` below is after dropping
near-ties on `Delta_0` or `Delta_h` (never counted as reversals/agreements
either way).

**A confound found and avoided, not just noted**: the existing
`orpt_pairs_*.diagnostics.json` files (the real data that trained the real
ORPT checkpoints) cannot be reused for this measurement. Real pair
construction drops every pair where the rollout-preferred candidate does
not also have the higher standalone score — i.e. it filters out reversals by
construction. Reusing those files would have reported a reversal rate of
exactly zero by survivorship bias, not because none occurred. Experiment B
recomputes `Delta_0`/`Delta_h` from scratch, before any such filter.

Reversal rate (`motivation_B_reversal.{pdf,png,csv}`):

| h | n (unfiltered) | reversal % | n (filtered) | reversal % |
|---|---|---|---|---|
| 1 | 413 | 38.3 [33.6, 42.9] | 317 | 38.5 [33.1, 43.8] |
| 2 | 413 | 47.2 [42.4, 52.0] | 285 | 47.4 [41.6, 53.2] |
| 3 | 409 | 46.9 [42.1, 51.8] | 254 | 51.6 [45.4, 57.7] |
| 5 | 415 | 53.3 [48.5, 58.1] | 218 | 55.0 [48.4, 61.6] |
| 10 | 396 | **59.8 [55.0, 64.7]** | 128 | **64.1 [55.8, 72.4]** |
| 20 | 294 | 44.6 [38.9, 50.2] | 52 | 51.9 [38.3, 65.5] |
| 50 | 188 | 39.9 [32.9, 46.9] | 22 | 31.8 [12.4, 51.3] |

**The mismatch is non-monotonic in h**: it roughly doubles from h=1 (38%)
to a peak at h=10 (~60-64%), then falls back toward h=1's level by h=50
(~32-40%). This holds in both the filtered and unfiltered series. A plausible
reading: at very long horizons, GP-UCB has enough budget left to converge to
similar incumbents regardless of which of the two candidates started the
pool, shrinking `Delta_h` back toward (reliable) agreement with the
standalone signal — but this is offered as a reading of the shape, not
claimed as confirmed by any further test.

Agreement with the H=1 training signal
(`motivation_B_horizon_agreement.{pdf,png,csv}`):

| h | agreement % [95% CI], unfiltered (n) | agreement % [95% CI], filtered (n) |
|---|---|---|
| 2 | 87.6 [84.4, 90.8] (410) | 94.4 [91.7, 97.1] (285) |
| 3 | 84.4 [80.9, 87.9] (404) | 90.9 [87.4, 94.5] (253) |
| 5 | 77.1 [73.0, 81.2] (406) | 80.6 [75.3, 85.8] (216) |
| 10 | **54.3 [49.3, 59.3] (381)** | **51.6 [42.9, 60.2] (128)** |
| 20 | 55.3 [49.5, 61.1] (284) | 63.5 [50.4, 76.5] (52) |
| 50 | 54.9 [47.7, 62.2] (182) | 68.2 [48.7, 87.6] (22) |

- **Revised read (the original write-up overstated this — corrected after
  the user questioned whether the trend supports the intended claim):**
  the well-powered unfiltered series is flat within noise from h=10 through
  h=50 — 54.3/55.3/54.9%, all three 95% CIs mutually overlapping and every
  one of them straddling 50% (chance). Read on its own, that is a clean
  **monotonic decline from ~85-94% agreement at nearby horizons to chance by
  h=10, then a plateau at chance through h=50** — not a dip-then-recovery.
- The filtered series' apparent uptick to 63.5/68.2% at h=20/50 is **not
  distinguishable from its own h=10 value or from chance**: its h=10/20/50
  CIs ([42.9,60.2], [50.4,76.5], [48.7,87.6]) all mutually overlap, and the
  h=20/50 lower bounds (50.4, 48.7) sit almost exactly at chance. This is
  the reliability filter shrinking `n` to 52 and 22 at h=20/50 (it
  increasingly excludes pairs whose `Delta_h` has shrunk toward zero at
  long horizons, where BO has had enough budget to equalize outcomes), not
  a real recovery — stated rather than smoothed over, and superseding the
  "partially recovers" framing this section originally shipped with.
- Net effect on the motivating claim: the generalization gap between the
  H=1 training horizon and longer downstream horizons is real, substantial,
  **and does not close** on this evidence — if anything a cleaner statement
  of the risk Experiment B set out to check than the original non-monotonic
  framing was.

No qualitative Branin-contour example figure was produced (the doc's
optional `motivation_B_example.pdf/png`) — deprioritized given time, not
attempted and abandoned.

## Experiment C — what should a learned initializer optimize?

**What it measures**: downstream BO trajectories at `T=50` for six
initializer variants, holding the base model, training tasks, held-out
tasks, proposal budget, downstream BO, and decoding protocol fixed, varying
only initialization. `motivation_C_initializer_objective.{pdf,png,csv}`,
`motivation_C_table.{csv,tex}`.

| Method | Init. | @5 | @10 | @50 |
|---|---|---|---|---|
| Random | 5.813 | 2.528 | 1.692 | 0.019 |
| Prior-best reuse | 0.016 | 0.016 | 0.016 | 0.010 |
| Context-to-optimum regression | **8.458** | 2.986 | 1.793 | 0.022 |
| BOLT (Top-K SFT) | 0.043 | 0.043 | 0.043 | 0.009 |
| H=0 preference | 0.021 | 0.021 | 0.021 | 0.010 |
| ORPT (H=1) | 0.003 | 0.003 | 0.003 | 0.003 |
| ORPT (H=3), secondary | 0.002 | 0.002 | 0.002 | 0.002 |

- **ORPT (H=1) is the only method that is already within 0.003 of the true
  optimum at initialization** and stays there — it does not need the BO
  budget at all on this task family.
- **A negative result, reported rather than hidden**: Context-to-optimum
  regression (k=3 nearest-neighbor on scalar `task_t`) has *worse*
  initialization regret (8.458) than pure Random (5.813). The simplest
  possible cross-task regressor does not just fail to help here — it
  actively hurts, on this benchmark.
- Prior-best reuse scores surprisingly well (0.016 at init) — **flagged as a
  likely benchmark-structure artifact, not a genuine transfer result**: this
  task family's global optimum location does not depend on `t` (see the
  verified-optimum note above), so "always propose the same point regardless
  of context" is close to a winning strategy specifically *because* of how
  this synthetic family was constructed. This should not be read as evidence
  that naive prior-best reuse generalizes to task families where the optimum
  location actually moves with context.
- H=0 preference (standalone-objective-only DPO) reaches 0.021 at init,
  worse than ORPT (H=1)'s 0.003 and H=3's 0.002 — consistent with
  Experiment B's finding that standalone quality and downstream utility are
  not the same signal, and that training on the latter (ORPT) rather than
  the former (H=0) produces a better initializer here.
- The table above is the citable source for the BOLT/H=0/ORPT(H=1)/ORPT(H=3)
  gap, since the main figure plots all six methods on one full-range axis
  and those four converge close enough to zero to be visually
  indistinguishable there. `motivation_C_initializer_objective.png` now
  overlays a zoomed inset (oracle calls 30-40) in the main panel's empty
  upper-right region to make the same gap visible on the figure itself, not
  just in the table.

**Where the init pools actually land** (`motivation_C_init_pools.pdf/png`,
not a doc-named figure — added on request): each method's raw `init_size`
points plotted on the Branin contour for three held-out tasks (t=0.0, 0.5,
0.95), against the numerically verified optimum.

- The 3-well valley structure is directly visible and confirms the verified-
  optimum note above from a different angle: at t=0.0 the three wells (near
  x1=-π, π, 3π) are sharply resolved, separated by clearly higher "passes"
  along the valley; at t=0.95 the same valley renders almost uniformly
  flat end-to-end, because the well-to-pass height is `2*s*(1-t)` — 20 at
  t=0 but only 1 at t=0.95. This is the geometric reason prior-best reuse
  and context regression are so hard to beat on this family: at high t
  nearly the whole valley floor is close to optimal, not just the three
  points.
- BOLT, ORPT-H0, ORPT-H1, ORPT-H3, and Prior-best reuse all cluster tightly
  around the *same* single well (the one near x1=π) at **all three** task
  values, rather than spreading across the three symmetric wells or
  shifting which one they target as t changes. None of the learned methods
  appear to condition which well they propose on task context either — only
  on finding a well at all.
- Random lands inside and outside the valley both, as expected from uniform
  sampling with no cross-task signal.
- Context-to-optimum regression's single predicted point is a concrete,
  visual instance of its reported negative result: it falls inside the good
  cluster at t=0.0 and t=0.95, but at t=0.5 it lands well outside the valley
  entirely (around x1≈1.1, x2≈5.7) — a visibly bad proposal, consistent with
  its worse-than-Random initialization regret reported above.
- Read as a diagnostic, not additional statistical evidence (same status as
  the doc's own optional qualitative figure for Experiment B).

A second, narrower render of the same underlying data,
`motivation_C_init_pools_orpt_variants.pdf/png` (`generate_init_pools(...,
methods=["Random", "BOLT", "ORPT-H0", "ORPT-H1", "ORPT-H3"])`), drops
Prior-best reuse and Context-to-optimum regression to isolate BOLT vs. the
three ORPT horizon variants vs. Random — requested as "ORPT(H=2)," read as
H=3 since H=2 does not exist in this package (deviation 6 above). Since a
scatter has no linestyle to carry the H=0/H=1/H=3 distinction the way the
regret-trajectory plots do, this view also gives each a distinct marker
shape (triangle/circle/square; `style.ABLATION_MARKER`) on top of their
shared red-family color, so overlapping points stay distinguishable.

A third render, `motivation_C_init_pools_bolt_orpth1_random.pdf/png`
(`methods=["Random", "BOLT", "ORPT-H1"]`), widens coverage from 3 to 10
held-out tasks (`task_indices=range(0, 20, 2)`, i.e. t=0.0, 0.1, ..., 0.9 —
every other task, evenly spaced, deterministic) to check the
same-well-every-task pattern noted above against more than 3 data points.
`generate_init_pools` was generalized to lay panels out on a wrapping grid
(`ncols = min(n, 5)`) rather than a single row for this. Over the wider
sample, BOLT and ORPT (H=1) keep landing in a good well far more reliably
than Random (which regularly lands outside the valley entirely), consistent
with the 3-task version.

**A reading caveat this wider version exposes clearly, not visible at n=3:**
the gold star marks whichever single x* `verify_optimum` happened to return,
one of 3 *exactly tied* global optima (x1 in {-π, π, 3π} — see the verified-
optimum note above) — which well it reports is an arbitrary artifact of that
search, not a property of the task. At t=0.7/0.8/0.9 the star itself jumps
from the left well to the right one between panels for exactly this reason.
Consequently a panel where, e.g., BOLT's cluster sits at a *different* well
than that panel's star (visible at a few of the 10 t values) does not mean
BOLT did worse there — it may have found an equally-optimal well the star
simply isn't marking. Not corrected here (would mean plotting all 3
symmetric wells per panel, a visualization choice rather than a bug fix) —
flagged so this figure isn't misread.

## Implementation deviations from the existing main experiment

Full technical detail in `IMPLEMENTATION_PLAN.md`; summarized here per the
doc's "any implementation deviations" requirement.

1. **Two real, reproducible bugs found and fixed in `synthetic_experiment/`
   itself** (not new code written for this experiment, but required before
   any of it could run on this machine):
   - `tune` (torchtune's CLI) was invoked as a bare string, resolving
     through `PATH` rather than the running interpreter — this machine has
     two Python environments each shipping a `tune` binary with a different
     torch build. Fixed with `steps.py::tune_executable()`.
   - All concurrent `tune run` launches defaulted to torchrun's rendezvous
     port 29500, a host-wide (not per-GPU) resource; running the BOLT-only
     chain alongside the three ORPT (H=0/1/3) chains caused one real
     `EADDRINUSE` crash and one *silent* corruption (a checkpoint "saved
     successfully" per torchtune's own log, immediately followed by this
     repository's own `if not final.exists(): raise` catching that the
     expected files were never produced — traced to FSDP sharding the save
     as if the job had two ranks, consistent with an accidental rendezvous
     merge between two unrelated concurrent single-process launches on the
     same port). Fixed with `steps.py::torchrun_master_port()`
     (`29500 + first visible GPU id`), mirroring `peptide_experiment`'s/
     `mol_experiment`'s identical existing fix for the same failure mode.
   - A separate, independent bug (not a collision): torchtune 0.4.0's
     `FullModelHFCheckpointer` on this install writes flat,
     epoch-suffixed files directly in `output_dir`, not the nested
     `epoch_{N}/` subdirectory `config.py`'s checkpoint-dir helpers assumed
     — ported `materialize_hf_checkpoint`/`cleanup_intermediate_epochs`
     from `mol_experiment`/`peptide_experiment` (which hit and fixed the
     identical torchtune-version mismatch first) rather than re-deriving a
     fix.
2. **MTBO/POGPE/SGPE/STBO run against the BOLT-only chain's trajectories**,
   via a new config (`configs/baselines_on_bolt_chain.yaml`) sharing that
   chain's `experiment_id`/run directory, specifically so they are not
   trained on ORPT-policy-shaped trajectories (a confound identified before
   running anything — see `IMPLEMENTATION_PLAN.md` section 3).
3. **Experiment B is new analysis code** (`core/ranking_reversal.py`) built
   from existing `mi_orpt/` primitives, not an existing script — none
   existed before this work (confirmed by inspection, `IMPLEMENTATION_PLAN.md`
   section 5).
4. **Experiment C's Prior-best-reuse and Context-to-optimum-regression are
   new, deliberately simple baselines** (`core/initializer_baselines.py`) —
   neither existed in the repository.
5. **Figure style was ported from `experiments/eval2/figures/style.py` as it
   exists on `main-orpt-mol`/`main-orpt-peptide`**, not from
   `synthetic_experiment/eval2/style.py` (a visibly stale copy of the same
   file). `main-orpt-branin` carries no `experiments/eval2` source at all
   (confirmed via `git ls-tree`). One deliberate addition: vector PDF output
   alongside PNG (`style.py::savefig` now returns `(pdf_path, png_path)`) —
   neither source style module produced PDF.
6. **H=2 does not exist.** The committed `..._h1.yaml`'s inline comment
   ("H=2, matching this run's cached MI evaluations") is treated as stale,
   not as evidence of an H=2 checkpoint: `mi_bo_steps=1` and the
   experiment_id's own `_h1` suffix are internally consistent with the h0/h3
   sibling configs' convention (`mi_bo_steps` *is* H). Only H=0, H=1, H=3
   exist; H=3 is used as the doc's permitted secondary/optional variant.
7. **Two post-hoc additions made after first viewing the figures, not part
   of the original plan**:
   - Experiment C's three new-baseline colors (`Random`/`PriorBestReuse`/
     `ContextRegression`) were originally three tints of one ColorBrewer
     "Purples" ramp (ported style, ultimately sequential-gradient colors
     repurposed for a qualitative comparison) and were hard to tell apart,
     worst exactly where two of their curves nearly coincide. Replaced with
     three distinct hues (gold/sea-green/purple); `ContextRegression` kept
     its original purple.
   - `figures/figure_c.py::generate_init_pools` (`motivation_C_init_pools.
     pdf/png`) — not one of the doc's named figures, added on request: for
     held-out tasks t=0.0/0.5/0.95 (first/middle/last of the 20, fixed and
     deterministic), each method's raw `init_size` initialization points
     plotted on that task's own Branin contour against the numerically
     verified optimum. See the Experiment C section above for what it shows.

## Reproducibility

All commands below assume the repo root and the venv at
`/bolt-storage/home/mol-venv` (this machine's environment; see
`mol_experiment/ENVIRONMENT.md` for why a dedicated venv exists here).

```bash
VENV=/bolt-storage/home/mol-venv
PKG=experiments/eval2_branin_motivation

# 1. Verify the global optimum (fast, CPU-only, ~2 minutes for 70 tasks)
$VENV/bin/python -c "
import sys; sys.path.insert(0, '.'); sys.path.insert(0, '$PKG')
from pathlib import Path
from synthetic_experiment.config import load_config
from core.global_optimum import verify_many, save_verified_optima
cfg = load_config('synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller.yaml')
save_verified_optima(verify_many(cfg.train_task_values, seed=0), Path('$PKG/results/verified_optima_train.json'))
save_verified_optima(verify_many(cfg.heldout_task_values, seed=1), Path('$PKG/results/verified_optima_heldout.json'))
"

# 2. Train the 4 chains (BOLT-only + ORPT H=0/H=1/H=3), ~1-2 hours on 8 idle GPUs
$VENV/bin/python -m synthetic_experiment trajectory_chain --config synthetic_experiment/configs/synthetic_50train_20heldout_bolt_inline_example_smaller.yaml
$VENV/bin/python -m synthetic_experiment trajectory_chain --config $PKG/configs/h1_gpu23.yaml
$VENV/bin/python -m synthetic_experiment trajectory_chain --config $PKG/configs/h0_gpu45.yaml
$VENV/bin/python -m synthetic_experiment trajectory_chain --config $PKG/configs/h3_gpu67.yaml

# 3. Baselines heldout (STBO/MTBO/POGPE/SGPE/BOLT), all 7 milestones
$VENV/bin/python -m synthetic_experiment heldout_eval --config $PKG/configs/baselines_on_bolt_chain.yaml --arm all

# 4. ORPT-only heldout for each H variant (avoids re-training the baselines under a confounded config)
for cfg in h1_gpu23 h0_gpu45 h3_gpu67; do
  for m in 2 5 10 20 30 40 50; do
    $VENV/bin/python -m synthetic_experiment heldout_eval --config $PKG/configs/$cfg.yaml --arm "ORPT-$m"
  done
done

# 5. Experiment C's two new baselines (Random/PriorBestReuse/ContextRegression)
$VENV/bin/python $PKG/core/run_initializer_baselines.py \
    --config synthetic_experiment/configs/synthetic_50train_20heldout_bolt_inline_example_smaller.yaml \
    --n-train-tasks 50 --arm all

# 6. Experiment B's horizon sweep (10 tasks x 10 candidates x 7 horizons, ~15-20 minutes)
$VENV/bin/python $PKG/core/run_ranking_reversal.py \
    --config $PKG/configs/h1_gpu23.yaml --n-tasks 10 --n-candidates 10 --reference-milestone 50

# 7. Figures (pure post-hoc, no GPU, seconds)
$VENV/bin/python -c "
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
"
```
