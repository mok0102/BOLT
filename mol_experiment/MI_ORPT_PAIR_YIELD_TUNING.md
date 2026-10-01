# mi_orpt pair-construction yield: tuning mol's config to match peptide's own validated fix

## Context

The first full `run_trajectory_chain` for mol (`mol_smoke_orpt.yaml`: `mi_num_backgrounds=8`,
`mi_max_candidates_per_task=4`, `mi_target_pairs_per_task=2`, `mi_z_min=1.96` -- the
`MolExperimentConfig` class default) produced a real, working ORPT-5/ORPT-10 checkpoint but
with a very thin yield: 1/10 possible target pairs at milestone 5 (5 tasks x target 2), 3/10
at milestone 10. Not a bug -- `pair_construction.py`'s reliability filter did exactly what
it's supposed to -- but thin enough to be worth checking against the exact same problem
peptide already solved for its own production run.

## peptide already diagnosed this exact failure mode

`peptide_experiment/configs/peptide_main_orpt_h1.yaml`'s own comment block documents a real
milestone-600 pair-yield collapse and its root causes, found empirically (not by reasoning
from first principles):

1. **U_1 dilution**: when pair construction reuses the deployment pool size (`init_size`) as
   its own shared background size, the candidate's own marginal contribution to
   `U_1 = max(background + candidate + BO batch)` gets diluted (~1/init_size incumbent rate).
   Fixed by `mi_background_size` -- a pair-construction-only pool size, decoupled from
   `init_size`, deliberately smaller.
2. **Arithmetic knife-edge at z_min=1.96**: whenever exactly 1 of M backgrounds has a nonzero
   diff and the rest are exactly 0, the paired SE formula gives `SE == |mean|` identically, so
   `z = |mean| / (SE + eps)` is always *just under* 1.0 regardless of the true effect size.
   `mi_num_backgrounds=8` (up from 4) reduces how often this pattern occurs but does not
   eliminate it. peptide's chosen production value is `mi_z_min: 1.0` (looser than the
   textbook 1.96), with the **class-level default left at 1.96** -- the relaxation is applied
   as a config-level, empirically-justified override, not a change to what the field defaults
   to when unset. `mol_experiment/config.py` is left unchanged for exactly the same reason.
3. **Not enough candidate pairs attempted**: raising `mi_max_candidates_per_task` (16 -> 24)
   combinatorially raises how many candidate *pairs* get a chance to clear the bar
   (`C(k,2)` pairs from `k` candidates), at a linear real-BO-call cost (`k * mi_num_backgrounds`).
   peptide's validated production values: `mi_num_backgrounds=8`, `mi_max_candidates_per_task=24`,
   `mi_background_size=20`, `mi_z_min=1.0`.

`pair_construction.py`/`background_sampler.py`/`candidate_bank.py`'s statistics (paired
Delta_1/SE/z, `ReferenceAlignedDistribution`) are byte-identical between peptide and mol (see
those files' own docstrings) -- this is a property of the shared algorithm, not of peptide's
domain, so there was no reason to expect mol would be exempt from either failure mode.

## Confirming it's the same failure mode in mol's own data (zero extra compute)

Retroactively re-examined the real per-pair `delta`/`se`/`z` values `mol_smoke_orpt`'s actual
run already printed (`mi_max_candidates_per_task=4`, `mi_num_backgrounds=8`, `mi_z_min=1.96`,
across both milestone 5 and milestone 10 real one-step-BO evaluations -- 60 real comparisons,
no new oracle calls needed):

- Only **9/60 (15%)** cleared `z >= 1.96`.
- **24/60 (40%)** would clear `z >= 1.0`.
- The exact same knife-edge artifact peptide found is directly visible in mol's own numbers:
  repeated `z=1.000`-printed values (`delta=0.0017 se=0.0017`, `delta=-0.0569 se=0.0569`, etc.)
  are cases where `se == |delta|` almost exactly, giving `z` just barely under 1.0 -- these do
  NOT clear even the relaxed `z_min=1.0` threshold, matching peptide's own description of this
  pattern as reduced-but-not-eliminated by `mi_num_backgrounds=8`.

This is strong evidence the mechanism transfers directly, not something to be independently
re-derived for mol.

## Live confirmation via mol's own build_pairs.py CLI

Two quick, targeted re-runs against task 0's existing trajectory + existing BOLT-5 checkpoint
(no SFT/DPO retraining, just the pair-construction step, GPU 0):

| `max_candidates_per_task` | `z_min` | kept pairs (target 5) |
|---|---|---|
| 6  | 1.0 | 1/5 |
| 12 | 1.0 | **5/5** |

Tripling `max_candidates_per_task` (4 -> 12) together with `z_min: 1.96 -> 1.0` took a single
task from "1 pair across all 5 tasks combined" (the original run) to "5/5 target pairs from
one task alone." Real-BO-call cost for one task at these settings: `12 * 8 = 96` (~4 min on
GPU 0, in-process `mol_run_bo` -- no subprocess spin-up cost, unlike peptide's own per-call
overhead).

## What changed

`mol_experiment/config.py`'s class-level defaults are **unchanged**, matching peptide's own
convention of keeping conservative/textbook defaults at the class level and applying validated
overrides in run configs. `mol_experiment/configs/mol_smoke_orpt.yaml` now sets:

- `mi_z_min: 1.0` (was implicitly 1.96)
- `mi_background_size: 20` (was implicitly `None` -> `init_size`; explicit now so a future
  `init_size` change can't silently reintroduce U_1 dilution -- matches this smoke config's
  own `init_size=20` exactly today, so no behavior change here, but the field is no longer
  load-bearing-by-coincidence)
- `mi_max_candidates_per_task: 12` (was 4) -- a deliberately smaller step than peptide's fully
  validated 24, sized for smoke-scale wall-clock; raise toward 24 at paper scale, following
  the same "watch early-milestone yield in the logs" discipline peptide's own comment
  recommends before trusting it at full schedule.
- `mi_target_pairs_per_task: 5` (was 2) -- matches the class default; no longer needs to be
  artificially suppressed now that yield isn't collapsing.
