# Finding: the pretrained SELFIES VAE's latent space does not preserve Tanimoto
similarity to seed, even locally (Milestone 4)

## The finding

`optimization/mol/_upstream/lolbo`'s official pretrained SELFIES VAE checkpoint
(the one the spec requires reusing "as literally as possible") reconstructs tiny
molecules exactly (Milestone 1: ethanol round-trips identically) but does **not**
preserve chemical similarity to a point's own encoding for real drug-like
molecules, even at the tightest radius tested.

Measured directly (`git log`/session record; reproducible via the diagnostic
inlined in this file's history): sampling `N(seed_z, radius)` around a real
training seed's own latent encoding and decoding:

| radius | mean Tanimoto(decoded, seed) |
|---|---|
| 0.1 (~seed's own encoding) | 0.149 |
| 0.3 | 0.160 |
| 0.8 (TuRBO's initial trust-region length) | 0.117 |
| 1.5 | 0.091 |
| 3.0 | 0.057 |

Background reference: mean Tanimoto similarity among 5 **completely unrelated**
drug-like molecules (aspirin, caffeine, ibuprofen, paracetamol, diphenhydramine) is
**0.142** -- statistically indistinguishable from the radius=0.1 number above.

## Why this matters

The spec's design (mirroring peptide's `APEXSimilarityConstraint`) wires the
Tanimoto-to-seed constraint into the constrained-BO machinery: one GP constraint
surrogate predicting feasibility from `z`, used to bias Thompson sampling toward
feasible regions (SCBO). That only works if there is *some* local structure in
latent space relating `z` to decoded-similarity-to-seed for the GP to learn. There
isn't, for this checkpoint. Confirmed empirically, not just by absence of a
correlation argument:

- A real constrained-BO smoke run (Milestone 4, `mol_bo_smoke.py` v1) found **zero**
  feasible candidates in 600 real oracle evaluations (100 init + 500 acquired),
  across two real training targets and two tau_mol values, both before and after
  fixing two real bugs found along the way (a missing per-round
  `update_surrogate_model()` call, and a SELFIES-vocab out-of-vocabulary gap in the
  init-candidate generator). Neither fix changed the outcome.
- This is *not* explained by the VAE being generally too lossy: it reconstructs
  small molecules exactly (Milestone 1). It's specific to how this VAE's latent
  geometry relates to Tanimoto similarity for larger, real drug-like molecules.
- Plausible reason (not verified, offered as context): the original LOL-BO paper's
  own benchmarks (GuacaMol MPO tasks, penalized logP) never needed
  decoded-neighborhood locality -- TuRBO's trust region there exploits local
  smoothness of the *objective* in latent space, not similarity-preservation in
  decoded space. This checkpoint was never trained or selected for the latter
  property.

## What was ruled out before concluding this

- **Not a device/plumbing bug**: the constrained-BO engine (ported byte-identical
  from peptide's own generic extension of upstream LOL-BO) runs end-to-end
  correctly once CPU/CUDA placement matched peptide's actual working convention
  (bookkeeping tensors CPU-resident; VAE/GP model CUDA-resident; each method
  cuda()s its own inputs and cpu()s its own outputs at the boundary, exactly
  mirroring `update_next`'s `z_next_.detach().cpu()`).
- **Not a smoke-budget-too-small artifact**: increased from 100 to 600 total
  evaluations (TuRBO's `failure_tolerance` for this latent dim is ~256/bsz
  rounds -- 600 evals safely clears that) with the surrogate retrained every
  round; the zero-feasible outcome was unchanged.
- **No larger/alternate official checkpoint exists**: `nataliemaus/robot` (the
  same authors' follow-up repo) ships the exact same checkpoint file, byte
  identical (sha256 `edffc4b4...`, verified via its Git LFS pointer). No other
  branches, tags, or releases exist in either repo.

## First resolution attempt (superseded): unconstrained latent BO + post-hoc mask

The first fix tried was: drop the constraint from the GP/acquisition entirely
(`train_c=None`), build the init pool only from Tanimoto-feasible candidates, and
mask BO's own trajectory by Tanimoto-to-seed only when computing the reported
"best feasible score" (mirroring `experiments/eval2/domains/peptide.py::
best_objective_at_k`'s never-report-an-unmasked-max rule).

This measurably failed: with `LOLBOState.acquisition()` still sourcing candidates
by Sobol-perturb-then-decode in raw z-space (unchanged from vanilla), **zero** of
500 further BO-proposed candidates were Tanimoto-feasible (100% pre-filtered init
pool, 0/500 BO-proposed). `best_feasible_score` was therefore capped at the init
pool's own best value -- headroom exactly 0.0 -- because latent BO's own proposals
essentially never land in the lead-like neighborhood at all (consistent with the
locality measurements above). Also: fully-unconstrained latent BO wandering
through the rest of the latent space is exactly the "unconstrained de-novo
generation / OOD oracle exploitation" failure mode the domain's lead-optimization
framing exists to avoid, so accepting this outcome (rely on masking alone) is not
a scientifically defensible fallback either, independent of the zero-headroom
result.

## Resolution (locked, user decision 2026-09-29): candidate sourcing swap, not a new BO engine

`optimization/mol/mol_acquisition.py::mol_acquisition_step` replaces
`LOLBOState.acquisition()`'s role for this domain. It changes only where the
per-round Thompson-sampling candidate pool comes from:

- **Vanilla** (`mol_lolbo/utils/bo_utils/turbo.py::generate_batch`, untouched,
  byte-identical port): Sobol-perturb around the trust-region center in raw
  z-space, then decode each candidate through the VAE.
- **Mol** (`mol_acquisition_step`): generate fresh Tanimoto-feasible SELFIES-edit
  candidates directly (`mol_init_candidates.py::
  generate_feasible_candidates_around_seed` -- the one generator shown to actually
  preserve seed-locality), encode them (`vae_forward`, no decode step at all), and
  Thompson-sample among their *known* (x, z) pairs using the existing
  `MaxPosteriorSampling` class unconstrained. The oracle is called only on the
  points Thompson sampling selects.

Because x is known by construction (never decoded from a perturbed z), the VAE's
own decode noise -- the actual source of the locality failure -- never enters the
loop. `LOLBOState.update_next()` (incumbent tracking, `train_x/y/z` growth, TuRBO
success/failure bookkeeping) is called exactly as before, unmodified.

**This is not a new/separate finite-space BO engine** (still forbidden by the
spec): Thompson-sampling over a finite candidate pool is exactly what vanilla
`generate_batch()` already does internally (`X_cand` there is Sobol-sourced but
still finite); only the candidate *source* changed. The GP surrogate only has to
learn `z -> potency`, which is the relationship the original LOL-BO paper's own
benchmarks validated as learnable -- never `z -> Tanimoto-feasibility`, which the
locality measurements above show has no learnable structure in this checkpoint.

**Confirmed working**, real oracle, two training targets, `tau_mol=0.4`, 100 init
+ 500 acquired (5 candidates/round Thompson-sampled from ~20-80 freshly generated
per round):

| target | best feasible @ init | best feasible after BO | headroom (p) | feasible fraction of all 600 evaluated |
|---|---|---|---|---|
| Q8WTS6 | 4.717 | 5.034 | **+0.317** | 600/600 (100%) |
| P48775 | 6.394 | 6.509 | **+0.115** | 600/600 (100%) |

At `tau_mol=0.5` (Q8WTS6): headroom +0.129, but generation yield drops from 18.4%
to 6.5% feasible-per-attempt (more mutation attempts needed per usable candidate,
still cheap since generation is pure RDKit/SELFIES, no oracle calls). `tau_mol=0.4`
is the better yield/selectivity balance and is the current working value; final
locking still requires the broader training-only sweep the original tau_mol smoke
study section describes, now run through this corrected mechanism.

**Same policy applies to the future LLM proposal generator** (not yet built):
LLM-proposed candidates must go through the same Tanimoto pre-filter before being
treated as usable, not be fed to the oracle unfiltered and masked only at report
time -- pre-filtering avoids spending oracle budget on candidates already known to
be out of the lead-like neighborhood.

**Known residual gap**: `objective.num_calls` accounting between the init-pool
path (manual dict assignment, doesn't increment `num_calls`) and
`mol_acquisition_step` (increments per kept candidate) is not yet unified to
peptide's exact `latent_space_objective.__call__` convention. Budget-accounting
correctness is a Milestone 5 concern (real ORPT/BOLT trajectories need exact,
auditable oracle-call counts); this smoke study only needed to establish that
headroom exists at all, which it does.
