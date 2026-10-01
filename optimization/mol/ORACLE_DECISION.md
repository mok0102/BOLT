# Oracle decision record (Milestone 3)

## Decision

The primary oracle is an **unweighted 4-model ensemble** over frozen, pretrained
DeepPurpose BindingDB-IC50 models:

- `morgan_cnn_bindingdb_ic50`
- `cnn_cnn_bindingdb_ic50`
- `daylight_aac_bindingdb_ic50`
- `morgan_aac_bindingdb_ic50`

**`mpnn_cnn_bindingdb_ic50` — the model originally named in the spec — is excluded.**

Implemented in `optimization/mol/mol_oracle.py::EnsembleMolOracle`. Locked by the
user 2026-09-29, after reviewing the evidence below.

## Why MPNN_CNN was dropped

`optimization/mol/test_oracle_model_comparison_all.py`, run on 40 randomly-sampled
BindingDB targets (>= 15 vocab-ok ligands each, sequence <= 1000 residues, 15
ligands sampled per target, fixed seeds):

| model | mean rho | median rho | frac rho>0.3 | frac rho>0.5 | mean MAE (p) |
|---|---|---|---|---|---|
| mpnn_cnn_bindingdb_ic50 | 0.036 | 0.039 | 0.125 | 0.025 | 2.766 |
| morgan_cnn_bindingdb_ic50 | 0.196 | 0.264 | 0.450 | 0.300 | 1.191 |
| cnn_cnn_bindingdb_ic50 | 0.160 | 0.211 | 0.425 | 0.175 | 0.998 |
| daylight_aac_bindingdb_ic50 | 0.169 | 0.218 | 0.325 | 0.175 | 0.909 |
| morgan_aac_bindingdb_ic50 | 0.311 | 0.325 | 0.550 | 0.300 | 1.131 |
| ensemble of all 5 | 0.159 | 0.186 | 0.350 | 0.150 | 1.042 |
| ensemble of MPNN_CNN + morgan_cnn | 0.087 | 0.039 | 0.250 | 0.025 | 1.613 |
| **ensemble of the 4 above (no MPNN)** | **0.280** | **0.382** | **0.550** | **0.425** | 0.968 |

MPNN_CNN's within-target rank correlation with real measured potency is
statistically indistinguishable from noise (mean rho 0.036; only 1 of 40 targets
exceeded rho=0.5). Two alternative explanations were tested and ruled out before
concluding this is a genuine model-quality issue:

1. **CNN target-encoder truncation** (`DeepPurpose.utils.trans_protein`:
   `MAX_SEQ_PROTEIN = 1000`, silently truncates longer sequences). Re-testing on
   targets restricted to <=1000 residues did not improve correlation (still
   negative for several: Q13547 rho=-0.571, O42275 rho=-0.786). Ruled out.
2. **Replicate/assay measurement noise** (BindingDB pools measurements from many
   labs/protocols per target). Measured directly: median p_stdev across
   >=3-replicate pairs is 0.153, 95th percentile 1.047 -- far too small to explain
   MPNN_CNN's 2.77 mean absolute error in p-scale. Ruled out.

Critically, **every ensemble that includes MPNN_CNN performs worse than the
ensemble without it** -- averaging in a near-noise model dilutes real signal from
the other four rather than being harmlessly averaged out. This is why MPNN_CNN is
excluded outright rather than down-weighted.

## Why the 4-model ensemble over the single best model (morgan_aac)

`morgan_aac_bindingdb_ic50` alone has the best mean rho (0.311). The 4-model
ensemble has a lower mean rho (0.280) but the best **median** rho (0.382) and by
far the best **fraction of well-behaved targets** (42.5% at rho>0.5, vs. 30% for
morgan_aac alone). Since the real deliverable is 950 individual optimization
tasks, the fraction of tasks with a genuinely informative oracle landscape matters
more than the single-model mean -- a task built on a target where the oracle has
near-zero rank signal cannot show real BO headroom regardless of the optimization
method. The ensemble mirrors peptide's own APEX oracle shape (an unweighted mean
over 8 independently trained models), so this is a return to established practice
in this codebase, not a novel design.

The averaging is unweighted, deliberately: weighting members by how well they
performed on *this same* 40-target benchmark would overfit the ensemble to this
one validation draw.

## Throughput consequence (accepted)

`optimization/mol/bench_oracle_throughput.py`, batch_size=512, CPU:

| config | molecules/sec |
|---|---|
| morgan_aac alone | 1309 |
| 4-model ensemble | 356 |

A GPU spot-check (opportunistic measurement while a live peptide `eval2`
pipeline stage was running on other devices -- not a verified-idle measurement,
see the run log) showed negligible GPU speedup for either config (1430/sec and
413/sec respectively): these models use CPU-side classical featurization
(Morgan fingerprints, amino-acid composition), not GPU-heavy neural encoders, so
GPU acceleration is not the lever here.

At the peptide-scale ceiling (900 tasks x up to 20,000 oracle evaluations each,
an upper bound rarely fully spent per task), the 4-model ensemble's worst-case
*single-threaded* cost is ~14 hours; the real pipeline runs BO as parallel
subprocesses across tasks (as peptide's `run_bo` already does), so actual
wall-clock is divided by however many parallel workers are used. Accepted as
tractable without further optimization.

## What still needs to happen (Milestone 4/5, not yet done)

- `EnsembleMolOracle` is not yet wired into the constrained-BO machinery
  (`train_c`/constraint surrogate) or into a `num_calls`-tracking LOL-BO objective
  wrapper -- both remain Milestone 5 work.
- **Oracle-overoptimization audit model** (spec's "Oracle-overoptimization audit"
  section) needs re-deciding: the original plan was "rescore with one
  differently-parameterized model not used as the primary oracle." All 5 available
  BindingDB_IC50 models are now accounted for -- 4 are primary ensemble members,
  and the 5th (MPNN_CNN) was excluded specifically for being unreliable, which
  makes it a poor audit reference too. This needs a decision before Gate 5:
  either source a genuinely independent DTI model from outside this roster, or
  explicitly narrow the audit's claim (e.g. audit robustness to a single held-out
  ensemble member rather than a fully independent model) and document why.
- Task manifest (`optimization/mol/task_manifest/mol_task_manifest.tsv`) was built
  independent of oracle choice and does not need to be regenerated for this
  decision.
