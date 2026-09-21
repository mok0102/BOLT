# experiments/eval2/

Standalone, paper-ready figure/table generator for `paper/experiments.tex`.
See `/root/.claude/plans/1-llvm-bubbly-jellyfish.md` for the full design
rationale (fully independent of `experiments/eval/`'s clutter-prone
`fig_*.py`/`tab_*.py`, one atomized PNG/tex per domain x panel).

Results go under `experiments/eval2/results/` (already covered by the
repo's existing `experiments/*/results/` gitignore glob -- no new pattern
needed).

All commands below use the **currently active experiment round**
(init_size=100/oracle_budget=500 for peptide, init_size=100/oracle_budget=200
for LLVM, decided 2026-09-21) and are ready to copy-paste. Run from the BOLT
repo root.

**Status as of this writing** -- several background jobs this data depends
on are still running; commands that need them will just print
`skipping`/`no rows` until those finish, not crash:
- Peptide Phase 1 (`main_v2_orpt_vs_bolt__lowbudget`): raw per-task
  trajectories exist (`main_bo` works today), but the run hasn't finished
  and its `summary_*.csv` files aren't written yet (`fewshot`/`scaling`
  need those -- wait for it to finish).
- Peptide Phase 2 (ORPT-H0 low-budget eval): not started -- gated on
  `runs/peptide_ablation_orpt_h0/checkpoints/ORPT-600` existing first.
- LLVM eval (`llvm_main_v1__gpu{4,5,6}`): still on step 1/4, no
  `summary_*.csv` written yet.
- LLVM ablation (`llvm_ablation_h0_vs_h1`): never triggered
  (`run_llvm_main_bolt_vs_orpt_mi_eval.sh`'s `RUN_ABLATION` defaults to 0).

---

## Peptide

### main_bo (fig:main-bo)
```bash
python experiments/eval2/cli.py main_bo --domain peptide --milestone 600 \
  --config peptide_experiment/configs/peptide_main_v2_eval_gpu3_lowbudget.yaml \
  --run-dir BOLT=runs/peptide_main_bolt_v2_lowbudget \
  --run-dir ORPT-H1=runs/peptide_main_orpt_h1_v2_lowbudget \
  --run-dir STBO=runs/peptide_main_bolt_v2_lowbudget \
  --run-dir MTBO=runs/peptide_main_bolt_v2_lowbudget \
  --run-dir POGPE=runs/peptide_main_bolt_v2_lowbudget \
  --run-dir SGPE=runs/peptide_main_bolt_v2_lowbudget \
  --run-dir OptFormer=runs/peptide_main_bolt_v2_lowbudget \
  --run-dir LLAMBO=runs/peptide_main_bolt_v2_lowbudget \
  --task-set heldout100 \
  --out-dir experiments/eval2/results/peptide
```
Once Phase 2 finishes, add: `--run-dir ORPT-H0=runs/peptide_ablation_orpt_h0_lowbudget`

### fewshot (fig:fewshot)
```bash
python experiments/eval2/cli.py fewshot --domain peptide --milestone 600 \
  --results-dir experiments/eval/results/main_v2_orpt_vs_bolt__lowbudget \
  --out-dir experiments/eval2/results/peptide
```

### scaling (fig:scaling)
```bash
python experiments/eval2/cli.py scaling --domain peptide \
  --results-dir experiments/eval/results/main_v2_orpt_vs_bolt__lowbudget \
  --out-dir experiments/eval2/results/peptide
```

### ablation (tab:ablation)
```bash
python experiments/eval2/cli.py ablation --domain peptide --milestone 600 \
  --results-dir experiments/eval/results/ablation_h0_vs_h1__lowbudget \
  --out-dir experiments/eval2/results/peptide
```
Needs Phase 2 (see plan file) run first -- `ablation_h0_vs_h1__lowbudget`
doesn't exist yet.

---

## LLVM

### main_bo (fig:main-bo)
```bash
python experiments/eval2/cli.py main_bo --domain llvm --milestone 600 \
  --config llvm_experiment/configs/llvm_main_eval_v1.yaml \
  --run-dir BOLT=runs/llvm_main_bolt_v1 \
  --run-dir ORPT-H1=runs/llvm_main_orpt_h1_v1 \
  --run-dir STBO=runs/llvm_main_bolt_v1 \
  --run-dir MTBO=runs/llvm_main_bolt_v1 \
  --run-dir POGPE=runs/llvm_main_bolt_v1 \
  --run-dir SGPE=runs/llvm_main_bolt_v1 \
  --run-dir OptFormer=runs/llvm_main_bolt_v1 \
  --run-dir LLAMBO=runs/llvm_main_bolt_v1 \
  --task-set heldout \
  --out-dir experiments/eval2/results/llvm
```
Note: LLVM's task-set is `heldout` (the 20-task set), not `heldout100` --
that's the eval launcher's own default (`TASK_SET=${TASK_SET:-heldout}`).

### fewshot (fig:fewshot)
```bash
python experiments/eval2/cli.py fewshot --domain llvm --milestone 600 \
  --results-dir experiments/eval/results/llvm_main_v1__gpu4,experiments/eval/results/llvm_main_v1__gpu5,experiments/eval/results/llvm_main_v1__gpu6 \
  --out-dir experiments/eval2/results/llvm
```

### scaling (fig:scaling)
```bash
python experiments/eval2/cli.py scaling --domain llvm \
  --results-dir experiments/eval/results/llvm_main_v1__gpu4,experiments/eval/results/llvm_main_v1__gpu5,experiments/eval/results/llvm_main_v1__gpu6 \
  --out-dir experiments/eval2/results/llvm
```

### ablation (tab:ablation)
```bash
python experiments/eval2/cli.py ablation --domain llvm --milestone 600 \
  --results-dir experiments/eval/results/llvm_ablation_h0_vs_h1 \
  --out-dir experiments/eval2/results/llvm
```
Needs `RUN_ABLATION=1` passed to `run_llvm_main_bolt_vs_orpt_mi_eval.sh` (or
the manifest run manually) first -- LLVM's H0/H1/BOLT are all already fully
trained, this just hasn't been evaluated yet.
