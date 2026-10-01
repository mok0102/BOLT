#!/usr/bin/env bash
set -euo pipefail

# Small exact-GP solves are substantially slower when BLAS/NumExpr fan out
# across every CPU core. Keep the smoke run deterministic and responsive.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# python -m synthetic_experiment trajectory_chain \
#   --config synthetic_experiment/configs/synthetic_smoke_mi_orpt.yaml

# python -m synthetic_experiment heldout_eval \
#   --config synthetic_experiment/configs/synthetic_smoke_mi_orpt.yaml \
#   --arm all

# python -m synthetic_experiment aggregate \
#   --config synthetic_experiment/configs/synthetic_smoke_mi_orpt.yaml

python -m synthetic_experiment trajectory_chain \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller_h0.yaml

python -m synthetic_experiment heldout_eval \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller_h0.yaml \
  --arm all

python -m synthetic_experiment aggregate \
  --config synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller_h0.yaml
