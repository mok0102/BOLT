"""Presentation-layer constants for the peptide figures and tables.

Only values that are genuinely about *how the paper reports results* live
here. Anything the experiment itself defines -- oracle_budget, init_size,
milestones, table_k_checkpoints -- is read from the run's own cfg instead,
so there is never a second copy to drift out of sync with the yaml that
actually produced the numbers.
"""

from __future__ import annotations

# Slug used in output filenames (main_bo_<slug>.png, ablation_<slug>.tex, ...).
DOMAIN_SLUG = "peptide"

# fig:scaling panel (a)'s representative k, per paper/experiments.tex.
SCALING_K = 200

# Arms fig:main-bo / fig:fewshot / fig:scaling expect, in paper-label space
# (see figures/labels.py::paper_arm -- ORPT-H1 displays as "ORPT").
MAIN_BO_ARMS = ("BOLT", "ORPT", "STBO", "MTBO", "POGPE", "SGPE", "OptFormer", "LLAMBO")
