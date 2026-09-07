"""Single mapping layer from this repo's internal checkpoint/arm identifiers
to paper/experiments.tex's exact terminology -- so a wording change only
touches this file, and every fig_*.py/tab_*.py script stays consistent.

The paper's own method is ORPT (\\ourmethod{} = "Optimization-Outcome-Ranked
Proposal Tuning", method.tex sec:orpt); BOLT is a baseline in the paper's
framing, even though this repo is named BOLT.
"""

from __future__ import annotations

# ORPT-MI: the already-trained, real main-results arm (900-task schedule,
# runs/peptide_main_orpt_mi). ORPT-H1: the new H-notation-consistent name
# (method.tex's U_H, H=1 = one-step outcome ranking) used by every new (v2+)
# manifest/config this session, including the ablation study's one-step arm.
# Both are the exact same method (mi_bo_steps=1, matched-intervention
# one-step BO utility) -- just two identifiers from two points in time.
PAPER_ARM_NAME = {"ORPT-MI": "ORPT", "ORPT-H1": "ORPT"}


def paper_arm(name: str) -> str:
    """Every other arm (BOLT, STBO, MTBO, POGPE, SGPE, OptFormer, LLAMBO)
    already matches the paper's exact spelling (see paper/experiments.tex's
    "Methods." paragraph) -- passed through unchanged."""
    return PAPER_ARM_NAME.get(name, name)


# paper/experiments.tex: "database query-plan optimization" / "antimicrobial
# peptide design" in prose; the paper's own commented-out table sketch
# abbreviates these to "Query plan"/"Peptide" for compact cells -- reused
# here for panel titles/table cells.
DOMAIN_PAPER_NAME = {"peptide": "Peptide", "query_plan": "Query plan"}

# Ablation Study (sec:ablations) mapping. ORPT-H0 (mi_bo_steps=0) is the
# literal zero-step ablation arm mi_orpt/pair_construction.py implements
# ("the free zero-step ablation", its own comment) -- ranks pairs by
# one_step_evaluator.py::zero_step_utility (the shared background pool's own
# best-known value, method.tex's U_0(C,S)), no real one-step BO call. ORPT-H1
# (mi_bo_steps=1) is the paper's actual ORPT method.
ABLATION_ARM_TO_SIGNAL = {
    "BOLT": "Supervised proposal",
    "ORPT-H0": "Zero-step outcome ranking",
    "ORPT-H1": "One-step outcome ranking",
}
# Paper's own row order (experiments.tex sec:ablations' itemized list) --
# tab_ablation.py must not sort these alphabetically.
ABLATION_ROW_ORDER = ["Supervised proposal", "Zero-step outcome ranking", "One-step outcome ranking"]

ABLATION_SCALE_NOTE = (
    "PoC-scale data (runs/peptide_poc20_bolt_v2, runs/peptide_ablation_orpt_h0, "
    "runs/peptide_ablation_orpt_h1; 5-task heldout_tasks_override) -- not the "
    "paper-fidelity main-scale run."
)
