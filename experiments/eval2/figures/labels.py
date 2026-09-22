"""Internal arm names -> the terminology paper/experiments.tex uses."""

from __future__ import annotations

# Internal arm name -> paper terminology. ORPT-MI and ORPT-H1 are both
# internal names for the paper's single "ORPT" method; ORPT-H0 is kept
# distinct here because ablation.py needs to tell it apart from ORPT-H1 --
# main_bo.py/fewshot.py/scaling.py should never see an ORPT-H0 row (that
# arm only appears in arm_specs/ablation.yaml and its results).
PAPER_ARM_NAME: dict[str, str] = {
    "ORPT-MI": "ORPT",
    "ORPT-H1": "ORPT",
}


def paper_arm(name: str) -> str:
    return PAPER_ARM_NAME.get(name, name)


# Rows for tab:ablation, in the paper's fixed order.
ABLATION_ROW_ORDER = ["Supervised proposal", "Zero-step outcome ranking", "One-step outcome ranking"]

ABLATION_ARM_TO_ROW: dict[str, str] = {
    "BOLT": "Supervised proposal",
    "ORPT-H0": "Zero-step outcome ranking",
    "ORPT-H1": "One-step outcome ranking",
}
