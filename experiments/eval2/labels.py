"""Own small copy of experiments/eval/paper_labels.py's arm/domain naming,
kept independent per this package's design (see the plan file). Small and
stable enough that duplicating it is cheaper than importing eval/.
"""

from __future__ import annotations

# Internal arm name -> paper terminology. ORPT-MI and ORPT-H1 are both
# internal names for the paper's single "ORPT" method; ORPT-H0 is kept
# distinct here because ablation.py needs to tell it apart from ORPT-H1 --
# main_bo.py/fewshot.py/scaling.py should never see an ORPT-H0 row (that
# arm only ever appears in the ablation manifest/results).
PAPER_ARM_NAME: dict[str, str] = {
    "ORPT-MI": "ORPT",
    "ORPT-H1": "ORPT",
}


def paper_arm(name: str) -> str:
    return PAPER_ARM_NAME.get(name, name)


DOMAIN_PAPER_NAME: dict[str, str] = {
    "peptide": "Peptide",
    "llvm": "LLVM",
}

# Rows for tab:ablation, in the paper's fixed order.
ABLATION_ROW_ORDER = ["Supervised proposal", "Zero-step outcome ranking", "One-step outcome ranking"]

ABLATION_ARM_TO_ROW: dict[str, str] = {
    "BOLT": "Supervised proposal",
    "ORPT-H0": "Zero-step outcome ranking",
    "ORPT-H1": "One-step outcome ranking",
}
