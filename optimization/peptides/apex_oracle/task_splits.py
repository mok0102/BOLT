"""Single source of truth for how REFERENCE_SEQUENCE's 1000 peptides are split.

Per the paper (p.6, p.18): 900 training peptides, last 100 held out for
validation. Table 11's "20 validation tasks" isn't specified further by the
paper; this reimplementation assumes it's the first 20 of the 100-task
held-out block (see imp_plan/01_peptide_reimplementation_plan.md).
"""

TRAIN_TASKS = range(0, 900)
HELDOUT20_TASKS = range(900, 920)
HELDOUT100_TASKS = range(900, 1000)
