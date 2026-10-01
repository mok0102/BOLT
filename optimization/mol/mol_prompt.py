"""The mol domain's two-slot LLM prompt (spec's "LLM interface" section): target
protein sequence + seed ligand SELFIES, output exactly one candidate SELFIES
string. Owned here (not duplicated at each call site) so the exact wording used
in sampling and in reference-model log-likelihood scoring
(mol_experiment/mi_orpt/likelihood.py) never drifts apart -- the same failure
mode experiments/eval2/domains/peptide.py's docstring warns about for the
similarity formula applies equally to prompt text.
"""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You propose valid SELFIES molecules for lead optimization. Given a target "
    "protein and a seed ligand, propose a structurally related molecule expected "
    "to improve predicted binding affinity to the target. Output only the "
    "candidate SELFIES string."
)


def render_context(target_sequence: str, seed_selfies: str) -> str:
    """The full "user" turn text -- this IS the `context` string
    mi_orpt/likelihood.py::score_sequences's `context` param expects, and what a
    real sampling call's user turn is built from. One render function, used by
    both sampling and reference-scoring, so they can never diverge."""
    return f"Target protein sequence:\n{target_sequence}\n\nSeed ligand:\n{seed_selfies}"


def make_messages(target_sequence: str, seed_selfies: str, candidate_selfies: str) -> list[dict]:
    """(system, user, assistant) message triple in the same shape
    fine-tuning/peptides/make_dpo_train_data_csv.py::make_messages builds for
    peptide -- SFT/DPO dataset construction and likelihood scoring both consume
    this shape."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_context(target_sequence, seed_selfies)},
        {"role": "assistant", "content": candidate_selfies},
    ]


def parse_candidate_selfies(raw_output: str) -> str | None:
    """Extracts exactly one SELFIES candidate from raw LLM output, or None if the
    output doesn't look like a bare SELFIES string. Deliberately strict: a
    malformed/decorated response (extra prose, code fences, multiple molecules)
    is rejected here rather than passed through for the oracle layer to fail on
    less legibly."""
    text = raw_output.strip()
    if not text or "\n" in text:
        return None
    if not (text.startswith("[") and text.endswith("]")):
        return None
    return text
