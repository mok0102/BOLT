"""The one canonical Morgan/ECFP4 fingerprint + Tanimoto similarity implementation
for the mol domain. Per the spec's "Fingerprint definition" section: owned by this
module and imported everywhere Tanimoto similarity is needed -- the BO constraint
(mol_constraint.py), the initial-candidate feasibility filter, the evaluation
reducer, and every diagnostic -- never reimplemented at a second call site,
mirroring how experiments/eval2/domains/peptide.py's module docstring treats its
own similarity formula as the one place that owns it. (Milestone 2's task manifest
did not need this: seed selection there was by within-target potency rank, not
chemical similarity -- Tanimoto-to-seed only enters as of Milestone 4's BO
constraint.)

Fixed definition: Morgan/ECFP4, radius=2, fpSize=2048 bits, using RDKit's current
generator API (rdFingerprintGenerator.GetMorganGenerator), not the deprecated
AllChem.GetMorganFingerprintAsBitVect.
"""

from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

MORGAN_RADIUS = 2
MORGAN_FPSIZE = 2048

_generator = rdFingerprintGenerator.GetMorganGenerator(radius=MORGAN_RADIUS, fpSize=MORGAN_FPSIZE)


def get_morgan_fingerprint(smiles: str):
    """Returns an RDKit ExplicitBitVect, or None if the SMILES doesn't parse.
    Callers decide what an unparseable input means for them (usually: infeasible/
    invalid, not a silent zero-similarity default)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _generator.GetFingerprint(mol)


def tanimoto_similarity(smiles_a: str, smiles_b: str) -> float | None:
    """Tanimoto similarity between the ECFP4 fingerprints of two SMILES. Returns
    None (not 0.0, not NaN-as-a-float-surprise) if either side fails to parse --
    an explicit sentinel a caller must handle, not a value that could be silently
    averaged into a similarity distribution."""
    fp_a = get_morgan_fingerprint(smiles_a)
    fp_b = get_morgan_fingerprint(smiles_b)
    if fp_a is None or fp_b is None:
        return None
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)
