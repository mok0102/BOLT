"""Frozen DeepPurpose BindingDB-IC50 oracle, wrapped behind the same
objective-function shape peptide's ApexObjective uses (optimization/peptides/
your_tasks/your_objective_functions.py::ApexObjective, read-only reference, not
imported -- see Section 0 / "couplings not to inherit" in
impl_plan/opus_bindingdb_implementation_prompt.md).

Convention (verified empirically in Milestone 1 against a real BindingDB record:
predicted 6.872 vs measured 6.752 p for a known pair, and confirmed by reading
DeepPurpose.utils.convert_y_unit's source):
    raw model output = p = -log10(IC50_in_molar)
    higher p == stronger predicted binding == already maximize-oriented.
No sign flip is applied (unlike peptide's -MIC), and this file must not add one.

Caching (per the spec's "Budget-accounting constraint" in the Oracle section):
this module's on-disk cache is a compute-saving memo ONLY, keyed by
(target_id, canonical_smiles). It has no notion of "run-local" or "num_calls" --
that budget accounting belongs one layer up, in the LOL-BO objective wrapper
(Milestone 5), which must dedupe through its own run-local xs_to_scores_dict
BEFORE ever calling into this module, exactly mirroring
optimization/peptides/lolbo/latent_space_objective.py::__call__. This module
must never be the thing deciding whether a call "counts" -- it just answers
"what is the score for this (target, molecule) pair", cached or not.
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import selfies as sf
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_UPSTREAM_LOLBO = Path(__file__).resolve().parent / "_upstream" / "lolbo"
if str(_UPSTREAM_LOLBO) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_LOLBO))

_PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained" / "pretrained_models"
DEFAULT_MODEL_DIR = _PRETRAINED_DIR / "mpnn_cnn_bindingdb_ic50"
DEFAULT_CACHE_DB = Path(__file__).resolve().parent / "oracle_cache.sqlite3"

# LOCKED PRIMARY ORACLE (Milestone 3 decision, evidence in
# ORACLE_DECISION.md): an unweighted 4-model ensemble over BindingDB-IC50
# pretrained DeepPurpose models, deliberately excluding MPNN_CNN_BindingDB_IC50.
#
# The spec originally named MPNN_CNN_BindingDB_IC50 as the oracle. A 40-target
# score-direction benchmark (test_oracle_model_comparison_all.py) found it
# essentially uncorrelated with real measured potency (mean Spearman rho=0.036,
# only 2.5% of targets above rho=0.5) -- ruled out as a truncation or sampling
# artifact (poor on targets both above and below the CNN target-encoder's
# MAX_SEQ_PROTEIN=1000 limit, and on both extreme-percentile and plain-random
# ligand samples). All 4 remaining BindingDB_IC50 models individually
# outperformed it, and averaging MPNN_CNN back in *hurt* every ensemble
# combination that included it -- so it is excluded here, not merely
# de-weighted. The 4-model mean was chosen over the single best individual
# model (morgan_aac, mean rho=0.311) because it had the best median rho (0.382)
# and by far the best fraction of well-behaved targets (rho>0.5 for 42.5% of
# targets vs 30% for morgan_aac alone) -- more of the 950 manifest tasks get a
# genuinely informative oracle landscape, which matters more for BO headroom
# than the single-model mean. Per F1, this is a locked, reported substitution,
# not a silent one.
ENSEMBLE_MODEL_NAMES = [
    "morgan_cnn_bindingdb_ic50",
    "cnn_cnn_bindingdb_ic50",
    "daylight_aac_bindingdb_ic50",
    "morgan_aac_bindingdb_ic50",
]


class ScoreCache:
    """Disk-backed (target_id, canonical_smiles) -> score memo. Safe for multiple
    BO subprocesses to share (WAL mode); a cache miss simply means "not yet
    computed", never an error."""

    def __init__(self, db_path: Path = DEFAULT_CACHE_DB):
        self.db_path = db_path
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS scores ("
            " target_id TEXT NOT NULL,"
            " canonical_smiles TEXT NOT NULL,"
            " score REAL NOT NULL,"
            " PRIMARY KEY (target_id, canonical_smiles)"
            ")"
        )
        self._conn.commit()

    def get_many(self, target_id: str, canonical_smiles: list[str]) -> dict[str, float]:
        if not canonical_smiles:
            return {}
        placeholders = ",".join("?" * len(canonical_smiles))
        rows = self._conn.execute(
            f"SELECT canonical_smiles, score FROM scores "
            f"WHERE target_id = ? AND canonical_smiles IN ({placeholders})",
            [target_id, *canonical_smiles],
        ).fetchall()
        return dict(rows)

    def put_many(self, target_id: str, items: dict[str, float]) -> None:
        if not items:
            return
        self._conn.executemany(
            "INSERT OR REPLACE INTO scores (target_id, canonical_smiles, score) VALUES (?, ?, ?)",
            [(target_id, smi, score) for smi, score in items.items()],
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


def decode_and_canonicalize(selfies_or_smiles: str, input_kind: str) -> str | None:
    """input_kind: 'selfies' or 'smiles'. Returns canonical SMILES, or None if the
    input is unparseable (F1/F5: no silent scoring of malformed input; the caller
    turns None into np.nan, never a fallback numeric value).

    Malformed-SELFIES trap found empirically while building this module:
    sf.decoder() does not raise on garbage input that contains no recognizable
    SELFIES tokens -- it silently returns "" (empty SMILES), and
    Chem.MolFromSmiles("") returns a technically-non-None *empty* Mol, which
    Chem.MolToSmiles renders back as "". An empty canonical SMILES is therefore
    rejected explicitly here; feeding it to the DTI featurizer produced DeepPurpose's
    own silent "Molecules not found and change to zero vectors.." fallback and a
    None prediction for that row -- exactly the silent-coercion failure mode F5/F6
    rule out, just one layer deeper (inside sf.decoder + DeepPurpose) than expected.
    """
    try:
        if input_kind == "selfies":
            smiles = sf.decoder(selfies_or_smiles)
        else:
            smiles = selfies_or_smiles
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    canon = Chem.MolToSmiles(mol)
    return canon or None


class MolOracle:
    """One instance is bound to a single (target_id, target_sequence) task, mirroring
    how peptide's ApexObjective is bound to a single reference/bacteria strain at
    construction time. Candidates are SELFIES strings by default (the domain's
    candidate representation, see "Domain definition" in the spec)."""

    def __init__(
        self,
        target_id: str,
        target_sequence: str,
        model_dir: Path = DEFAULT_MODEL_DIR,
        cache: ScoreCache | None = None,
        device: str = "cpu",
    ):
        from DeepPurpose import DTI, utils

        self.target_id = target_id
        self.target_sequence = target_sequence
        self.device = device
        self._utils = utils
        self.model = DTI.model_pretrained(path_dir=str(model_dir))
        if device != "cpu":
            self.model.device = device
            self.model.model = self.model.model.to(device)
        if cache is not None:
            self.cache = cache
        else:
            # The cache schema keys only on (target_id, canonical_smiles) -- silently
            # sharing one cache file across two different pretrained models would
            # mix their scores under the same key. Default to a per-model cache file
            # (named after model_dir) so this can't happen by accident; callers who
            # deliberately want to share a ScoreCache across MolOracle instances for
            # the *same* model may still pass one in explicitly.
            model_name = Path(model_dir).name
            self.cache = ScoreCache(DEFAULT_CACHE_DB.with_name(f"oracle_cache_{model_name}.sqlite3"))

    def query_oracle(self, candidates: list[str], input_kind: str = "selfies") -> list[float]:
        """Input: a list of candidate SELFIES (or SMILES if input_kind='smiles').
        Output: a list of the same length, each a maximize-oriented float score, or
        np.nan wherever the candidate does not decode/canonicalize to a valid
        molecule, OR (this model only, when self.model.drug_encoding == 'CNN')
        wherever the canonical SMILES exceeds DeepPurpose's own hardcoded
        MAX_SEQ_DRUG=100 character cap. Mirrors ApexObjective.query_black_box's
        contract exactly.

        The length check exists because DeepPurpose's CNN drug encoder
        (DeepPurpose.utils.trans_drug) does not reject an over-length SMILES --
        it silently truncates to the first 100 characters and scores that
        truncated string, producing a normal-looking (finite) number for the
        wrong molecule. That's invisible to the "any non-finite member -> NaN
        overall" safeguard in EnsembleMolOracle.query_oracle (a truncated score
        is finite, not NaN), so without this check a too-long molecule would
        silently corrupt this member's vote in the ensemble mean instead of
        being excluded the way a genuine decode failure already is. Confirmed
        real, not hypothetical: 9/900 task seeds already exceed 100 characters
        (up to 169) -- see mol_experiment/MI_ORPT_PAIR_YIELD_TUNING.md-adjacent
        investigation, 2026-10-01. Fingerprint-based drug encodings (Morgan,
        Daylight -- the other 3 ensemble members) are fixed-size regardless of
        SMILES length and are not affected."""
        canon_per_candidate: list[str | None] = [
            decode_and_canonicalize(c, input_kind) for c in candidates
        ]
        if self.model.drug_encoding == "CNN":
            from DeepPurpose.utils import MAX_SEQ_DRUG

            canon_per_candidate = [
                None if (canon is not None and len(canon) > MAX_SEQ_DRUG) else canon
                for canon in canon_per_candidate
            ]

        valid_canon = sorted({c for c in canon_per_candidate if c is not None})
        cached = self.cache.get_many(self.target_id, valid_canon)
        to_score = [c for c in valid_canon if c not in cached]

        newly_scored: dict[str, float] = {}
        if to_score:
            newly_scored = self._score_batch(to_score)
            self.cache.put_many(self.target_id, newly_scored)

        all_scores = {**cached, **newly_scored}
        return [all_scores[c] if c is not None else float("nan") for c in canon_per_candidate]

    def _score_batch(self, canonical_smiles: list[str]) -> dict[str, float]:
        """Raw DeepPurpose batched inference for a list of already-canonicalized,
        already-valid SMILES against this oracle's fixed target sequence."""
        train, val, test = self._utils.data_process(
            X_drug=canonical_smiles,
            X_target=[self.target_sequence],
            y=[0.0] * len(canonical_smiles),
            drug_encoding=self.model.drug_encoding,
            target_encoding=self.model.target_encoding,
            split_method="repurposing_VS",
        )
        preds = self.model.predict(test)
        if len(preds) != len(canonical_smiles):
            raise RuntimeError(
                f"DeepPurpose returned {len(preds)} predictions for "
                f"{len(canonical_smiles)} already-RDKit-validated canonical SMILES "
                f"against target {self.target_id!r} -- cannot align scores to inputs."
            )
        scores: dict[str, float] = {}
        for smi, p in zip(canonical_smiles, preds):
            score = float(p)
            if not np.isfinite(score):
                # F5: a NaN/inf here is an error, not a candidate to silently drop or
                # coerce -- every input reaching this point already passed RDKit
                # validation, so a non-finite prediction means the featurizer choked
                # on something RDKit accepted (e.g. an atom/bond type outside
                # DeepPurpose's MPNN feature vocabulary) and must be surfaced, not
                # papered over as if it were just another invalid molecule.
                raise RuntimeError(
                    f"DeepPurpose produced a non-finite score ({p!r}) for RDKit-valid "
                    f"canonical SMILES {smi!r} against target {self.target_id!r}."
                )
            scores[smi] = score
        return scores


class EnsembleMolOracle:
    """The locked primary oracle (see ENSEMBLE_MODEL_NAMES above): an unweighted
    mean across several independently-parameterized frozen DTI models, same
    query_oracle(candidates) -> list[float] contract as MolOracle so it is a
    drop-in replacement everywhere a single-model oracle would be used (Milestone
    5's LOL-BO objective wrapper does not need to know it's an ensemble).

    Mirrors peptide's APEX oracle shape: predict_APEX averages 8 independently
    trained ensemble members (optimization/peptides/apex_oracle/APEX_predict.py,
    read-only reference). The averaging here is likewise unweighted -- weighting
    members by their own performance on the benchmark that selected them would be
    circular (overfit to that same validation run), exactly as peptide's ensemble
    is not weighted by each member's individual validation score either.
    """

    def __init__(
        self,
        target_id: str,
        target_sequence: str,
        model_names: list[str] = ENSEMBLE_MODEL_NAMES,
        pretrained_dir: Path = _PRETRAINED_DIR,
        device: str = "cpu",
    ):
        self.target_id = target_id
        self.model_names = list(model_names)
        self.members = [
            MolOracle(target_id=target_id, target_sequence=target_sequence,
                      model_dir=pretrained_dir / name, device=device)
            for name in self.model_names
        ]

    def query_oracle(self, candidates: list[str], input_kind: str = "selfies") -> list[float]:
        """Per-candidate unweighted mean across member scores. A candidate that is
        NaN for one member (which only happens via decode/canonicalize failure,
        identical and deterministic across every member since that step doesn't
        depend on the model) is NaN for all members and therefore NaN overall --
        there is no member-subset averaging that could silently mask an invalid
        candidate as a partial score."""
        per_member_scores = [m.query_oracle(candidates, input_kind=input_kind) for m in self.members]
        out = []
        for i in range(len(candidates)):
            vals = [per_member_scores[m][i] for m in range(len(self.members))]
            if any(not np.isfinite(v) for v in vals):
                out.append(float("nan"))
            else:
                out.append(sum(vals) / len(vals))
        return out
