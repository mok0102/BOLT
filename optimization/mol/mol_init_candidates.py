"""Task-local initial candidate generation: bounded SELFIES-token edits around the
seed molecule, mirroring peptide's task-local initialization philosophy (spec:
"Mirror peptide's task-local initialization philosophy: valid SELFIES edits around
the seed, then decode/canonicalize, deduplicate, and record feasibility").

Deliberately does NOT reuse peptide's `max_mutation_distance = 1.0 -
similarity_threshold` parameterization (spec: "The parameterization does not carry
over ... peptide's similarity is a normalized edit distance" -- fingerprint
Tanimoto has no such linear relationship to token-edit count). Instead this
generates a spread of edit distances and lets the caller filter by *measured*
ECFP4 Tanimoto, exactly as the spec requires.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import selfies as sf
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_lolbo.utils.mol_utils.selfies_vae.model_positional_unbounded import SELFIESDataset  # noqa: E402

RDLogger.DisableLog("rdApp.*")

_VOCAB_TOKENS = None  # lazily populated: every non-special token in the VAE's fixed vocabulary
_VOCAB_SET = None


def _vocab_tokens() -> list[str]:
    global _VOCAB_TOKENS
    if _VOCAB_TOKENS is None:
        dataobj = SELFIESDataset()
        _VOCAB_TOKENS = [t for t in dataobj.vocab if t not in ("<start>", "<stop>")]
    return _VOCAB_TOKENS


def _vocab_set() -> set[str]:
    global _VOCAB_SET
    if _VOCAB_SET is None:
        _VOCAB_SET = set(SELFIESDataset().vocab2idx.keys())
    return _VOCAB_SET


def filter_vocab_safe(smiles_list: list[str]) -> list[bool]:
    """Returns a boolean mask, True for each SMILES whose re-encoded SELFIES
    (sf.encoder(smiles) -- the exact representation MoleculeObjective.vae_forward
    tokenizes, mol_objective.py) has every token in the VAE's fixed vocabulary.

    This is the one chokepoint every candidate source (LLM sampling, seed-
    mutation generation, oracle-ranked top-k, ...) funnels through before
    reaching the VAE, so it's the right place for a final defensive gate --
    not a replacement for each source's own vocab-check (which should still
    filter as early as possible, for yield-reporting accuracy), but a
    guarantee that a gap in any ONE of them can't crash vae_forward.
    Motivated by a real incident (2026-09-30): mol_sampling.py's own
    vocab-check validated the LLM's original SELFIES tokens, not the ones
    produced by re-encoding the canonical SMILES -- a different, unchecked
    representation that vae_forward actually uses -- which reached
    production and crashed a live run with an unhandled KeyError.

    Callers must apply this mask to every list that's index-aligned with
    smiles_list (scores, etc.) before calling vae_forward -- filtering
    inside vae_forward itself is NOT done here deliberately, since that
    would silently desync z's row count from a caller's already-built
    train_y/train_x without the caller knowing to re-filter those too."""
    vocab = _vocab_set()
    mask = []
    for smi in smiles_list:
        try:
            tokens = list(sf.split_selfies(sf.encoder(smi)))
        except Exception:
            mask.append(False)
            continue
        mask.append(all(t in vocab for t in tokens))
    return mask


def _mutate_tokens(tokens: list[str], rng: random.Random, n_edits: int) -> list[str]:
    tokens = list(tokens)
    vocab = _vocab_tokens()
    for _ in range(n_edits):
        if not tokens:
            op = "insert"
        else:
            op = rng.choice(["substitute", "insert", "delete"])
        if op == "substitute" and tokens:
            i = rng.randrange(len(tokens))
            tokens[i] = rng.choice(vocab)
        elif op == "insert":
            i = rng.randrange(len(tokens) + 1)
            tokens.insert(i, rng.choice(vocab))
        elif op == "delete" and len(tokens) > 1:
            i = rng.randrange(len(tokens))
            del tokens[i]
    return tokens


def decode_tokens_to_canonical_smiles(tokens: list[str]) -> str | None:
    selfies_str = "".join(tokens)
    try:
        smiles = sf.decoder(selfies_str)
    except Exception:
        return None
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def generate_feasible_candidates_around_seed(
    seed_smiles: str,
    tau_mol: float,
    n_feasible: int,
    rng_seed: int,
    max_edits: int = 4,
    max_attempts: int | None = None,
) -> tuple[list[tuple[str, str]], dict]:
    """Generate-and-filter (Milestone 4 redesign, see MOL_LATENT_SPACE_FINDING.md):
    latent BO does not enforce the Tanimoto constraint in-loop (empirically the
    pretrained SELFIES VAE's latent space does not preserve similarity to a point's
    own encoding even at the tightest radius tested -- fitting a GP constraint
    surrogate over Tanimoto-vs-z has nothing learnable to fit). Feasibility is
    instead enforced here, at candidate-generation time, by pre-filtering: every
    candidate returned already satisfies Tanimoto(candidate, seed) >= tau_mol.
    This is also the policy the future LLM proposal generator must follow (spec:
    "LLM proposals also go through the same candidate filtering").

    Returns (candidates, stats) where stats reports the true generation yield
    (spec: "report the proposals-per-feasible-candidate rate") -- a low yield is
    a reportable finding, not something to mask by retrying indefinitely."""
    if max_attempts is None:
        max_attempts = n_feasible * 200  # generous; yield itself is what's reported
    rng = random.Random(rng_seed)
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        raise ValueError(f"seed_smiles is not RDKit-valid: {seed_smiles!r}")
    seed_canon = Chem.MolToSmiles(seed_mol)
    seed_tokens = list(sf.split_selfies(sf.encoder(seed_canon)))

    from mol_fingerprint import tanimoto_similarity

    seen = {seed_canon}
    out: list[tuple[str, str]] = []
    n_attempts = 0
    n_valid_molecule = 0
    while len(out) < n_feasible and n_attempts < max_attempts:
        n_attempts += 1
        n_edits = rng.randint(1, max_edits)
        mutated_tokens = _mutate_tokens(seed_tokens, rng, n_edits)
        canon = decode_tokens_to_canonical_smiles(mutated_tokens)
        if canon is None or canon in seen:
            continue
        try:
            re_encoded = sf.encoder(canon)
        except Exception:
            continue
        if any(t not in _vocab_set() for t in sf.split_selfies(re_encoded)):
            continue
        n_valid_molecule += 1
        sim = tanimoto_similarity(canon, seed_smiles)
        if sim is None or sim < tau_mol:
            seen.add(canon)  # still dedupe -- no point re-generating a known-infeasible one
            continue
        seen.add(canon)
        out.append((canon, re_encoded))

    stats = {
        "n_attempts": n_attempts,
        "n_valid_molecule": n_valid_molecule,
        "n_feasible_found": len(out),
        "yield_feasible_per_attempt": len(out) / n_attempts if n_attempts else 0.0,
    }
    return out, stats


def generate_candidates_around_seed(
    seed_smiles: str,
    n_candidates: int,
    rng_seed: int,
    max_edits: int = 4,
) -> list[tuple[str, str]]:
    """Returns up to n_candidates (canonical_smiles, selfies_str) pairs, deduplicated
    by canonical SMILES (the seed itself excluded), generated by bounded random
    token edits around the seed's SELFIES encoding. A deterministic RNG seed is
    required -- this is called from both the tau_mol smoke study and (eventually)
    real task initialization, and both need reproducible pools."""
    rng = random.Random(rng_seed)
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        raise ValueError(f"seed_smiles is not RDKit-valid: {seed_smiles!r}")
    seed_canon = Chem.MolToSmiles(seed_mol)
    seed_tokens = list(sf.split_selfies(sf.encoder(seed_canon)))

    seen = {seed_canon}
    out: list[tuple[str, str]] = []
    attempts = 0
    max_attempts = n_candidates * 50  # generous but bounded -- this is a smoke-scale
                                       # utility, not expected to spin forever; a low
                                       # yield rate itself is a reportable finding
                                       # (spec: "report the proposals-per-feasible-
                                       # candidate rate"), not something to mask by
                                       # retrying indefinitely.
    while len(out) < n_candidates and attempts < max_attempts:
        attempts += 1
        n_edits = rng.randint(1, max_edits)
        mutated_tokens = _mutate_tokens(seed_tokens, rng, n_edits)
        canon = decode_tokens_to_canonical_smiles(mutated_tokens)
        if canon is None or canon in seen:
            continue
        try:
            # A molecule RDKit accepts is not always one selfies.encoder can
            # re-encode -- found empirically: RDKit will parse/canonicalize a
            # SMILES whose aromatic system it cannot kekulize, and sf.encoder
            # requires kekulization internally and raises. Treated as another
            # invalid-candidate case (skipped, not retried with a repaired
            # structure), not a crash.
            re_encoded = sf.encoder(canon)
        except Exception:
            continue
        # Vocab-membership check: canonicalization can restructure a molecule
        # enough that its natural SELFIES re-encoding uses tokens outside the
        # VAE checkpoint's fixed 97-token vocabulary, even though every token we
        # *inserted* during mutation came from that same vocabulary (found
        # empirically: KeyError on '[=I+1]' downstream in dataobj.encode() before
        # this check existed). This is the same phenomenon Milestone 2 found for
        # real BindingDB ligands (~40% out-of-vocabulary) -- a candidate failing
        # here is dropped, exactly like an unencodable seed is dropped at
        # task-construction time, never surfaced as a cryptic KeyError at BO time.
        if any(t not in _vocab_set() for t in sf.split_selfies(re_encoded)):
            continue
        seen.add(canon)
        out.append((canon, re_encoded))

    return out
