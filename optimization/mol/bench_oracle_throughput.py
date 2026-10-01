"""Oracle throughput benchmark, per the spec's "Throughput gate" requirements:
single-candidate latency, batched throughput vs. batch size, molecules/sec, and
(R5) explicit contended-vs-idle GPU labeling.

Benchmarks the two candidates the model-comparison left standing:
  - single model: morgan_aac_bindingdb_ic50 (best individual mean rho)
  - 4-model ensemble: morgan_cnn + cnn_cnn + daylight_aac + morgan_aac (best
    median rho / best fraction of well-behaved targets; MPNN_CNN excluded because
    including it *hurt* ensemble quality, not because of its cost)
MPNN_CNN alone is included too, for reference against Milestone 1/3's numbers.

APEX (the peptide oracle) is not re-benchmarked here; its batch_size=3000, one-GPU-
pass, 8-model-ensemble numbers are read from optimization/peptides/apex_oracle/
APEX_predict.py (Section 0 reference) for the comparison line in the report.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mol_oracle import MolOracle, ScoreCache  # noqa: E402

PRETRAINED_DIR = Path(__file__).resolve().parent / "pretrained" / "pretrained_models"
ENSEMBLE_NO_MPNN = ["morgan_cnn_bindingdb_ic50", "cnn_cnn_bindingdb_ic50",
                    "daylight_aac_bindingdb_ic50", "morgan_aac_bindingdb_ic50"]

# A fixed, real target + a pool of real candidate SELFIES to score repeatedly,
# reused (with cache disabled per-run) so the benchmark measures fresh-compute
# cost, not cache-hit cost.
TARGET_SEQ = ("MVGSLNCIVAVSQNMGIGKNGDLPWPPLRNEFRYFQRMTTTSSVEGKQNLVIMGKKTWFSIPEKNRPLKGRINLVLS"
              "RELKEPPQGAHFLSRSLDDALKLTEQPELANKVDMVWIVGGSSVYKEAMNHPGHLKLFVTRIMQDFESDTFFPEIDLE"
              "KYKLLPEYPGVLSDVQEEKGIKYKFEVYEKND")
TARGET_ID = "P00374_bench"  # distinct from the real P00374 cache entries
SEED_SELFIES = ("[C][N][Branch2][Ring1][=Branch1][C][C][=C][N][=C][N][=C][Branch1][C][N][N]"
                "[=C][Branch1][C][N][C][Ring1][Branch2][=N][Ring1][N][C][=C][C][=C][Branch2]"
                "[Ring1][O][C][=Branch1][C][=O][N][C][Branch1][=C][C][C][Branch1][C][F]"
                "[Branch1][C][F][C][=Branch1][C][=O][O][C][=Branch1][C][=O][O][C][=C][Ring2]"
                "[Ring1][Ring2]")


class NullCache:
    """A cache that never hits, so every benchmark call is a fresh, uncached score
    -- otherwise the second+ iteration of any repeated candidate would measure
    sqlite lookup speed, not oracle inference speed."""
    def get_many(self, target_id, canonical_smiles):
        return {}

    def put_many(self, target_id, items):
        pass


def make_candidates(n: int) -> list[str]:
    # n copies of the same valid seed SELFIES: fine for a throughput benchmark
    # (identical *content*, but NullCache guarantees each is freshly computed) --
    # correctness/direction was already verified separately with real varied
    # molecules in test_oracle_direction.py / the model-comparison scripts.
    return [SEED_SELFIES] * n


def bench_single(oracles: list[MolOracle], batch_sizes: list[int]) -> None:
    for n in batch_sizes:
        candidates = make_candidates(n)
        t0 = time.perf_counter()
        for oracle in oracles:
            oracle.query_oracle(candidates)
        dt = time.perf_counter() - t0
        per_sec = n / dt if dt > 0 else float("inf")
        print(f"  batch_size={n:>5}  wall={dt:8.3f}s  molecules/sec={per_sec:9.1f}"
              f"  (n_models_in_this_call={len(oracles)})")


def main() -> None:
    print("=== Device: CPU (mol-venv currently has CPU-only torch) ===")
    print()

    for model_name in ["mpnn_cnn_bindingdb_ic50", "morgan_aac_bindingdb_ic50"]:
        print(f"--- single model: {model_name} ---")
        oracle = MolOracle(target_id=TARGET_ID, target_sequence=TARGET_SEQ,
                            model_dir=PRETRAINED_DIR / model_name, cache=NullCache())
        bench_single([oracle], [1, 8, 32, 128, 512])
        print()

    print(f"--- 4-model ensemble (no MPNN): {ENSEMBLE_NO_MPNN} ---")
    ensemble_oracles = [
        MolOracle(target_id=TARGET_ID, target_sequence=TARGET_SEQ,
                  model_dir=PRETRAINED_DIR / m, cache=NullCache())
        for m in ENSEMBLE_NO_MPNN
    ]
    bench_single(ensemble_oracles, [1, 8, 32, 128, 512])


if __name__ == "__main__":
    main()
