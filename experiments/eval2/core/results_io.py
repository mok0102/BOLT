"""The contract between compute/ and figures/.

compute/ writes summary CSVs; figures/ reads them back through this module
and never imports compute/. That keeps the figure path pure post-hoc -- it
needs no GPU, no oracle, and no checkpoints, only the CSVs a previous run
left behind.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUMMARY_FIXED_TARGET_BO = "summary_fixed_target_bo.csv"
SUMMARY_INCUMBENT = "summary_incumbent_vs_pool_size.csv"


def _concat_csv(results_dirs: str | Path | list[str | Path], filename: str) -> pd.DataFrame:
    """Concatenate one summary file across several results dirs -- that is how
    a GPU-sharded eval run is reassembled. A missing shard prints and is
    skipped rather than raising, so a partially finished sweep still plots."""
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]
    frames = []
    for d in results_dirs:
        path = Path(d) / filename
        if not path.exists():
            print(f"[eval2] {path} not found, skipping")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_summary_fixed_target_bo(results_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    return _concat_csv(results_dirs, SUMMARY_FIXED_TARGET_BO)


def load_summary_incumbent(results_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    return _concat_csv(results_dirs, SUMMARY_INCUMBENT)
