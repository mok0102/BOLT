"""Pure-pandas readers for the aggregated CSV contract experiments/eval/'s
compute scripts (fixed_target_rejection_bo.py, incumbent_vs_pool_size.py)
already produce -- no experiments/eval/ imports needed for these, since
fewshot.py/scaling.py/ablation.py only ever read already-computed summary
CSVs.

main_bo.py is the one exception (see its own docstring): correctly
recomputing a dense running-best curve from raw per-task trajectory CSVs
requires peptide's feasibility/similarity check (which candidate rows are
even valid), and reimplementing that domain science independently here
would risk silently diverging from the authoritative
experiments/eval/domains.py implementation. main_bo.py imports
`domains.DOMAINS` for that one piece only; every other module in this
package stays fully independent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

SUMMARY_FIXED_TARGET_BO = "summary_fixed_target_bo.csv"
SUMMARY_INCUMBENT = "summary_incumbent_vs_pool_size.csv"


def _concat_csv(results_dirs: str | Path | list[str | Path], filename: str) -> pd.DataFrame:
    if isinstance(results_dirs, (str, Path)):
        results_dirs = [results_dirs]
    frames = []
    for d in results_dirs:
        path = Path(d) / filename
        if not path.exists():
            print(f"[eval2.data] {path} not found, skipping")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_summary_fixed_target_bo(results_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    return _concat_csv(results_dirs, SUMMARY_FIXED_TARGET_BO)


def load_summary_incumbent(results_dirs: str | Path | list[str | Path]) -> pd.DataFrame:
    return _concat_csv(results_dirs, SUMMARY_INCUMBENT)


def raw_trajectory_dir(run_dir: Path, task_set: str, arm: str, milestone: int, target_pool_size: int) -> Path:
    return run_dir / "eval_fixed_target_bo" / task_set / f"{arm}-{milestone}__target{target_pool_size}"
