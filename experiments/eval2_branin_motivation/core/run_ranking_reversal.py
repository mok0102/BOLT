"""Driver: collects Experiment B's raw pairs at real (non-smoke) scale and
saves both the raw pairs and the computed statistics (with and without the
reliability filter, per the doc's explicit request).

Task selection: the first N training tasks in manifest order (task_0000..),
a fixed, deterministic, non-cherry-picked choice -- not selected by any
downstream result.
"""

from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1]
BOLT_ROOT = PKG_ROOT.parents[1]
for _p in (BOLT_ROOT, PKG_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from synthetic_experiment.config import load_config  # noqa: E402

from core.ranking_reversal import (  # noqa: E402
    HORIZONS, TRAIN_HORIZON, collect_all, reversal_and_agreement_stats,
    save_raw_pairs, save_stats,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/eval2_branin_motivation/configs/h1_gpu23.yaml")
    parser.add_argument("--n-tasks", type=int, default=10)
    parser.add_argument("--n-candidates", type=int, default=10)
    parser.add_argument("--reference-milestone", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261002)
    parser.add_argument("--out-dir", default="experiments/eval2_branin_motivation/results/experiment_b")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    work_dir = out_dir / "mi_evaluations"

    rows = collect_all(
        cfg, task_indices=list(range(args.n_tasks)), reference_milestone=args.reference_milestone,
        n_candidates=args.n_candidates, horizons=HORIZONS, seed=args.seed, work_dir=work_dir,
    )
    raw_path = save_raw_pairs(rows, out_dir / "raw_pairs.json")
    print(f"wrote {len(rows)} raw pairs to {raw_path}")

    for reliability_filter in (False, True):
        stats = reversal_and_agreement_stats(
            rows, HORIZONS, reliability_filter=reliability_filter, z_min=cfg.mi_z_min,
        )
        suffix = "filtered" if reliability_filter else "unfiltered"
        stats_path = save_stats(stats, out_dir / f"stats_{suffix}.json")
        print(f"wrote stats ({suffix}) to {stats_path}")
        for h in HORIZONS:
            row = stats["by_h"][h]
            print(f"  h={h:>2d}  n={row['n_pairs']:>4d}  reversal_rate={row['reversal_rate']:.3f} "
                  f"[{row['ci95_low']:.3f},{row['ci95_high']:.3f}]")


if __name__ == "__main__":
    main()
