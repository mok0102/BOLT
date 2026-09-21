"""Thin CLI dispatcher for experiments/eval2/'s figure/table generators.
Run directly (matches experiments/eval/'s own convention -- eval2/ is not a
package, scripts rely on their own directory being on sys.path):

Usage (run from the BOLT repo root):
    python experiments/eval2/cli.py main_bo --domain peptide --milestone 600 \\
        --run-dir BOLT=runs/peptide_main_bolt_v2_lowbudget \\
        --run-dir ORPT-H1=runs/peptide_main_orpt_h1_v2_lowbudget \\
        --task-set heldout100 --out-dir experiments/eval2/out/peptide

    python experiments/eval2/cli.py fewshot --domain peptide --milestone 600 \\
        --results-dir experiments/eval/results/main_v2_orpt_vs_bolt__lowbudget \\
        --out-dir experiments/eval2/out/peptide

    python experiments/eval2/cli.py scaling --domain peptide \\
        --results-dir experiments/eval/results/main_v2_orpt_vs_bolt__lowbudget \\
        --out-dir experiments/eval2/out/peptide

    python experiments/eval2/cli.py ablation --domain peptide --milestone 600 \\
        --results-dir experiments/eval/results/ablation_h0_vs_h1__lowbudget \\
        --out-dir experiments/eval2/out/peptide
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ablation
import fewshot
import main_bo
import scaling


def _split_results_dirs(raw: str) -> list[str]:
    return [d.strip() for d in raw.split(",") if d.strip()]


def _parse_run_dirs(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for pair in pairs:
        arm, _, path = pair.partition("=")
        if not path:
            raise argparse.ArgumentTypeError(f"--run-dir expects ARM=PATH, got {pair!r}")
        out[arm] = Path(path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("main_bo")
    p.add_argument("--domain", required=True, choices=("peptide", "llvm"))
    p.add_argument("--milestone", type=int, required=True)
    p.add_argument("--config", required=True, type=Path, help="Any config from the domain's manifest -- only used for similarity_threshold/task universe, never to select a model")
    p.add_argument("--run-dir", action="append", required=True, help="Repeatable ARM=PATH")
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("fewshot")
    p.add_argument("--domain", required=True, choices=("peptide", "llvm"))
    p.add_argument("--milestone", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("scaling")
    p.add_argument("--domain", required=True, choices=("peptide", "llvm"))
    p.add_argument("--results-dir", required=True)
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("ablation")
    p.add_argument("--domain", required=True, choices=("peptide", "llvm"))
    p.add_argument("--milestone", type=int, required=True)
    p.add_argument("--results-dir", required=True)
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    args = parser.parse_args()

    if args.command == "main_bo":
        main_bo.generate(
            domain=args.domain,
            milestone=args.milestone,
            config_path=args.config,
            run_dirs=_parse_run_dirs(args.run_dir),
            task_set=args.task_set,
            out_dir=args.out_dir,
        )
    elif args.command == "fewshot":
        fewshot.generate(
            domain=args.domain,
            milestone=args.milestone,
            results_dirs=_split_results_dirs(args.results_dir),
            task_set=args.task_set,
            out_dir=args.out_dir,
        )
    elif args.command == "scaling":
        results_dirs = _split_results_dirs(args.results_dir)
        scaling.generate_init(domain=args.domain, results_dirs=results_dirs, task_set=args.task_set, out_dir=args.out_dir)
        scaling.generate_finalbo(domain=args.domain, results_dirs=results_dirs, task_set=args.task_set, out_dir=args.out_dir)
    elif args.command == "ablation":
        ablation.generate(
            domain=args.domain,
            results_dirs=_split_results_dirs(args.results_dir),
            task_set=args.task_set,
            milestone=args.milestone,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
