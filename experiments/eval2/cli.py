"""Single entry point for the whole peptide eval chain: sampling, the two
compute engines, the baseline reports, and every paper figure/table.

Run as a module from the BOLT repo root, which is what puts peptide_experiment
on sys.path:

    # compute (needs a GPU / the oracle)
    python -m experiments.eval2.cli generate_raw \\
        --config peptide_experiment/configs/peptide_main_bolt.yaml \\
        --arms-file experiments/eval2/arm_specs/main.yaml --task-sets heldout100

    python -m experiments.eval2.cli incumbent \\
        --config peptide_experiment/configs/peptide_main_bolt.yaml \\
        --arms-file experiments/eval2/arm_specs/main.yaml --task-sets heldout100 \\
        --out-dir experiments/eval2/results/main

    python -m experiments.eval2.cli fixed_target_bo \\
        --config peptide_experiment/configs/peptide_main_bolt.yaml \\
        --arms-file experiments/eval2/arm_specs/main.yaml --task-sets heldout100 \\
        --target-pool-sizes 100 --out-dir experiments/eval2/results/main

    # figures (pure post-hoc, no GPU)
    python -m experiments.eval2.cli fewshot --milestone 600 \\
        --results-dir experiments/eval2/results/main \\
        --out-dir experiments/eval2/results/peptide

Sharding across GPUs is expressed with --arms / --milestones / --cuda-visible-devices
against one arm-spec file; see pipelines/run_eval.sh.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from .compute import fixed_target_bo, generate_raw, incumbent, reports
from .core.arms import load_arms
from .core.pools import bo_k_checkpoints
from .domains import peptide
from .figures import ablation, fewshot, main_bo, scaling


def _csv_list(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [item.strip() for item in raw.split(",") if item.strip()]


def _int_list(raw: str | None) -> list[int] | None:
    items = _csv_list(raw)
    return [int(i) for i in items] if items else None


def _parse_run_dirs(pairs: list[str]) -> dict[str, Path]:
    out = {}
    for pair in pairs:
        arm, _, path = pair.partition("=")
        if not path:
            raise argparse.ArgumentTypeError(f"--run-dir expects ARM=PATH, got {pair!r}")
        out[arm] = Path(path)
    return out


def _load(args):
    """Shared setup for the compute subcommands: config, GPU pinning, arms."""
    cfg = peptide.load_config(args.config)
    if args.cuda_visible_devices is not None:
        cfg = dataclasses.replace(cfg, cuda_visible_devices=str(args.cuda_visible_devices))
    specs = load_arms(args.arms_file, arms=_csv_list(args.arms), milestones=_int_list(args.milestones))
    return cfg, specs, _csv_list(args.task_sets) or ["heldout"]


def _add_compute_args(p, with_out_dir: bool = True):
    p.add_argument("--config", required=True, help="Config supplying similarity_threshold / init_size / task universe")
    p.add_argument("--arms-file", required=True, help="e.g. experiments/eval2/arm_specs/main.yaml")
    p.add_argument("--arms", default=None, help="Comma-separated subset of arm names (sharding)")
    p.add_argument("--milestones", default=None, help="Comma-separated subset of milestones (sharding)")
    p.add_argument("--task-sets", default="heldout")
    p.add_argument("--cuda-visible-devices", default=None, help="Pin this shard to one GPU")
    if with_out_dir:
        p.add_argument("--out-dir", required=True, type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate_raw", help="Sample raw proposals from each arm's checkpoint")
    _add_compute_args(p, with_out_dir=False)

    p = sub.add_parser("incumbent", help="Best-of-k proposal quality + rejection rates (fig:fewshot data)")
    _add_compute_args(p)
    p.add_argument("--n-proposals-checkpoints", default=None, help="Default: cfg.table_k_checkpoints")

    p = sub.add_parser("fixed_target_bo", help="Fixed-size rejection-sampled pool + real BO (fig:main-bo data)")
    _add_compute_args(p)
    p.add_argument("--target-pool-sizes", default=None, help="Comma-separated fixed init-pool sizes")
    p.add_argument("--limit-tasks", type=int, default=None, help="Smoke test: first N tasks per task_set only")

    p = sub.add_parser("baselines_report", help="POGPE/SGPE and LLAMBO text reports")
    p.add_argument("--config", required=True)
    p.add_argument("--results-dir", required=True, help="Comma-separated")
    p.add_argument("--bo-calls", type=int, default=None, help="Default: cfg.oracle_budget")

    p = sub.add_parser("main_bo", help="fig:main-bo")
    p.add_argument("--milestone", type=int, required=True)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--run-dir", action="append", required=True, help="Repeatable ARM=PATH")
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--target-pool-size", type=int, default=None, help="Default: cfg.init_size")
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("fewshot", help="fig:fewshot")
    p.add_argument("--milestone", type=int, required=True)
    p.add_argument("--results-dir", required=True, help="Comma-separated")
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("scaling", help="fig:scaling (init + finalbo PNGs)")
    p.add_argument("--results-dir", required=True, help="Comma-separated")
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    p = sub.add_parser("ablation", help="tab:ablation")
    p.add_argument("--milestone", type=int, required=True)
    p.add_argument("--results-dir", required=True, help="Comma-separated")
    p.add_argument("--task-set", default="heldout100")
    p.add_argument("--out-dir", required=True, type=Path)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "generate_raw":
        cfg, specs, task_sets = _load(args)
        generate_raw.run(peptide, cfg, specs, task_sets)

    elif args.command == "incumbent":
        cfg, specs, task_sets = _load(args)
        incumbent.run(peptide, cfg, specs, task_sets, args.out_dir,
                      n_proposals_checkpoints=_int_list(args.n_proposals_checkpoints))

    elif args.command == "fixed_target_bo":
        cfg, specs, task_sets = _load(args)
        fixed_target_bo.run(peptide, cfg, specs, task_sets, args.out_dir,
                            target_pool_sizes=_int_list(args.target_pool_sizes),
                            limit_tasks=args.limit_tasks)

    elif args.command == "baselines_report":
        cfg = peptide.load_config(args.config)
        bo_calls = args.bo_calls if args.bo_calls is not None else max(bo_k_checkpoints(cfg))
        reports.run(_csv_list(args.results_dir), bo_calls)

    elif args.command == "main_bo":
        main_bo.generate(
            milestone=args.milestone,
            config_path=args.config,
            run_dirs=_parse_run_dirs(args.run_dir),
            task_set=args.task_set,
            out_dir=args.out_dir,
            target_pool_size=args.target_pool_size,
        )

    elif args.command == "fewshot":
        fewshot.generate(
            milestone=args.milestone,
            results_dirs=_csv_list(args.results_dir),
            task_set=args.task_set,
            out_dir=args.out_dir,
        )

    elif args.command == "scaling":
        results_dirs = _csv_list(args.results_dir)
        scaling.generate_init(results_dirs=results_dirs, task_set=args.task_set, out_dir=args.out_dir)
        scaling.generate_finalbo(results_dirs=results_dirs, task_set=args.task_set, out_dir=args.out_dir)

    elif args.command == "ablation":
        ablation.generate(
            results_dirs=_csv_list(args.results_dir),
            task_set=args.task_set,
            milestone=args.milestone,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
