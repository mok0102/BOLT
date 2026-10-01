"""Generate synthetic Branin heldout figures and tables from run trajectories.

Run from the BOLT repository root. See README.md for commands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ablation
import fewshot
import main_bo
import scaling

BOLT_ROOT = Path(__file__).resolve().parents[2]
if str(BOLT_ROOT) not in sys.path:
    sys.path.insert(0, str(BOLT_ROOT))
from synthetic_experiment.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('main_bo', 'fewshot', 'scaling', 'ablation', 'all'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--milestone', type=int, help='Required for main_bo, fewshot, and ablation; defaults to final milestone')
    parser.add_argument('--run-dir', action='append', required=True, metavar='ARM=PATH',
                        help='Repeat for each arm, e.g. BOLT=runs/bolt_id')
    parser.add_argument('--out-dir', type=Path, default=Path('synthetic_experiment/eval2/results/synthetic'))
    parser.add_argument('--main-bo-inset', action='store_true', help='Save the inset layout as main_bo_synthetic_new.png')
    args = parser.parse_args()
    cfg = load_config(args.config)
    milestone = args.milestone or cfg.milestones[-1]
    if milestone not in cfg.milestones:
        parser.error(f'milestone {milestone} is not in config milestones {cfg.milestones}')
    run_dirs = {}
    for item in args.run_dir:
        arm, sep, path = item.partition('=')
        if not sep or not arm or not path:
            parser.error(f'--run-dir expects ARM=PATH, got {item!r}')
        run_dir = Path(path)
        base, dash, number = arm.rpartition('-')
        explicit = dash and number.isdigit()
        arm_dir = arm if explicit or arm in {"STBO", "LLAMBO"} else f"{arm}-{milestone}"
        if not any((run_dir / "heldout" / arm_dir).glob("task_*.csv")):
            parser.error(f"no heldout trajectories for {arm_dir} under {run_dir / 'heldout' / arm_dir}")
        run_dirs[arm] = run_dir
    if args.command in ('main_bo', 'all'):
        main_bo.generate(cfg, milestone, run_dirs, args.out_dir, inset=args.main_bo_inset)
    if args.command in ('fewshot', 'all'):
        fewshot.generate(cfg, milestone, run_dirs, args.out_dir)
    if args.command in ('scaling', 'all'):
        scaling.generate(cfg, run_dirs, args.out_dir, final=False)
        scaling.generate(cfg, run_dirs, args.out_dir, final=True)
    if args.command in ('ablation', 'all'):
        ablation.generate(cfg, milestone, run_dirs, args.out_dir)


if __name__ == '__main__':
    main()
