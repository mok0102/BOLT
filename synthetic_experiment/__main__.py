"""Dependency-free command-line entry point (``python -m synthetic_experiment``)."""

from __future__ import annotations

import argparse

from .aggregate import arms, build_summary
from .config import load_config
from .heldout_eval import run_heldout_eval
from .orpt import build_orpt_pairs
from .trajectory_chain import run_trajectory_chain


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Branin BOLT/MI-ORPT experiment")
    subs = parser.add_subparsers(dest="command", required=True)
    for command in ("trajectory_chain", "aggregate", "train_pogpe", "train_sgpe", "train_optformer"):
        child = subs.add_parser(command)
        child.add_argument("--config", required=True)
        if command in {"train_pogpe", "train_sgpe", "train_optformer"}:
            child.add_argument("--milestone", required=True, type=int)
    pairs = subs.add_parser("build_orpt_pairs")
    pairs.add_argument("--config", required=True)
    pairs.add_argument("--milestone", required=True, type=int)
    heldout = subs.add_parser("heldout_eval")
    heldout.add_argument("--config", required=True)
    heldout.add_argument("--arm", default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.command == "trajectory_chain":
        run_trajectory_chain(cfg)
    elif args.command == "build_orpt_pairs":
        print(build_orpt_pairs(cfg, args.milestone))
    elif args.command in {"train_pogpe", "train_sgpe"}:
        from .gp_expert_transfer import train_gp_expert_pool
        print(train_gp_expert_pool(cfg, args.milestone))
    elif args.command == "train_optformer":
        from .optformer import train_optformer
        print(train_optformer(cfg, args.milestone))
    elif args.command == "heldout_eval":
        for arm in arms(cfg) if args.arm == "all" else [args.arm]:
            run_heldout_eval(cfg, arm)
    else:
        print(build_summary(cfg))


if __name__ == "__main__":
    main()
