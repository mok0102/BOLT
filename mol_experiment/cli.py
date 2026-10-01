"""Single entry point for the mol_experiment package: training only. Mirrors
peptide_experiment/cli.py's shape (read-only reference, never imported --
isolation contract), minus the baseline-arm commands (train_mtbo/
train_optformer/train_gp_expert_transfer) -- those baselines have no mol
port yet (mol_experiment/config.py's own module docstring: "out of scope
until a mol MTBO/OptFormer/etc. baseline is actually requested").

Usage (run from the BOLT repo root):
    python -m mol_experiment.cli trajectory_chain \\
        --config mol_experiment/configs/mol_main_bolt.yaml
"""

from __future__ import annotations

import fire

from .config import load_config
from .trajectory_chain import run_trajectory_chain


class CLI:
    def trajectory_chain(self, config: str) -> None:
        cfg = load_config(config)
        run_trajectory_chain(cfg)


if __name__ == "__main__":
    fire.Fire(CLI)
