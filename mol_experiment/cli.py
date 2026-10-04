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

    def eval(self, config: str, arm: str, milestone: int, shard: str = "0/1") -> None:
        """Heldout BO for one arm ("BOLT" or "ORPT-H1") at one milestone. The
        config must set eval_init_size and eval_oracle_budget. Parallelize by
        launching several processes with different CUDA_VISIBLE_DEVICES and
        --shard k/N."""
        from .eval_bo import run_heldout_eval

        run_heldout_eval(load_config(config), arm, milestone, shard)

    def eval_summary(self, config: str, arm: str, milestone: int) -> None:
        """Best-objective-at-k table (cfg.table_k_checkpoints) over the heldout
        tasks already evaluated for this arm/milestone; writes summary.json."""
        from .eval_bo import summarize_heldout_eval

        summarize_heldout_eval(load_config(config), arm, milestone)


if __name__ == "__main__":
    fire.Fire(CLI)
