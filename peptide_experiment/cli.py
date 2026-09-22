"""Single entry point for the peptide_experiment package: training only.

Evaluation lives in experiments/eval2/ and depends on this package, never the
reverse -- see experiments/eval2/pipelines/ for the train -> eval runbooks.

Usage (run from the BOLT repo root):
    python -m peptide_experiment.cli trajectory_chain \\
        --config peptide_experiment/configs/peptide_smoke.yaml

    python -m peptide_experiment.cli train_mtbo \\
        --config peptide_experiment/configs/peptide_main_bolt.yaml
"""

from __future__ import annotations

import fire

from .config import load_config
from .gp_expert_transfer import train_gp_expert_pool
from .mtbo import train_mtbo_surrogate
from .optformer import train_optformer as _train_optformer
from .trajectory_chain import run_trajectory_chain


class CLI:
    def trajectory_chain(self, config: str) -> None:
        cfg = load_config(config)
        run_trajectory_chain(cfg)

    def train_mtbo(self, config: str) -> None:
        cfg = load_config(config)
        for m in cfg.milestones:
            train_mtbo_surrogate(cfg, m)

    def train_optformer(self, config: str) -> None:
        cfg = load_config(config)
        for m in cfg.milestones:
            _train_optformer(cfg, m)

    def train_gp_expert_transfer(self, config: str) -> None:
        cfg = load_config(config)
        for n in cfg.gp_expert_counts:
            train_gp_expert_pool(cfg, n)


if __name__ == "__main__":
    fire.Fire(CLI)
