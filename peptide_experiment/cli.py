"""Single entry point for the peptide_experiment package, replacing
peptide_script.sh / peptide_script_dpo.sh.

Usage (run from the BOLT repo root):
    python -m peptide_experiment.cli trajectory_chain \\
        --config peptide_experiment/configs/peptide_smoke.yaml

    python -m peptide_experiment.cli init_only_eval \\
        --config peptide_experiment/configs/peptide_smoke.yaml \\
        --arm all --tasks heldout20

    python -m peptide_experiment.cli heldout_eval \\
        --config peptide_experiment/configs/peptide_smoke.yaml \\
        --arm all --tasks heldout100

    python -m peptide_experiment.cli aggregate \\
        --config peptide_experiment/configs/peptide_smoke.yaml --tasks both
"""

from __future__ import annotations

import fire

from .aggregate import _arms as _all_arms
from .aggregate import build_bo_scaling_curve, build_no_bo_milestone_eval
from .config import load_config
from .gp_expert_transfer import train_gp_expert_pool
from .heldout_eval import run_heldout_eval, run_init_only_eval
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

    def heldout_eval(self, config: str, arm: str, tasks: str) -> None:
        assert tasks in ("heldout20", "heldout100"), tasks
        cfg = load_config(config)
        arms = _all_arms(cfg) if arm == "all" else [arm]
        for a in arms:
            run_heldout_eval(cfg, a, tasks)

    def init_only_eval(self, config: str, arm: str, tasks: str) -> None:
        assert tasks in ("heldout20", "heldout100"), tasks
        cfg = load_config(config)
        arms = _all_arms(cfg) if arm == "all" else [arm]
        for a in arms:
            run_init_only_eval(cfg, a, tasks)

    def aggregate(self, config: str, tasks: str = "both") -> None:
        assert tasks in ("heldout20", "heldout100", "both"), tasks
        cfg = load_config(config)
        if tasks in ("heldout20", "both"):
            build_no_bo_milestone_eval(cfg)
        if tasks in ("heldout100", "both"):
            build_bo_scaling_curve(cfg)


if __name__ == "__main__":
    fire.Fire(CLI)
