from __future__ import annotations

import fire

from .aggregate import arms, build_summary
from .config import load_config
from .heldout_eval import run_heldout_eval
from .orpt import build_orpt_pairs
from .trajectory_chain import run_trajectory_chain


class CLI:
    def trajectory_chain(self, config: str) -> None:
        run_trajectory_chain(load_config(config))

    def build_orpt_pairs(self, config: str, milestone: int) -> None:
        print(build_orpt_pairs(load_config(config), milestone))

    def train_mtbo(self, config: str, milestone: int) -> None:
        from .mtbo import train_mtbo_surrogate
        print(train_mtbo_surrogate(load_config(config), milestone))

    def train_pogpe(self, config: str, milestone: int) -> None:
        from .gp_expert_transfer import train_gp_expert_pool
        print(train_gp_expert_pool(load_config(config), milestone))

    def train_sgpe(self, config: str, milestone: int) -> None:
        from .gp_expert_transfer import train_gp_expert_pool
        print(train_gp_expert_pool(load_config(config), milestone))

    def train_optformer(self, config: str, milestone: int) -> None:
        from .optformer import train_optformer
        print(train_optformer(load_config(config), milestone))

    def heldout_eval(self, config: str, arm: str = "all") -> None:
        cfg = load_config(config)
        for name in arms(cfg) if arm == "all" else [arm]:
            run_heldout_eval(cfg, name)

    def aggregate(self, config: str) -> None:
        print(build_summary(load_config(config)))


if __name__ == "__main__":
    fire.Fire(CLI)
