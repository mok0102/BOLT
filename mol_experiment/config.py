"""Experiment config for the mol (BindingDB molecular lead-optimization) domain.
Mirrors peptide_experiment/config.py's shape (read-only reference, never
imported -- see the isolation contract in
impl_plan/opus_bindingdb_implementation_prompt.md) but:

- zero sys.path insertion / zero import of anything under optimization/peptides
  (contrast peptide_experiment/config.py:17-18, which inserts optimization/peptides
  into sys.path and imports apex_oracle.task_splits as an import-time side effect).
  Task identity comes from optimization/mol/mol_tasks.py instead.
- tau_mol replaces similarity_threshold (locked at 0.4, see
  optimization/mol/MOL_LATENT_SPACE_FINDING.md); no task_specific_args field --
  mol's oracle is fixed per task via the manifest, not selected by an index into
  a bacteria-strain list.
- a single heldout task set (the manifest's own fixed 50), not peptide's
  heldout20/50/100 tiering -- mol's scale is locked at 900/50, one tier only.
- run_dir is runs/mol/<experiment_id>, never the flat runs/<experiment_id>
  peptide uses (isolation contract R6).
- base_checkpoint_dir defaults to peptide's OWN Qwen2.5-3B-Instruct checkpoint,
  referenced read-only (R6: "may reference it read-only ... but must not write
  there") -- this repo does not duplicate a second 3B base model on disk.

Deliberately NOT ported in this first skeleton: MTBO/OptFormer/GP-expert-transfer/
LLAMBO baseline-arm fields (peptide's build_mtbo/build_optformer/
build_gp_expert_transfer/llambo_* and their hyperparameters). Those are secondary
comparison methods, not part of the core BOLT/ORPT pipeline this milestone builds;
porting them is out of scope until a mol MTBO/OptFormer/etc. baseline is actually
requested.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "optimization" / "mol"))
# The only path mol_experiment inserts into sys.path -- optimization/mol -- per
# the isolation contract's "optimization/mol is the only path mol inserts into
# sys.path" rule. No apex_oracle/peptide import happens here or anywhere below.

BOLT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class MolExperimentConfig:
    experiment_id: str
    milestones: list[int]
    oracle_budget: int
    init_size: int
    bsz: int
    sft_epochs: int
    tau_mol: float = 0.4  # locked, Milestone 4 -- see MOL_LATENT_SPACE_FINDING.md
    # >1 -> mol_run_bo creates a persistent CPU worker pool (per task, reused
    # across all its BO steps -- optimization/mol/mol_generation_pool.py) and
    # splits each step's generate_feasible_candidates_around_seed call across
    # it. Default 1 = current serial behavior, backward compatible. See
    # mol_experiment/warm_task_pool.py's docstring for the real profiling
    # that motivated this (generation cost scales ~linearly with n_candidates,
    # not fixed-overhead dominated, so this is a real lever).
    generation_workers: int = 1
    torchtune_config: str = "qwen_2_5_3B_lora.yaml"
    torchtune_recipe: str = "lora_finetune_distributed"
    # Read-only reference to peptide's own base checkpoint (R6) -- not copied.
    base_checkpoint_dir: Path = field(
        default_factory=lambda: BOLT_ROOT / "fine-tuning" / "peptides" / "ckpt" / "Qwen2.5-3B-Instruct"
    )
    use_pretrained_vae: bool = True  # mol always uses the official LOL-BO SELFIES VAE; no "skip" mode
    max_train_tasks: int | None = None  # None -> max(milestones)
    cuda_visible_devices: str | None = None
    heldout_tasks_override: list[int] | None = None
    table_k_checkpoints: list[int] = field(default_factory=lambda: [1, 100, 200, 500, 1000])

    # ORPT (mirrors peptide's core knobs; ablation-only fields like
    # orpt_loss_type's dpo_ii_penalty/bpo variants and fa_orpt_* are deferred --
    # add them if/when a mol ablation actually needs them)
    build_orpt: bool = False
    orpt_epochs: int = 1
    orpt_pairs_per_task: int = 1000
    orpt_beta: float = 0.1
    orpt_lr: float = 3e-4  # placeholder -- spec requires retuning at smoke scale, not reusing peptide's value blindly
    orpt_pairing_mode: str = "feasible_only"
    orpt_torchtune_config: str = "qwen_2_5_3B_lora_dpo.yaml"
    orpt_torchtune_recipe: str = "lora_dpo_distributed"
    orpt_loss_type: str = "dpo"

    # Matched-intervention ORPT (mi_orpt/) -- same fields/semantics as peptide's,
    # since the algorithm is domain-generic (pool -> q_t -> matched one-step BO ->
    # z-filter -> JSONL); only the domain adapters mi_orpt/ calls into differ.
    #
    # Default differs from peptide's ("objective_ranked"): mol has no port of
    # make_dpo_train_data_csv.py (peptide's uniform-random-pairing baseline
    # arm) yet -- orpt.py::build_orpt_pairs raises NotImplementedError for
    # "objective_ranked" rather than silently approximating it. Defaulting to
    # "matched_intervention" (the paper's actual method, and the only pairing
    # path mi_orpt/ has been built and smoke-tested for) means the default
    # config exercises an implemented path instead of an unimplemented one.
    orpt_pair_source: str = "matched_intervention"
    mi_num_backgrounds: int = 8
    mi_background_size: int | None = None
    mi_tau_q: float = 1.0
    mi_z_min: float = 1.96
    mi_delta_t: float = 0.0
    mi_target_pairs_per_task: int = 5
    mi_max_candidates_per_task: int = 20
    mi_bo_steps: int = 1
    mi_require_h0: bool = True
    mi_max_tasks_per_milestone: int | None = None
    mi_parallel_gpus: list[str] | None = None
    mi_candidate_temperature: float | None = None
    mi_candidate_pool_size: int | None = None

    bolt_root: Path = BOLT_ROOT

    def __post_init__(self) -> None:
        self.base_checkpoint_dir = Path(self.base_checkpoint_dir)
        if not self.base_checkpoint_dir.is_absolute():
            self.base_checkpoint_dir = self.bolt_root / self.base_checkpoint_dir
        self.milestones = sorted(self.milestones)

    # -- task identity ----------------------------------------------------
    # Delegates to optimization/mol/mol_tasks.py (the manifest-backed
    # analogue of apex_oracle/refseqs.py + task_splits.py) rather than
    # keeping its own copy of the heldout list, so there is exactly one
    # place task order/scale is defined.

    def train_task_range(self):
        from mol_tasks import N_TRAIN_TASKS
        n = self.max_train_tasks if self.max_train_tasks is not None else max(self.milestones)
        return range(0, min(n, N_TRAIN_TASKS))

    def heldout_tasks(self, task_set: str = "heldout") -> list[int]:
        assert task_set == "heldout", "mol has exactly one heldout tier (50 tasks), not peptide's 20/50/100"
        if self.heldout_tasks_override is not None:
            return list(self.heldout_tasks_override)
        from mol_tasks import heldout_tasks as _heldout_tasks
        return _heldout_tasks()

    # -- derived paths ------------------------------------------------------
    # runs/mol/<experiment_id>, never the flat runs/<experiment_id> peptide uses
    # (isolation contract R6).

    @property
    def run_dir(self) -> Path:
        return self.bolt_root / "runs" / "mol" / self.experiment_id

    @property
    def trajectories_dir(self) -> Path:
        return self.run_dir / "trajectories"

    @property
    def trajectories_csv_dir(self) -> Path:
        return self.run_dir / "trajectories_csv"

    @property
    def checkpoints_dir(self) -> Path:
        return self.run_dir / "checkpoints"

    @property
    def milestones_dir(self) -> Path:
        return self.run_dir / "milestones"

    @property
    def orpt_pairs_dir(self) -> Path:
        return self.run_dir / "orpt_pairs"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    def heldout_dir(self, task_set: str = "heldout") -> Path:
        assert task_set == "heldout"
        return self.run_dir / task_set

    def milestone_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"BOLT-{milestone}"

    def orpt_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"ORPT-{milestone}"

    def ensure_dirs(self) -> None:
        for d in (
            self.trajectories_dir,
            self.trajectories_csv_dir,
            self.checkpoints_dir,
            self.milestones_dir,
            self.orpt_pairs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> MolExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        path = BOLT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = MolExperimentConfig(**raw)
    if cfg.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return cfg
