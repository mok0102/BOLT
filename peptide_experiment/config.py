"""Experiment config: one YAML per experiment, resolved into absolute paths.

See imp_plan/01_peptide_reimplementation_plan.md for why this replaces the
hardcoded shell variables the old root scripts used.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

BOLT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOLT_ROOT / "optimization" / "peptides"))
from apex_oracle.task_splits import HELDOUT20_TASKS, HELDOUT100_TASKS  # noqa: E402


@dataclass
class ExperimentConfig:
    experiment_id: str
    milestones: list[int]
    oracle_budget: int
    init_size: int
    bsz: int
    sft_epochs: int
    similarity_threshold: float = 0.75
    task_specific_args: str = "bacteria_0"
    torchtune_config: str = "qwen_2_5_3B_lora.yaml"
    torchtune_recipe: str = "lora_finetune_distributed"
    base_checkpoint_dir: Path = field(
        default_factory=lambda: BOLT_ROOT / "fine-tuning" / "peptides" / "ckpt" / "Qwen2.5-3B-Instruct"
    )
    # False (default): run_bo() passes --path_to_vae_statedict "" so the
    # LOLBO subprocess skips loading the pretrained UniRef-VAE checkpoint.
    use_pretrained_vae: bool = False
    max_train_tasks: int | None = None  # None -> max(milestones)
    # e.g. "5" to pin all subprocesses to one specific GPU (matches the
    # CUDA_VISIBLE_DEVICES=N pattern the old root scripts used). None -> don't set it.
    cuda_visible_devices: str | None = None
    # Smoke-test escape hatch: override both held-out sets with a tiny list of
    # task indices, so a smoke run doesn't try to sweep the real 20/100-task
    # paper held-out sets. None -> use the real paper splits.
    heldout_tasks_override: list[int] | None = None
    # Oracle-call checkpoints for the no_bo_milestone_eval / bo_scaling_curve
    # aggregations (paper's Table 11 / Figure 1-2). Paper default is
    # [1, 100, 200, 500, 1000] (Table 11); override to something smaller than
    # oracle_budget for a smoke run.
    table_k_checkpoints: list[int] = field(default_factory=lambda: [1, 100, 200, 500, 1000])

    # ORPT (Stage 4): when True, an ORPT-<m> DPO stage is trained on top of
    # every BOLT-<m>, and -- critically -- tasks sampled after the first
    # milestone come from the latest ORPT-<m> checkpoint rather than BOLT-<m>
    # (see trajectory_chain.py's checkpoint_to_sample_from). False by default
    # so existing BOLT-only configs are unaffected.
    build_orpt: bool = False
    orpt_epochs: int = 1
    orpt_pairs_per_task: int = 1000  # mirrors make_dpo_train_data_csv.py's own default
    # Only used when orpt_loss_type == "dpo_ii_penalty" -- mirrors
    # orpt_pairs_per_task, but for make_infeasible_singles_csv.py's
    # individually-sampled (not paired) infeasible completions.
    orpt_infeasible_singles_per_task: int = 1000
    orpt_beta: float = 0.1  # DPOLoss's beta (reference-relative logit scale)
    orpt_lr: float = 3e-4  # DPO optimizer learning rate
    # "feasible_only" is the only supported mode: preference pairs are built
    # only from similarity-constraint-feasible candidates, ranked by objective
    # score alone (see make_dpo_train_data_csv.py's sample_pairs()/
    # _pick_chosen_rejected()). Superseded candidate-level modes
    # ("lexicographic"/"feasibility_aware") were removed along with ORPT-LEX/
    # ORPT-FA -- see imp_plan/05_pool_orpt_phase1_plan.md.
    orpt_pairing_mode: str = "feasible_only"
    orpt_torchtune_config: str = "qwen_2_5_3B_lora_dpo.yaml"
    orpt_torchtune_recipe: str = "lora_dpo_distributed"
    # "dpo" (default, unchanged): torchtune.rlhf.loss.DPOLoss via the stock
    # lora_dpo_distributed recipe. "dpo_ii_penalty": an ablation that keeps
    # orpt_pairing_mode="feasible_only" and the stock DPOLoss completely
    # untouched, and additively suppresses individually-sampled infeasible
    # completions (fine-tuning/peptides/dpo_ii/loss.py's
    # InfeasibleSuppressionLoss, no pairing needed) on top; set
    # orpt_torchtune_recipe="dpo_ii/recipe.py" and orpt_torchtune_config=
    # "poc_qwen_2_5_3B_lora_dpo_ii_penalty.yaml" alongside this. (The
    # feasibility-aware "fa_orpt" loss type this ablation was built against
    # was removed along with ORPT-FA -- see imp_plan/05_pool_orpt_phase1_plan.md.)
    # "bpo": Balanced Preference Optimization (arxiv 2506.03557) --
    # fine-tuning/peptides/bpo_loss.py's BPOLoss, a drop-in loss._component_
    # replacement under the STOCK lora_dpo_distributed recipe (no recipe
    # override needed, unlike dpo_ii_penalty -- BPOLoss only changes the loss
    # formula on the same single (chosen, rejected) dataset). Set
    # orpt_torchtune_config="poc_qwen_2_5_3B_lora_dpo_bpo.yaml" alongside this.
    orpt_loss_type: str = "dpo"
    # dpo_ii_penalty's InfeasibleSuppressionLoss hyperparameters (see
    # fine-tuning/peptides/dpo_ii/loss.py's docstring for the notation);
    # unused when orpt_loss_type == "dpo". Field names kept as fa_orpt_* since
    # they were originally shared with the now-removed fa_orpt loss and
    # renaming would touch peptide_poc20_orpt_dpo_ii.yaml for no functional
    # gain.
    fa_orpt_gamma_i: float = 0.5
    fa_orpt_lambda_inf: float = 1.0
    # bpo's "gap adaptor" alpha (see bpo_loss.py's docstring); unused unless
    # orpt_loss_type == "bpo". Paper default 0.5 for the logistic-loss variant.
    orpt_bpo_alpha: float = 0.5

    # Matched-intervention ORPT (paper/method.tex, paper/appendix.tex --
    # actual one-step BO evaluator; peptide_experiment/mi_orpt/).
    # "objective_ranked" (default, unchanged) keeps build_orpt_pairs() on the
    # existing make_dpo_train_data_csv.py raw-score-ranking path.
    # "matched_intervention" switches it to peptide_experiment/mi_orpt/
    # build_pairs.py instead -- a different pair *labeling mechanism*, kept
    # separate from orpt_pairing_mode (which stays its own, narrower
    # "feasible_only"-only knob). Pool size m deliberately reuses init_size
    # rather than adding a new field; pair count is its own mi_* fields
    # below (unlike m, K's semantics genuinely differ between pairing
    # modes -- see mi_target_pairs_per_task's docstring).
    orpt_pair_source: str = "objective_ranked"
    mi_num_backgrounds: int = 8  # M, shared backgrounds sampled once per task and reused across every candidate evaluated for it
    mi_tau_q: float = 1.0  # reference-aligned distribution q_t's softmax temperature
    mi_z_min: float = 1.96  # SNR reliability threshold a pair's Delta_1 must clear
    mi_delta_t: float = 0.0  # min |Delta_1| (numerical tolerance only, not a tunable effect-size floor)
    # matched_intervention builds pairs via a target/max-candidates loop, not
    # a fixed proposal count: orpt_pairs_per_task (objective_ranked's field)
    # is a *guaranteed* final pair count for that pairing mode, but for
    # matched_intervention the reliability filter can reject any given
    # comparison, so reusing the same field would silently mean "up to this
    # many pairs, maybe fewer, maybe zero" -- not interpretable from the
    # config alone. These two fields keep the semantics honest: keep
    # evaluating new candidates (cached against the same mi_num_backgrounds
    # shared backgrounds, memos/suggestion.txt's design -- every newly
    # evaluated candidate is compared against every previously evaluated one
    # for free, no fresh evaluation per comparison) until either
    # mi_target_pairs_per_task reliable pairs are found, or
    # mi_max_candidates_per_task candidates have been evaluated. Each
    # candidate costs exactly mi_num_backgrounds real-BO calls (not
    # 2x mi_num_backgrounds -- there's no per-pair "arm" cost anymore).
    mi_target_pairs_per_task: int = 5
    mi_max_candidates_per_task: int = 20
    # 1 (default): the paper's actual method -- one real BO acquisition
    # round per matched pool, real oracle calls during pair construction.
    # 0: the zero-step ablation (paper/experiments.tex sec:ablations) --
    # ranks by the pool's own best already-known value, no BO round, no
    # additional oracle cost.
    mi_bo_steps: int = 1
    # CUDA device ids to round-robin across for concurrent one-step BO
    # calls during matched_intervention pair construction (mi_orpt/
    # one_step_evaluator.py). Each of a candidate's M background
    # evaluations is independent (fully separate LOLBO subprocesses),
    # so this cuts wall-clock cost by up to len(mi_parallel_gpus)x on a
    # multi-GPU machine. None (default): fully serial, one call at a time
    # on cuda_visible_devices -- today's behavior, unchanged. Independent
    # of cuda_visible_devices, which still governs everything else
    # (trajectory sampling, BOLT SFT, ORPT DPO training) -- these never
    # run concurrently with pair construction, so the two can overlap or
    # not without any runtime GPU contention.
    mi_parallel_gpus: list[str] | None = None
    # None (default): matched_intervention's candidate bank reads the same
    # cumulative trajectory data (trajectories_csv_dir) that milestone's
    # BOLT-<m> SFT dataset is also built from -- today's behavior, unchanged,
    # one shared sampling temperature for both. A float value (e.g. 1.5):
    # build a SEPARATE, dedicated candidate pool for mi_orpt specifically at
    # this sampling temperature (steps.py::sample_and_build_init's new
    # temperature param, orpt.py::build_orpt_pairs's matched_intervention
    # branch), leaving BOLT-SFT's own trajectory data at the default
    # temperature -- lets pair construction draw from a more diverse pool
    # without changing what BOLT-SFT itself trains on.
    mi_candidate_temperature: float | None = None
    # Target unique-candidate count for the dedicated mi_candidate_temperature
    # sampling pass (steps.py::sample_and_build_init's own init_size override,
    # via dataclasses.replace -- same idiom mi_orpt/one_step_evaluator.py
    # already uses for a scoped init_size override). None (default): computed
    # as 2x min_bank_size_needed(init_size, mi_max_candidates_per_task) --
    # deliberately NOT just init_size itself, since that's mathematically
    # guaranteed to fall short of min_bank_size_needed=init_size-1+
    # mi_max_candidates_per_task even before any candidates are filtered out
    # for infeasibility (real observed feasible-fraction-of-unique-draws at
    # temp=1.5/milestone 5: ~57-78%, hence the 2x margin rather than 1x).
    mi_candidate_pool_size: int | None = None

    # Shared-surrogate MTBO baseline (paper's DKT/FSBO-style comparison --
    # peptide_experiment/mtbo.py): False (default) leaves _arms() unchanged,
    # so no existing config/run is affected. True adds a per-milestone
    # MTBO-<m> arm, mirroring BOLT-<m>/ORPT-<m>'s own milestone-tiered shape
    # (the paper's own MTBO hyperparameters are explicitly milestone-tiered,
    # unlike STBO).
    build_mtbo: bool = False
    # Feasible top-N sequences pooled per training task to fit the shared
    # surrogate on -- mirrors make_train_data_csv.py's own top_n default for
    # BOLT-SFT's dataset, so the same "how much of each task's trajectory
    # counts as good data" notion is reused rather than reinvented.
    mtbo_top_n_per_task: int = 1000
    mtbo_num_inducing_points: int = 1024  # paper's value
    mtbo_lr: float = 0.01  # paper's value
    mtbo_epochs: int = 20  # paper's value

    # OptFormer baseline (paper's LLM-as-optimizer comparison, reimplemented
    # against this repo's own torchtune/Qwen infra rather than the paper's
    # GPT-4o-mini -- fine-tuning/peptides/make_optformer_train_data_csv.py,
    # peptide_experiment/optformer.py, peptide_experiment/
    # optformer_optimization.py): False (default) leaves _arms() unchanged.
    # True adds a per-milestone OptFormer-<m> arm (retrained at each
    # milestone on only that milestone's completed trajectories, mirroring
    # MTBO-<m>/BOLT-<m>/ORPT-<m>).
    build_optformer: bool = False
    optformer_epochs: int = 1  # paper's value
    # "100 trials in" (paper's value) -- how much recent trial history is
    # serialized into the prompt, both at train time (make_optformer_train_data_csv.py's
    # windows) and held-out inference time (run_optformer_bo).
    optformer_context_length: int = 100
    optformer_windows_per_task: int = 50
    # Paper uses 1000; this repo's much smaller per-milestone training
    # corpus uses fewer by default (see make_optformer_train_data_csv.py's
    # docstring) -- sparse bins would add label complexity without more
    # usable signal at this scale.
    optformer_num_score_bins: int = 100
    optformer_temperature: float = 0.7  # paper's value

    # GP-expert-transfer baseline (POGPE/SGPE, Schilling et al. 2016 --
    # peptide_experiment/gp_expert_transfer.py): an ensemble of per-task GP
    # experts combined via product-of-experts at inference, distinct from
    # shared-surrogate MTBO's single surrogate pooling all training tasks
    # together. False (default) leaves _arms() unchanged. gp_expert_counts
    # doubles as the milestone axis for this baseline (POGPE-5/SGPE-5 <->
    # milestone=5, etc.) rather than being crossed against cfg.milestones --
    # matches the paper's own framing ("the first 5/10/20 trajectories were
    # used to train the POGPE/SGPE expert models").
    build_gp_expert_transfer: bool = False
    gp_expert_counts: list[int] = field(default_factory=lambda: [5, 10, 20])  # paper's own configurations
    # Feasible top-N sequences collected per task to fit that task's own
    # expert on (NOT pooled across tasks, unlike mtbo_top_n_per_task).
    gp_expert_top_n_per_task: int = 1024
    gp_expert_num_inducing_points: int = 1024  # paper's value; inherited from MTBO's own (same appendix architecture)
    gp_expert_lr: float = 0.01  # paper's value; inherited from MTBO's own
    gp_expert_epochs: int = 20  # inherited from MTBO's own (not independently re-derived from the paper)

    # LLAMBO baseline (Liu et al. 2024), reimplemented against the same
    # unmodified, zero-shot base_checkpoint_dir every other arm fine-tunes
    # from (paper's own LLAMBO uses GPT-4o-mini out-of-the-box; no
    # fine-tuning stage here either, see peptide_experiment/
    # llambo_optimization.py). No build_llambo flag/no _arms() integration --
    # unlike every other baseline, LLAMBO has no milestone-dependent
    # checkpoint, so it's added directly as a single manifest entry in
    # fixed_target_rejection_bo.py rather than expanded per-milestone.
    llambo_max_input_tokens: int = 10_000_000  # paper's value
    llambo_alpha: float = 0.1  # paper's value (EI's xi / target-conditioning margin)
    llambo_m: int = 20  # paper's value: candidates sampled per round (half unconditioned, half target-conditioned)
    llambo_k_mc_samples: int = 10  # paper's value: in-context-surrogate MC samples per candidate
    llambo_context_length: int = 100  # trials shown in-prompt; mirrors optformer_context_length
    llambo_temperature: float = 0.7  # paper's value
    llambo_top_p: float = 0.95
    # LLAMBO's own algorithm scores one real candidate per round (sequential,
    # matching the paper) -- deliberately not reusing cfg.bsz, which every
    # other arm's batched round means something different for.
    llambo_batch_size: int = 1

    bolt_root: Path = BOLT_ROOT
    heldout20_tasks: list[int] = field(default_factory=lambda: list(HELDOUT20_TASKS))
    heldout100_tasks: list[int] = field(default_factory=lambda: list(HELDOUT100_TASKS))

    def __post_init__(self) -> None:
        self.base_checkpoint_dir = Path(self.base_checkpoint_dir)
        if not self.base_checkpoint_dir.is_absolute():
            self.base_checkpoint_dir = self.bolt_root / self.base_checkpoint_dir
        self.milestones = sorted(self.milestones)
        if self.heldout_tasks_override is not None:
            self.heldout20_tasks = list(self.heldout_tasks_override)
            self.heldout100_tasks = list(self.heldout_tasks_override)

    # -- derived paths --------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self.bolt_root / "runs" / self.experiment_id

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
    def aggregate_dir(self) -> Path:
        return self.run_dir / "aggregate"

    @property
    def orpt_pairs_dir(self) -> Path:
        return self.run_dir / "orpt_pairs"

    @property
    def optformer_dir(self) -> Path:
        """OptFormer's own training-data + frozen score-bin-edges sidecar,
        separate per milestone (see optformer.py::train_optformer) --
        distinct from checkpoints_dir, which holds the fine-tuned model
        itself."""
        return self.run_dir / "optformer"

    @property
    def tensorboard_dir(self) -> Path:
        return self.run_dir / "tensorboard"

    def heldout_dir(self, task_set: str) -> Path:
        assert task_set in ("heldout20", "heldout100")
        return self.run_dir / task_set

    def heldout_tasks(self, task_set: str) -> list[int]:
        assert task_set in ("heldout20", "heldout100")
        return self.heldout20_tasks if task_set == "heldout20" else self.heldout100_tasks

    def train_task_range(self) -> range:
        n = self.max_train_tasks if self.max_train_tasks is not None else max(self.milestones)
        return range(0, n)

    def milestone_checkpoint_dir(self, milestone: int) -> Path:
        return self.checkpoints_dir / f"BOLT-{milestone}" / f"epoch_{self.sft_epochs - 1}"

    def orpt_checkpoint_dir(self, milestone: int) -> Path:
        """ORPT-<milestone>'s final checkpoint: a DPO stage trained on top of
        that same milestone's own BOLT-<milestone>.
        """
        return self.checkpoints_dir / f"ORPT-{milestone}" / f"epoch_{self.orpt_epochs - 1}"

    def mtbo_checkpoint_dir(self, milestone: int) -> Path:
        """MTBO-<milestone>'s trained shared-surrogate state dict (see
        mtbo.py::train_mtbo_surrogate) -- not an LLM checkpoint dir like
        milestone_checkpoint_dir()/orpt_checkpoint_dir(), a single .pt file.
        """
        return self.checkpoints_dir / f"MTBO-{milestone}" / "surrogate_state_dict.pt"

    def optformer_checkpoint_dir(self, milestone: int) -> Path:
        """OptFormer-<milestone>'s final checkpoint: a from-scratch LoRA SFT
        stage trained on history-conditioned windows (see
        optformer.py::train_optformer), retrained at each milestone on only
        that milestone's completed trajectories.
        """
        return self.checkpoints_dir / f"OptFormer-{milestone}" / f"epoch_{self.optformer_epochs - 1}"

    def gp_expert_dir(self, n_experts: int) -> Path:
        """Root directory for POGPE-<n_experts>/SGPE-<n_experts>'s shared
        pretrained expert pool (see gp_expert_transfer.py::train_gp_expert_pool)."""
        return self.checkpoints_dir / f"GPExperts-{n_experts}"

    def gp_expert_checkpoint(self, n_experts: int, expert_idx: int) -> Path:
        return self.gp_expert_dir(n_experts) / f"expert_{expert_idx:02d}" / "surrogate_state_dict.pt"

    def gp_expert_manifest(self, n_experts: int) -> Path:
        """POGPE's ready-to-use {experts: [{path, weight}, ...]} manifest --
        SGPE additionally appends its own online target expert on top of this
        (see gp_expert_transfer.py::build_sgpe_manifest)."""
        return self.gp_expert_dir(n_experts) / "poe_manifest.json"

    def ensure_dirs(self) -> None:
        for d in (
            self.trajectories_dir,
            self.trajectories_csv_dir,
            self.checkpoints_dir,
            self.milestones_dir,
            self.aggregate_dir,
            self.orpt_pairs_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    if not path.is_absolute():
        path = BOLT_ROOT / path
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = ExperimentConfig(**raw)
    if cfg.cuda_visible_devices is not None:
        # Also pins any in-process CUDA use (e.g. apex_wrapper called directly
        # from steps.py), not just the subprocesses steps.py launches.
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    return cfg
