"""Offline GP-expert-transfer baseline (POGPE/SGPE, Schilling et al. 2016):
an ensemble of per-task GP experts, combined via product-of-experts at
inference (optimization/peptides/lolbo/utils/bo_utils/poe_gp.py::PoEGPModel),
rather than shared-surrogate MTBO's single surrogate pooling all training
tasks together.

POGPE: one expert per each of the first n_experts training tasks (NOT
pooled -- each expert only ever sees its own task's own data), equally
weighted. SGPE: POGPE's n_experts pretrained experts, plus one extra expert
fit on the held-out task's own init pool at eval time, weighted equal to the
sum of the pretrained experts' weights (Schilling et al. 2016's weighting
scheme: "the independent GP for the target dataset carries the same weight
as the entire set of experts").

Uses the same frozen, pretrained UniRef VAE as MTBO -- every ensemble member
must share one stable latent space for a product-of-experts combination at a
shared query point to be meaningful at all.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import gpytorch
import torch

from .config import ExperimentConfig
from .mtbo import PATH_TO_VAE_STATE_DICT, PEPTIDES_DIR  # also triggers mtbo.py's own sys.path setup

from lolbo.utils.bo_utils.ppgpr import GPModelDKL  # noqa: E402
from lolbo.utils.utils import update_surr_model  # noqa: E402
from uniref_vae.load_vae import load_vae, vae_forward  # noqa: E402
from make_train_data_csv import collect_top_sequences  # noqa: E402


def _fit_one_expert(
    seqs: list[str], scores: list[float], num_inducing_points: int, lr: float, epochs: int,
) -> tuple[dict, dict]:
    """Shared VAE-encode-then-fit-one-GPModelDKL routine (same recipe
    mtbo.py::train_mtbo_surrogate uses), factored out here so both the
    offline per-task pool (train_gp_expert_pool) and SGPE's online target
    expert (fit_sgpe_target_expert) share one implementation. Returns
    (state_dict, meta_dict); callers persist them via _save_expert.
    """
    if len(seqs) < 2:
        raise RuntimeError(f"need at least 2 sequences to fit an expert, got {len(seqs)}")

    # uniref_vae/data.py's DatasetKmers resolves its vocab/data CSVs via
    # relative paths, assuming CWD == optimization/peptides -- see
    # mtbo.py::train_mtbo_surrogate's identical comment/fix.
    prev_cwd = os.getcwd()
    os.chdir(PEPTIDES_DIR)
    try:
        vae, dataobj = load_vae(PATH_TO_VAE_STATE_DICT)
        with torch.no_grad():
            train_z, _vae_loss = vae_forward(seqs, dataobj, vae)
    finally:
        os.chdir(prev_cwd)
    train_z = train_z.detach().cpu()
    train_y = torch.tensor(scores, dtype=torch.float32)

    num_inducing = min(num_inducing_points, train_z.shape[0])
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = GPModelDKL(train_z[:num_inducing].cuda(), likelihood=likelihood).cuda()
    mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.size(-2))
    model = update_surr_model(model, mll, lr, train_z, train_y, epochs)

    meta = {"num_inducing_points": num_inducing, "n_train_points": len(seqs)}
    return model.state_dict(), meta


def _save_expert(state_dict: dict, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "surrogate_state_dict.pt"
    (out_dir / "surrogate_meta.json").write_text(json.dumps(meta, indent=2))
    torch.save(state_dict, ckpt_path)
    return ckpt_path


def train_gp_expert_pool(cfg: ExperimentConfig, n_experts: int) -> Path:
    """POGPE's/SGPE's shared pretrained pool: one independent expert per
    each of the first n_experts training tasks, NOT pooled across tasks like
    MTBO. Idempotent per-expert (skip any expert whose state dict already
    exists). Always (re)writes poe_manifest.json listing every expert that
    exists on disk for this n_experts, each weighted 1/n_experts (POGPE:
    "all experts weighted equally").
    """
    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    pool_dir = cfg.gp_expert_dir(n_experts)

    for task_idx in range(n_experts):
        ckpt_path = cfg.gp_expert_checkpoint(n_experts, task_idx)
        if ckpt_path.exists():
            print(f"[gp_expert_transfer N={n_experts}] expert {task_idx} already exists at {ckpt_path}, skipping")
            continue

        traj_csv = cfg.trajectories_csv_dir / f"task_{task_idx:04d}.csv"
        top_seqs, _rows_skipped, _rows_infeasible = collect_top_sequences(
            traj_csv, cfg.gp_expert_top_n_per_task, REFERENCE_SEQUENCE[task_idx], cfg.similarity_threshold,
        )
        if len(top_seqs) < 2:
            raise RuntimeError(
                f"[gp_expert_transfer N={n_experts}] task {task_idx} has only {len(top_seqs)} feasible "
                "sequences -- need at least 2 to fit an expert"
            )
        seqs = [seq for _score, seq in top_seqs]
        scores = [score for score, _seq in top_seqs]
        state_dict, meta = _fit_one_expert(
            seqs, scores, cfg.gp_expert_num_inducing_points, cfg.gp_expert_lr, cfg.gp_expert_epochs,
        )
        _save_expert(state_dict, meta, ckpt_path.parent)
        print(
            f"[gp_expert_transfer N={n_experts}] expert {task_idx} trained on {len(seqs)} sequences, "
            f"num_inducing_points={meta['num_inducing_points']}, saved to {ckpt_path}"
        )

    experts = [
        {"path": str(cfg.gp_expert_checkpoint(n_experts, task_idx)), "weight": 1.0 / n_experts}
        for task_idx in range(n_experts)
        if cfg.gp_expert_checkpoint(n_experts, task_idx).exists()
    ]
    manifest_path = cfg.gp_expert_manifest(n_experts)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"experts": experts}, indent=2))
    print(f"[gp_expert_transfer N={n_experts}] wrote manifest with {len(experts)} experts to {manifest_path}")
    return manifest_path


def fit_sgpe_target_expert(
    cfg: ExperimentConfig, task_idx: int, out_dir: Path, init_path: Path, scores_path: Path,
) -> Path:
    """SGPE's extra expert: fit fresh, at eval time, on exactly the held-out
    task's own (init_path, scores_path) pair -- the same build_mutation_init()
    output every other arm's init pool comes from. Not offline/pretrained,
    since it depends on which held-out task/pool is being evaluated.
    """
    seqs = [line for line in init_path.read_text().splitlines() if line.strip()]
    scores = [float(line) for line in scores_path.read_text().splitlines() if line.strip()]
    state_dict, meta = _fit_one_expert(
        seqs, scores, cfg.gp_expert_num_inducing_points, cfg.gp_expert_lr, cfg.gp_expert_epochs,
    )
    ckpt_path = _save_expert(state_dict, meta, out_dir)
    print(
        f"[sgpe task {task_idx}] target expert trained on {len(seqs)} sequences, "
        f"num_inducing_points={meta['num_inducing_points']}, saved to {ckpt_path}"
    )
    return ckpt_path


def build_sgpe_manifest(base_manifest_path: Path, target_expert_state_dict: Path, out_path: Path) -> Path:
    """Reads base_manifest_path's N pretrained-pool entries (weight 1/N
    each), appends the target expert with weight = sum of those N weights.
    Writes out_path.
    """
    base_experts = json.loads(base_manifest_path.read_text())["experts"]
    target_weight = sum(e["weight"] for e in base_experts)
    experts = base_experts + [{"path": str(target_expert_state_dict), "weight": target_weight}]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"experts": experts}, indent=2))
    return out_path
