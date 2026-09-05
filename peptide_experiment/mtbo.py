"""Offline shared-surrogate MTBO training (paper's DKT/FSBO-style baseline):
one PPGPR/GPModelDKL surrogate trained across many training tasks'
trajectories, later reused (peptide_experiment/heldout_eval.py's MTBO-<m>
branch, via optimization/peptides/lolbo/lolbo.py's pretrained_surrogate_path
hook) instead of a fresh per-task GP.

Uses the frozen, pretrained UniRef VAE (not the per-task, randomly
initialized/fine-tuned encoder every other arm's held-out run uses) to embed
pooled training-task sequences into a stable shared latent space -- a
"shared" surrogate is only meaningful against a latent space that doesn't
move per task.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gpytorch
import torch

from .config import ExperimentConfig

BOLT_ROOT = Path(__file__).resolve().parents[1]
PEPTIDES_DIR = BOLT_ROOT / "optimization" / "peptides"
for _p in (PEPTIDES_DIR, BOLT_ROOT / "fine-tuning" / "peptides"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lolbo.utils.bo_utils.ppgpr import GPModelDKL  # noqa: E402
from lolbo.utils.utils import update_surr_model  # noqa: E402
from uniref_vae.load_vae import load_vae, vae_forward  # noqa: E402
from make_train_data_csv import collect_top_sequences  # noqa: E402

PATH_TO_VAE_STATE_DICT = str(
    PEPTIDES_DIR
    / "uniref_vae"
    / "saved_models"
    / "dim128_k1_kl0001_eff256_dff256_pious-sea-2_model_state_epoch_118.pkl"
)


def train_mtbo_surrogate(cfg: ExperimentConfig, milestone: int) -> Path:
    """Pools cfg.mtbo_top_n_per_task feasible top sequences from each of
    tasks [0, milestone)'s completed trajectory, encodes them with the
    frozen pretrained VAE, and fits a GPModelDKL (paper's own DKT/FSBO
    architecture) on the pooled (z, score) pairs. Saves state_dict plus a
    sidecar recording the exact inducing-point count used -- that count is
    baked into the state dict's shapes and must match exactly when reloaded
    (see lolbo.py's pretrained_surrogate_path hook).
    """
    ckpt_dir = cfg.checkpoints_dir / f"MTBO-{milestone}"
    ckpt_path = ckpt_dir / "surrogate_state_dict.pt"
    meta_path = ckpt_dir / "surrogate_meta.json"
    if ckpt_path.exists():
        print(f"[mtbo milestone {milestone}] checkpoint already exists at {ckpt_path}, skipping")
        return ckpt_path

    from apex_oracle.refseqs import REFERENCE_SEQUENCE

    pooled_scores: list[float] = []
    pooled_seqs: list[str] = []
    for task_idx in range(milestone):
        traj_csv = cfg.trajectories_csv_dir / f"task_{task_idx:04d}.csv"
        top_seqs, _rows_skipped, _rows_infeasible = collect_top_sequences(
            traj_csv, cfg.mtbo_top_n_per_task, REFERENCE_SEQUENCE[task_idx], cfg.similarity_threshold,
        )
        for score, seq in top_seqs:
            pooled_scores.append(score)
            pooled_seqs.append(seq)

    if len(pooled_seqs) < 2:
        raise RuntimeError(
            f"[mtbo milestone {milestone}] only {len(pooled_seqs)} pooled feasible sequences across "
            f"{milestone} training tasks -- need at least 2 to fit a surrogate"
        )

    # uniref_vae/data.py's DatasetKmers resolves its vocab/data CSVs via
    # relative paths ("./uniref_vae/..."), assuming CWD == optimization/peptides
    # -- true for every existing call site (always a subprocess launched with
    # cwd=optimization/peptides/lolbo_scripts or similar), but this module
    # calls load_vae() in-process, so CWD must be set explicitly here.
    # vae_forward() has no internal batching -- passing all of pooled_seqs in
    # one call (up to mtbo_top_n_per_task * milestone, e.g. 126,000+ at
    # main-experiment scale, vs. PoC scale's much smaller actual pooled count)
    # overflows a CUDA kernel launch limit inside nn.TransformerEncoderLayer's
    # nested-tensor fastpath ("CUDA error: invalid configuration argument").
    # Chunk it -- encoding is per-sequence independent (the batch dimension
    # only affects padding-mask grouping), so this is exact, not approximate.
    VAE_ENCODE_CHUNK_SIZE = 1024

    prev_cwd = os.getcwd()
    os.chdir(PEPTIDES_DIR)
    try:
        vae, dataobj = load_vae(PATH_TO_VAE_STATE_DICT)
        z_chunks = []
        with torch.no_grad():
            for start in range(0, len(pooled_seqs), VAE_ENCODE_CHUNK_SIZE):
                chunk_z, _vae_loss = vae_forward(
                    pooled_seqs[start : start + VAE_ENCODE_CHUNK_SIZE], dataobj, vae,
                )
                z_chunks.append(chunk_z)
        train_z = torch.cat(z_chunks, dim=0)
    finally:
        os.chdir(prev_cwd)
    train_z = train_z.detach().cpu()
    train_y = torch.tensor(pooled_scores, dtype=torch.float32)

    num_inducing = min(cfg.mtbo_num_inducing_points, train_z.shape[0])
    likelihood = gpytorch.likelihoods.GaussianLikelihood().cuda()
    model = GPModelDKL(train_z[:num_inducing].cuda(), likelihood=likelihood).cuda()
    mll = gpytorch.mlls.PredictiveLogLikelihood(model.likelihood, model, num_data=train_z.size(-2))
    model = update_surr_model(model, mll, cfg.mtbo_lr, train_z, train_y, cfg.mtbo_epochs)

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    meta_path.write_text(
        json.dumps({"num_inducing_points": num_inducing, "n_train_points": len(pooled_seqs)}, indent=2)
    )
    print(
        f"[mtbo milestone {milestone}] trained on {len(pooled_seqs)} pooled sequences, "
        f"num_inducing_points={num_inducing}, saved to {ckpt_path}"
    )
    return ckpt_path
