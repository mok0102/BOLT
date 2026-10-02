"""Branin task manifest: generation, persistence, and loading.

Replaces the old regenerate-from-seed-on-every-``load_config()`` scalar-t
sampler with a frozen, content-hashed, index-keyed manifest of tasks, each
with its own random affine transform (shift/rotation/scale, see
``branin.py``) and numerically verified optimum (``global_optimum.py``).

Tasks are independent random draws (no sorted/swept sequence), rejection-
sampled against ``global_optimum.is_well_posed`` so every accepted task's
optimum is confirmed reachable inside the fixed search box. Once any model
has trained against a manifest, freeze it -- ``build_manifest`` is the only
way to produce one and ``save_manifest`` refuses to overwrite an existing
file; a range change makes a new, differently-named manifest rather than
silently mutating one already in use (this is also what rules out any
appearance of post-hoc-tuning a split to produce a favorable split).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .branin import BraninTask, BraninTaskTransform, canonical_global_minimizers
from .global_optimum import VerifiedOptimum, edge_margin, is_well_posed, verify_optimum
from .prompts import task_descriptor

SCHEMA = "branin_affine_task_manifest/v1"

# Sampling ranges -- see the design writeup for the geometric reasoning
# (well-spacing vs. GP lengthscale, box-fraction coverage, etc.), not
# repeated here as inline comments to avoid drifting out of sync with it.
SHIFT_RANGE: tuple[float, float] = (-3.0, 3.0)
ROTATION_RANGE_DEG: tuple[float, float] = (0.0, 360.0)
SCALE_LOG_RANGE: tuple[float, float] = (0.8, 1.25)
MIN_EDGE_MARGIN = 1.0
AGREEMENT_ATOL = 1e-6

_SCALE_LOG_SPAN = np.log(SCALE_LOG_RANGE[1] / SCALE_LOG_RANGE[0])


def task_name(index: int) -> str:
    return f"task_{index:04d}"


def task_context_embedding(transform: BraninTaskTransform) -> np.ndarray:
    """Leakage-free (normalized by the sampler's own ranges, not by observed
    data), rotation-aware embedding used for BOTH the k-NN context-
    regression baseline and MTBO's GP context feature. Each of the 5
    components contributes ~1/6 of the average squared distance between two
    random tasks -- matching the variance the OLD scalar `t ~ U[0,1]`
    feature itself contributed, so neither consumer's existing
    hyperparameters (k, GP lengthscale) need to change meaning or be
    retuned. Euclidean distance on this embedding is then a true chord
    distance for rotation (periodic, handles the 359.9/0.1-degree
    wraparound correctly -- raw-radian subtraction would not)."""
    shift_max = max(abs(SHIFT_RANGE[0]), abs(SHIFT_RANGE[1]))
    theta = transform.rotation_rad
    return np.array([
        transform.shift[0] / (2.0 * shift_max),
        transform.shift[1] / (2.0 * shift_max),
        np.cos(theta) / np.sqrt(12.0),
        np.sin(theta) / np.sqrt(12.0),
        np.log(transform.scale) / _SCALE_LOG_SPAN,
    ])


def task_context_distance(a: BraninTaskTransform, b: BraninTaskTransform) -> float:
    return float(np.linalg.norm(task_context_embedding(a) - task_context_embedding(b)))


@dataclass(frozen=True)
class TaskRecord:
    split: str  # "train" or "heldout"
    index: int  # local to its split -- NOT globally unique alone, see module docstring
    transform: BraninTaskTransform
    verified: VerifiedOptimum

    @property
    def name(self) -> str:
        return task_name(self.index)

    @property
    def descriptor(self) -> str:
        return task_descriptor(self.transform)

    def oracle(self, *, maximize: bool = True) -> BraninTask:
        return BraninTask(transform=self.transform, maximize=maximize)


@dataclass(frozen=True)
class TaskManifest:
    schema: str
    seed: int
    ranges: dict
    stats: dict
    train: tuple[TaskRecord, ...]
    heldout: tuple[TaskRecord, ...]
    token: str

    def tasks(self, split: str) -> tuple[TaskRecord, ...]:
        if split not in ("train", "heldout"):
            raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")
        return self.train if split == "train" else self.heldout

    def task(self, split: str, index: int) -> TaskRecord:
        return self.tasks(split)[index]


def _transform_to_dict(t: BraninTaskTransform) -> dict:
    return {"shift": list(t.shift), "rotation_deg": t.rotation_deg, "scale": t.scale}


def _transform_from_dict(d: dict) -> BraninTaskTransform:
    return BraninTaskTransform(shift=tuple(d["shift"]), rotation_deg=d["rotation_deg"], scale=d["scale"])


def _verified_to_dict(v: VerifiedOptimum) -> dict:
    return {
        "f_star": v.f_star, "x_star": list(v.x_star),
        "x_star_all": [list(p) for p in v.x_star_all],
        "edge_margin": v.edge_margin, "boundary_min": v.boundary_min,
        "matches_canonical": v.matches_canonical,
        "n_restarts_agreeing": v.n_restarts_agreeing, "n_restarts_total": v.n_restarts_total,
    }


def _verified_from_dict(d: dict) -> VerifiedOptimum:
    return VerifiedOptimum(
        f_star=d["f_star"], x_star=tuple(d["x_star"]),
        x_star_all=tuple(tuple(p) for p in d["x_star_all"]),
        edge_margin=d["edge_margin"], boundary_min=d["boundary_min"],
        matches_canonical=d["matches_canonical"],
        n_restarts_agreeing=d["n_restarts_agreeing"], n_restarts_total=d["n_restarts_total"],
    )


def _record_to_dict(r: TaskRecord) -> dict:
    return {"index": r.index, "name": r.name, "transform": _transform_to_dict(r.transform),
            "verified": _verified_to_dict(r.verified)}


def _record_from_dict(split: str, d: dict) -> TaskRecord:
    return TaskRecord(split=split, index=d["index"], transform=_transform_from_dict(d["transform"]),
                      verified=_verified_from_dict(d["verified"]))


def _payload_without_token(schema: str, seed: int, ranges: dict, stats: dict,
                           train: tuple[TaskRecord, ...], heldout: tuple[TaskRecord, ...]) -> dict:
    return {
        "schema": schema, "seed": seed, "ranges": ranges, "stats": stats,
        "train": [_record_to_dict(r) for r in train],
        "heldout": [_record_to_dict(r) for r in heldout],
    }


def _compute_token(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def build_manifest(
    num_train: int = 50,
    num_heldout: int = 20,
    seed: int = 0,
    *,
    shift_range: tuple[float, float] = SHIFT_RANGE,
    rotation_range_deg: tuple[float, float] = ROTATION_RANGE_DEG,
    scale_log_range: tuple[float, float] = SCALE_LOG_RANGE,
    min_edge_margin: float = MIN_EDGE_MARGIN,
    agreement_atol: float = AGREEMENT_ATOL,
    max_attempts_per_task: int = 200,
) -> TaskManifest:
    """Draw `num_train` + `num_heldout` independent random tasks (no sorted
    sweep), each rejection-sampled against `global_optimum.is_well_posed`.
    Train and held-out splits get independent random streams (via
    `np.random.SeedSequence(seed).spawn(2)`) so neither leaks into the
    other's draws and so a shorter `num_*_tasks` prefix stays stable."""
    ranges = {
        "shift_range": list(shift_range), "rotation_range_deg": list(rotation_range_deg),
        "scale_log_range": list(scale_log_range), "min_edge_margin": min_edge_margin,
        "agreement_atol": agreement_atol,
    }
    stats = {"attempts": 0, "rejected_prefilter": 0, "rejected_w1": 0, "rejected_w2": 0, "accepted": 0}

    train_seed_seq, heldout_seed_seq = np.random.SeedSequence(seed).spawn(2)

    def _draw_split(split: str, n: int, seed_seq: np.random.SeedSequence) -> list[TaskRecord]:
        rng = np.random.default_rng(seed_seq)
        records: list[TaskRecord] = []
        for index in range(n):
            accepted: TaskRecord | None = None
            for _attempt in range(max_attempts_per_task):
                stats["attempts"] += 1
                shift = (
                    round(float(rng.uniform(*shift_range)), 2),
                    round(float(rng.uniform(*shift_range)), 2),
                )
                rotation_deg = round(float(rng.uniform(*rotation_range_deg)), 1) % 360.0
                log_scale = rng.uniform(np.log(scale_log_range[0]), np.log(scale_log_range[1]))
                scale = round(float(np.exp(log_scale)), 3)
                transform = BraninTaskTransform(shift=shift, rotation_deg=rotation_deg, scale=scale)

                # Analytic pre-filter (can only reject, never accept): would
                # any canonical minimizer, mapped through this transform,
                # land with margin >= min_edge_margin? Cheap compared to the
                # full numeric search below.
                images = [transform.from_canonical(np.array(m)) for m in canonical_global_minimizers()]
                best_margin = max(edge_margin(tuple(float(v) for v in pt)) for pt in images)
                if best_margin < min_edge_margin:
                    stats["rejected_prefilter"] += 1
                    continue

                task = BraninTask(transform=transform, maximize=False)
                verify_seed = int(rng.integers(0, 2**31 - 1))
                verified = verify_optimum(task, seed=verify_seed, agreement_atol=agreement_atol)
                if not verified.matches_canonical:
                    stats["rejected_w1"] += 1
                    continue
                if verified.edge_margin < min_edge_margin:
                    stats["rejected_w2"] += 1
                    continue

                accepted = TaskRecord(split=split, index=index, transform=transform, verified=verified)
                break
            if accepted is None:
                raise RuntimeError(
                    f"{split} task {index}: no well-posed draw found in {max_attempts_per_task} attempts "
                    f"-- the sampling ranges may need revisiting (see stats so far: {stats})"
                )
            stats["accepted"] += 1
            records.append(accepted)
        return records

    train = tuple(_draw_split("train", num_train, train_seed_seq))
    heldout = tuple(_draw_split("heldout", num_heldout, heldout_seed_seq))
    stats["acceptance_rate"] = stats["accepted"] / stats["attempts"] if stats["attempts"] else 0.0

    payload = _payload_without_token(SCHEMA, seed, ranges, stats, train, heldout)
    token = _compute_token(payload)
    return TaskManifest(schema=SCHEMA, seed=seed, ranges=ranges, stats=stats, train=train, heldout=heldout, token=token)


def save_manifest(manifest: TaskManifest, path: str | Path, *, overwrite: bool = False) -> Path:
    path = Path(path)
    if path.suffix == ".manifest":
        raise ValueError(f"manifest must not use the '.manifest' extension (gitignored): {path}")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists -- manifests are frozen once built; use a new filename "
            "(e.g. bump the version suffix) instead of overwriting one that may already be in use"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _payload_without_token(manifest.schema, manifest.seed, manifest.ranges, manifest.stats,
                                      manifest.train, manifest.heldout)
    payload["token"] = manifest.token
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return path


def load_manifest(path: str | Path) -> TaskManifest:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"task manifest not found: {path} -- build one with "
            "`python -m synthetic_experiment.task_splits build`, this module never builds one implicitly"
        )
    raw = json.loads(path.read_text())
    stored_token = raw["token"]
    payload = {k: v for k, v in raw.items() if k != "token"}
    recomputed_token = _compute_token(payload)
    if recomputed_token != stored_token:
        raise ValueError(
            f"{path}: content hash mismatch (stored {stored_token}, recomputed {recomputed_token}) -- "
            "file was edited after being built, or is corrupt"
        )
    if raw["schema"] != SCHEMA:
        raise ValueError(f"{path}: unexpected schema {raw['schema']!r}, expected {SCHEMA!r}")

    train = tuple(_record_from_dict("train", d) for d in raw["train"])
    heldout = tuple(_record_from_dict("heldout", d) for d in raw["heldout"])
    for split_name, records in (("train", train), ("heldout", heldout)):
        if [r.index for r in records] != list(range(len(records))):
            raise ValueError(f"{path}: {split_name} indices are not a contiguous 0..N-1 range")
        for r in records:
            if not is_well_posed(r.verified, agreement_atol=raw["ranges"]["agreement_atol"],
                                 min_edge_margin=raw["ranges"]["min_edge_margin"]):
                raise ValueError(f"{path}: {split_name} task {r.index} fails its own recorded well-posedness gate")

    return TaskManifest(schema=raw["schema"], seed=raw["seed"], ranges=raw["ranges"], stats=raw["stats"],
                        train=train, heldout=heldout, token=stored_token)


def _cli_build() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build and freeze a Branin task manifest.")
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--out", required=True, help="output path, must end in .json")
    parser.add_argument("--num-train", type=int, default=50)
    parser.add_argument("--num-heldout", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = build_manifest(num_train=args.num_train, num_heldout=args.num_heldout, seed=args.seed)
    path = save_manifest(manifest, args.out)
    print(f"wrote {path} (token={manifest.token})")
    print(f"acceptance rate: {manifest.stats['acceptance_rate']:.1%} ({manifest.stats['accepted']}/{manifest.stats['attempts']} attempts)")
    if manifest.stats["acceptance_rate"] < 0.80:
        print("WARNING: acceptance rate below 80%% -- consider revisiting the sampling ranges")


if __name__ == "__main__":
    _cli_build()
