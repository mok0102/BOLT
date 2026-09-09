"""Serial real-DB acquisition; shared seeds and already-scored initial pools."""
import ast
import hashlib
import json
from dataclasses import replace
import pandas as pd
from ..steps import run_bo, _write_init_csv

def completed_utility(df):
    values = df.loc[df.censoring == 0, "train_y"]
    values = values[values.notna() & (values < 0)]
    if values.empty:
        raise RuntimeError("MI pool has no completed plan for utility")
    return float(values.max())

def zero_step_utility(background, candidate):
    return max(c.y for c in [*background, candidate])

def run_candidates_one_step(cfg, task_idx, backgrounds, candidates, work_dir_root, seeds, worker_pool=None):
    if worker_pool is not None:
        raise ValueError("Query-plan MI uses serial DB evaluation")
    values = []
    for candidate in candidates:
        results = []
        for bg, seed in zip(backgrounds, seeds):
            pool = [*bg, candidate]
            context = {"pool": [(c.seq,c.y) for c in pool], "seed": seed, "task": task_idx,
                       "config": repr(cfg), "vae_initialization": "random", "version": 3}
            key = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:20]
            work = work_dir_root / key
            work.mkdir(parents=True, exist_ok=True)
            init = work / "init.csv"
            _write_init_csv(init, [ast.literal_eval(c.seq) for c in pool], [c.y for c in pool], [0]*len(pool))
            # The explicit scored init CSV bypasses initial_plan_source entirely.
            # Use a fresh random VAE for MI, with the matched seed shared across
            # candidates. Keep deployment/trajectory checkpoint settings intact.
            run_cfg = replace(cfg, init_size=len(pool), oracle_budget=cfg.bsz,
                              use_pretrained_vae=False, vae_statedict_path=None,
                              initial_plan_source="bao")
            path = run_bo(run_cfg, task_idx, work, run_id=f"mi-{key}", init_csv_path=init,
                          seed=seed, max_bo_steps=1)
            utility = completed_utility(pd.read_csv(path))
            (work / "evaluation.json").write_text(json.dumps({**context,"utility":utility}, indent=2))
            results.append(utility)
        values.append(results)
    return values
