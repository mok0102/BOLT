"""Initialization/final BO table for every supplied synthetic arm."""
from pathlib import Path
import pandas as pd
from data import task_curves


def generate(cfg, milestone: int, run_dirs: dict[str, Path], out_dir: Path) -> Path | None:
    rows = []
    for arm, run_dir in run_dirs.items():
        curves = task_curves(run_dir, arm, milestone, cfg.num_heldout_tasks, cfg.init_size, cfg.oracle_budget)
        if curves.empty:
            continue
        initial_scores = curves.loc[0]
        final_scores = curves.loc[cfg.oracle_budget] if cfg.oracle_budget in curves.index else pd.Series(dtype=float)
        rows.append({'arm': arm, 'milestone': milestone, 'num_tasks': curves.shape[1],
                     'initialization_best_score': initial_scores.mean(),
                     'initialization_best_score_std': initial_scores.std(ddof=1),
                     'final_bo_best_score': final_scores.mean(),
                     'final_bo_best_score_std': final_scores.std(ddof=1)})
    if not rows:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / 'ablation_synthetic.csv', index=False)
    tex = frame[['arm', 'initialization_best_score', 'final_bo_best_score']].to_latex(
        index=False, header=['Method', 'Initialization $\\uparrow$', 'Final BO $\\uparrow$'], float_format='%.2f')
    path = out_dir / 'ablation_synthetic.tex'
    path.write_text(tex)
    return path
