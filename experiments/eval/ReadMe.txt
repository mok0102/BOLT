experiments/eval/
├── incumbent_vs_pool_size.py         # (2.1) compute: proposal-level incumbent-at-k, no BO
├── plot_incumbent_vs_pool_size.py    # (2.1) plot
├── fixed_target_rejection_bo.py      # (2.2) compute: resample-to-target pool, then real BO
├── plot_fixed_target_rejection_bo.py # (2.2) plot
├── fixed_budget_rejection_bo.py      # (2.3) thin driver reusing constraint_violation/'s
│                                      #       run_rejection_sampled_bo.py + summarize_rejection_sampled_bo.py
│                                      #       as-is; no new plot file (reuses plot_rejection_sampled_bo.py)
└── results/                          # output CSVs, one subdir per experiment_id (gitignored,
                                       # already covered by the repo's existing experiments/*/results/ glob)


