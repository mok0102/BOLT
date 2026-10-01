# Branin task family

This module defines a Branin family in which the canonical parameters
`a`, `b`, `c`, `r`, and `s` are shared by every task and only
`t ~ Uniform[0, 1)` changes between tasks.

```python
from synthetic_experiment import BraninTask, sample_branin_tasks

# A specific task context.
task = BraninTask(task_t=0.3)
score = task([0.0, 0.0])       # negative Branin: larger is better for BOLT
raw_value = task.raw([0.0, 0.0])  # conventional Branin: smaller is better

# A reproducible task set. Each object exposes task.task_t and task.task_id.
train_tasks = sample_branin_tasks(100, seed=0)
heldout_tasks = sample_branin_tasks(20, seed=1)
```

The input domain is `x1 in [-5, 10]`, `x2 in [0, 15]`, available from
`BRANIN_BOUNDS`. The evaluator accepts either one point of shape `(2,)` or a
batch of points with shape `(..., 2)`.

## Heldout baselines

The 50-task inline configuration enables STBO, MTBO, POGPE, SGPE, OptFormer, and LLAMBO
alongside BOLT/ORPT. All produce the same `train_x`/`train_y` CSV format with
`init_size + oracle_budget` rows when the run completes. STBO and LLAMBO write to
`heldout/STBO` and `heldout/LLAMBO`; MTBO, POGPE, and OptFormer include the milestone
in their arm directory. `train_y` is negative Branin, so larger is better. Baselines use the same heldout task
values and seeds. STBO uses uniform initial points and the deployment GP-UCB
loop. MTBO fits one shared RBF GP to top observations pooled across completed
training tasks, then adapts its predictions to target-task observations during
BO. POGPE trains one frozen exact RBF GP per completed training task, then
combines their predictive densities by a product of experts. SGPE uses the same pretrained experts plus an RBF GP fitted to the heldout initial points; the target expert has the same total weight as the pretrained pool. OptFormer trains
a separate history-conditioned Qwen checkpoint per milestone and proposes BO
points directly. LLAMBO uses the unfine-tuned base Qwen for history-conditioned
candidate generation, in-context score predictions, and expected improvement.

From the `BOLT` directory, after the trajectory chain has produced training
trajectories, run for milestone 50:

```bash
CONFIG=synthetic_experiment/configs/synthetic_50train_20heldout_inline_example_smaller.yaml
python -m synthetic_experiment train_mtbo --config "$CONFIG" --milestone 50
python -m synthetic_experiment train_pogpe --config "$CONFIG" --milestone 50
# SGPE uses the same expert pool; train_sgpe is an equivalent command.
python -m synthetic_experiment train_optformer --config "$CONFIG" --milestone 50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm STBO
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm MTBO-50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm POGPE-50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm SGPE-50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm OptFormer-50
python -m synthetic_experiment heldout_eval --config "$CONFIG" --arm LLAMBO
python -m synthetic_experiment aggregate --config "$CONFIG"
```

Repeat MTBO, POGPE, SGPE, and OptFormer training/evaluation for each milestone to plot
their scaling curves. STBO and LLAMBO need only one evaluation each. POGPE, SGPE,
and MTBO training also start automatically on their first heldout evaluation. OptFormer
requires its training command first. LLAMBO may stop before `oracle_budget`
when its input-token or retry budget is exhausted; a sidecar `.meta.json`
records the reason. The synthetic `eval2` README shows the plotting commands.
