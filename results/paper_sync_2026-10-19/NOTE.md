# Paper-sync simulations, 19 Oct 2026 (commit dea2dda, planner m3.3)

Run by Claude for the Overleaf paper. Linux sandbox, Python 3.10, numpy 2.2.6, osqp 1.1.3, scipy 1.15.3
(scipy 1.18.1 from pyproject.toml is not available for Python 3.10). Check: overtaker v4/score v4/seed 1000
reproduces the cached baseline (7290.428337) to 4e-9.

- `metrics/v1_{overtaker,lane_keeping}_{v0,v1,v2}_<seed>_kinematic.json`: scenario v1, seeds 0-9 (paper Table I,
  planner behaviour). `scripts/sweep.py`.
- `metrics/{v1,v4}_overtaker_v3_<seed>_dynamic.json`: linear MPC on the dynamic plant, seeds 0-4 (all crash).
  The nonlinear MPC was not rerun (acados is not available in the sandbox).
- `metrics/ppo_<model>_<best|final>_<score>_<seed>.json`: PPO models under score v7 (seeds 1000-1019) and v6 (seed
  1050, and seed 1000 as a check). torch could not be installed in the sandbox, so `scripts/nppolicy.py` re-implements
  the saved MaskablePPO actor (2x256 tanh, masked argmax) and VecNormalize scaling in numpy; it reproduces the saved
  eval_compare_v6.json scores for seed 1000 to 4e-8 (v6 final and 5M best). `scripts/ppoeval.py`.
- `scripts/clr.py`: pass clearances and the moments of minimum traffic clearance (v1, seeds 0-9).
- `figs/`: snapshot.pdf (seed 0, t = 36.0 s, plans) and episode_seed0.pdf, uploaded to Overleaf figs/
  (`scripts/mkfigs.py`).
