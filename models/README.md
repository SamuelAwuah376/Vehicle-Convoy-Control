# Trained PPO models

Each folder works like a `runs_rl/<run>/` folder: `nohesi.rl.evaluate` and `run_sim.py` (`POLICY = 'ppo'`, `PPO_RUN`) accept it.
All models use environment `e4` (273-value observation) on scenario v4 (20 cars, blind-merging traffic), linear MPC + kinematic sim.

| Folder | Trained | Use |
|---|---|---|
| `ppo_v4_e4_5m` | From scratch, 5M decisions, score v4. `best` = 5.0M checkpoint, `final` = end. | Safest: 0 crashes in 20 seeds (best). |
| `ppo_v6_e4_5m` | Fine-tuned from `ppo_v4_e4_5m/final`, 5M decisions, score v6, LR 1e-4. `best` = 2M checkpoint, `final` = end. | Most drafting and highest score (final), 15 % crashes. |
| `ppo_v7_e4_6m` | Fine-tuned from `ppo_v6_e4_5m/final`, 6M decisions, score v7 (team play), LR 1e-4, 200k value warm-up. `best` = 4.5M checkpoint, `final` = end. | Current team-play model (final): most draft-train play; starting point for v8. |

Files per folder: `{best,final}_model.zip` + `{best,final}_vecnormalize.pkl` (observation scaling; needed to run the model), `config.json`
(training and env config), `eval_history.csv` (10-seed evaluations during training), and the 20-seed evaluations
(`eval_*_default.json` on the run's own score, `eval_compare_v6.json` with all models on v6).
