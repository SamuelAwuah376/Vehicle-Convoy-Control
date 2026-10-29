# Trained PPO models

Each folder works like a `runs_rl/<run>/` folder: `nohesi.rl.evaluate` and `run_sim.py` (`POLICY = 'ppo'`, `PPO_RUN`) accept it.
The `e4` models (273-value observation) use the linear MPC + kinematic sim. They were trained on scenario v4 (20 cars, blind-merging traffic), except `ppo_v8_e4_10m` (scenario v5: BMW M3s, 45-60 mph traffic without walls). `ppo_v8_e5_dyn_b2c_3m` uses environment `e5` (306 values, 51 actions with the overtakes) on the dynamic car with the nonlinear MPC, scenario v7.

| Folder | Trained | Use |
|---|---|---|
| `ppo_v4_e4_5m` | From scratch, 5M decisions, score v4. `best` = 5.0M checkpoint, `final` = end. | Safest: 0 crashes in 20 seeds (best). |
| `ppo_v6_e4_5m` | Fine-tuned from `ppo_v4_e4_5m/final`, 5M decisions, score v6, LR 1e-4. `best` = 2M checkpoint, `final` = end. | Most drafting and highest score (final), 15 % crashes. |
| `ppo_v7_e4_6m` | Fine-tuned from `ppo_v6_e4_5m/final`, 6M decisions, score v7 (team play), LR 1e-4, 200k value warm-up. `best` = 4.5M checkpoint, `final` = end. | Team-play model on v7 (final). |
| `ppo_v8_e4_10m` | From scratch, 10M decisions, scenario v5, score v8, safety layer m3.7, LR 3e-4. `best` = 9.5M checkpoint, `final` = end. | Kinematic-car model (final): 83.4k on 70 unseen seeds against 46.4k for `ppo_v7_e4_6m`, 16 % crashes. |
| `ppo_v8_e5_dyn_b2c_3m` | Dynamic car, batches 1 → 2 → 2b → 2c (README, milestone 5): 3M decisions from batch 2b's 2M checkpoint, scenario v7, score v8, LR 3e-4. `best` = 3M checkpoint, `final` = end. | Dynamic-car model (final): 56.5k and 0 crashes on 50 seeds (`eval_50seeds_1000-1049.json`); batch 3 starts from it. |

Files per folder: `{best,final}_model.zip` + `{best,final}_vecnormalize.pkl` (observation scaling; needed to run the model), `config.json`
(training and env config), `eval_history.csv` (10-seed evaluations during training), and the 20-seed evaluations
(`eval_*_default.json` on the run's own score, `eval_compare_v6.json` with all models on v6). `ppo_v8_e4_10m/eval_compare_v8.json`
compares its best and final models, `ppo_v7_e4_6m/final` and the overtaker on 70 unseen seeds (scenario v5, score v8).
