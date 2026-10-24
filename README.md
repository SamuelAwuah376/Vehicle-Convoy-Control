# No Hesi Team Driving

A **No Hesi-style team game** in Python (`nohesi/`): three cars weave through highway traffic, scoring points for close passes, staying near teammates and lane-change maneuvers. Each car is driven by a model predictive controller (MPC) that turns high-level choices ("change left, speed up") into safe, smooth steering and throttle. The planned next stage is a reinforcement-learning (RL) policy choosing those high-level actions.

Everything uses SI units (metres, seconds, m/s, radians).

## Quick start

Requires Python 3.10+ (developed on 3.12/3.13, Windows).

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest -q              # run all tests
.venv\Scripts\python run_sim.py                # run one No Hesi simulation
```

The nonlinear MPC also needs acados built locally (see the `nmpc` extra in `pyproject.toml`): `pip install -e ".[test,nmpc]"` plus `pip install -e C:/acados/interfaces/acados_template`.

On macOS/Linux use `.venv/bin/python`. If `python` opens the Microsoft Store, use `py -3` to create the environment.

`run_sim.py` is the easiest way to run a simulation. Open it, change the settings at the top, and run it:

| Setting | Options |
|---|---|
| `SCENARIO` | `'v0'` 60 s, 8 traffic cars · `'v1'` 120 s, 10 cars · `'v2'` 120 s, 20 cars · `'v3'`/`'v4'` = v1/v2 with lane-changing traffic |
| `POLICY` | `'overtaker'` (lane changes + MPC) · `'lane_keeping'` (stays in its lane, no MPC) |
| `SCORE` | `'v0'` … `'v4'` (see [Scoring](#scoring)); `'v4'` is the latest (default) |
| `SEED` | any integer: a different traffic layout |
| `TRAFFIC` | override the number of traffic cars (`None` keeps the scenario's) |
| `PLANT` | `'kinematic'` (default) · `'dynamic'` (dynamic bicycle with tyre slip) |
| `MPC` | `'linear'` (OSQP, kinematic model) · `'nonlinear'` (acados, dynamic model; use with `PLANT = 'dynamic'`) |
| `ANIMATE` | `True` writes a GIF (takes a few extra seconds) |

Each run is saved to its own numbered folder, `results/run_NNN_<scenario>_<policy>_score<score>_seed<seed>/`, so runs never overwrite each other. `results/runs.csv` lists every run with its settings, versions, team score and collisions. The same runs are available from the command line, e.g. `.venv\Scripts\python -m nohesi --scenario v1 --policy overtaker --seed 0 --animate`.

## Milestones

| Milestone | What it added |
|---|---|
| No Hesi M1 | Straight 3-lane road, seeded traffic, collision detection, versioned scoring, lane-keeping policy. |
| No Hesi M2, part 1 | Lane changes executed by a per-car MPC. |
| No Hesi M2, part 2 | Lane-changing traffic, plus robustness fixes for crowded situations. |
| No Hesi M2, part 3 | CG reference, dynamic bicycle simulator and a nonlinear MPC (acados). |
| No Hesi M3 | PPO policy (MaskablePPO), score v4 with drafting, and stronger safety fallbacks. |
| No Hesi M3, part 2 | Trained PPO models, drafting-focused fine-tuning (scores v5–v6), faster simulator. |
| No Hesi M4 | Safety layer m3.7, team-play scores v7–v8, the v8 world (BMW M3s, faster traffic without walls), and a new 10M-decision model trained on all of it. |
| **No Hesi M4, part 2** | The stack on the dynamic car: realistic M3 braking physics, a grip budget, an MPC with grip hard and contact soft, a 100 Hz stability layer, an MPC emergency mode and plan verification (planner m4.2). |

**Milestone 1** builds the game world:
- **Road and traffic:** a straight 3-lane road, with seeded traffic that follows the car ahead and is recycled ahead of the team.
- **Collisions:** detected with footprints; touching traffic, a teammate or the road edge ends the episode.
- **Scoring:** versioned, and measured only from what actually happened.
- **Outputs:** an episode runner that saves metrics, histories, plots and GIFs.
- **Lane-keeping policy:** stays in lane, as the first baseline.

**Milestone 2, part 1** gives every car a planner that executes high-level choices safely:
- **Actions:** 45 choices per car: lane (two right, one right, keep, one left, two left) × speed (slower, hold, faster) × lean in the lane (right, centre, left).
- **Intention layer:** turns a lane choice into a locked target gap. It rejects unsafe requests, such as a gap that's too short, a faster car closing from behind, or a blocked middle lane during a two-lane move.
- **Per-car MPC:** a quadratic program solved with OSQP in about 2 ms that plans 3 s ahead, avoiding traffic and teammates within hard 0.25 m margins.
- **Safety net:** fallbacks for failed solves, plus a team-wide check of every plan after each planning cycle.
- **Scoring up to v3:** a team combo multiplier, teammate proximity and lane-change maneuvers.
- **Scripted overtaker:** a hand-written policy that exercises all of the above.
- **Debug outputs:** GIFs that draw each car's current plan.

**Results:** across seeds 0–4 on scenarios v1 and v2 there are 0 collisions. Under score v3 the overtaker averages about 17,000 points on v1 and 14,000 on v2, while the lane-keeping baseline scores almost nothing (it never passes closely). Reference run for the baseline: `results/run_006_v1_lane_keeping_scorev3_seed0/`.

**Milestone 2, part 2** makes traffic more realistic and fixes cases where cars got stuck:
- **Lane-changing traffic** (scenarios v3, v4): traffic changes lanes with the MOBIL model, moving over smoothly in 3 s, with an 8 s cooldown between a car's changes.
- **Predicting it:** the team predicts lane-changing traffic (`prediction.py`). A car that is moving over is expected to reach its target lane, and counts as a gap boundary there.
- **No more stuck lane changes:** a committed change that makes no sideways progress for 2 s, or takes longer than 6 s, is cancelled. A car can't re-request the same lane for 3 s afterwards.
- **Teammates make room:** a teammate beside a car changing into its lane drops back or pulls ahead, so the slot opens.
- **Fewer repeated re-locks:** a locked gap is only re-checked when it would actually be wrong at the end of the move.

**Results (milestone 2, part 2):** seeds 0–4 have 0 collisions on v1 and on v4 (20 cars, lane-changing traffic). The overtaker averages about 16,800 points on v1 and 21,200 on v4.

**Milestone 2, part 3** adds realistic vehicle physics and a controller that can handle it:
- **CG reference:** every vehicle is now tracked at its centre of gravity. The car is 60/40 front-biased by default (`Vehicle.cg_to_rear_axle = 1.62 m`).
- **Dynamic bicycle simulator** (`Vehicle.model = 'dynamic'`, `PLANT` in `run_sim.py`): the cars now slide sideways and lag in turning, driven by Fiala tyres with a friction ellipse that shares grip between braking and cornering. Below 3–6 m/s it blends into the kinematic model. The linear MPC can't drive it: it overshoots and crashes within seconds.
- **Nonlinear MPC** (`planning/nonlinear.py`, `Planner.mpc = 'nonlinear'`, `MPC` in `run_sim.py`): uses the same dynamic bicycle and tyres, solved with acados (one real-time SQP iteration per 0.1 s replan, Gauss-Newton Hessian). It reuses the linear MPC's collision constraints. A **soft lateral grip limit** keeps each axle's slip below 80 % of the saturation slip of its remaining grip; in force terms that is about 99 % of the available lateral grip. acados setup: see `pyproject.toml` (`nmpc` extra); generated code goes to `C:/dev/acados_build`.
- **Comparisons:** `run_sim.py` covers all three cases. `viz.animate_compare` and `viz.plot_compare` show episodes side by side, and results folders are numbered (`run_NNN_...`, indexed in `results/runs.csv`).

**Results (milestone 2, part 3):** scripted overtaker, seeds 0–4 on v1 and v4, score v3. Re-run with the current code (planner m3.3, commit `ec3f1d6`) by `experiments/nmpc_sweep.py`; the tables are in `results/run_022_nmpc_sweep_v1_v4_seeds0-4_scorev3/`.
- **Linear MPC + kinematic sim:** 0 collisions, about 16,800 points on both scenarios.
- **Linear MPC + dynamic sim:** crashes in all 10 runs, within 7–45 s.
- **Nonlinear MPC + dynamic sim:** 0 collisions, about 13,500 points on v1 and 10,100 on v4 (blind-merging traffic). Lane changes are slower because of solver-stabilising penalties. Planned motion matches the car to about 0.2–0.3 mm per 0.1 s, against about 5 mm for the linear MPC on its own kinematic sim. Solves take about 7 ms on average (12 ms p95), against 2.4 ms for the linear MPC, timed one episode at a time.

**Milestone 3** replaces the scripted overtaker with a reinforcement-learning (RL) policy:
- **Score v4 (default):** a close pass is worth more when teammates are near (up to 20 m), and **drafting** replaces proximity. Drafting means driving directly behind a teammate in the same lane, within 1 s, with no traffic in between.
- **RL environment** (`nohesi/rl/env.py`): one step is one team decision (0.5 s), where each car picks one of its 45 actions. Lane changes that the gap rule would reject right now are masked out. The observation includes, for each nearby traffic car, whether it could merge left or right right now.
- **Traffic merges blind:** a traffic car may cut in front of a team car closing from behind. It only holds back when a car in the target lane is alongside it or within half a car length.
- **Faster MPC:** the QP structure is reused between solves (`Planner.qp_reuse`), which makes solves about 30 % faster.
- **Stronger fallbacks:**
  - Emergency braking now uses the car's full 6 m/s².
  - The emergency escape tries full braking and holding speed in every nearby lane, and picks the option that stays clear longest.
  - An abandoned lane change keeps the clearest nearby lane instead of always returning to the start lane.
  - The gap rule counts a teammate leaning over a lane line as being in both lanes.

**Results (first 3M-decision PPO run, before the fallback changes):** on seeds 1000–1019 PPO scored about 47,800 points, against 18,200 for the overtaker. It crashed in 5 of 20 episodes. Replaying the same policy with the new fallbacks gave 2 crashes in 20 against traffic that looked before merging. With traffic merging blind it was back to 5. The policy was then retrained (part 2).

**Milestone 3, part 2** trains the policy that is saved in `models/`, then fine-tunes it toward team play:
- **5M-decision run** (`models/ppo_v4_e4_5m`): trained from scratch on score v4, with blind-merging traffic and the new observation (environment `e4`).
- **Fine-tuning** (`--init-from` / `INIT_FROM` in `train_rl.py`): a new run starts from another run's policy weights and observation scaling. The optimiser, learning-rate schedule and reward scaling start fresh.
- **Score v5:** drafting 5× more rewarding (25 points per second).
- **Score v6:** v5 plus **draft passes**. A close pass by a car that is drafting, or left the draft within the last 1 s (a slingshot), is worth 3×. The v6 model (`models/ppo_v6_e4_5m`) is the 5M model fine-tuned on v6 for 5M decisions at a learning rate of 1e-4.
- **Faster simulator:** about 15 % more decisions per second, with the same episodes. Traffic car-following and polygon distance checks are computed for all cars at once.
- **GPU learner:** the PPO network trains on the GPU; the simulator workers stay on the CPU.

**Results (milestone 3, part 2):** seeds 1000–1019, every model scored on v6:

| | Overtaker | 5M model (best) | v6 model (final) |
|---|---|---|---|
| Team score | 24,596 | 62,498 | **70,683** |
| Crash rate | 0.00 | **0.00** | 0.15 |
| Close passes | 40.0 | 55.5 | 54.0 |
| Draft passes | 4.3 | 4.6 | **7.9** |
| Draft time (s) | 21.7 | 13.2 | **23.0** |

- **The 5M model** beats the overtaker on all 20 seeds with 0 crashes. Under v4 it scores 53,800, against 47,800 and 25 % crashes for the first run.
- **The v6 fine-tune** drafts about 75 % more and scores 13 % more, but crashes on 3 of 20 seeds.
- **Seed 1050, unseen by all policies:** the v6 model scored 98,400, the 5M model 54,100, and the overtaker crashed at 29.5 s (`results/run_021_...`). Both PPO policies spread out to a median 105–110 m between the first and last car. The team close-pass bonus therefore averaged only about 1.06×, which is what v7 targets.

**Milestone 4** trains a new model on a new safety layer and on scores that reward the team moving together:
- **Safety layer m3.7** (after replaying all 11 crashes of the v7 model, 10 of which were checker gaps):
  - **Slot margin:** while a car changes lanes, its plan is compared with the gap's front and rear cars and with any vehicle merging into a lane it enters, each assumed to keep its speed. Wherever they would overlap sideways during the lane change, the time to contact is the bumper gap divided by the closing speed. If it falls below 0.5 s, the car backs out of the change.
  - **Smarter emergency escape:** it picks the option with the gentlest predicted impact (smallest speed difference), and can also speed up, not only brake or hold.
  - **Denser checks:** plans are checked 5 times per 0.1 s, so a fast pass between plan points is caught.
  - **Limits from the car and the scenario:** emergency braking uses the vehicle's full braking, and the planner's speed cap comes from the scenario.
  - **Result:** with the v7 model unchanged (it was trained without these checks), crashes on 80 seeds fell from 20 to 10, with the mean score 8 % lower (42.8k against 46.6k).
- **Score v7 (team play):** rewards the whole team together.
  - A three-car **draft train** (any lanes, within 0.5 s of each other) pays 25 pts/s per car and 3× passes; a two-car draft pays 5 pts/s and 1.5×.
  - The team close-pass bonus needs all three cars within 40 m, and a capped quadratic **team spread penalty** applies beyond 50 m.
  - A crash wipes 50 % of the team's banked points, and a car can re-score the same traffic car only after 10 s.
  - `models/ppo_v7_e4_6m` is the v6 model fine-tuned on v7 for 6M decisions (LR 1e-4, 200k-decision value-network warm-up).
- **v8 world (scenario v5):**
  - **Cars are BMW M3 Competitions** (`config.bmw_m3()`): 4.79 m long, 1,780 kg, 52.5/47.5 weight, launch-limited at 8 m/s² and power-limited above 26 m/s (375 kW), brakes at 11 m/s², corners at about 1 g.
  - **Traffic drives at 45–60 mph** and never forms a wall (three cars, one per lane, within two car lengths). Spawns and lane changes avoid walls, and the rear car of a forming wall drops back. On 5 seeds, walls are present 0.7 % of the time, against 33 % in v4.
  - **The team may command up to 134 mph** (`Scenario.team_speed_max`), starts at 70 mph about 0.6 s apart, and the scripted overtaker cruises at 90 mph.
- **Score v8 (fast team play):** v7 with draft-train points and train passes only above 80 mph, +10 pts/s per car above 120 mph (times the combo), the spread penalty doubled (capped at 400 pts/s per pair), and the team close-pass bonus within 50 m.
- **New model** (`models/ppo_v8_e4_10m`): trained from scratch for 10M decisions on scenario v5, score v8 and safety layer m3.7 (`train_rl.py`: LR 3e-4 decaying to 0, no warm-up). It took about 38 hours on the laptop (GPU learner, 16 CPU workers).

**Results (milestone 4):** 70 seeds that no model trained on or was selected on (1010–1019, 1100–1119, 1200–1239), scenario v5, every model scored on v8 (`models/ppo_v8_e4_10m/eval_compare_v8.json`):

| | Overtaker | v7 model (final) | v8 model (best, 9.5M) | v8 model (final) |
|---|---|---|---|---|
| Team score | −12,187 | 46,427 | 78,510 | **83,365** |
| Crash rate | **0.03** | 0.17 | 0.20 | 0.16 |
| Close passes | 49.6 | 59.3 | 58.2 | **60.8** |
| Draft passes | 10.0 | 14.3 | 24.0 | **24.7** |
| Train passes | 0.3 | 0.6 | 7.7 | **8.3** |
| Train time (s) | 0.3 | 0.2 | 4.0 | **4.4** |
| Mean speed (m/s) | 30.2 | 29.3 | 28.3 | 28.1 |

- **The v8 model (final) beats the v7 model** by 36.9k ± 5.1k per seed, and scores higher on 56 of 70 seeds. The gain holds on the 30 seeds used before (+42.8k) and the 40 fresh ones (+32.6k).
- **The gain comes from team motion:** the cars close-pass about as often as v7, but make 1.7× as many draft passes and 14× as many train passes.
- **Best against final:** the 9.5M checkpoint was selected on 10 seeds (99.5k there), but on 70 seeds it scores 4.9k below the final model and crashes more. `run_sim.py` therefore uses the final model.
- **Open issues:** about 1 episode in 6 still ends in a crash, the same rate as the v7 model. Mean speed stays near 28 m/s, with no time above 120 mph: in this traffic the team rarely reaches the 80 mph needed for train points.

**Milestone 4, part 2** makes the stack work on the dynamic car, the step before CARLA, whose physics are dynamic too. Each change came from replaying crashes and asking why they happened; in the order they were made:

1. **Transfer test.** The v8 model on the dynamic M3 with the nonlinear MPC crashed in 25 of 30 episodes (the overtaker in 7), against 5 on the kinematic car. A trace of the last 5 s of all 32 crashes found the cause: 27 had braking of 10 m/s² or more within a second before the car began to slide. Every layer could command 11 m/s², more than the tyres' μg = 10.3 m/s², and with the brakes split 50/50 the rear tyres were asked for 112 % of their grip. That leaves no cornering force, so the car could not steer, and slid or spun.
2. **Realistic M3 braking, in the plant and in the nonlinear MPC's model.**
   - Longitudinal load transfer (CG height 0.5 m, assumed).
   - Brakes split 72/28 front/rear, so the front reaches its limit first. With load transfer, a 65/35 split still locks the rear first, at about 9.2 m/s².
   - Rear-biased all-wheel drive at 30/70, like the M3's xDrive.
   - Each axle's force capped at what friction allows, so μg = 10.3 m/s² is now the M3's real braking limit.

   On its own this did not help (24 of 30): the planner still braked to the tyre limit.
3. **Grip budget.**
   - Braking is capped at 0.8 μg (8.2 m/s²) in both MPCs, the lane-keep fallback and the emergency escape. That leaves about 60 % of the grip for steering.
   - The nonlinear MPC got a friction-circle limit at 90 % of each axle's grip.
   - It also starts from the acceleration the car was last given, since a fallback may have braked after its last plan.

   Result: 16 of 30. A jerk limit and a stronger penalty on changes of acceleration were also tried. Neither helped (the stiffer penalty made braking late), so both were dropped.
4. **Grip hard, contact soft.** A probe of what the MPC planned showed it planning the slides itself. Obstacle avoidance was a hard constraint and grip only a penalty, so in a tight spot the solver paid the grip penalty to swerve. A plan beyond grip can't be executed anyway, so the priorities were swapped:
   - the friction-circle limit is now hard;
   - obstacles and road edges are soft, with a very large penalty. This is an exact penalty: whenever a plan exists that avoids everything within grip, that plan is the one returned.

   Result: 11 of 30, and the overtaker 1 of 30. MPC solve failures fell from 1.6 % to 0.01 %, and open-loop replays of the previous plan fell from 23 per episode to none.
5. **Stability layer** (`stability.py`). An inner loop at 100 Hz (every plant substep) between every command source and the car. It makes minor corrections only, using steering, throttle and overall braking; no per-wheel braking.
   - It tracks the MPC's planned yaw rate and sideslip; for fallbacks, the car's steady-state response to their steering.
   - It corrects over-rotation and excess sideslip, by at most 0.1 rad of steering.
   - It keeps the front tyre within its saturation slip.
   - Braking and throttle get only the grip the axle has left after cornering, like ABS and traction control.
   - It posts the car's state to a buffer that the MPC replans from.

   A steady-state steering cap first applied to every command blocked the MPC's own transient steering and caused edge crashes. It now applies to fallback commands only. Result: 9 of 30, score 48.2k.
6. **MPC emergency mode.** The emergency escape, the lane-keep law with kinematic rollouts, had become the weakest part. It is now an MPC plan:
   - A few candidate manoeuvres: brake, hold or gain speed, in the car's lane or one lane to either side.
   - Each candidate is the reference of an emergency solve, which keeps only the contact rows, runs more solver iterations and uses lighter smoothing weights. Contact has one large, flat penalty.
   - The first clear plan (else the one with the least contact) is driven like any plan.
   - The escaping car plans first in the next cycle.

   The lane-keep law is now only the last resort. Result: 4 of 30, score 56.9k.
7. **Plan verification.** Plotting one MPC step against the stability layer's 100 Hz substeps (`results/run_034_mpc_step_stability_layer_seed1010_car0_t84.6/`) showed why the MPC still planned slides with grip hard:
   - each replan runs one solver iteration, which enforces nonlinear constraints only to first order (here, a plan at 99 % grip from a car at 62 %);
   - and the plans asked for about 1.4× the yaw rate that grip can hold.

   Every solved plan's true grip use is now checked. While it breaks the limit, up to 4 more iterations run: a deterministic cap, plus an optional wall-clock budget for real-time runs that is off in simulation. A plan that still fails, or whose extra iterations fail, goes to the emergency mode. A car that starts a replan past the limit gets 0.3 s to come back, as long as it doesn't get worse. Result: 2 of 30, overtaker 0 of 30, and the time spent sliding fell from 0.47 to 0.07 s per episode.

**Results (milestone 4, part 2):** dynamic M3 with the nonlinear MPC, seeds 1010–1019 and 1100–1119, scored on v8, v8 model (final):

| Step | v8 model: crashes | Team score | Time sliding (>3°) per episode | Overtaker: crashes |
|---|---|---|---|---|
| Start (planner m3.7) | 25 of 30 | 20,197 | – | 7 of 30 |
| + realistic braking, grip budget | 16 of 30 | 32,400 | 0.80 s | 6 of 30 |
| + grip hard, contact soft | 11 of 30 | 41,700 | 0.55 s | 1 of 30 |
| + stability layer | 9 of 30 | 48,200 | 0.76 s | 4 of 30 |
| + MPC emergency mode | 4 of 30 | 56,900 | 0.47 s | 2 of 30 |
| **+ plan verification (planner m4.2)** | **2 of 30** | **56,000** | **0.07 s** | **0 of 30** |
| *Kinematic car, same seeds* | *5 of 30* | *89,800* | – | *1 of 30* |

- **On these seeds, the dynamic car now crashes less often than the kinematic car** (2 against 5 of 30), but it scores lower (56k against 90k). The v8 model was trained on the kinematic car and plans moves the real car can't make as tightly, so fine-tuning it on the dynamic car is next.
- **A few crashes in 30 is within run-to-run noise**, because tiny numerical differences change how an episode unfolds. The trend from 25 to 2 is not.
- **The kinematic, linear-MPC path is unchanged** (bit-identical results), so the v8 model's training environment is too.
- **Compute:** per 120 s episode, the v8 model's cars need about 230 extra solver iterations and 16 emergency plans, about half a second of solver time.
- **Proposed next:**
  - a yaw-rate and sideslip envelope in the MPC (no turn faster than grip can hold);
  - relaxing the first step's grip limit when the car starts past it;
  - clipping the stability layer's reference to the same envelope;
  - then fine-tuning the policy on the dynamic car.

**Next:** CARLA (Town04 highway) is planned later, for visualisation and evaluation only. A curved, Town04-like road was started and is paused until the dynamic stack is settled.

## Training (milestone 3)

Needs the `rl` extra (`pip install -e ".[test,rl]"`). For GPU learning (`DEVICE = 'cuda'` in `train_rl.py`), install the CUDA build: `pip install --force-reinstall --no-deps torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130`. `train_rl.py` works like `run_sim.py`: edit the settings at the top (`RUN_ID`, `SCENARIO`, `SCORE`, `N_ENVS`, `TOTAL_TIMESTEPS`, ...) and run it, or use `.venv\Scripts\python -m nohesi.rl.train --run-id ppo_v4_001 --timesteps 10000000` (`--resume runs_rl/<id>/checkpoints/model_<t>.zip` continues a run; `--init-from <model>.zip` starts a new run from a trained model, for fine-tuning; `--vf-warmup N` then trains only the value network for the first N decisions, so its estimates fit the new score before the policy changes). It trains sb3-contrib `MaskablePPO` on `nohesi.rl.env.NoHesiEnv`. One timestep is one team decision (0.5 s), and invalid lane commands are masked.

- **Setup:** 16 `SubprocVecEnv` workers × 512 steps (8192 per update), batch 1024, 10 epochs, lr 3e-4 decaying linearly to 0, clip 0.2, entropy 0.01, γ 0.99, λ 0.95. Separate 256×256 tanh networks for policy and value. `VecNormalize` normalises observations and rewards (clip 10), and its statistics are saved with every model.
- **Threads:** each worker uses one thread (`OMP/MKL/OPENBLAS_NUM_THREADS=1`), and the learner uses 1 torch thread while collecting rollouts and 4 during PPO updates. With 16 envs, training runs at about 80 timesteps/s; the environment (mostly the MPC) is the bottleneck.
- **Evaluation:** every 500k timesteps on held-out seeds 1000–1009 (`--seeds` for the final evaluation, default 1000–1019), with deterministic actions and masks applied. The model with the best mean team score is saved as `best_model.zip` + `best_vecnormalize.pkl`.
- **Saved models:** `models/<name>/` holds the trained models kept in git: best and final models with their `vecnormalize` files, `config.json`, `eval_history.csv` and the 20-seed evaluations. `run_sim.py` (`POLICY = 'ppo'`, `PPO_RUN`) and `nohesi.rl.evaluate` accept these folders like a run folder.
- **Outputs:** everything goes to `runs_rl/<run_id>/` (git-ignored): `config.json` (configs, library versions, git commit), TensorBoard logs in `tb/` (per-term reward, aux penalties, episode and eval metrics; view with `.venv\Scripts\tensorboard --logdir runs_rl`), `checkpoints/`, and `eval_history.csv`.

`.venv\Scripts\python -m nohesi.rl.evaluate runs_rl/<run_id>` evaluates the best model on seeds 1000–1019. It compares the result with the scripted overtaker on the same seeds, scenario and score (cached in `runs_rl/baselines/`). Options: `--mpc nonlinear --vehicle dynamic` for transfer to the dynamic plant; `--store` saves the first seed to `results/` like `run_sim.py` (numbered folder + GIF, label `<scenario>_ppo-<run_id>_score<score>_seed<seed>`).

## How the No Hesi simulation works

```
traffic + team state ──► policy (every 0.5 s) ──► intention layer ──► MPC per car (every 0.1 s) ──► vehicles (every 0.05 s)
                         chooses an action        locks a gap          plans 3 s of steering            kinematic or dynamic
                                                  or rejects           and throttle                     bicycle model
                                                                                   │      ▲ dynamic car: stability layer
                                                                                   │      │ (100 Hz) under every command
                                     scoring ◄── only what actually happened ◄─────┘
```

Key ideas, in the order things happen:

1. **Road and traffic** (`road.py`, `traffic.py`, `world.py`): a straight 3-lane road. Traffic cars drive at 18–25 m/s and slow down for cars ahead using the Intelligent Driver Model. In scenarios v3/v4 they also change lanes (MOBIL: when it gains them speed without making the new follower brake hard), moving over smoothly in 3 s. Cars that fall far behind the team respawn ahead. Positions are in road coordinates: `s` along the road and `d` across it (positive = left, lane 0 = rightmost).
2. **Policy** (`rl/scripted.py`): picks one of the 45 actions per car. The scripted *overtaker* changes lanes when stuck behind slower traffic and leans toward cars it passes. Later, RL will replace it.
3. **Intention layer** (`intentions.py`): a lane request locks the **gap** (the space between two cars) in the target lane where the car will be 3 s later (4 s for a two-lane move). A lane change is *pending* until the car touches the lane line, then *committed* (it can no longer be cancelled by the policy), then *done*.
   The team predicts traffic (`prediction.py`) at constant speed, and a car that is moving over is predicted to keep moving to its target lane.
4. **MPC** (`planning/single.py`): model predictive control. Every 0.1 s it solves a small optimisation: "which steering and throttle over the next 3 s best reach the target lane and speed, without breaking the vehicle's limits or getting within 0.25 m of anyone?" Only the first command is applied, and then it re-plans. The three cars plan one after another, each avoiding the plans already made.
5. **Safety net** (`planning/fallback.py`, `coordination/centralized.py`): if a solve fails, the car follows its previous plan for up to 0.5 s, and otherwise returns to its original lane and follows the car ahead. After each planning cycle, every plan is checked against traffic and teammates. If two plans conflict, the lower-priority car aborts its lane change or brakes. These checks use *predicted* traffic, so they reduce risk but are not a guarantee.
   With the dynamic car and the nonlinear MPC (milestone 4, part 2), braking stays within a grip budget, every plan's grip use is verified, an emergency is planned by the MPC itself, and the stability layer (`stability.py`) carries out every command at 100 Hz within grip.
6. **Scoring** (`scoring/`): measured only from the motion that actually happened, kept separate from the planner so the rules can change freely.

### Nonlinear MPC (optional, `Planner(mpc='nonlinear')`)

`planning/nonlinear.py` plans with the same dynamic bicycle as the `dynamic` plant (Fiala tyres, friction ellipse), solved by acados (SQP-RTI, HPIPM) in about 3–6 ms. It uses the same targets, costs and collision constraints as the linear MPC, so everything around it is unchanged. Set `MPC = 'nonlinear'` with `PLANT = 'dynamic'` in `run_sim.py`.
- **Setup (Windows):** acados built in `C:/acados`, `pip install -e C:/acados/interfaces/acados_template`, MSYS2 UCRT64 gcc in `C:/msys64/ucrt64/bin` (see the `nmpc` extra in `pyproject.toml`). Generated C code goes to `C:/dev/acados_build/<config hash>` (it must be a path without spaces; override with `NOHESI_ACADOS_BUILD`). It compiles once (about 10–30 s), then every process just loads it. Parallel sweeps are safe because a file lock lets only one process compile.
- **Differences from the linear MPC:** steering and throttle are states and their rates are the inputs. The speed cost is on progress along the road. Small costs on sideslip and yaw rate are needed for the solver to converge. A soft slip-angle limit keeps plans off the tyres' grip limit. There is no grip-proxy steering cap (`nmpc_steer_cap` turns it back on).
- **Since milestone 4, part 2 (planner m4.2):**
  - the model has load transfer and the M3's brake and drive split;
  - braking is capped at `grip_brake` μg;
  - the friction-circle limit (`nmpc_grip_limit`) is hard, while obstacles and road edges are soft (`nmpc_w_obstacle`);
  - every plan's true grip use is verified, with up to `nmpc_verify_iter` extra iterations;
  - emergencies are solved as candidate manoeuvres (`nmpc_emergency`).
- **Results** (seeds 0–4, overtaker, score v3, dynamic plant): 0 collisions on v1 and v4, where the linear MPC on the same dynamic plant crashes in all 10 episodes (9 off the road edge, 1 into traffic). Mean scores are 13,500 (v1) and 13,000 (v4), against 16,600 and 16,900 for the linear MPC on the kinematic plant. The next-step plan error is under 0.03 m.

## Scoring

Scores are versioned, and each result records which version it used. v4 is the default. v5 is v4 with drafting 5× more rewarding; v6 is v5 plus draft passes. v7 rewards the whole team moving together, and v8 (used to train the current model) makes that fast.

| Points for | How |
|---|---|
| **Close pass** | Overtaking a traffic car with less than 1.5 m clearance: about 100–200 points, more when closer and faster. |
| **Team close pass** (v4+) | A close pass with teammates nearby earns up to 2× (both teammates at contact), falling linearly to no bonus at 20 m. In v7 it only counts when the whole team is together: all three cars within 40 m of each other (graded over 40 m); in v8, within 50 m. |
| **Team combo** (v1+) | Every close pass raises a team multiplier, up to 5×. It resets after 5 s without a close pass. |
| **Teammate proximity** (v1–v3) | Points per second while teammates drive within 20 m of each other. Closer earns more (v2+). |
| **Drafting** (v4+, replaces proximity) | 5 points per second (v4) or 25 (v5) for each car directly behind a teammate (same lane, no traffic between) within 1 s, both above 20 m/s. Multiplied by the team combo. |
| **Draft pass** (v6+) | A close pass by a car that is drafting a teammate, or left the draft within the last 1 s (a slingshot), is worth 3× more, on top of the team pass bonus and the combo. In v7 a two-car draft pass is worth 1.5×, and a pass by any car of a draft train 3×. |
| **Draft train** (v7) | All three cars in a row along the road, in any lanes, each within 0.5 s bumper to bumper of the car ahead, all above 20 m/s. Every car earns 25 points per second (times the combo) while it holds; a two-car draft earns 5. In v8 train points and train passes need every car above 80 mph (35.8 m/s). |
| **Lane-change maneuvers** (v3+) | Only scored with traffic within 10 m and speed above 25 m/s. 300 for a two-lane change → close pass → another lane change, 150 for the same with a single change, 90/60 for a lane change with a close pass, 25 for a lane change alone. Each step of a chain must happen within 3 s. |
| **Speed** | A small amount for driving above 25 m/s (v1+). |
| **Top speed** (v8) | +10 points per second for each car above 120 mph (53.6 m/s), times the combo. |
| **Team spread penalty** (v7) | For each pair of teammates more than 50 m apart: −2 × ((distance − 50 m) / 10 m)² points per second, at most 200 (reached at 150 m), split between the pair (−18 pts/s at 80 m, −50 at 100 m). Not multiplied by the combo. v8 doubles it: −4 × ((distance − 50 m) / 10 m)², at most 400. |
| **Collision** | −500 to each car involved, and the episode ends. In v7 a collision also wipes 50 % of the whole team's banked points (No Hesi-style run reset). |
| **Re-pass cooldown** (v7) | A team car scores the same traffic car again only after 10 s (no farming one slow car). |

## Outputs

Each run folder contains:
- `team.gif`: top-down animation. Each car's current plan is drawn faintly ahead of it, dashed when the car is on a fallback.
- `summary.png`: speed, lane position and score over time.
- `metrics.json`: scores by term, collisions, lane changes, rejections, fallbacks, solve times, and every setting used.
- `events.json`: every pass, maneuver, lane-change decision and fallback, with times.
- `trajectory.npz` and `team.csv`: raw histories, including every MPC plan.

`results/run_001_v1_overtaker_scorev2_seed0_failed_m2.4/` keeps a run that exposed planner bugs, which are now fixed, with an explanation in its `NOTE.md`.

## Code map (`nohesi/`)

| File | Responsibility |
|---|---|
| `config.py` | All settings, plus versioned `SCENARIOS` and `SCORES` |
| `road.py`, `traffic.py`, `world.py`, `dynamics.py`, `geometry.py` | Road, traffic, simulation step, vehicle models (kinematic and dynamic), footprints and clearances |
| `controllers.py`, `prediction.py` | Pure pursuit and gap control for lane keeping; traffic prediction |
| `intentions.py` | Action space, gap rule, lane-change phases |
| `planning/single.py`, `planning/constraints.py` | Per-car linear MPC; the collision constraints both MPCs share |
| `planning/nonlinear.py` | Optional per-car nonlinear MPC (acados, dynamic bicycle): grip hard, contact soft, plan verification, emergency solves |
| `stability.py` | Stability layer: 100 Hz inner loop under every command on the dynamic car (grip governor, yaw/sideslip corrections, state buffer) |
| `planning/fallback.py` | Fallback plans and plan conflict checks |
| `coordination/centralized.py` | Runs the policy, intentions and the three MPCs together |
| `rl/scripted.py` | Lane-keeping and overtaker policies |
| `scoring/events.py`, `scoring/score.py` | Measure what happened, then turn it into points |
| `evaluation/` | Episode runner, metrics, file output and plots/GIFs |

**Vehicle models:** all positions (`s`, `d`) are centre-of-gravity (CG) positions. The CG is 60/40 front-biased by default (`Vehicle.cg_to_rear_axle = 1.62 m`). There are two models:
- **Kinematic bicycle:** slip angle β = atan((l_r/L)·tan δ). The default for the simulator, and the linear MPC's model.
- **Dynamic bicycle:** Fiala tyres with a friction ellipse, longitudinal load transfer (`Vehicle.cg_height`), brake and drive split (`brake_bias_front`, `drive_bias_front`) and axle forces capped at μFz. Used by `PLANT = 'dynamic'` and by the nonlinear MPC.

**Limitations:**
- The road is straight.
- The dynamic model has longitudinal load transfer only (quasi-static), no left/right tyres and no wheel-speed (ABS) dynamics; the stability layer's grip governor stands in for ABS and traction control.
- The nonlinear MPC needs a local acados build.
- Traffic changes lanes only in v3/v4.
- Safety checks are sampled every 0.05 s rather than continuous.

**Planned next steps:**
- fine-tuning the policy on the dynamic car;
- a yaw-rate and sideslip envelope in the MPC;
- curved roads (a Town04-like road is drafted);
- CARLA/Assetto Corsa evaluation.
