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
| **No Hesi M3, part 2** | Trained PPO models, drafting-focused fine-tuning (scores v5–v6), faster simulator, team spread penalty (v7). |

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
- **Safety layer m3.5** (after replaying all 11 crashes of the v7 model, 10 of which were checker gaps):
  - **Smarter emergency escape:** it picks the option with the gentlest predicted impact (smallest speed difference), and can also speed up, not only brake or hold.
  - **Denser checks:** plans are checked 5 times per 0.1 s, so a fast pass between plan points is caught.

**Results (first 3M-decision PPO run, before the fallback changes):** on seeds 1000–1019 PPO scored about 47,800 points, against 18,200 for the overtaker. It crashed in 5 of 20 episodes. Replaying the same policy with the new fallbacks gave 2 crashes in 20 against traffic that looked before merging. With traffic merging blind it was back to 5. The policy was then retrained (part 2).

**Milestone 3, part 2** trains the policy that is saved in `models/`, then fine-tunes it toward team play:
- **5M-decision run** (`models/ppo_v4_e4_5m`): trained from scratch on score v4, with blind-merging traffic and the new observation (environment `e4`).
- **Fine-tuning** (`--init-from` / `INIT_FROM` in `train_rl.py`): a new run starts from another run's policy weights and observation scaling. The optimiser, learning-rate schedule and reward scaling start fresh.
- **Score v5:** drafting 5× more rewarding (25 points per second).
- **Score v6:** v5 plus **draft passes**. A close pass by a car that is drafting, or left the draft within the last 1 s (a slingshot), is worth 3×. The v6 model (`models/ppo_v6_e4_5m`) is the 5M model fine-tuned on v6 for 5M decisions at a learning rate of 1e-4.
- **Score v7 (ready, not trained yet):** rewards the whole team together. A three-car **draft train** (any lanes, within 0.5 s of each other) pays 25 pts/s per car and 3× passes; a two-car draft 5 pts/s and 1.5×. The team close-pass bonus needs all three cars within 40 m, and a capped quadratic **team spread penalty** applies beyond 50 m. A crash wipes 50 % of the team's banked points (so training no longer needs the extra 5× crash penalty), and a car can re-score the same traffic car only after 10 s. `train_rl.py` is set up to fine-tune the v6 model on it for 6M decisions: LR 1e-4, with a 200k-decision value-network warm-up.
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

**Next:** fine-tune the v6 model on v7. CARLA (Town04 highway) is planned later, for visualisation and evaluation only.

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
                         chooses an action        locks a gap          plans 3 s of steering            kinematic bicycle
                                                  or rejects           and throttle                     model
                                                                                   │
                                     scoring ◄── only what actually happened ◄─────┘
```

Key ideas, in the order things happen:

1. **Road and traffic** (`road.py`, `traffic.py`, `world.py`): a straight 3-lane road. Traffic cars drive at 18–25 m/s and slow down for cars ahead using the Intelligent Driver Model. In scenarios v3/v4 they also change lanes (MOBIL: when it gains them speed without making the new follower brake hard), moving over smoothly in 3 s. Cars that fall far behind the team respawn ahead. Positions are in road coordinates: `s` along the road and `d` across it (positive = left, lane 0 = rightmost).
2. **Policy** (`rl/scripted.py`): picks one of the 45 actions per car. The scripted *overtaker* changes lanes when stuck behind slower traffic and leans toward cars it passes. Later, RL will replace it.
3. **Intention layer** (`intentions.py`): a lane request locks the **gap** (the space between two cars) in the target lane where the car will be 3 s later (4 s for a two-lane move). A lane change is *pending* until the car touches the lane line, then *committed* (it can no longer be cancelled by the policy), then *done*.
   The team predicts traffic (`prediction.py`) at constant speed, and a car that is moving over is predicted to keep moving to its target lane.
4. **MPC** (`planning/single.py`): model predictive control. Every 0.1 s it solves a small optimisation: "which steering and throttle over the next 3 s best reach the target lane and speed, without breaking the vehicle's limits or getting within 0.25 m of anyone?" Only the first command is applied, and then it re-plans. The three cars plan one after another, each avoiding the plans already made.
5. **Safety net** (`planning/fallback.py`, `coordination/centralized.py`): if a solve fails, the car follows its previous plan for up to 0.5 s, and otherwise returns to its original lane and follows the car ahead. After each planning cycle, every plan is checked against traffic and teammates. If two plans conflict, the lower-priority car aborts its lane change or brakes. These checks use *predicted* traffic, so they reduce risk but are not a guarantee.
6. **Scoring** (`scoring/`): measured only from the motion that actually happened, kept separate from the planner so the rules can change freely.

### Nonlinear MPC (optional, `Planner(mpc='nonlinear')`)

`planning/nonlinear.py` plans with the same dynamic bicycle as the `dynamic` plant (Fiala tyres, friction ellipse), solved by acados (SQP-RTI, HPIPM) in about 3–6 ms. It uses the same targets, costs and collision constraints as the linear MPC, so everything around it is unchanged. Set `MPC = 'nonlinear'` with `PLANT = 'dynamic'` in `run_sim.py`.
- **Setup (Windows):** acados built in `C:/acados`, `pip install -e C:/acados/interfaces/acados_template`, MSYS2 UCRT64 gcc in `C:/msys64/ucrt64/bin` (see the `nmpc` extra in `pyproject.toml`). Generated C code goes to `C:/dev/acados_build/<config hash>` (it must be a path without spaces; override with `NOHESI_ACADOS_BUILD`). It compiles once (about 10–30 s), then every process just loads it. Parallel sweeps are safe because a file lock lets only one process compile.
- **Differences from the linear MPC:** steering and throttle are states and their rates are the inputs. The speed cost is on progress along the road. Small costs on sideslip and yaw rate are needed for the solver to converge. A soft slip-angle limit keeps plans off the tyres' grip limit. There is no grip-proxy steering cap (`nmpc_steer_cap` turns it back on).
- **Results** (seeds 0–4, overtaker, score v3, dynamic plant): 0 collisions on v1 and v4, where the linear MPC on the same dynamic plant crashes in all 10 episodes (9 off the road edge, 1 into traffic). Mean scores are 13,500 (v1) and 13,000 (v4), against 16,600 and 16,900 for the linear MPC on the kinematic plant. The next-step plan error is under 0.03 m.

## Scoring

Scores are versioned, and each result records which version it used. v4 is the default. v5 is v4 with drafting 5× more rewarding; v6 is v5 plus draft passes. v6 is used for PPO fine-tuning.

| Points for | How |
|---|---|
| **Close pass** | Overtaking a traffic car with less than 1.5 m clearance: about 100–200 points, more when closer and faster. |
| **Team close pass** (v4+) | A close pass with teammates nearby earns up to 2× (both teammates at contact), falling linearly to no bonus at 20 m. In v7 it only counts when the whole team is together: all three cars within 40 m of each other (graded over 40 m). |
| **Team combo** (v1+) | Every close pass raises a team multiplier, up to 5×. It resets after 5 s without a close pass. |
| **Teammate proximity** (v1–v3) | Points per second while teammates drive within 20 m of each other. Closer earns more (v2+). |
| **Drafting** (v4+, replaces proximity) | 5 points per second (v4) or 25 (v5) for each car directly behind a teammate (same lane, no traffic between) within 1 s, both above 20 m/s. Multiplied by the team combo. |
| **Draft pass** (v6+) | A close pass by a car that is drafting a teammate, or left the draft within the last 1 s (a slingshot), is worth 3× more, on top of the team pass bonus and the combo. In v7 a two-car draft pass is worth 1.5×, and a pass by any car of a draft train 3×. |
| **Draft train** (v7) | All three cars in a row along the road, in any lanes, each within 0.5 s bumper to bumper of the car ahead, all above 20 m/s. Every car earns 25 points per second (times the combo) while it holds; a two-car draft earns 5. |
| **Lane-change maneuvers** (v3+) | Only scored with traffic within 10 m and speed above 25 m/s. 300 for a two-lane change → close pass → another lane change, 150 for the same with a single change, 90/60 for a lane change with a close pass, 25 for a lane change alone. Each step of a chain must happen within 3 s. |
| **Speed** | A small amount for driving above 25 m/s (v1+). |
| **Team spread penalty** (v7) | For each pair of teammates more than 50 m apart: −2 × ((distance − 50 m) / 10 m)² points per second, at most 200 (reached at 150 m), split between the pair (−18 pts/s at 80 m, −50 at 100 m). Not multiplied by the combo. |
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
| `planning/nonlinear.py` | Optional per-car nonlinear MPC (acados, dynamic bicycle) |
| `planning/fallback.py` | Fallback plans and plan conflict checks |
| `coordination/centralized.py` | Runs the policy, intentions and the three MPCs together |
| `rl/scripted.py` | Lane-keeping and overtaker policies |
| `scoring/events.py`, `scoring/score.py` | Measure what happened, then turn it into points |
| `evaluation/` | Episode runner, metrics, file output and plots/GIFs |

**Vehicle models:** all positions (`s`, `d`) are centre-of-gravity (CG) positions. The CG is 60/40 front-biased by default (`Vehicle.cg_to_rear_axle = 1.62 m`). There are two models:
- **Kinematic bicycle:** slip angle β = atan((l_r/L)·tan δ). The default for the simulator, and the linear MPC's model.
- **Dynamic bicycle:** Fiala tyres with a friction ellipse. Used by `PLANT = 'dynamic'` and by the nonlinear MPC.

**Limitations:**
- The road is straight.
- The dynamic model has no load transfer: axle loads are static, and the bicycle model has no left/right tyres.
- The nonlinear MPC needs a local acados build.
- Traffic changes lanes only in v3/v4.
- Safety checks are sampled every 0.05 s rather than continuous.

**Planned next steps:** an RL policy, curved roads, and CARLA/Assetto Corsa evaluation.
