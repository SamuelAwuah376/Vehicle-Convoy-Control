# Vehicle Convoy Control

Two Python driving simulators in one repository:

| Package | What it is | Status |
|---|---|---|
| `nohesi/` | A **No Hesi-style team game**: three cars weave through highway traffic, scoring points for close passes, staying near teammates and lane-change maneuvers. Each car is driven by a planner (MPC) that turns high-level choices ("change left, speed up") into safe, smooth steering and throttle. | Active (milestone 2, part 1) |
| `convoy/` | The original **three-car convoy**: a leader follows a curved road and two followers keep a safe gap. | Finished baseline, kept for comparison |

Everything uses SI units (metres, seconds, m/s, radians).

## Quick start

Requires Python 3.10+ (developed on 3.12/3.13, Windows).

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest -q              # run all tests
.venv\Scripts\python run_sim.py                # run one No Hesi simulation
```

On macOS/Linux use `.venv/bin/python`. If `python` opens the Microsoft Store, use `py -3` to create the environment.

`run_sim.py` is the easiest way to run a simulation. Open it, change the settings at the top, and run it:

| Setting | Options |
|---|---|
| `SCENARIO` | `'v0'` 60 s, 8 traffic cars · `'v1'` 120 s, 10 cars · `'v2'` 120 s, 20 cars |
| `POLICY` | `'overtaker'` (lane changes + MPC) · `'lane_keeping'` (stays in its lane, no MPC) |
| `SCORE` | `'v0'` … `'v3'` (see [Scoring](#scoring)); `'v3'` is the latest |
| `SEED` | any integer: a different traffic layout |
| `TRAFFIC` | override the number of traffic cars (`None` keeps the scenario's) |
| `ANIMATE` | `True` writes a GIF (takes a few extra seconds) |

Results go to `results/nohesi_<scenario>_<policy>_score<score>_seed<seed>/`. The same runs are available from the command line, e.g. `.venv\Scripts\python -m nohesi --scenario v1 --policy overtaker --seed 0 --animate`.

## Milestones

| Milestone | What it added |
|---|---|
| Convoy baseline | Curved single-lane convoy, pure pursuit steering, gap control, optional leader steering MPC. |
| No Hesi M1 | Straight 3-lane road, seeded traffic, collision detection, versioned scoring, lane-keeping policy. |
| **No Hesi M2, part 1** | Lane changes executed by a per-car MPC, as described below. |

**Milestone 2, part 1** gives every car a planner that executes high-level choices safely:
- **Actions:** 45 choices per car: lane (two right, one right, keep, one left, two left) × speed (slower, hold, faster) × lean in the lane (right, centre, left).
- **Intention layer:** turns a lane choice into a locked target gap. It rejects unsafe requests, such as a gap that's too short, a faster car closing from behind, or a blocked middle lane during a two-lane move.
- **Per-car MPC:** a quadratic program solved with OSQP in about 2 ms that plans 3 s ahead, avoiding traffic and teammates within hard 0.25 m margins.
- **Safety net:** fallbacks for failed solves, plus a team-wide check of every plan after each planning cycle.
- **Scoring up to v3:** a team combo multiplier, teammate proximity and lane-change maneuvers.
- **Scripted overtaker:** a hand-written policy that exercises all of the above.
- **Debug outputs:** GIFs that draw each car's current plan.

**Results:** across seeds 0–4 on scenarios v1 and v2 there are 0 collisions. Under score v3 the overtaker averages about 17,000 points on v1 and 14,000 on v2, while the lane-keeping baseline scores almost nothing (it never passes closely). Reference runs: `results/nohesi_v1_overtaker_scorev3/` and `results/nohesi_v1_lane_keeping_scorev3/`.

**Next:** milestone 2, part 2, then milestone 3: replace the scripted overtaker with a reinforcement-learning (RL) policy.

## How the No Hesi simulation works

```
traffic + team state ──► policy (every 0.5 s) ──► intention layer ──► MPC per car (every 0.1 s) ──► vehicles (every 0.05 s)
                         chooses an action        locks a gap          plans 3 s of steering            kinematic bicycle
                                                  or rejects           and throttle                     model
                                                                                   │
                                     scoring ◄── only what actually happened ◄─────┘
```

Key ideas, in the order things happen:

1. **Road and traffic** (`road.py`, `traffic.py`, `world.py`): a straight 3-lane road. Traffic cars stay in their lanes at 18–25 m/s and slow down for cars ahead using the Intelligent Driver Model. Cars that fall far behind the team respawn ahead. Positions are in road coordinates: `s` along the road and `d` across it (positive = left, lane 0 = rightmost).
2. **Policy** (`rl/scripted.py`): picks one of the 45 actions per car. The scripted *overtaker* changes lanes when stuck behind slower traffic and leans toward cars it passes. Later, RL will replace it.
3. **Intention layer** (`intentions.py`): a lane request locks the **gap** (the space between two cars) in the target lane where the car will be 3 s later (4 s for a two-lane move). A lane change is *pending* until the car touches the lane line, then *committed* (it can no longer be cancelled by the policy), then *done*.
4. **MPC** (`planning/single.py`): model predictive control. Every 0.1 s it solves a small optimisation: "which steering and throttle over the next 3 s best reach the target lane and speed, without breaking the vehicle's limits or getting within 0.25 m of anyone?" Only the first command is applied, and then it re-plans. The three cars plan one after another, each avoiding the plans already made.
5. **Safety net** (`planning/fallback.py`, `coordination/centralized.py`): if a solve fails, the car follows its previous plan for up to 0.5 s, and otherwise returns to its original lane and follows the car ahead. After each planning cycle, every plan is checked against traffic and teammates. If two plans conflict, the lower-priority car aborts its lane change or brakes. These checks use *predicted* traffic, so they reduce risk but are not a guarantee.
6. **Scoring** (`scoring/`): measured only from the motion that actually happened, kept separate from the planner so the rules can change freely.

## Scoring

Scores are versioned, and each result records which version it used.

| Points for | How |
|---|---|
| **Close pass** | Overtaking a traffic car with less than 1.5 m clearance: about 100–200 points, more when closer and faster. |
| **Team combo** (v1+) | Every close pass raises a team multiplier, up to 5×. It resets after 5 s without a close pass. |
| **Teammate proximity** (v1+) | Points per second while teammates drive within 20 m of each other. Closer earns more (v2+). |
| **Lane-change maneuvers** (v3) | Only scored with traffic within 10 m and speed above 25 m/s. 300 for a two-lane change → close pass → another lane change, 150 for the same with a single change, 90/60 for a lane change with a close pass, 25 for a lane change alone. Each step of a chain must happen within 3 s. |
| **Speed** | A small amount for driving above 25 m/s (v1+). |
| **Collision** | −500 to each car involved, and the episode ends. |

## Outputs

Each run folder contains:
- `team.gif`: top-down animation. Each car's current plan is drawn faintly ahead of it, dashed when the car is on a fallback.
- `summary.png`: speed, lane position and score over time.
- `metrics.json`: scores by term, collisions, lane changes, rejections, fallbacks, solve times, and every setting used.
- `events.json`: every pass, maneuver, lane-change decision and fallback, with times.
- `trajectory.npz` and `team.csv`: raw histories, including every MPC plan.

`results/failed_v1_overtaker_scorev2_seed0_m2.4/` keeps a run that exposed planner bugs, which are now fixed, with an explanation in its `NOTE.md`.

## Code map (`nohesi/`)

| File | Responsibility |
|---|---|
| `config.py` | All settings, plus versioned `SCENARIOS` and `SCORES` |
| `road.py`, `traffic.py`, `world.py`, `dynamics.py`, `geometry.py` | Road, traffic, simulation step, vehicle model, footprints and clearances |
| `intentions.py` | Action space, gap rule, lane-change phases |
| `planning/single.py` | Per-car MPC |
| `planning/fallback.py` | Fallback plans and plan conflict checks |
| `coordination/centralized.py` | Runs the policy, intentions and the three MPCs together |
| `rl/scripted.py` | Lane-keeping and overtaker policies |
| `scoring/events.py`, `scoring/score.py` | Measure what happened, then turn it into points |
| `evaluation/` | Episode runner, metrics, file output and plots/GIFs |

**Limitations:** the road is straight and the vehicle model is kinematic, which is fine up to about 35 m/s. Traffic never changes lanes, and safety checks are sampled every 0.05 s rather than continuous. Planned next steps include a dynamic vehicle model for higher speeds, curved roads, and Assetto Corsa integration.

## Convoy baseline (`convoy/`)

A leader drives a curved road (`y = 12 sin(x/45) + 4 sin(x/22)`, about 430 m) at 8 m/s, following it with **pure pursuit** steering, and stops 4 m before the end. Two followers keep a safe gap with `a = 0.65·(gap − 5 − 1.2·v) + 1.3·(v_ahead − v)`, i.e. a 5 m standstill gap plus 1.2 s of headway. Scenarios: `nominal`, `braking` (the leader slows to 2 m/s at 20–27 s) and `offset` (cars start off the road centre).

```powershell
.venv\Scripts\python -m convoy --scenario braking --out results/braking --animate
.venv\Scripts\python -m pip install -e ".[test,mpc]"           # optional: CasADi MPC for leader steering
.venv\Scripts\python -m convoy --controller mpc --out results/mpc
```

Tests require zero collisions or lane violations, a minimum gap above 3 m and small tracking errors. Measured results are in `VALIDATION.md` and `results/`. Without CasADi the MPC test is skipped.
