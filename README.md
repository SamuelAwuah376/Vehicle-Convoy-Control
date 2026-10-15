# Three-vehicle autonomous convoy simulator

A working **deterministic** baseline with a curved single-lane route, kinematic bicycle vehicles, leader path tracking, follower gap control, bounded actuators, endpoint stopping, quantitative tests, and saved plots/animation. Optional CasADi MPC replaces **leader steering only**; longitudinal control and follower steering remain the baseline.

The earlier ChatGPT architecture replies could not be retrieved: they appeared as content references. The defaults below were proposed in this chat and accepted by the user. They are not represented as recovered original parameters.

## Run from this directory

Use Python 3.10 or later (verified here with Python 3.12 on Windows). Create an isolated environment and install the package:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[test]"
.venv\Scripts\python -m pytest -q --basetemp=work/pytest-temp
.venv\Scripts\python -m convoy --scenario nominal --out results/nominal
.venv\Scripts\python -m convoy --scenario braking --out results/braking --animate
.venv\Scripts\python -m convoy --scenario offset --out results/offset
.venv\Scripts\python -m convoy --scenario braking --dt 0.1 --out results/braking_dt01
```

On macOS/Linux, use `.venv/bin/python` instead. If Windows `python` resolves to an unavailable Microsoft Store alias, use an installed Python executable or `py -3` for the environment creation command.

Optional MPC:

```powershell
.venv\Scripts\python -m pip install -e ".[test,mpc]"
.venv\Scripts\python -m pytest -q --basetemp=work/pytest-temp
.venv\Scripts\python -m convoy --controller mpc --out results/mpc
```

Without CasADi, the MPC test is explicitly skipped and the baseline has no CasADi import dependency. The native solver test runs in a subprocess with file output handles to avoid Windows IPOPT/pytest capture conflicts. Python-level solver failures use pure pursuit and increment `mpc_failures`; native process failures cannot be caught by that fallback. MPC construction or missing solver dependencies produce an explicit error.

Every run saves `metrics.json`, `trajectory.csv`, `trajectory.npz`, and `summary.png`; `--animate` adds `convoy.gif`. NPZ state shape is `(samples, 3, 4)` with state order `[x, y, yaw, speed]`; controls are `(samples-1, 3, 2)` in `[acceleration, steering]` order, applied during the interval beginning at the corresponding state sample. CSV vehicle IDs 0, 1, 2 identify the leader and followers. Angles are radians; SI units are used throughout. The GIF runs about six times faster than simulation time and shows the timestamp.

## No Hesi team simulation (`nohesi/`, milestone 1)

A separate package for the No Hesi-style team game; `convoy/` is unchanged. Milestone 1 adds a straight 3-lane road, seeded IDM traffic (8 cars, 18–25 m/s, recycled ahead of the team), collision and road-edge detection, versioned scoring, and a scripted lane-keeping policy. There is no MPC or RL yet.

```powershell
.venv\Scripts\python -m nohesi --scenario v0 --seed 0 --out results/nohesi_v0 --animate
```

Outputs: `metrics.json` (seed, scenario and score versions, full scenario config, team/individual scores by term, collisions, overtakes, proximity time, minimum clearances, policy compute time), `events.json` (every overtake with its minimum clearance, scored close passes, collisions), `trajectory.npz`, `team.csv`, `summary.png` and optionally `team.gif`.

Scoring v0 (`nohesi/config.py: Score`): close pass `100·(1+(1.5−c)/1.5)·(v/28)` when footprint clearance `c < 1.5 m` while overtaking from behind; teammate proximity 10 pts/s per pair within 12 m (both above 20 m/s), split between the pair; 1 pt/s per m/s above 20 m/s; collision (traffic, teammate or road edge) −500 to each car involved and the episode ends. Team score is the sum. Detection (`scoring/events.py`) reports raw measurements; only `scoring/score.py` applies the formula. Clearances are sampled at simulation steps, not continuous.

## Architecture and parameters

| Module | Responsibility |
|---|---|
| `config.py` | Immutable physical/control parameters and validation |
| `route.py` | Sampled curved reference, arc-length interpolation and segment projection |
| `dynamics.py` | Rear-axle bicycle dynamics and actuator bounds |
| `controllers.py` | Pure pursuit, leader speed/stop control, predecessor gap control |
| `mpc.py` | Optional nominal CasADi/IPOPT leader steering MPC |
| `simulation.py` | Synchronous three-vehicle loop, histories and performance metrics |
| `geometry.py` | Oriented vehicle rectangles, collision and lane checks |
| `visualization.py` | Route, speed, gap, tracking plots and GIF |
| `__main__.py` | Command-line simulation and exports |
| `tests/test_convoy.py` | Model checks and closed-loop acceptance tests |

| Parameter | Default |
|---|---:|
| Vehicles | 3 |
| Wheelbase / length / width | 2.7 / 4.5 / 1.8 m |
| Lane width | 3.6 m |
| Cruise speed | 8 m/s |
| Standstill bumper gap | 5 m |
| Time headway | 1.2 s |
| Acceleration limits | −4 to +2 m/s² |
| Steering angle / rate limits | ±0.5 rad / 0.6 rad/s |
| Simulation step / duration | 0.05 s / 90 s |
| Route sampling | 0.2 m in x; ~430.24 m arc length |

The route is `y(x) = 12 sin(x/45) + 4 sin(x/22)` for `0 ≤ x ≤ 420 m`. Initial rear-axle arc lengths are 43.2, 24.1 and 5 m, with all vehicles traveling at 8 m/s. This establishes the desired 14.6 m bumper gap at cruise. The leader stops 4 m before the route endpoint, leaving room for the full body; followers stop behind it at 5 m bumper gaps. This is a route-following demonstration with a precomputed geometric path, not an obstacle-avoidance planner.

The plant is `xdot = v cos(yaw)`, `ydot = v sin(yaw)`, `yawdot = v tan(steering)/wheelbase`, `vdot = acceleration`. Midpoint integration uses bounded commands; a braking step that reaches zero speed integrates only until the stop time and never moves backward. Each vehicle occupies a rectangle extending 0.9 m behind and 3.6 m ahead of its rear axle.

Pure pursuit targets an arc-length lookahead `3 + 0.65v` m. Follower acceleration is `0.65(gap − 5 − 1.2v) + 1.3(v_predecessor − v)`, before actuator saturation. Gap is the difference in rear-axle arc lengths minus vehicle length. This curved-route approximation is checked separately against oriented rectangle collisions. All commands are computed from one shared state snapshot before any vehicle advances.

The leader uses speed feedback and critically damped endpoint feedback: `a = min(1.2(v_target − v), remaining − 2v)`, then actuator saturation. In the braking scenario its target drops to 2 m/s during seconds 20–27, then returns to 8 m/s. The offset scenario introduces ±0.5 m lateral offsets and +0.05 rad yaw error on all vehicles. There is no noise or random seed dependency.

MPC has 12 steps of 0.15 s, a constant measured speed prediction, bicycle heading/position dynamics, steering magnitude/rate constraints, and position, heading, steering and steering-change costs. It replans every simulation step and warm-starts steering. It does not constrain intervehicle gaps or lane boundaries inside the optimizer; those are evaluated afterward. MPC implementation uses the [official CasADi Opti/IPOPT interface](https://web.casadi.org/docs/).

## Acceptance criteria and measured results

The baseline tests require zero sampled rectangle overlaps and lane-boundary violations, >3 m minimum bumper gap, <0.65 m maximum rear-axle cross-track error, <1.5 m gap-error RMS after 10 s, all final speeds <0.05 m/s, leader goal error <0.2 m, forward progress, and actuator/rate bounds. These thresholds are explicit regression criteria for the supplied scenarios, not formal safety guarantees. Tests also check straight motion, exact stopping distance within a step, circular motion against an analytical solution, route projection, rectangle geometry and equilibrium gap control.

Checked results and their complete configurations are saved in `results/`. See `VALIDATION.md` for the measurements. All geometry checks are at recorded simulation times: they do not certify continuous swept-volume safety. Lane checks use projected rectangle corners, appropriate to this gentle route, and do not model tire slip or road-edge interaction.

## Scope and next steps

This version assumes exact states, instantaneous predecessor state availability, fixed known dynamics, no obstacles and no communications delay. The route and controller form a tested nominal baseline. There is **no uncertainty-aware control, RL, perception, dynamic bicycle model, or formal string-stability proof**. Empirical recovery in the offset/braking cases is the extent of robustness established here. Parameter changes outside these tested conditions require fresh evaluation.

Useful next work: acceleration MPC with explicit gap constraints; solver timing and deadline handling; delayed/noisy observations and deterministic disturbance experiments; a dynamic bicycle model and friction limits. Add uncertainty-aware or RL claims only after implementing and evaluating those capabilities against this baseline.
