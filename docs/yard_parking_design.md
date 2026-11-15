# Convoy parking in a staging yard: design (v1)

This document gives the reasons behind the yard parking stack in `nohesi/yard/`. `docs/yard_interfaces.md` is the
module contract and `nohesi/yard/config.py` holds every parameter. Values quoted here are config versions `t1`, `s1`,
`y1`, `p1` (scenario `yard1`) and env `ye1`. "Start" marks a starting value to tune, not a validated one.

**Why.** The haul-road test (`results/haul_obstacles_2026-11-15/REPORT.md`) ran the road ('No Hesi') stack on static
obstacles. A car stopped 0.2 m behind an obstacle had no forward-only way out and no reverse: on `h1` seed 1000 all
three cars sat still from about 65 s to the end, through 6,522 infeasible solves. The planner also broke at 5–7 m/s,
below its tyre model's 5 m/s floor (`nmpc_vx_floor`). The yard stack therefore has reverse at two levels (in planned
paths, and as the BACK_OFF action) and tracks with a kinematic model that is valid down to standstill. The road stack
stays untouched.

## 1. Goal and operating concept

**Goal.** A convoy of rigid haul trucks arrives at a staging yard (a pad beside the haul road, or a log landing). The
followers park autonomously in spots the leader's driver chose, with no collision, filling the yard in the least time.

| Actor | Role in v1 |
|---|---|
| Driver (in the leader) | Drives the leader to the scan pose, starts the scan and checks the map on the cab display. Picks one spot per follower (up to 3). Confirms the yard is closed to people and light vehicles. Holds the remote stop for all followers. |
| Leader truck (human-driven) | Scans the yard (roof lidar) into an occupancy grid. Runs the spot check, the coordinator and the policy. Broadcasts the map, spots and stop limits over V2V. Stays parked at its scan pose. |
| Followers (autonomous, 1–3) | Queue on the access road, 14 m apart. Each plans its own path, drives it within the shield's limits, and reports its pose and near-field detections, which update the shared map. A follower stops on its supervisor, on a remote stop, or when its permission lapses (section 2). |
| Fleet / site system | Supplies the yard boundary, berm lines and entrance (here `Site`), RTK corrections, the site radio network and the remote-stop infrastructure. Declares the yard an autonomous zone, logs, allows remote monitoring. Implicit in simulation. |

**Sequence.**

1. The convoy drives the access road (60 m × 16 m) to the entrance, a 16 m gap in the yard's west berm.
2. The leader pulls in to its scan pose, (12, 56) m, heading 0. The followers wait on the road.
3. The leader scans. The display shows free, occupied and unknown ground.
4. The driver picks a spot per follower, in queue order. The check accepts the pick or names why it is rejected.
5. Each accepted spot comes with a Hybrid A* path from that follower's position.
6. The followers drive. The coordinator gives each truck the last path sample it may reach, and every 0.5 s the
   policy picks HOLD, CREEP, GO, BACK_OFF or REPLAN for each follower.
7. A follower is parked when it stands within 0.5 m and 0.1 rad of its spot.
8. The episode ends when all are parked, on a collision, at 300 s, or after 40 s with no truck moving.

**Operational design domain (v1).**

| Item | v1 value or assumption |
|---|---|
| Yard | 100 m × 70 m pad, 2 m berms, one 16 m entrance. Flat (2D). |
| Trucks | Rigid, about 40 t payload class: 9.5 m × 3.8 m, wheelbase 4.2 m, full lock 0.48 rad (rear-axle turning radius 8.1 m). A leader plus 1–3 followers. |
| Speeds | At most 3.0 m/s forward and 1.5 m/s reverse. CREEP 0.7 m/s. Lateral acceleration capped at 0.8 m/s². |
| Obstacles | Static only: berms, 2–6 rock piles (2.5–6 m), 0–3 parked machines (8 × 3.5 m), 0–2 windrows (12–20 × 1.5 m). The scan misses rocks and windrows with probability 0.15. |
| People, light vehicles | None in the yard while followers move: site procedure, confirmed by the driver. Not modelled. |
| Dust | Moderate: lidar range derated to 60 m. Denser dust is outside the ODD: rescan or wait (design choice). |
| GNSS | An RTK fixed solution is needed for autonomous motion. On losing it, a controlled stop on INS and odometry (design choice). Simulation: exact poses. |
| Light | Day and night: lidar, radar and ultrasonics need no light, and the driver picks on the rendered scan. |
| Weather | Dry or light rain. Heavy rain, snow and fog are outside the ODD until tested. |
| Radio | V2V across the yard (up to about 150 m) and a separate remote-stop radio. Simulation: exact, no latency. |

**Out of scope for v1:** articulated haulers and truck-trailers; slopes and terrain; moving people and vehicles;
sensor noise, latency and localisation error; a real cab UI; rescans or a leader that moves after its scan; leaving
the yard; more than 3 followers.

## 2. Architecture

```mermaid
flowchart TD
  LS["Leader sensors: roof lidar, RTK-GNSS/INS<br/>sensors.initial_map"] --> MAP["Leader map: occupancy grid 0.5 m<br/>sensors.OccupancyGrid"]
  MAP --> V2V["V2V shared map + truck poses<br/>sensors.OccupancyGrid, sensors.reveal"]
  V2V --> UI["Driver UI pick<br/>spots.Driver (scripted driver in simulation)"]
  UI --> CHK["Spot safety check<br/>spots.validate"]
  CHK -->|rejected, with reason| UI
  CHK -->|accepted| HA["Hybrid A*, forward and reverse<br/>hybrid_astar.HybridAStar"]
  HA --> CO["Coordination: conflict zones + token shield<br/>coordination.conflict_zones, Coordinator"]
  CO --> POL["Policy: HOLD / CREEP / GO / BACK_OFF / REPLAN<br/>env.YardEnv + MaskablePPO, or env.scripted_actions"]
  POL -->|REPLAN| HA
  POL --> MPC["Kinematic LTV-MPC with cusps<br/>mpc.PathTracker"]
  CO -->|stop index| MPC
  MPC --> SUP["Safety supervisor: near-field stop<br/>episode.YardRun"]
  NF["Follower near field: radar + ultrasonic<br/>sensors.near_field"] --> SUP
  NF -->|missed obstacles| V2V
  SUP --> TR["Truck<br/>truck.TruckPlant, world.YardWorld"]
  TR --> NF
```

| Box | Module | Rate in simulation |
|---|---|---|
| Leader scan | `sensors.initial_map` | Once, at the scan pose |
| Shared map | `sensors.OccupancyGrid`, `reveal`, `ClearanceMap` (for the planner) | Reveal every `dt` (0.1 s) |
| Driver pick, spot check | `spots.Driver`, `spots.validate`, `spots.candidates` | Per pick, and when a spot is reassigned |
| Path planning | `hybrid_astar.HybridAStar.plan`, `collision_free` | Per accepted pick and per REPLAN |
| Coordination | `coordination.conflict_zones`, `Coordinator.update` | Zones per new path; tokens and stop limits every `mpc_step` (0.2 s) |
| Policy | `env.YardEnv`, trained by `nohesi/yard/train.py` (MaskablePPO); baseline `env.scripted_actions` | Every `decision_interval` (0.5 s) |
| Tracking | `mpc.PathTracker.command` | Every `mpc_step`, predicting 15 × 0.2 s = 3 s |
| Safety supervisor | `episode.YardRun` (true geometry within `near_range`) | Every `dt` |
| Plant, ground truth | `truck.TruckPlant`, `world.YardWorld` | Every `dt` |
| Loop, record, plots | `episode.YardRun`, `YardEpisode`, `viz.py`; parameters `config.py`, `scenarios.py` | |

**Where it runs on real trucks (design choice).**

- **Leader computer:** the map, the UI, the spot check, the coordinator and the policy. The policy is centralised: one
  network picks all three followers' actions.
- **Each follower:** plans its own path on the shared map, and runs its MPC and supervisor on board.
- **Stop limits are leases.** A follower with no fresh stop limit within a timeout (start: 1 s, to be set from the
  measured radio latency) comes to a controlled stop before its next gate, like a movement permission that expires.
  Losing the link stops a follower, and losing the leader stops all of them: safe, but costly in availability.

## 3. Sensor suite

**Market.** Retrofit kits on 40–100 t rigid haul trucks in quarries and mid-size mines: dusty sites, sometimes poor
GNSS in pits, cost-sensitive buyers.

**Context.** Public descriptions of commercial autonomous haulage combine high-precision GNSS with inertial sensing,
radar and lidar for obstacle detection, a central fleet management system and a remote stop. ISO 17757:2019, the
safety standard for autonomous machines in earth-moving and mining, requires that a machine can be stopped from a
safe remote distance, and that restarting motion after such a stop needs operator intervention. Retrofit autonomy
kits for quarry and mid-size mining fleets are on sale (for example from Pronto, SafeAI and Epiroc). Their public
descriptions range from camera-first kits for quarries to GNSS + lidar + radar stacks for mines. We did not verify
their specifications.

**Principles (design choices).** Radar carries the safety function in dust: at 77 GHz the wavelength is about 4 mm,
far larger than dust particles, so dust scatters it far less than light. Lidar maps the geometry (berms, rocks,
equipment). It degrades in dust and can return the edge of a dust cloud as a surface (Phillips et al., 2017), so the
simulation derates its range. GNSS/INS gives the pose, and dual antennas give heading at standstill, where every
follower plans from. Cameras serve people (the display, remote monitoring), not the safety function. The remote stop
has its own radio.

**Leader (human-driven: the scan and the driver display).**

| Sensor | Count | Placement | Purpose | Dust and vibration | Simulation |
|---|---|---|---|---|---|
| RTK-GNSS, dual antenna, with INS and wheel speed | 1 system, 2 antennas | Antennas as far apart as the roof allows; IMU on the frame near mid-truck | Scan pose and map frame; heading at rest | GNSS is unaffected by dust. Pit walls cause multipath and outages, which the INS bridges. Isolated, rated mounts. | Exact pose |
| 360° lidar (3D, multi-beam) | 1 (a 2nd if the body hides the rear arc) | High: a mast at the front deck or the canopy edge (on most rigid haul trucks the body's canopy covers the cab) | The yard scan | Range falls in dust; dust clouds can show as obstacles. Window cleaning (air knife or wiper), isolated mount. | 2D ray cast, 360°, 0.5° rays, 60 m, 6 m ahead of the rear axle, no self-occlusion; low obstacles missed with p = 0.15 |
| Surround cameras | 4–6 | Corners, rear | Display overlay, follower video, records | Lenses foul; work lights at night; not safety-relevant | Not modelled |
| Cab display (rugged touchscreen) | 1 | Cab | Scan view, picks, rejection reasons, follower status | Glare- and glove-proof | `spots.Driver` model |
| V2V radio | 1 | Roof | Map, spots, poses, stop limits | Unaffected by dust | Exact, no latency |
| Remote-stop transmitter | 1 | Driver's hand or dash | Stops every follower | Independent of V2V | Not modelled |
| Compute | 1 | Cab | Mapping, check, planning, coordination, policy | Rugged, sealed | Not modelled |

**Each follower (autonomous).**

| Sensor | Count | Placement | Purpose | Dust and vibration | Simulation |
|---|---|---|---|---|---|
| RTK-GNSS, dual antenna, INS, wheel odometry | 1 system, 2 antennas | As on the leader | Localisation, heading at rest, a controlled stop on losing the fix | Wheel slip on gravel limits odometry | Exact pose |
| Forward long-range 4D imaging radar (or lidar) | 1 | Front bumper or deck | Path check ahead, mainly for the haul road; a second view in the yard | Radar preferred in dust | Not separate (10 m near field) |
| Rear long-range 4D imaging radar (or lidar) | 1 | Rear, below the body | Reversing toward berms and trucks at 1.5 m/s | As above; keep clear of spill from the body | Not separate |
| Short-range corner radars | 4 | Four corners | 360° near field, the supervisor's main input | Robust in dust; multipath near metal and berms, weak returns from small low objects | `near_field`: 10 m, 8 rays; supervisor on true geometry within 10 m |
| Ultrasonic arrays | 8–12 | Rear bumper and corners, front bumper | The last metres reversing in; low objects at the bumper | A few metres' range; mud cake and noise; cheap | Part of the 10 m near field |
| Surround cameras | 4–6 | Corners, rear | Remote monitoring by the driver or a control room; records | As on the leader | Not modelled |
| V2V radio + remote-stop receiver | 1 + 1 | Roof | Map, poses, stop limits; a stop that does not depend on V2V | | Exact |
| Autonomy computer + safety controller | 1 + 1 | Cab or electronics box | Planner and MPC; a separate safety-rated controller for the supervisor, remote stop and drive-by-wire watchdog | Sealed, vibration-rated | Not modelled |
| Drive-by-wire (actuation, not sensing) | 1 set | Steering, throttle, service and park brakes, direction | Executes the MPC's commands | | `TruckPlant`: steer-rate limit, 0.4 s acceleration lag, direction selector |

An optional roof lidar per follower would add its own scans to the shared map; v1 does not model it. For V2V we
assume the site provides a private LTE/5G network or an industrial mesh (Rajant is one mesh vendor used on mine
sites). GNSS/INS receivers of this class come from vendors such as Trimble, NovAtel and Septentrio, and 360° lidars
from vendors such as Ouster and Hesai. These names are examples of categories, not recommendations.

**Cost.** Relative tiers per truck only, not prices:

| Tier | Components |
|---|---|
| Low | Ultrasonic arrays, corner radars, cameras |
| Medium | 4D imaging radars, radios, compute |
| Medium to high | 360° lidar; RTK dual-antenna GNSS with INS (the IMU grade drives the cost) |

Prices vary widely with volume, ruggedisation and supplier. We expect integration (drive-by-wire, mounting, wiring,
the safety case and certification) to cost more than the sensors. That is an expectation, not a measured figure.

## 4. Spot selection and safety rules

The display shows the grid, the leader, the queued followers and the candidate spots: a lattice every 3.0 m with four
headings, filtered by the static checks. The driver taps a spot and can rotate it. The default heading is nose-out
(section 5). `spots.validate` returns the first failing reason, cheapest check first. The driver cannot override a
rejection in v1 (design choice). In simulation, 30 % of `spots.Driver`'s picks are free clicks (`driver_bad_pick`),
which are usually unsafe and exercise the rejection path. Its other picks favour spots along a berm with the nose out
(3:1).

| Reason | Rule | Value | Why |
|---|---|---|---|
| `outside` | The footprint is not inside the yard shrunk by `spot_margin` | 1.0 m | Keeps the body off the berm toe and the yard edge |
| `entrance` | The footprint is within `keep_clear` of the entrance centre | 18 m | A parked truck must not block arrivals or the way out |
| `leader` | Within `spot_spacing` of the leader's footprint | 2.5 m | The leader has a driver in it |
| `too_close` | Within `spot_spacing` of another assigned spot | 2.5 m | Both trucks' clearance and parking error (below); access on foot |
| `occupied` | A cell within `spot_margin` of the footprint is OCCUPIED | 1.0 m | A known obstacle or berm. Checked before `unscanned`, since the lidar maps only an obstacle's near face. |
| `unscanned` | A cell within `spot_margin` is UNKNOWN (or off the grid) | 1.0 m | Unknown is not free (below) |
| `unreachable` | Hybrid A* finds no path from the follower's pose | 6,000 expansions | The path avoids the map, the leader, followers later in the queue where they stand, and the spots already assigned |

**Why unknown ground is not free.** A ray certifies only what it saw: free space up to its first hit. The ground
behind a parked machine or a rock pile was never seen, and a rock, a hole or a person may be there. Part of the yard is
simply out of range: from the scan pose about 69 % of the yard lies within the 60 m lidar range, before occlusion. If
unknown counted as free, the planner would route through unseen ground and learn about it from the 10 m near field: at
3 m/s about 3 s of warning, against a 9.9 m comfortable stop.

What remains is the missed obstacles (`miss_prob`: low rocks and windrows read as free). The followers' near field
reveals them into the shared map. A path they block makes the truck stop 4 m short (`BLOCK_BUFFER`) and wait for a
REPLAN, and a spot they make unsafe is reassigned by the driver (event 'reassigned').

**The margins add up.** The planner keeps 0.6 m between the body and anything non-free. The cover disks over-cover
the body by up to 0.34 m at the sides, and the clearance map subtracts half a cell diagonal (0.35 m). The 2.5 m spot
spacing covers both trucks' planning clearance (2 × 0.6 m) and park tolerance (2 × 0.5 m), with 0.3 m left. A truck
parked at the heading tolerance (0.1 rad) swings its front corners up to about 0.74 m, which can block its neighbour's
path ('unreachable' or a failed replan). Tune the spacing on the `too_close` and `replan_failed` counts.

## 5. Path planning (Hybrid A*)

**The search** (`hybrid_astar.py`, after Dolgov et al., 2010).

- **Nodes:** continuous rear-axle (x, y, yaw), keyed by 1.0 m cells, 5° yaw bins (72) and direction of motion. Each
  key keeps its cheapest state.
- **Primitives:** 2.0 m arcs at steer fractions −1, −0.5, 0, 0.5, 1 of `steer_max`, forward and reverse (10). A
  full-lock arc turns 14.2° and a half-lock arc (17.2 m radius) 6.7°, so every turning primitive leaves its parent's
  yaw bin. Samples at most 0.5 m apart are checked with the 4 cover disks (radius 2.24 m) against the clearance map,
  plus 0.6 m.
- **Cost (metre-equivalents):** arc length, × 1.5 in reverse; + 6.0 per change of direction (stop, shift, restart:
  about 6 s from and back to 1.5 m/s); + 0.3 per unit of steer fraction; + 0.3 per unit change of steer fraction.
- **Heuristic:** f = g + 1.5 h (`heuristic_weight`), h = max(Euclidean, 2D grid distance to the goal).
  - The grid distance is an 8-connected Dijkstra over about 1 m cells where the rear axle has at least `width/2`
    (1.9 m) of clearance. Nodes on cells it cannot reach are pruned.
  - Every collision-free pose meets that clearance, and reverse costs at least 1 per metre. The grid distance is
    therefore a lower bound up to the 8-connected grid's error (at most about 8 %).
  - The weight trades cost for speed: paths are within about 1.5 times the optimum.
- **Analytic shots** close the search. Every 4 expansions within 45 m of the goal, a Dubins shot at full-lock radius
  is tried forward, and in reverse (the forward Dubins path from the goal to the node, driven backward). The cheaper
  collision-free one ends the path exactly at the spot.
- **Failures:** 'start_blocked', 'goal_blocked', 'no_path' (exhausted), 'budget' (6,000 expansions).

**Geometry.** A forward U-turn at full lock sweeps the outer front corner through a 12.4 m radius. With margins that
needs about 26 m of clear width, so most manoeuvres between obstacles need a cusp. The lateral cap limits speed on a
full-lock arc to 2.5 m/s.

**Reverse-in parking (nose out).** The spot's heading decides the manoeuvre, and the planner ends at whatever pose it
is given. With the nose toward the entrance, a truck arriving from it usually drives past, stops at a cusp and
reverses in. This is the default because the reverse then happens at the best time: once, just after a fresh scan,
at 1.5 m/s or less, with every truck's pose known and the yard closed to traffic. Departure, later, when the yard may
have changed, is forward with the forward sensors leading. Along a berm the rear faces the known berm, and the rear
ultrasonics cover the last metres. Site rules differ: some mine traffic plans ask light vehicles to reverse-park so
they drive out forward, and others forbid backing into a space. The heading is therefore the driver's choice, and
nose-out is only the driver model's preference.

**Known limits and upgrades.**

- **Dubins shots are single-direction.** A Reeds-Shepp shot (Reeds and Shepp, 1990) can contain cusps, so it would
  close a forward-then-reverse park directly instead of building the cusp from primitives. Adding the obstacle-free
  Reeds-Shepp distance to the heuristic (as in Dolgov et al.) would also guide the final heading, which h ignores now.
- **Curvature jumps.** Lock to lock takes 3.8 s at 0.25 rad/s (11.5 m at 3 m/s). The tracker caps speed where the
  reference steering changes sharply, so the steering can follow within 4 m of path (`STEER_WINDOW` in `mpc.py`).
  That costs time on S-bends; a curvature-continuous smoother is the upgrade.
- **Resolution.** The 2 m step and 1 m cells can miss passages only a little wider than the truck needs. Refine them
  only where a test shows it.

## 6. Multi-agent coordination

**Path-velocity decomposition** (Kant and Zucker, 1986). Each truck's path is fixed, and coordination decides only how
far along it each truck may go. Truck-to-truck collisions reduce to conflicts along known paths.

**Conflict zones** (`conflict_zones`). Two samples, one on each path, conflict when a cover disk of one comes within
`zone_margin` (1.0 m) of a cover disk of the other. Connected runs of conflicting samples, merged where they overlap on
either path, give the zones: pairs of sample ranges [a_in, a_out] and [b_in, b_out]. Every conflicting pair of samples
lies inside one zone on both paths.

**Token shield** (`Coordinator.update`, every 0.2 s). Each zone has one token.

1. **Holding.** A truck whose progress lies in a zone's hold range, [in − gate, out] with gate 2.0 m (4 samples),
   holds its token. An owner outside that range loses it once it has passed out or no longer wants to move. A truck
   still rolling counts as wanting (`episode.py`), so no token is released while its owner brakes.
2. **Requesting.** A truck that wants to move requests a zone within gate + 6 m (`LOOKAHEAD`) of it. Requests are
   served in `Coordinator.order` (the queue order), so ties go to the earlier truck. Zones closer than a gate apart on
   a truck's path form a cluster, which is requested whole: all its tokens or none.
3. **Stopping.** A truck is stopped at the gate (in − gate − 1 sample) of the first cluster where another truck holds
   a token. The MPC holds 0.3 m short of that sample (`HOLD_MARGIN`).

**Safety argument.** No two trucks are ever inside the same zone. Outside zones their cover disks, and so their bodies,
are at least `zone_margin` apart. This holds under four conditions, which the tests must cover:

1. **Updates are frequent enough.** A truck moves at most 0.6 m between updates (3 m/s × 0.2 s), less than the 2 m
   gate, so no truck goes from outside a gate to inside a zone unseen.
2. **Tracking error stays small.** The argument is about path samples. It holds for the bodies only while the two
   trucks' tracking errors sum to less than 1.0 m. `track_tolerance` is 1.5 m, so between 0.5 m of error per truck
   and 'off_path', the supervisor is the backstop. Measure tracking error in the MPC tests, then tighten
   `track_tolerance` or derive `zone_margin` from it.
3. **A denied truck can stop before its gate.** Tokens are exclusive and kept while wanted, so a truck is denied at
   the latest at its request point, 6 m before its gate.
   - The tracker brakes harder than comfort (up to `brake_max`) when a stop is closer than the comfort distance.
   - From 3 m/s it needs 3.0 m at 2.0 m/s² with the lag, inside the about 5.2 m left (6 m less one sample and the
     hold margin).
   - Keep `LOOKAHEAD` above the service stopping distance at the top speed, and test denials at the request point.
4. **A parked truck holds its zones for good.** That is correct, since it stays there, but see the margin problem
   below.

The shield gives safety, not progress. A deadlock is safe and ends the episode as 'stalled' after 40 s.

**Margin consistency (to fix).** Measured with `conflict_zones` on straight test paths: beside a parked truck, a path
conflicts with its final pose up to about 1.7 m of body gap, and behind its rear bumper up to about 2.5 m (the end
disks reach 1.05 m past the bumpers). The planner keeps only 0.6 m. A path planned around a parked truck (a REPLAN, or
a first path past an earlier truck's spot) can therefore be blocked by it for good. Expect it as a source of stalls.
The recommended fix: when a truck parks, remove its path from the coordinator and treat it as a static obstacle. Any
remaining path it blocks under `collision_free` is marked blocked (stop 4 m short, REPLAN), the same flow as a
revealed obstacle, so one rule covers everything static. The alternative is to inflate parked trucks in planning to
the zone test's reach.

**Where deadlocks come from.** Because clusters are requested whole, no truck waits while holding a zone, so there is
no hold-and-wait deadlock, pairwise or cyclic, except through a truck that stands inside a zone:

- **Parked in a zone.** A's path was planned before B's spot existed and crosses it. If B parks first, A waits forever.
- **Halted inside a zone** by a HOLD, a block or a safety stop.
- **A zone put under a standing truck** by a replan, including the margin case above.
- **The queue:** a later truck's path passes where an earlier one still stands on the access road.

**What resolves them.** *Ordering*: do not let a truck park in a spot another truck's path still crosses; hold it
back, or let the deep spots go first. The policy can learn this, and the baseline enforces it with `priority_wait`.
*REPLAN*: plan around the truck in the way, now an obstacle where it stands, when another route exists. *BACK_OFF*:
retrace up to 6 m of one's own path to leave a zone and release it, with no new route needed.

**Upgrade path.** Space-time reservation (cooperative A*, Silver 2005) plans each truck against the others' reserved,
time-indexed footprints instead of fixed paths. Conflict-based search (Sharon et al., 2015, with car-like variants such
as CL-CBS) searches all paths jointly and resolves conflicts by constraints: complete for the discretised problem, but
costlier. Either would replace the fixed-path assumption, with the shield and supervisor kept as the safety layer.

## 7. Execution

**Why a kinematic model is enough here.** Lateral acceleration is capped at 0.8 m/s² (0.08 g). Gravel at μ about 0.6
(the haul test's h2 value) offers about 5.9 m/s², so at most about 14 % of the grip is used. Slip angles stay small,
and the kinematic bicycle at the rear axle is accurate: x' = v cos(yaw), y' = v sin(yaw), yaw' = v tan(steer) /
wheelbase. It has no singularity at v = 0 and works the same for v < 0. The road stack is the contrast: it needs a
dynamic bicycle with Fiala tyres and load transfer because the M3 uses up to about 1 g at 20–60 m/s, and that model
floors at 5 m/s, where the haul test broke. Load stability is not modelled. The rollover threshold of a laden high-CG
truck is commonly quoted at a few tenths of a g; v1 relies on the 0.8 m/s² cap, and a rollover constraint is on the
roadmap.

**LTV-MPC** (`mpc.PathTracker`, OSQP) tracks one constant-direction segment at a time. Its state is
[x, y, yaw, v, steer, a], where a is the plant's lagged acceleration: the 0.4 s lag is discretised exactly, so the
prediction starts from the acceleration the truck really has. Inputs are the commanded acceleration and steering rate.
The horizon is 15 × 0.2 s = 3 s (9 m of preview at 3 m/s, 4.5 m in reverse). Weights: position 4, heading 8, speed 1,
acceleration 0.5, steering rate 4. Steering angle, steering rate and the gear's acceleration range are hard bounds.
Speed bounds are soft, because the lagged acceleration carries the linear model through zero speed at a stop.

The speed profile is the minimum of the action's limit (0, 0.7 m/s or the yard limits), the segment's direction
limit, the lateral cap on curvature and the steering-rate cap. It brakes at `brake_comfort` (0.5 m/s²) to zero at a
cusp or the path's end and to 0.3 m short of the stop index. It brakes harder, up to `brake_max`, only when a stop is
closer than the comfort distance.

**Cusps and the direction selector.** No prediction mixes directions: the reference ends at the cusp with zero speed.
Once the truck stands within 0.4 m of the cusp below 0.05 m/s, the next segment starts, unless the stop index forbids
passing the cusp. `TruckPlant` enforces the gear: in +1 the truck never rolls backward, in −1 never forward, and in 0
it brakes to a stop.

Steering at standstill is allowed in v1, limited only by the 0.25 rad/s rate (lock to lock up to 3.8 s at a cusp). On
a real truck it wears tyres and ground. Whether to forbid it, and plan the curvature change while rolling, is an open
question.

**Fallback.** When OSQP fails, pure pursuit at the rear axle takes over: a 2.5–6 m lookahead, mirrored behind the
truck in reverse, with the reference's acceleration. A truck more than 1.5 m off its path brakes to a stop with its
steering held, and waits for a REPLAN.

**Safety supervisor** (`episode.YardRun`, every 0.1 s). It brakes at `brake_emergency` (3.5 m/s²) when the truck's
footprint, or its path ahead in the direction of motion, comes within 0.3 m of an obstacle in near-field range (missed
ones too), the leader or another follower. It looks ahead v²/(2 × `brake_max`) + 1 m, which is 3.25 m at 3 m/s.

Stopping distances on `TruckPlant`, with its 0.4 s lag, from the moment braking is commanded:

| From | Comfort, 0.5 m/s² | Service, 2.0 m/s² | Emergency, 3.5 m/s² |
|---|---|---|---|
| 3.0 m/s forward | 9.9 m, 6.3 s | 3.0 m, 1.8 s | 2.0 m, 1.2 s |
| 1.5 m/s reverse | 2.7 m, 3.3 s | 0.9 m, 1.1 s | 0.6 m, 0.7 s |
| 0.7 m/s (CREEP) | 0.7 m, 1.7 s | 0.25 m, 0.6 s | 0.17 m, 0.5 s |

The emergency stop from 3 m/s (2.0 m) fits inside the 3.25 m look-ahead. A real system adds sensing and actuation
latency, still to be measured: each 0.1 s adds 0.3 m at 3 m/s. In v1 the supervisor sees the true geometry (no false
negatives, no false stops), and while a truck backs off it projects a straight line instead of the retraced path.
Every onset is logged ('safety_stop') and costs reward, so the policy learns to stay out of near misses rather than
rely on them.

## 8. RL formulation

**What is left to learn.** The planner, shield and supervisor make every allowed action safe, so the policy learns
order and timing: who goes first through a shared stretch, when to creep, when to back off and let another truck by,
when to replan. This replaces the road env's 51 actions per car (5 lane commands × 3 speeds × 3 lean offsets, plus 6
overtakes, one slot of which is FOLLOW). The lean and overtake variants are gone.

**Actions** (per follower slot, `MultiDiscrete([5, 5, 5])`, masks `(15,)` from `episode.YardRun.masks`).

| Action | Effect | Allowed when |
|---|---|---|
| HOLD | Stop at 0.5 m/s² and stay stopped; wants no zones once stopped | Always; the only action for an absent or parked follower |
| CREEP | Follow the path at up to 0.7 m/s, either direction; requests zones | The follower has a path and is not parked |
| GO | Follow the path at the yard limits (3.0 forward, 1.5 reverse, lateral cap); requests zones | As CREEP |
| BACK_OFF | Retrace up to 6 m of the path just driven at creep speed, then rejoin the same path where it stopped; continues while chosen | At least 1 m of the path driven, and the retrace does not enter a stretch another truck holds; continuing is allowed |
| REPLAN | Hybrid A* from where the truck stands, around the leader, all other trucks where they stand and higher-priority unparked spots; a new spot from the driver if the spot became unsafe | Stopped (below 0.2 m/s), 2 s since its last replan, not backing off, not parked |

A truck stops when a revealed obstacle blocks its path (4 m short), when it is off its path, or when it has no path.
Recovering is the policy's REPLAN: nothing replans on its own.

**Observation** (`env.py`): 152 floats, 3 slots × 50 plus 2 global. Slot k is follower k in queue order; an absent
follower is all zeros. Features are scaled to about ±1, and VecNormalize standardises them.

| Group | Size | Contents |
|---|---|---|
| Own state and path | 16 | Present, parked, speed, steer, direction of the current segment, remaining path, distance to the segment's end (cusp or spot), distance and heading error to the spot, lateral error, time since it last moved, safety stop, blocked, can replan, backing off, cusps left |
| Near field | 8 | Free distance along 8 rays in the truck frame, / 10 m |
| Each other follower | 2 × 11 | Present, relative x and y in the own frame, sin and cos of the relative heading, speed, parked, shares a zone ahead, own distance to that zone's gate, the other's distance to its gate (negative inside), the other holds that zone's token |
| Leader | 4 | Relative x and y, sin and cos of the relative heading |
| Global | 2 | Time / duration, followers / 3 |

No reward term depends on the clock, so try the time input off as an ablation. The road training turned its clock
input off (`OBS_CLOCK`).

**Reward per decision** (`YardEnvConfig` ye1; all start values).

| Term | Weight | Per |
|---|---|---|
| Time | −0.02 | Unparked follower-second |
| Progress | +0.02 | Metre of path completed |
| Parked | +5 | Follower parked |
| Collision | −20 | Collision (ends the episode) |
| Safety stop | −0.5 | Supervisor onset |
| Replan | −0.2 | Successful REPLAN |
| Failed replan | −0.5 | REPLAN without a path |
| Back-off | −0.02 | Metre reversed in BACK_OFF |
| Stall | −5 | Episode stalled |

**How the terms balance.** 1 m of progress equals 1 s of one truck's time: a truck at 3 m/s nets +0.04/s, and a
waiting truck pays 0.02/s. The park bonus equals 250 s of one truck waiting. Backing off d metres and driving them
again costs 0.02 d and earns it back, so only the time is lost.

**Two checks before the first long run.**

- **The collision weight is too low.** It should exceed what an episode can still cost: up to 0.02 × 3 × 300 = 18 of
  time, plus up to 15 of park bonuses. At −20, a policy facing a long wait gains almost nothing by avoiding a crash.
  The road training kept its crash penalty above a whole episode's return for the same reason. Start at −50 or lower,
  and raise `clip_reward` with it.
- **Progress can be farmed.** ye1 counts only decreases of the remaining length. A REPLAN that adds L metres is
  earned back when driven, and it pays once L exceeds about 15 m (0.02 L, minus the time at 3 m/s, minus 0.2). The
  fix is potential-based progress, which also charges increases. An episode's total progress then equals the initial
  remaining length, whatever the route.

**Episode end.** Terminated on 'parked' (all followers) or 'collision'. Truncated on 'timeout' (300 s) or 'stalled'
(40 s with no truck moving, −5). MaskablePPO bootstraps truncated episodes from the final state's value
(`sb3_contrib/ppo_mask/ppo_mask.py`), so stalling saves no time cost.

**Scripted baseline** (`env.scripted_actions`), the benchmark as the scripted overtaker was for the road policy. Queue
order is priority. A follower GOes along its path, and HOLDs within 1 m of the gate of a zone it shares with a
higher-priority truck that has not yet passed it (`priority_wait`). Blocked, off its path or without one, it drives up
to the block, then REPLANs once stopped. It never uses BACK_OFF or CREEP, so the policy's gain from those is measured
against it.

**Evaluation protocol.**

- **Seeds.** Training seeds are drawn at 2000 or above. During training, `nohesi/yard/train.py` evaluates the policy
  (and the baseline, once) on seeds 1000–1019 and keeps the best model by success rate, then mean duration. The final
  report uses 100 seeds that no model was trained or selected on (start: 1100–1199).
- **Runs.** `yard1` and `yard1_clear` (perfect scan), split by follower count (1, 2, 3), with policy and baseline on the
  same seeds.
- **Metrics:** success rate (all parked, no collision); collisions (must be 0); time to park all, median and p90,
  paired on seeds both solve; safety stops; replans and failed replans; back-off metres; metres driven against the
  first planned path length; stalls and timeouts.
- **Comparison.** The paired difference per seed with its standard error, as the road README reports. Picks rejected
  and spots reassigned come from the driver model and the scan, so check that they match between policy and baseline.
- **Records.** A GIF (`viz.animate`) of every stall and every collision.

**Training settings to start with** (`YardTrainConfig` defaults in `nohesi/yard/train.py`; all start values).

| Setting | Start | Note |
|---|---|---|
| Algorithm | MaskablePPO (sb3-contrib) | Invalid actions are masked, not penalised |
| Workers | 8 SubprocVecEnv | The road runs used 24 on the training machine |
| Rollout | `n_steps` 256 per env (2,048 decisions per update) | 128 s of yard time per env |
| Minibatch, epochs | 512, 10 | |
| Learning rate | 3e-4, linear to 1e-5 | |
| γ, GAE λ | 0.99, 0.95 per 0.5 s decision | A 50 s horizon; try 0.995 if parking 100 s ahead is under-credited |
| Entropy | 0.01 | 5 actions; lower later, as the road runs did (0.001) |
| Clip, target KL | 0.2, none | Add 0.1 if updates jump |
| Network | 256 × 256 tanh, separate policy and value | |
| Normalisation | VecNormalize on observations and rewards, clip 10 | Raise the reward clip with the collision weight |
| Budget | 2M decisions; evaluations every 50k, checkpoints every 100k | About 7,000–20,000 episodes at 100–300 decisions each |
| Scenarios | `yard1`, 1–3 followers per seed | Optionally start on `yard1_clear` |

## 9. Validation plan and roadmap

**Stage 1: simulation tests** (pytest, `tests/test_yard_*.py`), before any training.

- **Plant:** no rolling against the gear; the stopping distances in section 7.
- **Scan:** unknown behind obstacles; a miss rate close to `miss_prob` over many seeds; reveals within 10 m.
- **Spots:** one constructed case per rejection reason.
- **Hybrid A*:** exact goal reach; a reverse-in park with a cusp; a sealed goal gives 'goal_blocked' or 'no_path';
  every sample collision-free; the budget respected.
- **MPC:** straight, S-curve and cusp paths, forward and in reverse: lateral error well under 0.5 m (set the bound
  from the first runs), parking within 0.5 m and 0.1 rad, and the fallback on a forced solver failure.
- **Coordinator:** crossing paths under random request timings never give two holders of a zone; a denial at the
  request point at full speed stops before the gate (section 6, condition 3); the parked-truck margin case; deadlocks
  resolved by REPLAN and BACK_OFF.
- **Supervisor:** a missed obstacle on the path, driven at 3 m/s; the truck stops before contact.
- **Whole episodes:** the baseline on 100 seeds with 0 collisions; every stall inspected in a GIF; the same seed gives
  the same episode.

**Stage 2: a physics simulator with a truck model**, CARLA through the repo's bridge or another one. CARLA ships a few
large vehicles but, as far as we know, no haul truck, so a custom vehicle asset may be needed. Add what v1 leaves out:
radio latency and dropouts, RTK fix, float and outage errors, lidar dust attenuation and false returns, radar
multipath, actuation delay and grades. Run the same seeds and compare tracking error, stopping distances and success
rate with stage 1.

**Stage 3: a closed test pad.**

1. One retrofitted follower with a safety driver in the cab and the remote stop: CREEP only, then GO.
2. Foam or cone obstacles, some placed after the scan to exercise the near field.
3. Two followers on crossing paths, then three.
4. Widen the ODD step by step: dust, night, GNSS masking.

Before the pad, check the design against ISO 17757 (autonomous machine safety), ISO 19014 (functional safety of
earth-moving machinery) and ISO 16001 (object detection systems).

**Extensions (roadmap).**

- **Articulated haulers:** articulated-steering kinematics, the articulation angle in the planner state and the MPC.
- **Logging truck-trailers:** the trailer angle in the state, hitch-angle limits, and reversing that is unstable
  without feedback on that angle.
- **Slopes:** a terrain model from a 3D scan, grade limits in planning, a rollover constraint and braking on grades.
- **Moving people and light vehicles:** prediction, classification in the supervisor, yielding rules.
- **Latency and localisation error:** zone and spot margins derived from an error budget.
- **A real driver UI:** map rendering, touch picks, confirmation, rejections explained in words.
- **Wider scope:** leaving the yard; rescans and a moving leader; more than 3 followers (an input that scales with the
  truck count); fleet system integration.

**Risks and open questions.**

- **The 2D scan cannot see negative obstacles** (holes, soft ground, a pad's drop-off edge) or judge height. A 3D
  lidar with ground segmentation is needed before a pad test.
- **Dust is conservative, except for low objects.** Dust returns read as occupied, which is safe but removes spots.
  Missed low objects are the real risk, and the near field must catch them.
- **Safety rests on tracking error staying below `zone_margin`/2** (section 6). If real tracking is worse, raise the
  margins.
- **The policy must stay out of the safety case.** The case rests on the shield, the supervisor and the remote stop,
  and the learned component only picks among actions they made safe. Whether certifiers accept this argument is open.
- **Transfer and the real world.** Real yards will differ from the generator: safety does not depend on the policy,
  but its gain over the baseline may shrink. The kinematic model may fail on loose or wet ground or when reversing
  uphill (wheel slip), and steering at standstill (section 7) is unresolved. Real false stops, for example from radar
  multipath near berms, cost time and through the reward could teach the policy to avoid berms.
- **The central coordinator is a single point of failure.** A lost link stops everything: safe, but costly.
- **The core question:** does the policy beat the prioritized baseline enough to justify RL, or would a better
  scripted coordinator (CBS, reservations) do as well? The evaluation protocol is built to answer it.

## References

- Dolgov, Thrun, Montemerlo, Diebel. *Path planning for autonomous vehicles in unknown semi-structured environments.*
  IJRR, 2010 (Hybrid A*).
- Dubins. *On curves of minimal length with a constraint on average curvature.* American Journal of Mathematics, 1957.
- Reeds, Shepp. *Optimal paths for a car that goes both forwards and backwards.* Pacific Journal of Mathematics, 1990.
- Kant, Zucker. *Toward efficient trajectory planning: the path-velocity decomposition.* IJRR, 1986.
- Silver. *Cooperative pathfinding.* AIIDE, 2005.
- Sharon, Stern, Felner, Sturtevant. *Conflict-based search for optimal multi-agent pathfinding.* Artificial
  Intelligence, 2015.
- Phillips, Guenther, McAree. *When the dust settles: the four behaviors of LiDAR in the presence of fine airborne
  particulates.* Journal of Field Robotics, 2017.
- Huang, Ontañón. *A closer look at invalid action masking in policy gradient algorithms.* 2020 (MaskablePPO).
- ISO 17757:2019. *Earth-moving machinery and mining: autonomous and semi-autonomous machine system safety.*
