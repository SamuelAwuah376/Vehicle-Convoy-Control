# Perception and team belief: design (draft for approval)

Status: **proposal, not implemented.** Nothing here changes code until it is approved. Every part sits behind a flag
that is off by default, so scenario v8, v9 and the milestone 8 results reproduce exactly.

## 1. Goal

Give the team a realistic perception stack and plan on what it believes, not on the simulator's truth:

- a sensor suite on each team car: two lidars (front, rear) and four radar + camera pairs around the car;
- real processing algorithms on that sensor data (clustering, box fitting, data association, Kalman tracking);
- one **belief shared by the three cars**: every traffic car as a mean and a covariance, plus an existence
  probability and a probability over the lane it is heading to, updated by an IMM-UKF;
- planners (MPC, gap and overtake checks, action masks, fallback) that plan on that belief and use its uncertainty;
- a PPO policy that observes the belief, including its uncertainty, so it can learn to act on it.

The work uses two tiers that share one processing and tracking pipeline:

| | Tier A: our sensor simulator | Tier B: CARLA sensors |
|---|---|---|
| Use | training and the 50-seed evaluations | validation and calibration of Tier A |
| Speed | about real time × 25 environments | a few frames per second, offline |
| Lidar | 2D ray casting against car boxes | `sensor.lidar.ray_cast`, 3D point cloud |
| Radar | detection-level physical model | `sensor.other.radar` |
| Camera | detection-level (projected boxes) | RGB frames plus a real image detector |

## 2. What reads the true state today

Everything reads one `Snapshot` (`nohesi/world.py`): `traffic_s`, `traffic_d`, `traffic_v`, `traffic_vd`, `traffic_id`
and `traffic_target_lane`, the true lane each traffic car is heading to.

| Consumer | Where | What it reads |
|---|---|---|
| MPC obstacles | `coordination/centralized.py::_obstacles` through `prediction.traffic_prediction` | positions, speeds, lateral speed, target lane |
| Gap, overtake, follow and lead checks | `intentions.py` (`lane_gap`, `overtake_check`, `follow_check`, `_corridor`) | positions, speeds, target lane |
| Action masks | `rl/env.py::action_masks`, `_lane_mask`, `_overtake_options` | the checks above |
| Policy observation | `rl/env.py::_obs`: 8 traffic slots × 7 features per car, plus `merge_possible` | positions, speeds, lateral speed, target lane |
| Stuck reward and escapes | `rl/env.py::stuck`, `escapes` | positions and speeds |
| Fallback planner | `planning/fallback.py` through `vehicle_ahead_incl_merging` | positions, speeds, target lane |
| Collisions, scoring, traffic behaviour | `world.py`, `scoring/`, `traffic.py` | **stay on the true state** |

`traffic_target_lane` is the clearest leak: the planner knows a traffic car's intent the moment it decides to change
lane. Under perception, the IMM's lane probabilities estimate it instead.

## 3. Architecture

```
               true World state (traffic, team)
                 |                         \
                 |                          +--> collisions, scoring, traffic reactions   (unchanged, true state)
                 v
   Tier A sensor simulator, per team car                   [nohesi/perception/sim/]
   (front and rear lidars, 4 radars, 4 cameras; own RNG)
                 | timestamped raw measurements, each tagged with the sensor pose
                 v
   Per-sensor processing                                    [nohesi/perception/processing/]
   lidar: motion compensation, clustering, L-shape box fit
   radar: point detections with closing speed
   camera: box to bearing and ground-plane range
                 | detections with covariances
                 v
   Team tracker: one centralized belief                     [nohesi/perception/tracking/]
   IMM-UKF in road coordinates, association, existence, track management,
   delayed measurements, the three cars' data fused once
                 | Belief (means, covariances, p_exist, lane probabilities)
                 v
   Belief adapter                                           [nohesi/perception/belief.py]
   +-- perceived Snapshot (means)  --> every consumer in section 2, unchanged code paths
   +-- uncertainty (σ per car and horizon step) --> MPC margins, gap checks, masks, policy observation
```

Each team car's own state stays true in the first version (localization error is section 11, question 3). Teammates
come over V2V: each car's own state, delayed by the link latency. The tracker uses those V2V states to remove
teammates from the detections, so they never become traffic tracks.

## 4. Sensor suite (proposed defaults, all configurable)

Mount positions are in the car's frame (x forward from the rear axle, y left). The car is 4.79 m × 1.90 m.

| Sensor | Mount | Field of view | Range | Rate | Noise and resolution | Latency |
|---|---|---|---|---|---|---|
| Front lidar | front bumper, 0.6 m high | 120° | 200 m | 10 Hz | 0.1° beam spacing, range σ 3 cm, 2 % dropout | 50 ms |
| Rear lidar | rear bumper | 180° | 120 m | 10 Hz | 0.2° beam spacing, range σ 3 cm, 2 % dropout | 50 ms |
| Front radar | front, facing ahead | ±45° (±9° beyond 70 m) | 250 m | 20 Hz | range σ 0.25 m, bearing σ 0.3°, closing speed σ 0.1 m/s | 30 ms |
| Rear radar | rear, facing back | ±60° | 150 m | 20 Hz | range σ 0.25 m, bearing σ 0.5°, closing speed σ 0.1 m/s | 30 ms |
| Left and right radars | side mirrors, facing out | ±75° | 80 m | 20 Hz | range σ 0.3 m, bearing σ 1.0°, closing speed σ 0.15 m/s | 30 ms |
| Front camera | windscreen, 1.4 m high | 60° | ~150 m (detection limit) | 20 Hz | 1920 px wide, box-corner σ 2 px | 80 ms |
| Rear camera | rear window | 90° | ~80 m | 20 Hz | 1280 px wide, σ 2 px | 80 ms |
| Left and right cameras | side mirrors | 100° | ~60 m | 20 Hz | 1280 px wide, σ 2 px | 80 ms |

The simulation step is 0.05 s (20 Hz), so radars and cameras report every step and the lidars every other step.

Coverage note: the policy observes traffic out to 250 m ahead and 80 m behind (`EnvConfig.ahead` and `behind`).
Beyond about 200 m only the front radar sees a car; beyond 250 m nothing does. That limit is intended, and because the
three cars share one belief, the leading car's sensors extend what its teammates know.

## 5. Tier A sensor models (`nohesi/perception/sim/`)

All models read the true state at the measurement time, draw from a perception-only `numpy.random.Generator`
(seeded from the episode seed and a fixed tag), and return measurements stamped with their sensor time. The world's
own random draws, and so the traffic, are unchanged whether perception is on or off.

**Lidar (ray casting).** A 2D planar scan at the mount height against every traffic and team car's box. The road is
flat and every car is the same height, so extra vertical layers would only repeat points; a layer count sets how many
points each hit produces, to keep point counts realistic.
- Vectorized slab test for rays against boxes; the nearest hit wins, so cars hidden behind other cars are occluded
  correctly.
- Range noise, beam dropout, and a range limit tied to reflectivity (a car at 200 m returns with the specified
  probability; detection fades beyond).
- Scan skew: each beam is cast at its own time within the 100 ms sweep, against car positions interpolated to that
  time. At 40 m/s relative speed that is up to 4 m of smear; the processing must compensate for it.
- Road edges return nothing (no guardrail model in the first version).

**Radar (detection level).** Each visible car produces 1–3 reflection points near its visible corners and edges.
- Each point gives range, bearing and closing speed (the radial component of relative velocity), with noise.
- Detection probability depends on received power, set by the car's reflectivity (10 dBsm) and range⁻⁴, and on how
  much of the car is visible. Radar sees part of the way past an occluding car: a hidden car is detected with reduced
  probability instead of none.
- Resolution: two points closer than 0.4 m in range and 4° in bearing merge into one, at their mean.
- False alarms: Poisson clutter, about 2 per scan per radar, spread uniformly over the field of view.

**Camera (detection level).** Each car's 3D box is projected through a pinhole model with the camera's mount and
resolution, and drawn against the other boxes to find what is visible.
- Detected if its image height is at least 15 px and at least 30 % of it is visible; base detection probability 0.95.
- Box corners get 2 px of noise. Range then comes from where the box meets the road. On a flat road its error grows
  with range squared (σ_r ≈ r² σ_px / (f h)), the usual weakness of a single camera.
- One false box per 10 frames per camera.

## 6. Processing and tracking (`processing/`, `tracking/`)

**Lidar processing** (front and rear, every 0.1 s). The input is a list of beams: angle, range, and time within the
sweep.
1. **Motion compensation (de-skewing).** At 40 m/s the car travels 4 m during one sweep. Every point is moved into
   the car's frame at the scan end time `t_e`, assuming constant speed and yaw rate over the sweep (constant twist):
   - **Lever arm first.** The point is moved from the sensor frame to the car frame with the mount transform, so a
     turn's swing of a bumper-mounted sensor is included.
   - **Car motion.** Over `τ = t_e − t_i` the car turns `Δψ = r τ` and moves
     `Δp = [v_x sinΔψ − v_y(1 − cosΔψ), v_x(1 − cosΔψ) + v_y sinΔψ] / r`. This is the SE(2) exponential of the twist;
     for |r| < 1e-4 the straight-line limit `[v_x, v_y] τ` is used.
   - **Move the point:** `p_e = R(Δψ)ᵀ (p_i − Δp)`.
   - **Inputs.** `v_x`, `v_y` and `r` come from the car's state (true in the first version, from the localization
     filter once localization error is added), taken at the scan's mid time.
   - **When constant twist is not enough.** During hard braking or a fast steering input, the poses at the start,
     middle and end of the sweep are interpolated instead (translation linear, heading by angle). This is what LIO-type
     systems do with their IMU propagation.
   - **Other cars.** Their motion is unknown before tracking, so each cluster is stamped with its mean beam time
     instead of `t_e`. Optional second pass: after association, the cluster's points are de-skewed with the track's
     predicted velocity and the box is refitted.
2. **Into the road frame,** using the mount position.
3. **Remove known objects.** Points inside a teammate's box (from its V2V state, predicted to the beam time) are
   dropped. Points beyond the road edges are dropped too.
4. **Segmentation.** Walk the beams in scan order and split wherever the range jumps by more than an adaptive
   threshold (adaptive breakpoint detection, Borges and Aldon, 2004). The threshold grows with range, because at
   200 m neighbouring 0.1° beams are already 0.35 m apart. Two segments are merged into one car when a nearer car
   splits it and together they fit one car's size.
5. **Box fitting.** Search-based L-shape fitting (Zhang et al., 2017) with the variance criterion, over headings
   within ±20° of the road (a car on a highway never points further off).
   - **Anchor on what is seen.** The visible corner, or the visible face when only one side shows, is accurate. The
     known car size fixes the hidden sides, so the centre is the anchor plus half the size along them.
   - **Refinement (optional):** point-to-line ICP of the cluster against the fitted box outline (model-to-scan
     registration), with RANSAC line fits to reject stray points such as mirrors or noise. The L-shape search gives
     the starting pose and ICP removes the search's 1° heading steps.
   - **Covariance.** It depends on the point count, the range noise and which sides were seen. A car ahead seen only
     from behind has a sharp along-road position and lateral centre but a weak heading.
6. **Output:** box centre `(x, y)` and heading, with a covariance and a time stamp.

**Radar processing** (4 radars, every 0.05 s). The input is detections: range, bearing and closing speed. Both tiers
start here; neither simulates the radar's signal processing.
1. **Into the car frame,** using the mount position.
2. **Stationary rejection.** A stationary object's closing speed is fixed by the car's own speed and the bearing
   (`-v cos θ`). Detections that match it within 3σ are flagged as not vehicles: clutter, and in CARLA guardrails and
   signs. This is the standard Doppler test.
3. **Clustering.** One car gives 1–3 points, but the tracker's assignment is one measurement to one track. So DBSCAN
   on position and closing speed groups the points (radius about 1.5 m and 1 m/s). A slow and a fast car side by side
   stay apart because of their speed difference.
4. **Output:** one measurement per cluster: range, bearing and closing speed of the cluster centre, with the noise
   inflated by the cluster's spread.
5. **Measurement model.** Radar points lie on the near surface of a car, not its centre. So the predicted measurement
   is the point of the track's box nearest the sensor, not its centre; using the centre would bias range by 2 m on a
   car ahead.

**Camera processing** (4 cameras, every 0.05 s). The input is 2D boxes: pixel extents, class and score. In Tier A the
simulator gives them; in Tier B the image detector does.
1. **Remove known objects.** Boxes overlapping a teammate's projected box (from V2V) are dropped.
2. **Main method: update in the image itself.** The tracker does not convert a box into a 3D position first.
   - **Prediction.** For each track, the 8 corners of its 3D box (known car size, at the track's position and
     heading) are projected through the camera, `u = K [R | t] X`, and the predicted box is their pixel extent
     `[u_left, u_right, v_top, v_bottom]`.
   - **Update.** The UKF compares that with the detected box, with pixel noise as the measurement noise.
   - **Why.** The geometry then handles itself: bearing comes out sharp and range rough, the box's near face is
     modelled correctly (the box bottom is the near bumper, not the centre), and side cameras that see a car's flank
     get the right shape.
3. **For new tracks and gating, a 3D position from one box.** A new car has no track to project yet, so its first
   position comes from two geometric cues, fused by inverse variance:
   - **Ground-plane back-projection.** Cast the ray through the bottom-centre pixel, `d = R_cam K⁻¹ [u, v, 1]ᵀ`, from
     the camera centre `C` at height `h`, and intersect it with the road: `P = C + (−h / d_z) d`. That gives `(x, y)`
     in one step. Its range error grows with r²: `σ_r ≈ r² σ_px / (f h)`. It is very sensitive to camera pitch:
     `δr ≈ r² δθ / h`.
   - **Known size.** `r = f H / h_px`, with the car height H = 1.45 m. Height is used rather than width because
     width changes with the viewing angle. The error is relative, `σ_r / r = σ_px / h_px`, and does not depend on
     pitch.
   - **Example, front camera (f ≈ 1,660 px, h = 1.4 m), car at 50 m:**
     - both cues are about ±2 m from pixel noise alone;
     - a 0.2° pitch error adds about 6 m to the ground-plane cue, so the size cue carries most of the weight unless
       pitch is known well.
   - The fused range and the bearing `θ = atan((u_c − c_x) / f)` give the first position, with its covariance
     propagated from the pixel noise through these formulas (first-order Jacobian). The covariance is long along the
     ray and narrow across it.
4. **Pitch.** The road is flat, but the car pitches under braking and acceleration: about 1° per 0.5 g. The first
   version adds a pitch error from the car's longitudinal acceleration to Tier A's camera model. Without it, the
   cameras would look more accurate than real ones.
5. **Several views.** The three cars' cameras see the same car from different places, and one camera sees it from
   different places over time. Because the image updates are bearing-sharp, the tracker effectively triangulates
   bearings across the team and over time without a separate step.
6. **Tier B options, with real images:**
   - the same 2D detector plus the same image-plane update;
   - or a learned monocular 3D detector (SMOKE, FCOS3D or MonoFlex) that regresses the 3D box directly, used as a
     position measurement with an error calibrated from CARLA's ground truth.

   The geometric method of Mousavian et al. (2017) sits between the two: the network regresses the size and heading,
   and the 3D box is placed so its projection fits the 2D box tightly.

**How the measurements meet the tracker.** Each sensor type has its own measurement model from the track state
`[s, d, v_s, v_d]` to that sensor's frame:

| Sensor | Model output |
|---|---|
| Lidar | box centre and heading |
| Radar | range, bearing and closing speed of the nearest box point |
| Camera | the track's 3D box projected into the image: left, right, top and bottom pixel edges |

The UKF applies each measurement at its own time stamp (late ones are replayed, as described under Team fusion
below). Each sensor has its own χ²
gate, and its field of view, range and the occlusion feed the existence update: a track that is outside a sensor's
view or hidden is not penalized for being missed by it.

**Tracker.**
- **State, in road coordinates:** `[s, d, v_s, v_d]` along and across the road, which is what the planner and
  prediction already use. Measurements are made in the sensor frame; the UKF handles the nonlinear conversion between
  the two.
- **IMM with three motion models:**
  - lane keep: `d` pulled back to the lane centre, `v_s` constant with process noise;
  - change left;
  - change right (both: lateral motion toward the next lane centre).

  The switching probabilities come from configuration, defaulting to a typical lane-change rate rather than the
  simulator's own `lane_change_prob`. The model probabilities give `P(target lane)` for each track; that replaces the
  true `traffic_target_lane`.
- **Association:** global nearest neighbour (Hungarian assignment, `scipy.optimize.linear_sum_assignment`) with a 99 %
  χ² gate on the Mahalanobis distance. JPDA is an option if dense traffic shows wrong pairings.
- **Existence:** an IPDA-style existence probability per track. Its detection probability is **occlusion-aware**: a
  track predicted to be hidden behind another car is expected to go unseen, so missing it does not lower its existence.
  That keeps the occluded cars of dense traffic tracked, with a covariance that grows while they stay hidden.
- **Track management:** new tracks start from unassociated lidar boxes or radar points; a track is confirmed after 3 of
  5 detections, and deleted when its existence falls below 0.1 or it leaves the region of interest. Only confirmed
  tracks reach the planner.
- **Starting a track from one detection.** Measured parts of the state come from the detection, with its covariance
  mapped into road coordinates by an unscented transform. Unmeasured parts come from a wide prior.

  | Born from | Position | Speed along the road | Lateral speed |
  |---|---|---|---|
  | Lidar box | box centre | prior | prior: 0 ± 1 m/s |
  | Radar cluster | range and bearing | from the closing speed: the car's own speed − closing speed / cos θ (prior instead when \|cos θ\| < 0.3: a car alongside shows almost no closing speed) | prior: 0 ± 1 m/s |
  | Camera box | ground-plane and size cues (section 6, camera) | prior | prior: 0 ± 1 m/s |

  - **Speed prior.** The median speed of the confirmed tracks in that lane, ± 5 m/s. If the lane has none, the car's
    own speed ± 10 m/s. It is never the scenario's configured lane speeds, which a real car would not know. The
    following updates collapse it within a few scans.
  - **IMM.** The models start at lane keep 0.9 and change left or right 0.05 each.
  - **Existence.** It starts by source: 0.5 for a lidar box with at least 10 points, 0.3 for other lidar boxes and
    camera boxes, 0.2 for radar (the most clutter).
- **No duplicates across sensors.** Births happen at the end of each fusion cycle. Unassigned detections from
  different sensors and cars that gate with each other are grouped first, so one car seen by several sensors gives
  one new track. Two tracks that stay inside each other's gate for 5 cycles are merged.
- **Episode start.** The belief must not start empty: at t = 0 the planner would see an empty road. Instead each
  episode begins with a 1 s synthetic pre-roll.
  - Every traffic car's state is run backwards 1 s at constant speed.
  - The sensors and tracker run on that history, then the episode starts at t = 0 with confirmed tracks.
  - The world itself does not run, so the traffic and its random draws are unchanged.
- **Team fusion:** one centralized tracker, as the coordinator is already centralized. All three cars' measurements go
  in once, each with its own sensor pose and time, so the update is exact and nothing is counted twice. Measurements
  that arrive late (V2V 100 ms plus sensor latency) are handled by keeping 0.3 s of filter history and re-running the
  updates from the late measurement's time.
- **Ablation:** each car tracks with only its own sensors. This measures what sharing is worth.

## 7. The belief and its interface (`belief.py`)

```python
@dataclass(frozen=True)
class Belief:
    time: float
    id: np.ndarray          # (n,) track ids, stable while a track lives
    mean: np.ndarray        # (n, 4) [s, d, v_s, v_d]
    cov: np.ndarray         # (n, 4, 4)
    p_exist: np.ndarray     # (n,)
    lane_prob: np.ndarray   # (n, lanes) probability over the lane each car is heading to
    last_seen: np.ndarray   # (n,) time of the last associated measurement
    confirmed: np.ndarray   # (n,) bool

    def predict(self, times) -> tuple:   # means and covariances at `times` (the MPC horizon), per IMM model
    def snapshot(self, truth) -> Snapshot:
        # confirmed tracks as traffic_* (means; traffic_target_lane = the most likely lane),
        # team fields from the car's own state and V2V, collisions and done from the true state
```

`Snapshot` gains optional fields `traffic_sig_s`, `traffic_sig_d`, `traffic_sig_v` and `traffic_p_exist`; `None`
means a certain state, as now. With perception off, the world's own snapshot passes through unchanged: the first
build step must reproduce the 50-seed references byte for byte.

## 8. Planning on the belief

**Prediction with uncertainty.** `traffic_prediction` gains a belief variant that returns each track's predicted
mean and σ along the road and across it for every horizon step, from its model-mixed covariance.
- A track whose lane-change probability is above a threshold (default 0.2) becomes two obstacles, one keeping its lane
  and one changing, capped at one extra obstacle per car.
- Lane-change uncertainty is therefore handled by alternatives rather than by one very wide box.

**MPC margins (no solver rebuild).** `planning/constraints.py::obstacle_constraints` builds its linear rows in Python
with fixed gaps (`length + margin_long`, `width + margin_lat`). Per obstacle and per horizon step these become

```
long gap = length + margin_long + min(k σ_s(t), cap_long)
lat gap  = width  + margin_lat  + min(k σ_d(t), cap_lat)
```

- `k` is the chance level: `Planner.belief_k`, where 0 means off and 2 ≈ 97.7 % one-sided. The caps (defaults 3 m
  and 0.5 m) stop a far, uncertain car from closing a whole lane.
- These rows are already filled at run time, so both the QP and nonlinear MPC take the change without a rebuild.
- Tracks below `Planner.belief_min_exist` (default 0.5) are left out.

**Checks and masks.** `lane_gap`, `overtake_check`, `follow_check`, `_corridor`, `merge_possible`, the stuck rule and
the fallback planner read the perceived snapshot. The gap checks add the same `k σ` to the gaps they require. So the
action masks come from the belief and never show the policy the true state.

**Unseen space (optional, phase 3).** For a lane change, a target-lane stretch that no sensor covers can hold a
phantom car at the edge of visibility, driving at that lane's typical speed. This is the usual occlusion-aware
planning rule. It is off unless the first results show cars changing into blind spots.

## 9. PPO on the belief

The partially observed problem becomes an ordinary one once the observation summarizes the belief, and PPO applies
unchanged. The Kalman filter already holds the memory, so the policy stays feed-forward. (sb3-contrib has no
maskable recurrent PPO, so a recurrent version would mean writing our own.)

1. **Belief features.** For each car and each of its 8 traffic slots, five values are added:
   - `log σ_s`, `log σ_d`, `log σ_v`;
   - `p_exist`;
   - the time since the car was last seen (0–1 s, clipped).

   That is 3 × 8 × 5 = 120 inputs, **appended at the end**, taking the observation from 312 to 432 values. This is
   the same pattern as the e5 and follow features. The existing slot features read the belief means; the "present"
   flag means a confirmed track.
2. **Warm start from milestone 8.** `rl/train.py::load_policy_weights` already handles an appended observation: it
   copies the old weights, gives the new inputs zero weights, and extends the normalization statistics. At step 0 the
   policy acts exactly like milestone 8 on the belief means, then learns to use the uncertainty.
3. **Masks** come from the belief (section 8).
4. **Reward** is unchanged: it scores what really happened (true passes, crashes, stuck time).
5. **Privileged critic (optional).** The value network also sees the true slot features and the cars the belief is
   missing; the policy sees only the belief. This usually speeds up learning when the state is partly hidden, and it
   is legitimate because only the policy runs at test time. It needs a custom SB3 policy class (about a day's work).
   Added only if step 1's fine-tune learns slowly.
6. **Expected behaviour.** The policy can learn to act on uncertainty: hold back from an uncertain gap and commit to
   a sharp one. It learns to gather information (for example moving sideways to see past a truck) only to the extent
   that this pays off in score. There is no explicit search over future beliefs, and nothing in this stack needs one.

## 10. Tier B: CARLA validation (`nohesi/perception/carla_sensors.py`)

- Attach the same sensor suite (section 4) to the team cars in the CARLA bridge (`nohesi/carla/world.py`): the
  ray-cast lidar with matching beams and range, the radar, and the RGB cameras. Add instance segmentation cameras for
  ground-truth labels.
- Known quirk: this CARLA build advances 1–3 frames per `world.tick()` while rendering. Measurements are stamped from
  each sensor's own frame time, never from the tick count.
- Lidar: the same pipeline after one extra step, ground removal (RANSAC plane fit), because CARLA's point cloud is 3D.
- Radar: CARLA's detections (altitude, bearing, depth, speed) go into the same tracker.
- Camera: a real image detector on the RGB frames, chosen in phase 5. Licences matter: torchvision models are
  BSD-licensed, while Ultralytics YOLO is AGPL.
- Calibration: run the same episodes through both tiers. Compare detection rates against range, error distributions
  and tracking metrics, then tune Tier A's noise and detection parameters until the two match. That is the evidence
  that Tier A's results carry over.

## 11. Configuration and reproducibility

- `Scenario.perception: Perception`, with sub-configurations `Lidar`, `Radar`, `Camera`, `Tracker` and `Sharing` and
  `enabled=False` by default. It goes into `config_hash` and the saved `config.json` (`write_config`).
- Planner flags: `belief_k` (0 = off), `belief_min_exist`, `belief_cap_long`, `belief_cap_lat` and
  `belief_lane_split`. `Planner.version` is bumped when any of them is on.
- `EnvConfig.belief_obs: bool` adds the 120 belief inputs.
- The perception RNG is separate from the world RNG (section 5).
- Results runs and 50-seed json files record the perception configuration.

## 12. Evaluation

**Perception metrics.** On 50 v9 seeds, inside the observation window:
- GOSPA, split into position error, missed cars and false tracks;
- NEES and NIS consistency, i.e. whether the filter's covariances are honest. This matters because the planner trusts
  them;
- track identity switches;
- detection probability against range, per sensor.

**Driving metrics.** The usual 50-seed paired comparisons on v9, plus v8, against milestone 8 with the true state:
score, crashes, fallbacks, stuck time and train time.

**Experiments:**

| Step | Set-up | Question |
|---|---|---|
| E0 | perception on, ideal sensors (no noise, unlimited range, no occlusion) | Does the belief track the truth (means equal within tolerance, results nearly unchanged)? |
| E1 | milestone 8 policy and planner on the belief means | What does realistic perception cost? |
| E2 | E1 plus uncertainty margins in the MPC and checks (`belief_k` 1, 2, 3) | Does planning on the uncertainty recover it? |
| E3 | E2 plus belief observation, fine-tuned from milestone 8 | Does the policy learn to use the uncertainty? |
| Ablations | no sharing; no radar; no lidar; cameras only; latency × 2; noise × 2 | Which parts matter? |

## 13. Speed budget

Training throughput is set by the MPC solves. Rough sensor-side estimate per simulation step:
- lidar: 3 cars × about 2,100 beams × about 30 boxes in range ≈ 190k slab tests every other step, ≈ 1–2 ms
  amortized in vectorized numpy;
- radar and cameras: under 1 ms;
- tracker: about 60 tracks, a few hundred measurements, a batched UKF over tracks plus Hungarian assignment on gated
  blocks, ≈ 3–6 ms.

Target: no more than 25 % added time per environment step, profiled in phase 2. If Python loops dominate, the hot
paths can be vectorized further. `numba` is not installed today; adding it would be a new dependency and needs your
approval.

Measured in phase 2 (v9 seed 1000-1001, 15 s, one process): perception added about 95 ms per step to a 31 ms step
with the numpy code, mostly Python overhead in per-cluster, per-box and per-track loops. numba (0.61.2, approved
2026-11-19, the optional dependency `perception`) now compiles the lidar box fit (L-shape search, RANSAC, point-to-line
ICP and covariance in one kernel), the camera box processing and the tracker's prediction, gating, update, coverage
and existence: 41 ms added per step. The stages, numpy -> compiled (ms per step): lidar processing 43.9 -> 9.6,
camera processing 11.1 -> 2.2, tracker 18.2 -> 7.3; unchanged: lidar sim 9.1, camera sim 8.8, radar sim 4.2, radar
processing 3.7. The compiled kernels match the numpy code to rounding (tests/test_perception_jit_*.py), and
NOHESI_JIT=0 runs the numpy code. The simulators followed: the lidar sweep's ray casting (one kernel, its truth and
boxes computed once per firing for all six sweeps), the camera projection, occlusion and frame assembly, and the radar
view, visibility, reflection points, merge and noise, with every random draw left in Python in the same order and
size, so the compiled simulators give bit-identical scans and frames on this machine; TruthHistory.at builds each
snapshot's state once and matches the traffic by id without a Python loop (identical results). Then (v9 seed
1000-1001, 15 s, a lighter machine load than above, ms per step): lidar sim 1.2, radar sim 1.1, camera sim 0.6 (22 in
all before), lidar processing 5.9, radar processing 2.3, camera processing 1.0, tracker 4.7: about 17 ms of perception
on a 24 ms step. Still above the 25 % target: the Python bookkeeping around the processing and tracker kernels.

## 14. Phases (each ends with a check-in; training only on your go)

| Phase | Work | Done when |
|---|---|---|
| P0 | `nohesi/perception/` skeleton, `Perception` configuration, `Belief` and the Snapshot adapter, pass-through mode | the test suite passes; the v8 and v9 50-seed references reproduce byte for byte |
| P1 | Tier A sensor models, plus a viewer overlay (fields of view, points, detections) | unit tests (ray casting against analytic hits, occlusion, detection rates); within the speed budget |
| P2 | processing, IMM-UKF, association, existence, team fusion, delayed measurements | NEES and NIS within 95 % bounds on 50 v9 episodes; GOSPA reported; E0 passes |
| P3 | belief prediction, MPC margins, belief-based checks and masks; E1 and E2 | 50-seed v9 and v8 results for E1 and E2 |
| P4 | belief observation, warm start, optional privileged critic; E3 fine-tune | a training batch you approve; 50-seed results |
| P5 | Tier B CARLA sensors, the camera detector, calibration of Tier A | Tier A vs Tier B comparison report; Tier A parameters updated |

Tests go in `tests/test_perception_*.py`: ray casting, L-shape fits on synthetic clusters, UKF and IMM consistency on
synthetic tracks, association edge cases, occlusion-aware existence, delayed-measurement replay, and the byte-identical
pass-through.

## 15. Questions for your approval

1. **Sensor layout and specifications (section 4):** radar + camera pairs at front, rear, left and right, with a
   long-range front radar. Or would you rather have four corner radars and keep the cameras front, rear, left and right?
2. **Cameras in Tier A are detection-level only;** algorithms on actual pixels run in Tier B (CARLA). Is that enough,
   or is the camera processing itself part of the research?
3. **Each team car's own state stays true** in the first version, with teammates over V2V at 100 ms. Should
   localization error (GNSS and IMU) come later as its own step?
4. **One centralized team tracker** (exact fusion, matching the centralized coordinator), with per-car tracking as an
   ablation. Or do you want decentralized fusion (each car its own filter, covariance intersection between cars) as
   the main design?
5. **Turn signals:** traffic cars have none today. A visible signal would help the IMM, but signalling before a lane
   change changes the traffic model, so v9 would change. Leave it out?
6. **Phantom cars for unseen space** (section 8): only if E1 or E2 shows cars changing into blind spots?

## 16. Out of scope for now

Weather and lighting effects on Tier A sensors, curved roads (the road is straight today), road-edge and guardrail
detection, lane-marking perception, sensor faults, and the communication link dropping out (it is a fixed delay).
