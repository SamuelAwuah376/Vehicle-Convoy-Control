# Yard parking stack: module interfaces

This is the contract between the `nohesi/yard/` modules. It is written first so the modules can be built in
parallel. Everything is in SI units. Truck states are referenced at the **rear axle centre**: `[x, y, yaw, v]`, with
`v < 0` when reversing. `yaw` is always the truck's heading, never the direction of motion.

## Already in place: don't change without updating this file

- **`config.py`**: `Truck`, `Sensors`, `Site`, `Planning` and `YardScenario`, frozen dataclasses. Read their docstrings and defaults.
- **`truck.py`**:
  - `corners(truck, states) -> (M, 4, 2)`: footprint corners.
  - `disk_offsets(truck, n=4) -> (offsets (n,), radius)`: the cover disks along the centreline.
  - `disk_centres(truck, states) -> (M, n, 2)`.
  - `body_centre(truck, states) -> (M, 2)`.
  - `TruckPlant(truck, dt).step(state, steer, accel, steer_cmd, accel_cmd, gear) -> (state, steer, accel)`:
    - `gear` is +1, -1 or 0.
    - `accel` is along the truck's +x, so speeding up in reverse is a negative accel.
    - In gear +1 the truck never rolls backward; in gear -1 never forward; in gear 0 it brakes to a stop.
  - `wrap(angle)`.
- **`world.py`**:
  - `box`, `obb`.
  - `Obstacles(corners (K, 4, 2), kind list, missable (K,) bool)`. Berms are obstacles of kind `'berm'`.
  - `YardWorld(scenario, seed, n_followers=None)`, with attributes:
    - `.obstacles`;
    - `.leader` (4,): parked at its scan pose for the whole episode;
    - `.states` (n, 4), `.steer` (n,), `.accel` (n,);
    - `.n`, `.rng`, `.scenario`.
  - `YardWorld.footprints()`.
  - `YardWorld.collisions() -> [(follower, 'obstacle' | 'leader' | 'truck', other)]`.
- **`paths.py`**: `Path(x, y, yaw, direction)`.
  - Properties `s`, `states`, `length`, `cusps`.
  - Methods `segments()`, `segment_of(k)`, `nearest(xy, lo, hi)`, `remaining(k)`, `sub(i0, i1)`, `retrace(k, i_to)`, `Path.concat(parts)`.
  - `direction[k]` is the motion from sample k to k+1. A cusp is where it flips. Samples are at most `Planning.path_sample` (0.5 m) apart.

## `sensors.py`: leader scan, shared map, near field

```python
UNKNOWN, FREE, OCCUPIED = 0, 1, 2          # uint8 cell states

class OccupancyGrid:
    x0: float; y0: float; res: float; cells: np.ndarray   # (ny, nx) uint8; cell (iy, ix) centre = (x0+(ix+.5)res, y0+(iy+.5)res)
    @classmethod
    def for_site(cls, site, sensors) -> 'OccupancyGrid'   # covers the yard, the access road and the berms + 2 m, all UNKNOWN
    def copy(self) -> 'OccupancyGrid'
    def cell_of(self, xy) -> (iy, ix, inside)              # xy (..., 2) -> int arrays and a bool mask
    def mark_polygon(self, corners, value)                 # rasterise a convex polygon (4, 2): every cell whose centre is inside
    def known_free(self, corners, margin=0.0) -> bool      # every cell within `margin` of the polygon is FREE
    def clearance(self, extra=()) -> 'ClearanceMap'        # distance from each cell to the nearest non-FREE cell (UNKNOWN and
                                                           # OCCUPIED both block); `extra` polygons (K, 4, 2) are rasterised
                                                           # OCCUPIED in a copy first (the leader, parked or waiting trucks)

class ClearanceMap:
    dist: np.ndarray; x0: float; y0: float; res: float
    def at(self, xy) -> np.ndarray     # (...) clearance in m at points (..., 2): nearest-cell value minus res*sqrt(2)/2
                                       # (conservative), 0 outside the grid

def initial_map(world, sensors, rng) -> (OccupancyGrid, missed (K,) bool)
    # Leader roof lidar ray-cast (360 deg, sensors.lidar_resolution, sensors.lidar_range, origin lidar_mount ahead of the
    # leader's rear axle): FREE along each ray up to its first hit, OCCUPIED at the hit. Obstacles that are missable and
    # draw rng.random() < miss_prob are `missed`: they neither block rays nor show. The access road is marked FREE (the
    # convoy drove it), and the leader's and followers' own footprints FREE (they are known through V2V, not mapped).

def reveal(grid, world, missed, revealed, sensors) -> list[int]
    # Missed obstacles within sensors.near_range (footprint clearance) of any follower get rasterised OCCUPIED in the
    # shared grid; `revealed` (K,) bool is updated in place; returns the newly revealed indices.

def near_field(world, i, sensors, n=8) -> np.ndarray
    # (n,) free distance (m, capped at near_range) from follower i's footprint edge along n rays in the truck frame
    # (0 = straight ahead, counter-clockwise), against the true obstacles (missed ones too), the leader and other followers.
```

## `hybrid_astar.py`: forward and reverse path planning

```python
@dataclass
class PlanResult:
    path: Path | None
    reason: str        # '' (found), 'start_blocked', 'goal_blocked', 'no_path' (search exhausted), 'budget' (max_expansions)
    expansions: int
    time_s: float

def collision_free(truck, planning, clearance, states) -> np.ndarray   # (M,) bool: every cover disk's clearance >= its
                                                                        # radius + planning.clearance
class HybridAStar:
    def __init__(self, truck, planning)
    def plan(self, clearance, start, goal) -> PlanResult
        # start, goal: (3,) rear-axle [x, y, yaw]. The path ends EXACTLY at goal (an analytic Dubins shot, forward or
        # reverse, closes the search). Primitives: forward and reverse arcs at Planning.steer_fractions of steer_max,
        # Planning.step long. Costs: reverse_cost, switch_cost, steer_cost, steer_change_cost. Heuristic: max(Euclid,
        # 2D grid distance from the goal over cells where the rear axle has clearance >= width/2, computed with
        # scipy.sparse.csgraph.dijkstra). Path samples <= Planning.path_sample apart.
```

## `mpc.py`: path tracking, forward and reverse

```python
class PathTracker:
    def __init__(self, truck, planning)
    def set_path(self, path, progress=0)    # new path (or a replan / a back-off retrace); progress = start sample
    progress: int                           # sample index the truck is at (monotone within a segment)
    arrived: bool                           # at the path's end, standing still
    def command(self, state, steer, accel, speed_limit, stop_index=None) -> (steer_cmd, accel_cmd, gear, info)
        # speed_limit >= 0 (m/s): 0 means stop at Truck.brake_comfort. The speed profile respects speed_limit,
        # v_forward_max / v_reverse_max per segment, lat_accel_max on curvature, brake_comfort to a stop at every cusp,
        # at the path end and at stop_index (a sample the rear axle must not pass; None: no limit), and accel_max.
        # A linear time-varying MPC (OSQP, kinematic bicycle at the rear axle, Planning.mpc_horizon x mpc_step,
        # weights w_pos, w_yaw, w_speed, w_accel, w_steer_rate) tracks the reference of the current segment. At a cusp
        # it stops, then shifts (gear flips) and starts the next segment. If OSQP fails: a pure-pursuit fallback.
        # info: {'progress', 'segment', 'lateral_error', 'heading_error', 'status'}, where status is 'tracking', 'cusp',
        # 'holding' (at stop_index), 'arrived', 'off_path' (lateral error > track_tolerance) or 'fallback'.
```

## `coordination.py`: multi-truck conflict zones and the token shield

```python
@dataclass
class Zone:
    a: int; b: int                 # follower indices, a < b
    a_in: int; a_out: int          # sample range on a's path whose footprint conflicts with b's path (zone_margin)
    b_in: int; b_out: int          # ... on b's path

def conflict_zones(truck, planning, a, path_a, b, path_b) -> list[Zone]
    # Connected runs of conflicting samples (cover disks, zone_margin), merged where they overlap on either path.

class Coordinator:
    def __init__(self, truck, planning, n)
    def set_path(self, i, path, progress)   # (re)computes zones of truck i against every other truck with a path; a truck
                                            # without a path (None) conflicts with nothing
    def update(self, progress, wants) -> stop_index list
        # progress: (n,) current sample per truck; wants: (n,) bool, the truck wants to move along its path now.
        # A zone is held by a truck while its progress lies in [in - gate, out], gate = gate_buffer/path_sample samples.
        # A truck may enter only if the other truck is outside its side of the zone (before in - gate, or past out)
        # and does not hold it. Simultaneous requests: the lower index in self.order wins (self.order is the priority
        # list, default range(n)). Returns per truck the last sample it may reach now (gate start of the first zone it
        # may not enter; None for no limit).
    order: list
    def blocking(self, i) -> list[int]      # trucks currently holding a zone that stops truck i
    def priority_wait(self, i, progress) -> bool
        # The scripted baseline's rule: truck i must wait while any higher-priority truck (earlier in self.order) shares
        # a zone with it that the higher truck has not yet passed (progress < out).
```

## `spots.py`: driver picks and the safety check (integrator)

```python
def validate(scenario, grid, pose, assigned, others) -> str   # '' = accepted, else the reason the pick is rejected
def candidates(scenario, grid) -> (M, 3)                       # lattice of candidate spot poses in known-free ground
class Driver:                                                  # scripted leader driver
    def pick(self, follower, world, grid, assigned) -> (pose, reason_list)
```

## `episode.py` and `env.py`: the loop, the RL env and the record (integrator)

```python
ACTIONS = ('HOLD', 'CREEP', 'GO', 'BACK_OFF', 'REPLAN')   # per follower slot (3 slots; absent or parked: HOLD only)

@dataclass
class YardEpisode:                 # what viz.py draws
    scenario; seed: int; n: int
    time: np.ndarray               # (T,)
    states: np.ndarray             # (T, n, 4)
    steer: np.ndarray              # (T, n)
    actions: np.ndarray            # (T, n) int: the action in force (index into ACTIONS)
    safety_stop: np.ndarray        # (T, n) bool: the safety supervisor is braking that truck
    obstacles: Obstacles; missed: np.ndarray; revealed_at: np.ndarray   # (K,) time revealed (inf: never)
    leader: np.ndarray             # (4,)
    grid0: OccupancyGrid           # after the leader's scan
    spots: list                    # per follower: final assigned (3,) pose or None
    picks: list                    # [{'time', 'follower', 'pose', 'accepted', 'reason'}]
    paths: list                    # [(time, follower, Path)]: every plan and replan, in order
    events: list                   # [{'time', 'follower', 'kind', 'detail'}], kind: 'parked', 'collision', 'replan',
                                   # 'replan_failed', 'safety_stop', 'reassigned', 'back_off'
    end_reason: str                # 'parked', 'collision', 'timeout' or 'stalled'
```

### The RL env (`env.py`)

```python
@dataclass(frozen=True)
class YardEnvConfig:
    version: str = 'ye1'
    scenario: str = 'yard1'        # key into nohesi.yard.scenarios.YARD_SCENARIOS
    # reward weights (see env.py), ...

class YardEnv(gymnasium.Env):
    def __init__(self, cfg=YardEnvConfig(), seed_pool=None, rng_seed=None)
    action_space = MultiDiscrete([5, 5, 5])                   # ACTIONS per follower slot
    observation_space = Box(-inf, inf, (D,), float32)
    def reset(self, *, seed=None, options=None) -> (obs, info)       # seed None: next seed_pool entry, else a draw >= 2000
    def step(self, action) -> (obs, reward, terminated, truncated, info)
        # info['episode_metrics'] at episode end: {'seed', 'success' (all parked), 'parked', 'followers', 'collision',
        # 'end_reason', 'duration_s', 'replans', 'replans_failed', 'safety_stops', 'back_off_m', 'picks_rejected',
        # 'reassigned', 'path_length_m', 'driven_m'}
    def action_masks(self) -> np.ndarray                      # (15,) bool, flattened per slot
    def episode(self) -> YardEpisode

def make_env(cfg, rank=0, base_seed=0, seed_pool=None)        # thunk for SubprocVecEnv
def scripted_actions(env) -> np.ndarray                       # (3,) the prioritized baseline's actions for the current decision
```

## `viz.py`

```python
def animate(ep, path, fps=10, stride=None)   # top-down GIF of a YardEpisode
def plot(ep, path)                           # one PNG: the scan, the spots, the paths, the trucks' tracks
```
