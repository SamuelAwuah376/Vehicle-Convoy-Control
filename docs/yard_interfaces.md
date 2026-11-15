# Yard parking stack: module interfaces

This is the contract between the `nohesi/yard/` modules. It was written first, so the modules could be built in
parallel, and now records their interfaces as built. `docs/yard_parking_design.md` gives the reasons. Everything is in
SI units. Truck states are referenced at the **rear axle centre**: `[x, y, yaw, v]`, with `v < 0` when reversing.
`yaw` is always the truck's heading, never the direction of motion. Planned paths keep `yaw` continuous (unwrapped
from the start's yaw); the plant and `YardWorld.states` keep it wrapped to [−π, π).

## Foundations: don't change without updating this file

- **`config.py`**: `Truck`, `Sensors`, `Site`, `Planning` and `YardScenario`, frozen dataclasses validated in
  `__post_init__`. Read their docstrings and defaults. `Truck.front_overhang` and `Truck.min_radius` (rear-axle radius
  at full lock, 8.1 m) are properties.
- **`scenarios.py`**: `YARD_SCENARIOS = {'yard1': yard1, 'yard1_clear': yard1_clear}`, factories of `YardScenario`
  (`yard1_clear`: `miss_prob` 0).
- **`truck.py`**:
  - `corners(truck, states) -> (M, 4, 2)`: footprint corners, counter-clockwise from rear right.
  - `disk_offsets(truck, n=4) -> (offsets (n,), radius)`: the cover disks along the centreline (`DISKS = 4`).
  - `disk_centres(truck, states, n=4) -> (M, n, 2)`.
  - `body_centre(truck, states) -> (M, 2)`.
  - `TruckPlant(truck, dt).step(state, steer, accel, steer_cmd, accel_cmd, gear) -> (state, steer, accel)`:
    - `gear` is +1, -1 or 0.
    - `accel` is along the truck's +x, so speeding up in reverse is a negative accel. Driving effort is capped at
      `accel_max`, braking at `brake_emergency`, with a first-order lag (`accel_lag`).
    - In gear +1 the truck never rolls backward; in gear -1 never forward; in gear 0 it brakes to a stop at
      `brake_max`.
    - The returned yaw is wrapped.
  - `wrap(angle)`: to [−π, π).
- **`world.py`**:
  - `KINDS = ('berm', 'rock', 'equipment', 'windrow')`; `box(x0, x1, y0, y1)`, `obb(cx, cy, length, width, yaw)`.
  - `Obstacles(corners (K, 4, 2), kind list, missable (K,) bool)`, with `len()`. Berms are obstacles of kind `'berm'`.
  - `berms(site)`, `leader_state(site) -> (4,)`, `follower_starts(truck, site, n) -> (n, 4)`,
    `generate(scenario, rng, n_followers) -> (Obstacles, starts)`.
  - `YardWorld(scenario, seed, n_followers=None)`, with attributes:
    - `.obstacles`;
    - `.leader` (4,): parked at its scan pose for the whole episode;
    - `.states` (n, 4), `.steer` (n,), `.accel` (n,);
    - `.n`, `.rng`, `.seed`, `.scenario`.
  - `YardWorld.footprints(states=None) -> (n, 4, 2)`.
  - `YardWorld.collisions() -> [(follower, 'obstacle' | 'leader' | 'truck', other)]`.
- **`paths.py`**: `Path(x, y, yaw, direction)`.
  - Properties `s`, `states`, `length`, `cusps`; `len(path)` is the sample count.
  - Methods `segments()`, `segment_of(k)`, `nearest(xy, lo=0, hi=None)`, `remaining(k)`, `sub(i0, i1)`,
    `retrace(k, i_to=0)`, `Path.concat(parts)`.
  - `direction[k]` is the motion from sample k to k+1 (the last repeats N−2's). A cusp is where it flips. Samples are
    at most `Planning.path_sample` (0.5 m) apart.

## `sensors.py`: leader scan, shared map, near field

```python
UNKNOWN, FREE, OCCUPIED = 0, 1, 2          # uint8 cell states

class OccupancyGrid:                       # dataclass(eq=False)
    x0: float; y0: float; res: float; cells: np.ndarray   # (ny, nx) uint8; cell (iy, ix) centre = (x0+(ix+.5)res, y0+(iy+.5)res)
    shape                                                  # property: cells.shape
    @classmethod
    def for_site(cls, site, sensors) -> 'OccupancyGrid'   # covers the yard, the access road and the berms + 2 m, all UNKNOWN
    def copy(self) -> 'OccupancyGrid'
    def cell_of(self, xy) -> (iy, ix, inside)              # xy (..., 2) -> int arrays clipped into the grid (always safe
                                                           # to index with) and the mask of the points on the grid
    def mark_polygon(self, corners, value)                 # rasterise a convex polygon (V, 2): every cell whose centre is inside
    def known_free(self, corners, margin=0.0) -> bool      # every cell within `margin` of the polygon is FREE (False if that
                                                           # ground reaches off the grid)
    def clearance(self, extra=()) -> 'ClearanceMap'        # distance from each cell to the nearest non-FREE cell (UNKNOWN,
                                                           # OCCUPIED and the grid's outside all block); `extra` polygons
                                                           # (K, 4, 2) are rasterised OCCUPIED in a copy first (the leader,
                                                           # parked or waiting trucks); this grid is unchanged

class ClearanceMap:                        # dataclass(eq=False)
    dist: np.ndarray; x0: float; y0: float; res: float    # treat dist as read-only: at() caches a padded copy of it
    def at(self, xy) -> np.ndarray     # (...) clearance in m at points (..., 2): the value of the cell holding the point
                                       # minus res*sqrt(2)/2 (conservative), clipped at 0; 0 outside the grid and at NaN

def lidar_origin(world, sensors) -> np.ndarray   # (2,) the roof lidar: lidar_mount ahead of the leader's rear axle

def initial_map(world, sensors, rng) -> (OccupancyGrid, missed (K,) bool)
    # Exactly K rng.random() draws, one per obstacle in index order (missable or not): an obstacle is missed when it is
    # missable and its draw is < miss_prob; missed obstacles neither block rays nor show. 360 deg of rays,
    # lidar_resolution apart, from lidar_origin() out to lidar_range: FREE along each ray up to its first hit, OCCUPIED
    # at the hit. The access road (berm cells excluded) is FREE (the convoy drove it), and so are the leader's and the
    # followers' footprints (known through V2V, not mapped). Hit cells are written last: an obstacle seen in a cell wins.

def reveal(grid, world, missed, revealed, sensors) -> list[int]
    # Missed obstacles not yet revealed within sensors.near_range (footprint clearance) of any follower are rasterised
    # OCCUPIED in the shared grid; `revealed` (K,) bool is updated in place; returns the newly revealed indices, ascending.

def near_field(world, i, sensors, n=8) -> np.ndarray
    # (n,) free distance (m, clipped to [0, near_range]) from follower i's footprint edge along n rays from its body
    # centre in the truck frame (0 = straight ahead, counter-clockwise), against the true obstacles (missed ones too),
    # the leader and the other followers.
```

## `hybrid_astar.py`: forward and reverse path planning

```python
@dataclass
class PlanResult:
    path: Path | None
    reason: str        # '' (found), 'start_blocked', 'goal_blocked', 'no_path' (search exhausted, or the start
                       # unreachable on the grid heuristic), 'budget' (max_expansions)
    expansions: int
    time_s: float

def collision_free(truck, planning, clearance, states) -> np.ndarray   # (M,) bool for states (M, >= 3) or (3,): every
                                                                        # cover disk's clearance >= its radius + planning.clearance

def dubins_words(start, goal, radius) -> dict          # {word: (3,) segment lengths} of the Dubins words that apply
def dubins_sample(start, word, lengths, radius, sample) -> np.ndarray   # (N, 3) poses, exact arc integration
def dubins(start, goal, radius, sample, direction=1) -> Path | None     # shortest Dubins path; direction -1: the reverse
                                                                        # path (the forward curve driven backward)

class HybridAStar:
    staging = (8.0, 16.0)           # m: straight reverse legs of the staged shot (class attribute)
    reverse_turn_max = math.pi/2    # rad of total heading change for a reverse shot that may end the search
    def __init__(self, truck, planning)
    def plan(self, clearance, start, goal) -> PlanResult
        # start, goal: (>= 3,) rear-axle [x, y, yaw, ...]. The path ends EXACTLY at goal in x and y, and in yaw modulo
        # 2 pi (the yaw is continuous from the start's). Primitives: forward and reverse arcs at Planning.steer_fractions
        # of steer_max, Planning.step long. Costs: reverse_cost, switch_cost, steer_cost, steer_change_cost. Heuristic:
        # max(Euclid, 2D grid distance from the goal over ~1 m cells where the rear axle has clearance >= width/2,
        # scipy.sparse.csgraph.dijkstra). Path samples <= Planning.path_sample apart.
        # The search ends with the cheapest collision-free analytic shot (every analytic_every-th expansion within
        # analytic_range, turning radius Truck.min_radius): forward Dubins, reverse Dubins, or the staged shot per leg L
        # of `staging` (forward Dubins to the pose L ahead of the goal, then L straight in reverse). A node within
        # goal_tolerance (and 0.1 goal_tolerance rad) of the goal also ends it, with the goal appended.
        # A reverse shot turning by more than reverse_turn_max does not end the search: the cheapest collision-free one
        # is kept and returned (reason '') only if the search would fail with 'budget' or 'no_path'.
        # staging = () and reverse_turn_max = inf give plain Hybrid A* with forward and reverse Dubins shots.
```

## `mpc.py`: path tracking, forward and reverse

```python
class PathTracker:
    def __init__(self, truck, planning)     # ValueError if Planning.mpc_horizon < 2
    def set_path(self, path, progress=0)    # new path (or a replan / a back-off retrace); progress = start sample
    path: Path; segment: int                # index into path.segments() of the segment being tracked
    progress: int                           # sample index the truck is at (monotone within a segment)
    arrived: bool                           # at the path's end within park_tolerance and 0.1 rad, standing still
    prediction                              # (N+1, 6) last QP state trajectory [x, y, yaw, v, steer, a], or None
    stats: dict                             # {'solves', 'solve_time' (s, total), 'fallbacks'}
    last_error                              # repr of the last exception a solve raised, or None
    def command(self, state, steer, accel, speed_limit, stop_index=None) -> (steer_cmd, accel_cmd, gear, info)
        # RuntimeError before set_path. speed_limit >= 0 (m/s; None: no limit): 0 means stop at Truck.brake_comfort.
        # The speed profile respects speed_limit, v_forward_max / v_reverse_max per segment, lat_accel_max on curvature,
        # a steering-rate cap on sharp changes of the path's steering, brake_comfort to a stop at every cusp, at the path
        # end and HOLD_MARGIN (0.3 m) short of stop_index (a sample the rear axle must not pass; None: no limit), and
        # accel_max; it brakes harder, up to brake_max, only when a stop is closer than the comfort distance.
        # A linear time-varying MPC (OSQP, kinematic bicycle at the rear axle with the plant's acceleration lag,
        # Planning.mpc_horizon x mpc_step, weights w_pos, w_yaw, w_speed, w_accel, w_steer_rate) tracks the reference of
        # the current segment. At a cusp it stops, then shifts (gear flips) and starts the next segment; a stop_index at
        # or before the segment's last sample forbids the shift. A stopped truck whose steering is off the reference
        # steers in place first. A new path whose first segment runs against the truck's motion starts by braking in
        # the current gear. If OSQP fails: a pure-pursuit fallback.
        # info: {'progress', 'segment', 'lateral_error', 'heading_error', 'status'}, where status is
        #   'tracking';
        #   'cusp' (stopping for a cusp, shifting, steering in place before driving off, braking to reverse for a new path);
        #   'holding' (stopping or stopped for stop_index, or speed_limit 0);
        #   'arrived';
        #   'off_path' (lateral error > track_tolerance, or standing at or past the path's end outside park_tolerance:
        #     this path cannot park it, replan);
        #   'fallback' (pure pursuit this step).
```

## `coordination.py`: multi-truck conflict zones and the token shield

```python
COARSE = 1.0       # m of arc per cell of conflict_matrix's coarse screen
LOOKAHEAD = 6.0    # m before a zone's gate within which a truck that wants to move requests it

@dataclass
class Zone:
    a: int; b: int                 # follower indices, a < b
    a_in: int; a_out: int          # sample range on a's path whose footprint conflicts with b's path (zone_margin)
    b_in: int; b_out: int          # ... on b's path
    def side(self, x) -> (int, int)   # (in, out) on truck x's path
    def other(self, x) -> int         # the partner of truck x

def conflict_matrix(truck, planning, path_a, path_b) -> np.ndarray
    # (Na, Nb) bool: samples conflict when a cover disk of each is closer than 2 radius + zone_margin. Exact at the full
    # resolution; screened on COARSE cells first (triangle inequality), only undecided cell pairs tested per sample.

def conflict_zones(truck, planning, a, path_a, b, path_b) -> list[Zone]
    # a and b in any order (the zones carry a < b), sorted along a's path. Connected runs of conflicting samples,
    # merged to a fixed point where they overlap on either path: the ranges are disjoint on both paths, and every
    # conflicting sample pair lies inside one zone.

class Coordinator:
    def __init__(self, truck, planning, n)
    order: list                 # priority list, default range(n) (the queue order)
    gate: int                   # gate_buffer/path_sample samples (4)
    look: int                   # LOOKAHEAD/path_sample samples (12)
    paths: list; progress: np.ndarray
    zones: list[Zone]; owner: list   # owner parallel to zones: the token holder (truck index) or None
    def set_path(self, i, path, progress=0)
        # (Re)computes truck i's zones against every other truck with a path; None: truck i has no path and conflicts
        # with nothing (episode.py calls set_path(i, None) when truck i parks, and when the driver finds it no spot:
        # either way it becomes a static obstacle). A partner's reservation (not a hold of i) carries over to the new
        # zones that overlap, on the partner's path, a zone it had reserved.
    def update(self, progress, wants) -> list
        # progress: (n,) current sample per truck (clipped to its path); wants: (n,) bool, the truck wants to move along
        # its path now. Returns per truck the last sample it may reach now (None: no limit). Token rules below.
    def blocking(self, i) -> list[int]      # trucks holding a zone that stops truck i, as of the last update
    def priority_wait(self, i, progress) -> bool
        # The scripted baseline's rule: truck i must wait while a truck earlier in self.order shares a zone with it that
        # i has not passed and the earlier truck has not yet passed (progress < out).
    def zones_of(self, i) -> list
        # Length n, as of the last update (or set_path). Entry j is None for j == i, when truck i has no path, or when
        # no live zone with j lies ahead; else, for the nearest such zone along i's path, (m along i's path to its
        # in-gate, m along j's path to j's in-gate, j holds the zone's token). A distance is negative once that truck
        # is past its gate.
```

**Token rules** (`update`; `episode.py` calls it at the start of every decision and then every `mpc_step`). Tokens
persist from one update to the next.

1. **Hold.** A truck whose progress lies in a zone's hold range, [in − gate, out], holds its token. If both trucks are
   inside (the shield never allows it, but a replan can), the current owner keeps it, else the one earlier in `order`.
   An owner outside its hold range keeps the token (a reservation) until it has passed out or no longer wants to move.
   Callers must report a truck that is still rolling as wanting (`episode.py` does): releasing its token while it
   brakes could let the partner in.
2. **Request.** Zones closer than a gate apart along a truck's path, with any partners, form a cluster. In `order`, a
   truck that wants to move requests its clusters in turn along its path: every one whose entry gate (in − gate of
   its first zone) lies within `look` samples of its progress, also while it is still inside an earlier one, up to
   the first denial. It gets a cluster's every token or none. A denied truck drops the reservations it has in that
   cluster and beyond, so its reservations are always a prefix of its clusters.
3. **Stop.** A truck is stopped at the entry gate (in − gate − 1, never behind its progress) of the first cluster on
   its path in which another truck holds a zone. Already inside the cluster, it stops at the gate of the first zone it
   does not hold.

Zones the truck or its partner has passed are ignored. A moving truck learns of every stop at least `look` − 1 samples
(5.5 m) ahead, at the latest when its request for that cluster is denied (barring a zone that another truck's replan
creates closer ahead of it), and the stop lies past every cluster it holds (clusters are more than a gate apart). No
truck waits while holding a zone, so there is no hold-and-wait deadlock except through a truck that stands inside a
zone or a replan that puts a zone under a truck.

## `spots.py`: driver picks and the safety check

```python
REASONS = ('outside', 'entrance', 'leader', 'too_close', 'occupied', 'unscanned', 'unreachable')   # check order
YAWS = (0.0, pi/2, pi, -pi/2)     # candidate headings
ATTEMPTS = 12                     # driver picks per follower before giving up

def grid_status(truck, grid, poses, margin) -> (occupied (M,), unknown (M,))   # any cell within margin OCCUPIED /
                                                                                # UNKNOWN (off the grid counts UNKNOWN)
def static_reasons(scenario, grid, poses, assigned=()) -> np.ndarray   # (M,) object: first failing static check, '' if none
def validate(scenario, grid, pose, assigned=(), others=(), planner=None, start=None) -> (reason, plan_result)
    # reason '' = accepted, else the first failing REASONS entry. assigned: poses (3,) accepted for other followers;
    # others: (K, 4, 2) extra footprints the path must avoid (the leader, waiting followers). When the static checks
    # pass and planner (HybridAStar) and start (the follower's pose) are given, the spot must be reachable: plan_result
    # is the PlanResult on grid.clearance(extra=others + the assigned spots' footprints), and reason 'unreachable' when
    # it has no path. Otherwise plan_result is None (episode.YardRun.replan uses the static checks only).
def lattice(scenario) -> (M, 3)                        # every lattice spot: body centres every spot_lattice m, the four YAWS
def candidates(scenario, grid, assigned=()) -> (M, 3)  # lattice poses that pass the static checks (no planning)
def default_others(world, follower) -> (K, 4, 2)       # the leader and the followers later in the queue where they stand

class Driver:                                          # scripted leader driver; all randomness from rng
    def __init__(self, scenario, rng, attempts=ATTEMPTS)
    def preference(self, poses) -> (M,)                # weight PREFER (3) along a berm and nose out, else 1
    def free_click(self) -> (3,)                       # uniform pose in the yard, any heading
    def pick(self, follower, world, grid, assigned=(), others=None, planner=None) -> (pose (3,) | None, picks)
        # Up to `attempts` picks until the check accepts one: each a free click with probability driver_bad_pick, else
        # a candidate drawn by preference (never the same candidate twice). picks: one {'pose', 'accepted', 'reason', 'result'} per attempt,
        # 'result' the attempt's PlanResult (None if a static check rejected it, or no planner): the accepted pick's
        # result carries the follower's path. others None: default_others.
    def assign_all(self, world, grid, planner=None) -> (spots, log)   # queue order, each accepted spot joining
                                                                      # `assigned`; log: every pick with its 'follower'
```

## `episode.py`: the loop and the record

```python
ACTIONS = ('HOLD', 'CREEP', 'GO', 'BACK_OFF', 'REPLAN')   # HOLD, CREEP, GO, BACK_OFF, REPLAN = range(5)
BACK_OFF_DISTANCE = 6.0   # m reversed per BACK_OFF manoeuvre
SAFETY_MARGIN = 0.3       # m: the supervisor's stop margin
BLOCK_BUFFER = 4.0        # m: a blocked truck stops this far before the first blocked sample
MOVING = 0.05             # m/s: slower counts as standing still
REASSIGN_AFTER = 2        # failed replans in a row after which the driver picks the follower a new spot
REPLAN_IDLE = 10.0        # s standing still after which REPLAN is offered without a block (deadlock breaking)

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
    paths: list                    # [(time, follower, Path)]: every plan and replan, in order (back-off retraces not)
    events: list                   # [{'time', 'follower', 'kind', 'detail'}], kind: 'parked', 'collision', 'replan'
                                   # (detail 'new spot' when the driver assigned one), 'replan_failed', 'safety_stop',
                                   # 'reassigned', 'no_spot' (the driver found no acceptable spot)
    end_reason: str                # 'parked', 'collision', 'timeout', 'stalled' ('running' before the end)

class YardRun:                     # one episode, advanced one decision at a time
    def __init__(self, scenario, seed, n_followers=None)
        # The leader scans; the driver assigns spots in queue order, each pick checked and planned (spots.Driver.pick
        # with the planner); the accepted pick's result is the follower's first path. A follower the driver finds no
        # spot for starts without a path ('no_spot').
    n; world; grid; grid0; missed; revealed; revealed_at; planner; coordinator; driver
    f: list                        # per follower a Follower: tracker, spot, path, progress, parked, blocked, backing,
                                   # last_replan, failed_replans, failed_key (the inputs of the last failed replan),
                                   # last_remaining (the remaining length when its last path was dropped), last_move,
                                   # safety, action, status, lateral_error, driven
    time; stop_index; picks; paths; events; end_reason
    initial_path_length: float     # m: the first paths' summed length (metrics 'path_length_m')
    done: bool                     # property
    def masks(self) -> np.ndarray  # (n, 5) bool: parked -> HOLD only; no path -> HOLD and REPLAN when allowed
    def needs_replan(self, i) -> bool    # a replan can change something: no path, the path blocked, the tracker
                                         # 'off_path', or standing still >= REPLAN_IDLE while trying to move (below)
    def can_replan(self, i) -> bool      # not parked, |v| < 0.2 m/s, replan_cooldown since the last, not backing off,
                                         # and needs_replan(i)
    def can_back_off(self, i) -> bool    # standing still (|v| <= MOVING), >= 1 m along the path, and the retrace
                                         # touches no zone another truck holds or could request now (below);
                                         # continuing a back-off in progress is always allowed
    def replan(self, i) -> str     # REPLAN: 'ok' (a new path, a new spot's included), 'failed' (no path, same spot
                                   # kept) or 'no_spot' (reassigned, and the driver found none) (see below)
    def step(self, actions) -> dict
        # Applies one decision's actions (n,) (a masked action becomes HOLD; a REPLAN runs at once, then the follower
        # HOLDs for the decision) and simulates decision_interval; returns the decision's tally {'parked',
        # 'safety_stops', 'replans' (replan() 'ok'), 'replans_failed' ('failed' or 'no_spot'), 'back_off_m',
        # 'collision', 'progress_m'}. RuntimeError once the episode is over.
    def remaining(self) -> list    # per follower: path length left (m), 0 when parked; without a path the last
                                   # remaining length (last_remaining), None if it never had a path
    def episode(self) -> YardEpisode
    def metrics(self) -> dict      # the episode_metrics dict below
```

Semantics the env and the baseline rely on:

- **Control.** `_control` runs at the start of every decision, so the actions take effect at once, and then every
  `mpc_step` (0, 0.2 and 0.4 s into each 0.5 s decision). It first refreshes the progress of each truck that has a
  path and is neither parked nor backing off (the nearest path sample from its current progress to the end of its
  current segment; the trackers' own is one `mpc_step` old), then updates the coordinator, then commands the trackers.
- **Wants.** The coordinator gets `wants[i]` = has a path, not parked, not backing off, and (action GO or CREEP, or
  still rolling above `MOVING`).
- **Parking.** A follower parks standing still within `park_tolerance` and 0.1 rad of its spot, at its path's last 3
  samples. It then leaves the coordinator (`Coordinator.set_path(i, None)`) and becomes a static obstacle.
- **No path.** A follower the driver finds no spot for (at the start, or after a reassignment) has no path: it leaves
  the coordinator (`set_path(i, None)`), brakes to a standstill at `brake_comfort` and stays there, and counts as a
  static obstacle. Its REPLAN asks the driver for a spot again.
- **Blocks.** After every reveal, every parking and every 'no_spot' after t = 0, `_check_blocks` sets
  `Follower.blocked` to the first sample of the remaining path that fails `collision_free` on the map plus the
  footprints that stay put (`_static_feet`: the leader, the parked followers and the followers standing without a
  path), or None. A blocked truck stops `BLOCK_BUFFER` short and needs a REPLAN.
- **REPLAN.** Hybrid A* from where the truck stands to its spot, around the footprints that stay put (`_static_feet`)
  and the spots of unparked followers earlier in `coordinator.order` (prioritized planning). Followers that are not
  parked and have a path are not obstacles: the new path's conflict zones keep the trucks apart (a follower queued on
  the access road would otherwise wall the road off for the ones behind it). The planner is deterministic, so a
  repeat of the last failed replan (`failed_key`: the same pose, rounded to 0.1 m and 0.1 rad, spot, number of
  revealed obstacles and obstacle footprints) is not searched again and fails at once (reason 'repeat'). With no
  path, the spot is checked again (`spots.validate`, static checks). A reason other than 'unreachable' or 'too_close'
  gets a new spot from the driver at once ('reassigned'). Otherwise the replan fails ('replan_failed', returns
  'failed'), and after `REASSIGN_AFTER` failures in a row the driver picks a new spot ('reassigned', detail
  'unreachable after 2 replans'). A new spot is planned around the same obstacles plus every assigned spot; if the
  driver finds none, `replan` returns 'no_spot'.
- **REPLAN mask.** `can_replan` also requires `needs_replan(i)`: no path, a blocked path, tracker status 'off_path',
  or standing still (`time − last_move`) for at least `REPLAN_IDLE` while the action is GO or CREEP and no coordinator
  zone is what stops it (`coordinator.blocking(i)` empty). Otherwise the planner would return about the same path, so
  REPLAN is masked as a nuisance action.
- **BACK_OFF.** It starts only from standstill (|v| ≤ `MOVING`, so the coordinator's progress stays true) at least
  1 m along the path, toward a sample up to `BACK_OFF_DISTANCE` behind; the tracker drives the retrace
  (`Path.retrace`) at `v_creep`, and `progress` follows it back down the planned path. `_back_conflicts` refuses a
  retrace that touches the hold range [in − gate, out] of a zone that another truck holds, or whose partner could
  request it now (the partner's progress within [in − gate − look, out] on its own path). While a back-off runs,
  `_control` checks the retrace up to the stopping distance (v²/(2 `brake_max`)) + 0.5 m ahead the same way and stops
  the back-off before such a zone. Another action also stops it; it ends at the target or once stopped, the truck
  rejoins its path at the sample reached, and a BACK_OFF still in force becomes HOLD.
- **Supervisor.** Every `dt`, `_supervise` tests the current pose, a pose 0.25 m ahead in the direction of motion (or
  of the gear, from standstill) and the path ahead (every second sample within v²/(2 `brake_max`) + 1 m, in the
  direction of motion; while backing off, the retrace) against the obstacles whose polygon distance from the footprint
  is within `near_range` + that reach (missed ones too), the leader and the other followers. It brakes at
  `brake_emergency` when the step ahead is within `SAFETY_MARGIN` and closer than the current pose, or any pose ahead
  is within `SAFETY_MARGIN`. The current pose alone never triggers a stop, so a truck already within the margin can
  still move away.
- **Progress.** `progress_m` = sum over followers of (remaining before − remaining after) over the decision.
  `remaining()` is the potential: 0 once parked, and without a path the last remaining length, which carries over
  until a new path. Only a follower that has never had a path (None) is skipped. It is negative when a replan, a new
  spot or a back-off lengthens the remaining path (potential-based progress).

## `env.py`: the RL env and the scripted baseline

```python
SLOTS = 3
EGO, NEAR, OTHER, LEADER, GLOBAL = 16, 8, 11, 4, 2   # observation groups; SLOT = 16 + 8 + 2 x 11 + 4 = 50 per slot
def obs_dim() -> int                                 # 3 x 50 + 2 = 152

@dataclass(frozen=True)
class YardEnvConfig:               # bump version whenever the observation, the masks or the reward change
    version: str = 'ye1'
    scenario: str = 'yard1'        # key into nohesi.yard.scenarios.YARD_SCENARIOS
    followers: tuple = ()          # (min, max) followers per episode; () keeps the scenario's (1, 3)
    w_time: float = 0.02           # per unparked follower-second
    w_progress: float = 0.02       # per metre of remaining path removed (potential-based)
    w_park: float = 5.0
    w_collision: float = 50.0
    w_safety: float = 0.5          # per safety-stop onset
    w_replan: float = 0.2
    w_replan_fail: float = 0.5
    w_back: float = 0.0            # per metre backed off: 0, the progress potential already charges the reversed
                                   # metres and pays them back when driven again, so a back-off costs only its time
    w_stall: float = 5.0

class YardEnv(gymnasium.Env):
    def __init__(self, cfg=YardEnvConfig(), seed_pool=None, rng_seed=None)
    action_space = MultiDiscrete([5, 5, 5])                   # ACTIONS per follower slot
    observation_space = Box(-inf, inf, (152,), float32)
    run: YardRun | None
    step_started: float | None                                # monotonic start of the step in progress (watchdog)
    def reset(self, *, seed=None, options=None) -> (obs, info)       # seed None: next seed_pool entry, else a draw >= 2000;
                                                                     # info {'seed'}
    def step(self, action) -> (obs, reward, terminated, truncated, info)
        # terminated on 'parked' (all followers) or 'collision'; truncated on 'timeout' or 'stalled'.
        # info['tally'] every step (YardRun.step's tally). info['episode_metrics'] at episode end: {'seed', 'success'
        # (all parked), 'parked', 'followers', 'collision', 'end_reason', 'duration_s', 'replans' (including 'new spot'
        # replans), 'replans_failed' ('replan_failed' events plus 'no_spot' events after t = 0), 'safety_stops',
        # 'back_off_m', 'picks_rejected', 'reassigned', 'path_length_m' (the first paths), 'driven_m'}
    def action_masks(self) -> np.ndarray                      # (15,) bool, flattened per slot (absent slots: HOLD only)
    def episode(self) -> YardEpisode

def make_env(cfg=YardEnvConfig(), rank=0, base_seed=0, seed_pool=None)
    # Thunk for SubprocVecEnv: env RNG seeded base_seed + rank. In a worker process it starts the step watchdog
    # (nohesi.rl.watchdog.watch), which dumps hung steps when NOHESI_HANG_DIR is set.
def scripted_actions(env) -> np.ndarray
    # (3,) the prioritized baseline's actions for the current decision (docs/yard_parking_design.md, section 8): GO;
    # HOLD before the gate of a zone a higher-priority truck has not cleared, from v^2/(2 brake_comfort) +
    # v decision_interval + 1 m ahead, never inside a zone. Blocked: GO until 6 m before the blocked sample (the tracker
    # stops 4 m short); then, or off its path or without one: REPLAN when can_replan, else HOLD.
```

## `train.py`: MaskablePPO training

```python
@dataclass
class YardTrainConfig:             # run_id, env (a YardEnvConfig), env_module, n_envs, PPO and VecNormalize settings,
    ...                            # eval_every, eval_seeds (1000-1019), baseline, checkpoint_every, seed, device, resume
def train(cfg, make_env=None, scripted_fn=None) -> MaskablePPO       # outputs in runs_rl/<run_id>/ (module docstring)
def evaluate(model, normalize, env_cfg, seeds=EVAL_SEEDS, ...) -> {'per_seed', 'summary'}
def baseline(env_cfg, seeds=EVAL_SEEDS, *, workers=1, make_env=None, scripted_fn=None, ...) -> {'seeds', 'per_seed', 'summary'}
def summarize(m, decisions=None) -> dict; def aggregate(rows) -> dict; def rank(summary) -> tuple
```

The trainer talks to an "env module" (default `nohesi.yard.env`) that provides `YardEnvConfig`,
`make_env(cfg, rank=0, base_seed=0, seed_pool=None)` and `scripted_actions(env)`, and envs that follow the `YardEnv`
contract above (`reset(seed=s)` plays seed s, `action_masks()`, `info['episode_metrics']` at the end). The CLI is
`python -m nohesi.yard.train`; `train_yard.py` at the repo root runs it with settings at the top.

## `viz.py`

```python
def animate(ep, path, fps=10, stride=None)   # top-down GIF of a YardEpisode
def plot(ep, path)                           # one PNG: the scan, the spots, the paths, the trucks' tracks
```

`run_yard.py` at the repo root runs one episode and writes `yard.gif`, `yard.png`, `metrics.json` and `events.json` to
`results/run_NNN_<scenario>_<policy>_seed<seed>_followers<n>/`.
