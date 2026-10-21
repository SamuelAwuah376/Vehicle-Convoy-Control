from dataclasses import dataclass, field, replace
from math import isfinite


def _positive(obj, *names):
    for name in names:
        value = getattr(obj, name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive')


@dataclass(frozen=True)
class Road:
    """Straight multilane road. Lane 0 is rightmost; d is positive to the left."""
    lanes: int = 3
    lane_width: float = 3.6
    length: float = 5000.0

    def __post_init__(self):
        _positive(self, 'lanes', 'lane_width', 'length')


@dataclass(frozen=True)
class Vehicle:
    # Vehicle states are referenced at the centre of gravity (CG); the body overhangs both axles equally.
    wheelbase: float = 2.7
    cg_to_rear_axle: float = 1.62   # l_r (m); l_f = wheelbase - l_r. 1.62 = 60/40 front-biased weight (l_f = 1.08)
    length: float = 4.5
    width: float = 1.8
    accel_max: float = 3.0
    brake_max: float = 6.0
    steer_max: float = 0.5
    steer_rate: float = 0.6
    power_speed: float = 20.0       # above this, accel limit falls as accel_max*power_speed/v
    lateral_accel_max: float = 4.0  # ~0.4 g grip proxy, applied as a steering cap
    standstill_gap: float = 5.0
    headway: float = 1.2
    # Simulator plant: 'kinematic' (KinematicPlant) or 'dynamic' (DynamicPlant; the MPC stays kinematic).
    model: str = 'kinematic'
    mass: float = 1500.0                          # kg
    yaw_inertia: float = 2250.0                   # kg m^2
    cornering_stiffness_front: float = 80000.0    # N/rad per axle
    cornering_stiffness_rear: float = 80000.0
    friction: float = 1.0                         # tyre-road mu
    dynamic_blend_low: float = 3.0                # m/s: pure kinematic below, pure dynamic above high
    dynamic_blend_high: float = 6.0
    dynamic_substep: float = 0.01                 # s, integration step inside each sim step

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k != 'model'))
        if self.cg_to_rear_axle >= self.wheelbase:
            raise ValueError('cg_to_rear_axle must lie within the wheelbase')
        if self.model not in ('kinematic', 'dynamic'):
            raise ValueError(f'unknown vehicle model {self.model!r}')
        if self.dynamic_blend_high <= self.dynamic_blend_low:
            raise ValueError('dynamic_blend_high must exceed dynamic_blend_low')


def bmw_m3(model='kinematic'):
    """BMW M3 Competition (G80), approximate published figures: 4.79 x 1.90 m, 2.857 m wheelbase, 52.5/47.5 weight
    (l_r = 0.525 L), 1,780 kg; 375 kW (503 hp) -> 211 W/kg, launch-limited at 8 m/s^2 (0-60 mph ~3.4 s) up to
    26.3 m/s, power-limited above; 60-0 mph in ~100 ft -> 11 m/s^2; ~1.0 g skidpad -> 9.5 m/s^2; performance tyres."""
    return Vehicle(wheelbase=2.857, cg_to_rear_axle=1.50, length=4.79, width=1.90, accel_max=8.0, brake_max=11.0,
                   power_speed=26.3, lateral_accel_max=9.5, model=model, mass=1780.0, yaw_inertia=2800.0,
                   cornering_stiffness_front=110000.0, cornering_stiffness_rear=125000.0, friction=1.05)


@dataclass(frozen=True)
class Traffic:
    count: int = 8
    speed_min: float = 18.0
    speed_max: float = 25.0
    spawn_behind: float = 80.0      # initial spawn window relative to the team
    spawn_ahead: float = 250.0
    recycle_behind: float = 150.0   # cars this far behind the rearmost teammate respawn ahead
    respawn_ahead: float = 250.0
    min_gap: float = 25.0           # CG spacing between spawned cars in a lane
    team_lane_clear_ahead: float = 100.0
    team_lane_clear_behind: float = 40.0
    idm_accel: float = 1.5
    idm_brake: float = 2.0
    idm_brake_max: float = 8.0
    idm_gap: float = 2.0
    idm_headway: float = 1.5
    # MOBIL lane changes (off by default; when off, no extra rng draws, so v0-v2 traffic is unchanged).
    lane_changes: bool = False
    mobil_politeness: float = 0.3
    mobil_threshold: float = 0.2      # m/s^2 incentive needed to change
    mobil_safe_brake: float = 3.0     # m/s^2 most braking imposed on the new follower
    lane_change_time: float = 3.0     # s, lane centre to lane centre (quintic profile)
    lane_change_interval: float = 1.0 # s between MOBIL evaluations per car (seeded phase)
    lane_change_gap: float = 0.5      # car lengths: no lane change with a target-lane car alongside or within this bumper gap
    # No walls: never one traffic car per lane within wall_window (CG spread along the road) of each other. Spawns and
    # lane changes avoid it; once one car per lane comes within twice that, the rearmost drops wall_slowdown below its
    # current speed until they spread out.
    no_walls: bool = False
    wall_window: float = 9.0          # m (two car lengths)
    wall_slowdown: float = 3.0        # m/s off the rearmost wall car's desired speed
    lane_change_cooldown: float = 8.0 # s after finishing a lane change before the same car may start another
    keep_right_bias: float = 0.0      # m/s^2 added to the incentive for moving right, subtracted for left

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k not in ('lane_changes', 'mobil_politeness', 'mobil_threshold',
                                                             'keep_right_bias', 'no_walls')))
        if any(not isfinite(getattr(self, k)) or getattr(self, k) < 0 for k in ('mobil_politeness', 'mobil_threshold')):
            raise ValueError('mobil_politeness and mobil_threshold must be non-negative')
        if not isfinite(self.keep_right_bias):
            raise ValueError('keep_right_bias must be finite')
        if self.speed_min > self.speed_max:
            raise ValueError('speed_min must not exceed speed_max')


@dataclass(frozen=True)
class Timing:
    dt: float = 0.05
    duration: float = 60.0

    def __post_init__(self):
        _positive(self, 'dt', 'duration')


@dataclass(frozen=True)
class Score:
    """Game score formula. Bump version whenever any value changes."""
    version: str = 'v0'
    pass_base: float = 100.0
    pass_threshold: float = 1.5     # minimum footprint clearance (m) below which a pass scores
    pass_ref_speed: float = 28.0
    proximity_rate: float = 10.0    # points/s per teammate pair, split between the pair
    proximity_distance: float = 12.0
    proximity_min_speed: float = 20.0
    speed_rate: float = 1.0         # points/s per m/s above speed_floor
    speed_floor: float = 20.0
    collision_penalty: float = 500.0
    collision_wipe: float = 0.0     # share of the whole team's banked points a collision also removes (No Hesi run reset)
    repass_cooldown: float = 0.0    # s: a team car scores the same traffic car again only after this long (0: always)
    combo_step: float = 0.0         # team combo: multiplier gain per scored close pass (0 disables)
    combo_max: float = 1.0          # multiplier cap
    combo_window: float = 5.0       # seconds without a scored close pass before the combo resets
    proximity_falloff: str = 'flat' # 'flat': full rate inside the radius; 'linear': rate*(1 - distance/radius)
    # Lane-change maneuvers (0 points disables; detection always runs). Base points, times the combo multiplier.
    lc_base: float = 0.0              # lane change (single or two-lane) without a close pass
    lc_pass_single: float = 0.0       # single lane change + close pass, no follow-up change
    lc_pass_double: float = 0.0       # two-lane change + close pass, no follow-up change
    lc_chain_single: float = 0.0      # single lane change + close pass + another lane change
    lc_chain_double: float = 0.0      # two-lane change + close pass + another lane change (highest)
    lc_chain_window: float = 4.0      # s: pass within this of the change settling; follow-up within this of the pass
    lc_traffic_distance: float = 20.0 # m: a traffic car within this along the road in a lane involved in the move
    lc_min_speed: float = 25.0        # m/s: minimum speed throughout the lane change
    # Team close pass: points*(1 + team_pass_gain*sum over teammates of max(0, 1 - distance/team_pass_distance)).
    team_pass_gain: float = 0.0       # 0 disables
    team_pass_distance: float = 20.0  # m, body-centre distance at the pass step
    team_pass_require_all: bool = False  # True: no bonus unless all three teammate pairs are within team_pass_distance
    # Drafting: following a teammate in the same lane within draft_headway, both above draft_min_speed,
    # earns draft_rate pts/s times the combo multiplier.
    draft_rate: float = 0.0           # 0 disables
    draft_headway: float = 1.0        # s: bumper gap <= draft_headway*follower speed
    draft_min_speed: float = 20.0     # m/s
    # Draft pass: a close pass by a car that was drafting (as above) at the pass or within draft_pass_window s
    # before it (pulling out of the draft to pass) is worth draft_pass_gain times more (on top of team pass and combo).
    draft_pass_gain: float = 1.0      # 1 disables
    draft_pass_window: float = 1.0    # s
    # Draft train: all three cars in a row along the road (any lanes), each within train_headway bumper to bumper of the
    # car ahead, all above draft_min_speed. Each car earns train_rate pts/s (times the combo) instead of draft_rate, and
    # a close pass by a train member (or within draft_pass_window s after the train breaks) is worth train_pass_gain.
    train_rate: float = 0.0           # 0 disables
    train_headway: float = 0.5        # s
    train_pass_gain: float = 1.0      # 1 disables
    train_min_speed: float = 0.0      # m/s every train car must exceed for train points/passes (0: draft_min_speed)
    # Top speed bonus: top_speed_rate pts/s per car above top_speed_threshold, times the combo multiplier.
    top_speed_rate: float = 0.0       # 0 disables
    top_speed_threshold: float = 53.64  # m/s (120 mph)
    # Team spread penalty: each teammate pair loses spread_penalty_rate*((distance - spread_free_distance)/spread_scale)^2
    # pts/s beyond spread_free_distance (body-centre distance), split between the pair; not multiplied by the combo.
    spread_penalty_rate: float = 0.0  # 0 disables
    spread_free_distance: float = 30.0  # m
    spread_scale: float = 10.0        # m
    spread_penalty_cap: float = float('inf')  # pts/s: most a pair can lose per second

    NON_NEGATIVE = ('combo_step', 'proximity_rate', 'team_pass_gain', 'draft_rate', 'spread_penalty_rate', 'train_rate',
                    'collision_wipe', 'repass_cooldown', 'train_min_speed', 'top_speed_rate')

    def __post_init__(self):
        if not self.spread_penalty_cap > 0:   # inf allowed (no cap)
            raise ValueError('spread_penalty_cap must be positive')
        if not 0 <= self.collision_wipe <= 1:
            raise ValueError('collision_wipe must be in [0, 1]')
        _positive(self, *(k for k in vars(self) if k not in ('version', 'proximity_falloff', 'team_pass_require_all',
                                                             'spread_penalty_cap', *self.NON_NEGATIVE)
                          and not k.startswith('lc_') or k in ('lc_chain_window', 'lc_traffic_distance', 'lc_min_speed')))
        if any(getattr(self, k) < 0 for k in vars(self) if k.startswith('lc_')):
            raise ValueError('lane-change points must be non-negative')
        for k in self.NON_NEGATIVE:
            if not isfinite(getattr(self, k)) or getattr(self, k) < 0:
                raise ValueError(f'{k} must be non-negative')
        if self.proximity_falloff not in ('flat', 'linear'):
            raise ValueError('proximity_falloff must be flat or linear')
        if self.combo_max < 1:
            raise ValueError('combo_max must be at least 1')


@dataclass(frozen=True)
class Planner:
    """Intention layer, MPC and fallback settings. Saved with results; bump version on change."""
    version: str = 'm3.7'
    horizon: int = 30               # MPC steps (3 s at step 0.1)
    step: float = 0.1               # MPC step and replan interval (s)
    decision_interval: float = 0.5  # policy decisions (s)
    max_iter: int = 4000            # OSQP iteration budget (reproducible "timeout")
    qp_eps: float = 1e-4            # OSQP absolute/relative tolerance
    qp_reuse: bool = True           # one OSQP instance, fixed sparsity, data updated per solve (False: setup per solve)
    qp_soft_slots: int = 6          # qp_reuse: soft rows per step (v4: up to 7, >6 in 0.2 % of solves: those use a setup)
    speed_step: float = 2.0         # target speed change per speed action
    speed_min: float = 20.0
    speed_max: float = 35.0
    gap_lookahead: float = 3.0      # gap rule prediction time (s)
    margin_long: float = 0.25       # hard bumper-to-bumper margin (m)
    margin_lat: float = 0.25        # hard side-to-side margin (m)
    edge_margin: float = 0.3
    headway: float = 0.3            # soft time headway to traffic (s), penalised through slack
    teammate_headway: float = 0.2   # soft headway to teammates (drafting)
    lean_offset: float = 0.6        # lean action shifts the target this far from the lane centre (m)
    relevance: float = 60.0         # obstacles farther than this along the road are ignored
    side_zone: float = 5.0          # beyond bumper overlap, distance where a laterally clear car gets a stay-beside constraint
    done_tolerance: float = 0.3     # lane change done within this distance of the target d (centre + lean)
    max_shifted_steps: int = 5      # shifted-plan fallback is open loop; limit how long it is used
    merge_window: float = 30.0      # reject entering a lane a teammate enters from the other side this close
    rear_closing_time: float = 0.3  # gap rule: extra rear room per m/s the rear gap car closes on the ego (s)
    double_gap_lookahead: float = 4.0   # two-lane change: destination gap prediction time (s)
    middle_window_start: float = 0.8    # two-lane change: middle lane must be clear over this crossing window (s)
    middle_window_end: float = 2.5
    commit_progress_eps: float = 0.1    # committed change: lateral progress below this (m) does not count
    commit_stall_time: float = 2.0      # cancel a committed change without progress for this long (s)
    commit_max_time: float = 6.0        # cancel a committed change older than this (s)
    pending_max_time: float = 4.0       # cancel a pending change not committed within this (s)
    retry_cooldown: float = 3.0         # after a stalled/timed-out/forced cancel, the same car can't re-request that lane (s)
    check_margin: float = 0.1       # post-cycle footprint check clearance (QP margins are 0.25 m per axis)
    emergency_decel: float = 12.0   # braking of the emergency escape's braking candidates (the plant clips to brake_max)
    check_substeps: int = 5         # footprint checks per plan step (0.1 s): plans are interpolated between nodes
    escape_accel: float = 5.0       # m/s: speed-up candidates of the emergency escape aim this far above current speed
    slot_check: bool = True         # back out of a lane change whose target slot is closing (see slot_margin)
    slot_margin: float = 0.5        # s: least time to contact with the gap cars or vehicles merging into the lane
    slot_lateral: float = 0.1       # m: lateral clearance beyond the car widths that still counts as sharing the slot
    w_lateral: float = 1.0
    w_speed: float = 0.5
    w_heading: float = 5.0
    w_accel: float = 0.1
    w_steer: float = 10.0
    w_accel_rate: float = 1.0
    w_steer_rate: float = 2000.0
    w_slack: float = 100.0
    lateral_tolerance: float = 0.05  # |dd| this close below the side gap still counts as laterally clear (side bound unchanged)
    yield_window: float = 10.0      # yield to a teammate changing into the lane this close beyond bumper overlap (m)
    yield_margin: float = 1.0       # yielding car aims this far past the changer's side zone (m)
    w_yield: float = 100.0          # slack weight of the soft get-clear constraint toward that teammate
    mpc: str = 'linear'             # 'linear' (LaneChangeMPC, OSQP) or 'nonlinear' (acados)
    # Nonlinear MPC (planning/nonlinear.py, acados SQP-RTI on the dynamic bicycle).
    nmpc_hard_slots: int = 14       # hard obstacle rows per stage (v4 needs up to 11); excess nearest-first, counted
    nmpc_soft_slots: int = 10       # soft (headway/yield) rows per stage (v4: up to 7)
    nmpc_sqp_iter: int = 1          # SQP-RTI iterations per replan
    nmpc_integrator_steps: int = 2  # RK4 steps per MPC step
    nmpc_steer_cap: bool = False    # also apply the kinematic grip-proxy steer cap (plant.steer_limit)
    nmpc_slip_limit: float = 0.8    # soft bound on |tan slip| as a fraction of the Fiala saturation slip (99 % of peak)
    nmpc_w_slip: float = 1000.0     # its slack weight
    nmpc_w_vy: float = 10.0         # weight on body lateral speed vy: without it RTI/SQP diverges (GN misses r*vy coupling)
    nmpc_w_yaw_rate: float = 10.0   # weight on yaw rate r
    nmpc_levenberg_marquardt: float = 0.0   # acados LM regularisation of the Gauss-Newton Hessian

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k not in ('version', 'rear_closing_time', 'mpc', 'nmpc_steer_cap',
                                                             'qp_reuse', 'nmpc_levenberg_marquardt')
                          and not isinstance(getattr(self, k), bool)))   # on/off switches may be False
        if not isfinite(self.nmpc_levenberg_marquardt) or self.nmpc_levenberg_marquardt < 0:
            raise ValueError('nmpc_levenberg_marquardt must be non-negative')
        if self.mpc not in ('linear', 'nonlinear'):
            raise ValueError(f'unknown mpc {self.mpc!r}')
        if not isfinite(self.rear_closing_time) or self.rear_closing_time < 0:
            raise ValueError('rear_closing_time must be non-negative (0 disables the closing-speed rule)')
        if self.speed_min > self.speed_max:
            raise ValueError('speed_min must not exceed speed_max')
        if self.middle_window_start > self.middle_window_end:
            raise ValueError('middle_window_start must not exceed middle_window_end')


@dataclass(frozen=True)
class Scenario:
    version: str = 'v0'
    road: Road = field(default_factory=Road)
    vehicle: Vehicle = field(default_factory=Vehicle)
    traffic: Traffic = field(default_factory=Traffic)
    timing: Timing = field(default_factory=Timing)
    score: Score = field(default_factory=Score)
    team_speed: float = 28.0
    team_lane: int = 1
    team_start_s: float = 100.0
    team_speed_max: float = 35.0      # m/s: highest target speed the team may command (sets Planner.speed_max)
    team_start_headway: float = 0.0   # s: start formation spacing headway (0: vehicle.headway)
    overtaker_speed: float = 32.0     # scripted overtaker cruise speed (m/s)
    overtaker_look: float = 60.0      # scripted overtaker look-ahead for slow cars (m)

    def __post_init__(self):
        _positive(self, 'team_speed', 'team_start_s', 'team_speed_max', 'overtaker_speed', 'overtaker_look')
        if self.team_start_headway < 0:
            raise ValueError('team_start_headway must be non-negative')
        if not 0 <= self.team_lane < self.road.lanes:
            raise ValueError('team_lane outside road')
        if self.vehicle.width >= self.road.lane_width:
            raise ValueError('vehicle must fit in lane')


def scenario_v1():
    """v0 with 120 s episodes and 10 traffic cars (from 8) in the same spawn window; longer road to fit."""
    return Scenario(version='v1', road=Road(length=10000.0), traffic=Traffic(count=10),
                    timing=Timing(duration=120.0))


def scenario_v2():
    """v1 with 20 traffic cars (double v1's density) in the same spawn window."""
    return replace(scenario_v1(), version='v2', traffic=Traffic(count=20))


def scenario_v3():
    """v1 with MOBIL lane-changing traffic."""
    s = scenario_v1()
    return replace(s, version='v3', traffic=replace(s.traffic, lane_changes=True))


def scenario_v4():
    """v2 with MOBIL lane-changing traffic."""
    s = scenario_v2()
    return replace(s, version='v4', traffic=replace(s.traffic, lane_changes=True))


def scenario_v5():
    """v4 with BMW M3 cars (bmw_m3), traffic at 45-60 mph (20.1-26.8 m/s), no traffic walls across all lanes
    (two car lengths); the team starts at 70 mph (31.3 m/s) 0.6 s apart and may command up to 134 mph (60 m/s);
    the scripted overtaker cruises at 90 mph (40 m/s) looking 80 m (2 s) ahead."""
    s = scenario_v4()
    car = bmw_m3()
    return replace(s, version='v5', team_speed=31.3, vehicle=car, team_speed_max=60.0, team_start_headway=0.6,
                   overtaker_speed=40.0, overtaker_look=80.0,
                   traffic=replace(s.traffic, speed_min=20.1, speed_max=26.8, no_walls=True, wall_window=2*car.length))


SCENARIOS = {'v0': Scenario, 'v1': scenario_v1, 'v2': scenario_v2, 'v3': scenario_v3, 'v4': scenario_v4,
             'v5': scenario_v5}


def score_v1():
    """Rewards continuous close, fast driving: team combo on close passes, same-lane teammate
    proximity (20 m), and much smaller steady speed points."""
    return Score(version='v1', proximity_distance=20.0, speed_rate=0.25, speed_floor=25.0,
                 combo_step=0.5, combo_max=5.0, combo_window=5.0)


def score_v2():
    """v1 with proximity graded by distance: 20 pts/s per pair at contact, falling linearly to 0 at 20 m
    (10 pts/s at 10 m, about where 0.2 s drafting sits)."""
    return replace(score_v1(), version='v2', proximity_rate=20.0, proximity_falloff='linear')


def score_v3():
    """v2 plus lane-change maneuvers: a two-lane change, a close pass and a follow-up lane change scores most."""
    return replace(score_v2(), version='v3', lc_base=25.0, lc_pass_single=60.0, lc_pass_double=90.0,
                   lc_chain_single=150.0, lc_chain_double=300.0, lc_chain_window=3.0, lc_traffic_distance=10.0)


def score_v4():
    """v3 with drafting in place of proximity (5 pts/s following a teammate in the same lane within 1 s,
    both above 20 m/s) and team close passes (up to 2x with both teammates at contact, 0 bonus beyond 20 m)."""
    return replace(score_v3(), version='v4', proximity_rate=0.0, team_pass_gain=0.5, team_pass_distance=20.0,
                   draft_rate=5.0, draft_headway=1.0, draft_min_speed=20.0)


def score_v5():
    """v4 with drafting 5x more rewarding (25 pts/s)."""
    return replace(score_v4(), version='v5', draft_rate=25.0)


def score_v6():
    """v5 plus draft passes: a close pass by a car drafting a teammate (or within 1 s of leaving the draft) is worth 3x."""
    return replace(score_v5(), version='v6', draft_pass_gain=3.0, draft_pass_window=1.0)


def score_v7():
    """Team play: a three-car draft train (each car within 0.5 s bumper to bumper of the car ahead along the road, any
    lanes) earns 25 pts/s per car and 3x draft passes; a two-car draft keeps 5 pts/s and 1.5x passes. The team close-pass
    bonus needs all three cars within 40 m, and each pair beyond 50 m apart loses 2*((d - 50 m)/10 m)^2 pts/s, at most
    200 pts/s per pair (reached at 150 m). A collision also wipes 50 % of the team's banked points, and a car
    scores the same traffic car again only after 10 s."""
    return replace(score_v6(), version='v7', draft_rate=5.0, draft_pass_gain=1.5, train_rate=25.0, train_headway=0.5,
                   train_pass_gain=3.0, team_pass_distance=40.0, team_pass_require_all=True,
                   spread_penalty_rate=2.0, spread_free_distance=50.0, spread_scale=10.0, spread_penalty_cap=200.0,
                   collision_wipe=0.5, repass_cooldown=10.0)


def score_v8():
    """v7 made fast: draft-train points and train passes only above 80 mph (35.8 m/s); +10 pts/s per car above
    120 mph (53.6 m/s), times the combo; spread penalty doubled (4*((d - 50 m)/10 m)^2, capped at 400 pts/s per pair);
    team close-pass bonus with all three cars within 50 m (a 0.5 s train at 80 mph is about 45 m long)."""
    return replace(score_v7(), version='v8', train_min_speed=35.76, top_speed_rate=10.0, top_speed_threshold=53.64,
                   spread_penalty_rate=4.0, spread_penalty_cap=400.0, team_pass_distance=50.0)


SCORES = {'v0': Score, 'v1': score_v1, 'v2': score_v2, 'v3': score_v3, 'v4': score_v4, 'v5': score_v5, 'v6': score_v6,
          'v7': score_v7, 'v8': score_v8}
