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
    lane_change_cooldown: float = 8.0 # s after finishing a lane change before the same car may start another
    keep_right_bias: float = 0.0      # m/s^2 added to the incentive for moving right, subtracted for left

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k not in ('lane_changes', 'mobil_politeness', 'mobil_threshold',
                                                             'keep_right_bias')))
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
    # Drafting: following a teammate in the same lane within draft_headway, both above draft_min_speed,
    # earns draft_rate pts/s times the combo multiplier.
    draft_rate: float = 0.0           # 0 disables
    draft_headway: float = 1.0        # s: bumper gap <= draft_headway*follower speed
    draft_min_speed: float = 20.0     # m/s

    NON_NEGATIVE = ('combo_step', 'proximity_rate', 'team_pass_gain', 'draft_rate')

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k not in ('version', 'proximity_falloff', *self.NON_NEGATIVE)
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
    version: str = 'm3.3'
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
    emergency_decel: float = 6.0    # braking of the emergency escape's braking candidates (= brake_max)
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
                                                             'qp_reuse', 'nmpc_levenberg_marquardt')))
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

    def __post_init__(self):
        _positive(self, 'team_speed', 'team_start_s')
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


SCENARIOS = {'v0': Scenario, 'v1': scenario_v1, 'v2': scenario_v2, 'v3': scenario_v3, 'v4': scenario_v4}


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


SCORES = {'v0': Score, 'v1': score_v1, 'v2': score_v2, 'v3': score_v3, 'v4': score_v4}
