"""Settings for the convoy parking (staging-yard) problem: the truck, the simulated sensor suite, the site layout, the
planner and the coordination layer. Everything in SI units (m, s, m/s, rad). Bump a version when a value changes.

The truck is referenced at the REAR AXLE centre (x, y, yaw, v), with v signed: v < 0 is reversing. Forward and reverse
paths are then the same geometric curves (Dubins/Reeds-Shepp conventions), and the kinematic bicycle needs no slip
angle: x' = v cos(yaw), y' = v sin(yaw), yaw' = v tan(steer)/wheelbase.
"""
from dataclasses import dataclass, field
from math import isfinite, tan


def _positive(obj, *names):
    for name in names:
        value = getattr(obj, name)
        if not isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be positive')


@dataclass(frozen=True)
class Truck:
    """Rigid haul truck, representative of a ~40 t payload quarry / mid-size mining class (approximate, public-brochure
    scale figures: replace with the OEM's data for a real fleet). Yard manoeuvring limits, not road limits: at
    walking-to-jogging speeds the tyres are far from their grip limit, so a kinematic bicycle is the right model."""
    version: str = 't1'
    length: float = 9.5             # bumper to bumper (m)
    width: float = 3.8              # over the tyres / body (m)
    wheelbase: float = 4.2          # front axle to rear axle (m)
    rear_overhang: float = 2.1      # rear axle to rear bumper (m); front overhang = length - wheelbase - rear_overhang
    steer_max: float = 0.48         # rad: rear-axle turning radius wheelbase/tan(steer_max) = 8.1 m
    steer_rate: float = 0.25        # rad/s: hydraulic steering
    v_forward_max: float = 3.0      # m/s (10.8 km/h): yard speed limit forward
    v_reverse_max: float = 1.5      # m/s (5.4 km/h): reversing limit
    v_creep: float = 0.7            # m/s: CREEP action speed (both directions)
    accel_max: float = 0.6          # m/s^2: loaded truck at yard speeds
    brake_comfort: float = 0.5      # m/s^2: planned stops (cusps, holds, the spot)
    brake_max: float = 2.0          # m/s^2: service brake
    brake_emergency: float = 3.5    # m/s^2: the safety supervisor's stop
    accel_lag: float = 0.4          # s: first-order lag of the drivetrain / brakes on the commanded acceleration
    lat_accel_max: float = 0.8      # m/s^2: comfort and load-stability cap on v^2 * curvature

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k != 'version'))
        if self.wheelbase+self.rear_overhang >= self.length:
            raise ValueError('wheelbase + rear_overhang must leave a front overhang')
        if self.v_creep > min(self.v_forward_max, self.v_reverse_max):
            raise ValueError('v_creep must not exceed the forward or reverse limit')

    @property
    def front_overhang(self):
        return self.length-self.wheelbase-self.rear_overhang

    @property
    def min_radius(self):
        """Rear-axle turning radius at full lock (m)."""
        return self.wheelbase/tan(self.steer_max)


@dataclass(frozen=True)
class Sensors:
    """The simulated sensor suite (docs/yard_parking_design.md has the hardware it stands for).

    Leader scan: a 360 deg roof lidar on the leader's cab, ray-cast in 2D against the obstacles; occluded space stays
    unknown, and a low or dust-hidden obstacle is missed with probability miss_prob (it does not block rays and reads as
    free). Followers: 360 deg near-field coverage (radar + ultrasonic) within near_range, which reveals missed obstacles
    to the shared map (V2V) and feeds the safety supervisor. Poses come from RTK-GNSS/INS, exact in this version."""
    version: str = 's1'
    lidar_range: float = 60.0       # m: usable range in dust (well below a clear-air spec sheet)
    lidar_resolution: float = 0.5   # deg between rays
    lidar_mount: float = 6.0        # m ahead of the rear axle (cab roof)
    miss_prob: float = 0.15         # chance a small obstacle (rock pile, log) is not in the leader's scan
    near_range: float = 10.0        # m: follower near-field sensing (radar + ultrasonic), 360 deg
    grid_resolution: float = 0.5    # m per occupancy-grid cell

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k not in ('version', 'miss_prob')))
        if not 0 <= self.miss_prob < 1:
            raise ValueError('miss_prob must lie in [0, 1)')


@dataclass(frozen=True)
class Site:
    """A staging yard (a pad beside the haul road, a log landing) entered through a gap in its berm from an access
    road. x along the access road into the yard, y across; the yard spans [0, length] x [0, width]."""
    version: str = 'y1'
    length: float = 100.0           # m along x
    width: float = 70.0             # m along y
    entrance_width: float = 16.0    # m: gap in the yard's west berm, centred on the access road
    road_length: float = 60.0       # m: access road west of the yard (x in [-road_length, 0])
    road_width: float = 16.0
    berm_width: float = 2.0
    rock_piles: tuple = (2, 6)      # (min, max) count: 2.5-6 m squares, missable
    parked: tuple = (0, 3)          # parked equipment: 8 x 3.5 m, never missed
    windrows: tuple = (0, 2)        # long low rock / log windrows: 12-20 x 1.5 m, missable
    keep_clear: float = 18.0        # m: no obstacle (and no spot) this close to the entrance centre inside the yard
    leader_pose: tuple = (12.0, 56.0, 0.0)   # the leader's scan pose (rear axle), pulled aside just inside the yard
    follower_gap: float = 14.0      # m between queued followers' rear axles on the access road
    max_followers: int = 3

    def __post_init__(self):
        _positive(self, 'length', 'width', 'entrance_width', 'road_length', 'road_width', 'berm_width', 'keep_clear',
                  'follower_gap')
        if not 1 <= self.max_followers <= 3:
            raise ValueError('max_followers must be 1-3 (the policy has three follower slots)')


@dataclass(frozen=True)
class Planning:
    """Hybrid A*, spot validation, coordination and MPC settings."""
    version: str = 'p1'
    # Hybrid A* (hybrid_astar.py)
    xy_resolution: float = 1.0      # m: closed-set cell (the map itself is Sensors.grid_resolution)
    yaw_bins: int = 72              # 5 deg
    step: float = 2.0               # m of arc per motion primitive
    steer_fractions: tuple = (-1.0, -0.5, 0.0, 0.5, 1.0)   # of steer_max
    reverse_cost: float = 1.5       # x arc length driven in reverse
    switch_cost: float = 6.0        # m-equivalent per change of direction (a cusp: stop, shift, restart)
    steer_cost: float = 0.3         # m-equivalent per unit |steer fraction| per primitive
    steer_change_cost: float = 0.3  # ... per unit change of steer fraction
    analytic_every: int = 4         # try a Dubins shot (forward and reverse) every this many expansions
    analytic_range: float = 45.0    # m: ... when the goal is this close
    max_expansions: int = 6000
    heuristic_weight: float = 1.5   # f = g + w h (w > 1: greedier, faster, paths at most w times the optimum)
    clearance: float = 0.6          # m: body-to-obstacle margin of a planned path
    goal_tolerance: float = 0.3     # m / (x 0.1 rad): a path ends exactly at the spot (analytic shot) or this close
    # Spots (spots.py)
    spot_margin: float = 1.0        # m of known-free ground around a spot's footprint
    spot_spacing: float = 2.5       # m between two assigned spots' footprints
    spot_lattice: float = 3.0       # m between the driver model's candidate spots
    # Coordination (coordination.py)
    zone_margin: float = 1.0        # m: two footprints closer than this conflict
    gate_buffer: float = 2.0        # m of path before a conflict zone where a truck without the zone must stop
    path_sample: float = 0.5        # m between path samples
    # MPC (mpc.py)
    mpc_horizon: int = 15
    mpc_step: float = 0.2           # s
    w_pos: float = 4.0
    w_yaw: float = 8.0
    w_speed: float = 1.0
    w_accel: float = 0.5
    w_steer_rate: float = 4.0
    track_tolerance: float = 1.5    # m: farther than this from its path, a truck stops and must replan
    park_tolerance: float = 0.5     # m (and 0.1 rad): parked within this of its spot, standing still
    # Episode
    dt: float = 0.1                 # s: simulation step
    decision_interval: float = 0.5  # s: policy decisions
    duration: float = 300.0         # s
    stall_time: float = 40.0        # s: no truck makes progress for this long -> the episode ends (stalled)
    replan_cooldown: float = 2.0    # s between a truck's replans

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k not in ('version', 'steer_fractions', 'yaw_bins', 'max_expansions',
                                                             'analytic_every', 'mpc_horizon')))
        if round(self.decision_interval/self.dt) != self.decision_interval/self.dt or \
                round(self.mpc_step/self.dt) != self.mpc_step/self.dt:
            raise ValueError('decision_interval and mpc_step must be multiples of dt')


@dataclass(frozen=True)
class YardScenario:
    version: str = 'yard1'
    truck: Truck = field(default_factory=Truck)
    sensors: Sensors = field(default_factory=Sensors)
    site: Site = field(default_factory=Site)
    planning: Planning = field(default_factory=Planning)
    followers: tuple = (1, 3)       # (min, max) followers per episode, drawn per seed
    driver_bad_pick: float = 0.3    # chance a driver pick is a free click (often unsafe: exercises the rejection path)

    def __post_init__(self):
        lo, hi = self.followers
        if not 1 <= lo <= hi <= self.site.max_followers:
            raise ValueError('followers must satisfy 1 <= min <= max <= site.max_followers')
        if not 0 <= self.driver_bad_pick < 1:
            raise ValueError('driver_bad_pick must lie in [0, 1)')
        if self.truck.width+2*self.planning.clearance >= self.site.entrance_width:
            raise ValueError('the truck must fit through the entrance with its clearance')
