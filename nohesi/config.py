from dataclasses import dataclass, field
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
    # Attribute names match convoy.config.Config so convoy dynamics/geometry/controllers reuse them.
    wheelbase: float = 2.7
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

    def __post_init__(self):
        _positive(self, *vars(self))


@dataclass(frozen=True)
class Traffic:
    count: int = 8
    speed_min: float = 18.0
    speed_max: float = 25.0
    spawn_behind: float = 80.0      # initial spawn window relative to the team
    spawn_ahead: float = 250.0
    recycle_behind: float = 150.0   # cars this far behind the rearmost teammate respawn ahead
    respawn_ahead: float = 250.0
    min_gap: float = 25.0           # rear-axle spacing between spawned cars in a lane
    team_lane_clear_ahead: float = 100.0
    team_lane_clear_behind: float = 40.0
    idm_accel: float = 1.5
    idm_brake: float = 2.0
    idm_brake_max: float = 8.0
    idm_gap: float = 2.0
    idm_headway: float = 1.5

    def __post_init__(self):
        _positive(self, *vars(self))
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

    def __post_init__(self):
        _positive(self, *(k for k in vars(self) if k != 'version'))


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
