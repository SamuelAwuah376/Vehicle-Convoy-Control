from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Config:
    dt: float = 0.05
    duration: float = 90.0
    wheelbase: float = 2.7
    length: float = 4.5
    width: float = 1.8
    lane_width: float = 3.6
    cruise: float = 8.0
    standstill_gap: float = 5.0
    headway: float = 1.2
    accel_max: float = 2.0
    brake_max: float = 4.0
    steer_max: float = 0.5
    steer_rate: float = 0.6

    def __post_init__(self):
        for name, value in vars(self).items():
            if not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be positive')
        if self.width >= self.lane_width:
            raise ValueError('vehicle must fit in lane')
