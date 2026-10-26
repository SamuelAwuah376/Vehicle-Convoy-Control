"""A calibrated CARLA car and its low-level controller (no CARLA import: testable without the client)."""
from dataclasses import dataclass, replace
import json
from pathlib import Path
import numpy as np

CALIBRATIONS = Path(__file__).with_name('calibrations')


@dataclass(frozen=True)
class Calibration:
    """A CARLA car (tuned by vehicle.tune) measured on test roads: our model's parameters, where its CG and measured
    velocity sit relative to the actor origin, and the maps the low-level controller inverts."""
    blueprint: str
    vehicle: dict          # config.Vehicle fields of our model of the car
    x_cg: float            # m, CG ahead of the actor origin
    x_velocity: float      # m, point whose velocity CARLA reports, ahead of the actor origin
    steer_speed: tuple     # m/s grid of steer_gain
    steer_gain: tuple      # rad of front wheel angle per unit steer command at those speeds
    speed: tuple           # m/s grid of the maps
    throttle: tuple        # throttle levels, increasing from 0 ...
    throttle_accel: tuple  # ... and their accelerations (levels, speeds), non-decreasing in the level
    brake: tuple           # brake levels, 0 first, with the throttle at brake_throttle ...
    brake_accel: tuple     # ... and their accelerations (levels, speeds), non-increasing in the level
    brake_throttle: tuple = ()  # throttle (by speed) where the engine's net torque is ~0: the floor, held while braking
    manual_gear: bool = False   # drive in first gear without the gearbox automation (vehicle.tune(autobox=False))

    @classmethod
    def load(cls, name):
        data = json.loads((CALIBRATIONS/f'{name}.json').read_text(encoding='utf-8'))
        return cls(**{k: tuple(map(tuple, v)) if isinstance(v, list) and v and isinstance(v[0], list) else
                      tuple(v) if isinstance(v, list) else v for k, v in data.items()})

    def apply(self, vehicle):
        """vehicle (config.Vehicle) with this car's measured parameters."""
        return replace(vehicle, **self.vehicle)

    def steer_gain_at(self, v):
        return float(np.interp(v, self.steer_speed, self.steer_gain))

    def floor_at(self, v):
        """Throttle floor at speed v: brake_throttle, or the lowest throttle level without it."""
        return float(np.interp(v, self.speed, self.brake_throttle)) if self.brake_throttle else float(self.throttle[0])


class Longitudinal:
    """Acceleration command -> (throttle, brake) for one car: the calibrated maps inverted at the current speed, plus
    an integral of the acceleration error (measured from the speed change per tick, within +-limit). The throttle never
    drops below the calibration's floor, braking included: in manual first gear less throttle brakes through the
    engine and the rear-biased drive (up to 6 m/s^2 at 40 m/s), which can destabilise the car; with the automatic
    gearbox it disengages the drivetrain, which re-engages with a jolt."""
    def __init__(self, cal, tick, k_i=4.0, limit=2.0):
        self.cal, self.tick, self.k_i, self.limit = cal, tick, k_i, limit
        self.speed = np.asarray(cal.speed)
        self.th, self.th_a = np.asarray(cal.throttle), np.asarray(cal.throttle_accel)
        self.br, self.br_a = np.asarray(cal.brake), np.asarray(cal.brake_accel)
        self.bias, self.v_prev, self.a_prev = 0.0, None, None

    def _at(self, table, v):
        return np.array([np.interp(v, self.speed, row) for row in table])

    def __call__(self, a, v):
        if self.v_prev is not None:
            err = self.a_prev-(v-self.v_prev)/self.tick
            self.bias = min(max(self.bias+self.k_i*err*self.tick, -self.limit), self.limit)
        self.v_prev, self.a_prev = v, a
        want = a+self.bias
        floor = self.cal.floor_at(v)
        up = self._at(self.th_a, v)
        if want >= np.interp(floor, self.th, up):
            return max(float(np.interp(want, up, self.th)), floor), 0.0
        down = self._at(self.br_a, v)[::-1]
        return floor, float(np.interp(want, down, self.br[::-1]))
