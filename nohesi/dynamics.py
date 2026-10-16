from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
from convoy.dynamics import step as kinematic_step


class KinematicPlant:
    """Stage A plant: convoy's kinematic bicycle plus speed-dependent drive and grip limits.

    State is [x, y, yaw, v] at the rear axle. The grip cap clips the steering command
    before convoy's rate limit, so lateral acceleration can briefly exceed it while
    steering unwinds after a speed increase.
    """
    def __init__(self, vehicle, dt):
        self.params = SimpleNamespace(**asdict(vehicle), dt=dt)

    # Limits accept scalars or arrays (the MPC evaluates them along the horizon).
    def accel_limit(self, speed):
        p = self.params
        return p.accel_max*np.minimum(1.0, p.power_speed/np.maximum(speed, 1e-6))

    def steer_limit(self, speed):
        p = self.params
        return np.minimum(p.steer_max, np.arctan(p.lateral_accel_max*p.wheelbase/np.maximum(speed, 1.0)**2))

    def step(self, state, acceleration, steering, previous_steering):
        a = min(acceleration, self.accel_limit(state[3]))
        cap = self.steer_limit(state[3])
        return kinematic_step(state, a, float(np.clip(steering, -cap, cap)), previous_steering, self.params)
