import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def step(state, acceleration, steering, previous_steering, cfg):
    """Rear-axle state [x,y,yaw,v]; bounded controls; midpoint integration.

    Velocity clips at zero without allowing artificial reverse travel.
    """
    a = float(np.clip(acceleration, -cfg.brake_max, cfg.accel_max))
    delta = float(np.clip(steering, previous_steering-cfg.steer_rate*cfg.dt,
                          previous_steering+cfg.steer_rate*cfg.dt))
    delta = float(np.clip(delta, -cfg.steer_max, cfg.steer_max))
    x, y, yaw, v = state
    active_dt = min(cfg.dt, v/-a) if a < 0 else cfg.dt
    vm = max(0, v + 0.5*a*active_dt)
    yaw_rate = vm*np.tan(delta)/cfg.wheelbase
    heading = yaw + 0.5*yaw_rate*active_dt
    result = np.array([x+vm*np.cos(heading)*active_dt,
                       y+vm*np.sin(heading)*active_dt,
                       wrap(yaw+yaw_rate*active_dt), max(0, v+a*cfg.dt)])
    return result, a, delta
