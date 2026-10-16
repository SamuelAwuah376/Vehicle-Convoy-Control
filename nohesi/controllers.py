"""Low-level tracking laws used by lane keeping and the fallback rollouts."""
import numpy as np


def wrap(angle):
    return (angle + np.pi) % (2*np.pi) - np.pi


def pursuit(state, progress, route, cfg):
    """Rear-axle pure pursuit: state [x, y, yaw, v] at the rear axle, progress = its arc length on `route`."""
    lookahead = 3.0 + 0.65*state[3]
    target, _ = route.at(progress+lookahead)
    offset = target-state[:2]
    alpha = wrap(np.arctan2(offset[1], offset[0])-state[2])
    return np.arctan2(2*cfg.wheelbase*np.sin(alpha), max(np.linalg.norm(offset), 0.1))


def follower_acceleration(state, preceding, gap, cfg):
    """Gap control: standstill gap plus time headway, damped on the speed difference to the vehicle ahead."""
    desired = cfg.standstill_gap + cfg.headway*state[3]
    return 0.65*(gap-desired) + 1.3*(preceding[3]-state[3])
