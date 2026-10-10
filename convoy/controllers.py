import numpy as np
from .dynamics import wrap


def pursuit(state, progress, route, cfg):
    lookahead = 3.0 + 0.65*state[3]
    target, _ = route.at(progress+lookahead)
    offset = target-state[:2]
    alpha = wrap(np.arctan2(offset[1], offset[0])-state[2])
    return np.arctan2(2*cfg.wheelbase*np.sin(alpha), max(np.linalg.norm(offset), 0.1))


def leader_acceleration(state, progress, time, route, cfg, scenario):
    # Stop the rear axle 4 m before the end so the entire body fits the route.
    # Critically damped position feedback avoids crossing the endpoint at low speed.
    remaining = route.total-progress-4.0
    target = cfg.cruise
    if scenario == 'braking' and 20 <= time < 27:
        target = min(target, 2.0)
    return min(1.2*(target-state[3]), remaining-2.0*state[3])


def follower_acceleration(state, preceding, gap, cfg):
    desired = cfg.standstill_gap + cfg.headway*state[3]
    return 0.65*(gap-desired) + 1.3*(preceding[3]-state[3])
