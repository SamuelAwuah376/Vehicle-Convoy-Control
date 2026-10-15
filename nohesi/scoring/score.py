"""Game score: turns Measurements into per-vehicle points using a versioned Score config."""
import numpy as np
from ..world import TEAM
from .events import PAIRS

TERMS = ('close_pass', 'proximity', 'speed', 'collision')


def close_pass_points(min_clearance, speed, cfg):
    if min_clearance >= cfg.pass_threshold:
        return 0.0
    closeness = 1+(cfg.pass_threshold-min_clearance)/cfg.pass_threshold
    return cfg.pass_base*closeness*speed/cfg.pass_ref_speed


class Scorer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.total = {term: np.zeros(TEAM) for term in TERMS}
        self.awarded = []   # (Overtake, points)

    def update(self, m):
        """Return this step's per-term (3,) increments and add them to the totals."""
        cfg, inc = self.cfg, {term: np.zeros(TEAM) for term in TERMS}
        for o in m.overtakes:
            points = close_pass_points(o.min_clearance, o.speed, cfg)
            if points:
                inc['close_pass'][o.vehicle] += points
                self.awarded.append((o, points))
        for (i, j), distance in zip(PAIRS, m.pair_distance):
            if distance < cfg.proximity_distance and min(m.speeds[i], m.speeds[j]) > cfg.proximity_min_speed:
                inc['proximity'][[i, j]] += 0.5*cfg.proximity_rate*m.dt
        inc['speed'] += cfg.speed_rate*np.maximum(m.speeds-cfg.speed_floor, 0)*m.dt
        for c in m.collisions:
            inc['collision'][c.vehicle] -= cfg.collision_penalty
            if c.kind == 'team':
                inc['collision'][c.other] -= cfg.collision_penalty
        for term in TERMS:
            self.total[term] += inc[term]
        return inc

    def individual(self):
        return sum(self.total.values())

    def team(self):
        return float(self.individual().sum())
