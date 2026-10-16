"""Game score: turns Measurements into per-vehicle points using a versioned Score config."""
import numpy as np
from ..world import TEAM
from .events import PAIRS

TERMS = ('close_pass', 'proximity', 'speed', 'collision', 'maneuver')
EPS = 1e-9


def close_pass_points(min_clearance, speed, cfg):
    if min_clearance >= cfg.pass_threshold:
        return 0.0
    closeness = 1+(cfg.pass_threshold-min_clearance)/cfg.pass_threshold
    return cfg.pass_base*closeness*speed/cfg.pass_ref_speed


class Scorer:
    """Team combo: each scored close pass earns points*multiplier (multiplier from the combo before it),
    then raises the combo. Proximity is multiplied too; speed and collisions are not. The combo resets
    after combo_window s without a scored close pass and on any collision.

    Maneuvers: an eligible lane change L1 (traffic within lc_traffic_distance, speed above lc_min_speed)
    links the car's first unused scored pass in [L1.start_time, L1.time+lc_chain_window]; an eligible
    follow-up change starting at/after the pass and settling within lc_chain_window of it makes a chain.
    Each maneuver earns one award (chain, pass, or base by L1 size) times the multiplier when it resolves
    (follow-up, window expiry, or the car's next eligible change); awards never change the combo. Each car
    has at most one pending maneuver; every eligible change starts a new one. Collisions drop all pending."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.total = {term: np.zeros(TEAM) for term in TERMS}
        self.awarded = []   # (Overtake, points incl. multiplier)
        self.time, self.last_pass = 0.0, 0.0
        self.combo, self.max_combo = 0, 0
        self.passes = {}      # vehicle -> scored pass times
        self.used = set()     # (vehicle, pass time) already linked to a maneuver
        self.pending = {}     # vehicle -> [LaneChange, linked pass time or None]
        self.maneuvers = []   # resolved awards (dicts)

    @property
    def multiplier(self):
        return min(self.cfg.combo_max, 1+self.cfg.combo_step*self.combo)

    def update(self, m):
        """Return this step's per-term (3,) increments and add them to the totals."""
        cfg, inc = self.cfg, {term: np.zeros(TEAM) for term in TERMS}
        self.time += m.dt
        if self.combo and self.time-self.last_pass > cfg.combo_window:
            self.combo = 0
        for o in m.overtakes:
            points = close_pass_points(o.min_clearance, o.speed, cfg)
            if points:
                points *= self.multiplier
                inc['close_pass'][o.vehicle] += points
                self.awarded.append((o, points))
                self.passes.setdefault(o.vehicle, []).append(o.time)
                self.combo += 1; self.last_pass = self.time
                self.max_combo = max(self.max_combo, self.combo)
        for (i, j), distance in zip(PAIRS, m.pair_distance):
            if distance < cfg.proximity_distance and min(m.speeds[i], m.speeds[j]) > cfg.proximity_min_speed:
                weight = 1-distance/cfg.proximity_distance if cfg.proximity_falloff == 'linear' else 1.0
                inc['proximity'][[i, j]] += 0.5*cfg.proximity_rate*weight*m.dt*self.multiplier
        inc['speed'] += cfg.speed_rate*np.maximum(m.speeds-cfg.speed_floor, 0)*m.dt
        for c in m.collisions:
            inc['collision'][c.vehicle] -= cfg.collision_penalty
            if c.kind == 'team':
                inc['collision'][c.other] -= cfg.collision_penalty
        if m.collisions:
            self.combo = 0; self.pending.clear()
        else:
            self._maneuvers(m, inc['maneuver'])
        for term in TERMS:
            self.total[term] += inc[term]
        return inc

    def _link(self, v, lo, hi):
        t = min((t for t in self.passes.get(v, ()) if lo-EPS <= t <= hi+EPS and (v, t) not in self.used), default=None)
        if t is not None:
            self.used.add((v, t))
        return t

    def _award(self, v, kind, inc, follow=None):
        l1, pass_time = self.pending.pop(v)
        if kind != 'base':
            kind += '_double' if l1.size >= 2 else '_single'
        points = getattr(self.cfg, f'lc_{kind}')*self.multiplier
        inc[v] += points
        self.maneuvers.append(dict(time=self.time, vehicle=v, kind=kind, points=points, start_time=l1.start_time,
                                   lane_change_time=l1.time, from_lane=l1.from_lane, to_lane=l1.to_lane,
                                   size=l1.size, pass_time=pass_time, follow_up_time=follow))

    def _maneuvers(self, m, inc):
        cfg, window = self.cfg, self.cfg.lc_chain_window
        for v, p in self.pending.items():   # passes this step link to a pass-less pending maneuver
            if p[1] is None:
                p[1] = self._link(v, p[0].start_time, p[0].time+window)
        for lc in getattr(m, 'lane_changes', ()):
            if lc.min_traffic_ds > cfg.lc_traffic_distance or lc.min_speed <= cfg.lc_min_speed:
                continue
            v = lc.vehicle
            if v in self.pending:
                pass_time = self.pending[v][1]
                if pass_time is None:
                    self._award(v, 'base', inc)
                elif lc.start_time >= pass_time-EPS and lc.time <= pass_time+window+EPS:
                    self._award(v, 'chain', inc, lc.time)
                else:
                    self._award(v, 'pass', inc)
            self.pending[v] = [lc, self._link(v, lc.start_time, lc.time)]
        for v, (l1, pass_time) in list(self.pending.items()):
            if pass_time is None and self.time > l1.time+window+EPS:
                self._award(v, 'base', inc)
            elif pass_time is not None and self.time > pass_time+window+EPS:
                self._award(v, 'pass', inc)

    def individual(self):
        return sum(self.total.values())

    def team(self):
        return float(self.individual().sum())
