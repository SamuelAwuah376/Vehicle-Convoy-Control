"""Game score: turns Measurements into per-vehicle points using a versioned Score config."""
import numpy as np
from ..world import TEAM
from .events import PAIRS

TERMS = ('close_pass', 'proximity', 'speed', 'collision', 'maneuver', 'draft', 'spread', 'top_speed')
EPS = 1e-9


def close_pass_points(min_clearance, speed, cfg):
    if min_clearance >= cfg.pass_threshold:
        return 0.0
    closeness = 1+(cfg.pass_threshold-min_clearance)/cfg.pass_threshold
    return cfg.pass_base*closeness*speed/cfg.pass_ref_speed


def team_pass_multiplier(vehicle, pair_distance, cfg):
    """1 + team_pass_gain * sum over the passer's teammates of max(0, 1 - body-centre distance/team_pass_distance).
    With team_pass_require_all, 1 unless every teammate pair is within team_pass_distance (the whole team is close)."""
    if not cfg.team_pass_gain:
        return 1.0
    if cfg.team_pass_require_all and any(d >= cfg.team_pass_distance for d in pair_distance):
        return 1.0
    near = sum(max(0.0, 1-d/cfg.team_pass_distance) for (i, j), d in zip(PAIRS, pair_distance) if vehicle in (i, j))
    return 1+cfg.team_pass_gain*near


class Scorer:
    """Team combo: each scored close pass earns points*multiplier (multiplier from the combo before it),
    then raises the combo. A pass near teammates is first scaled by team_pass_multiplier. Proximity and
    drafting (following the nearest teammate ahead in the same lane within draft_headway) are multiplied
    too; speed and collisions are not. A close pass by a car drafting now or within draft_pass_window s
    (a draft pass) is also scaled by draft_pass_gain. While all three cars form a draft train (in a row along the road,
    any lanes, each within train_headway of the car ahead), every car earns train_rate instead of draft_rate, and a pass
    by any car while the train holds (or within draft_pass_window after) is scaled by train_pass_gain instead.
    A pass of a traffic car the same team car scored within repass_cooldown s scores nothing (and leaves the combo).
    A collision costs collision_penalty per involved car, plus collision_wipe of every car's banked points (the team's run). The team spread penalty (pairs beyond spread_free_distance,
    quadratic in the excess) is not multiplied. The combo resets
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
        self.team_pass = [] # team_pass_multiplier of each awarded pass
        self.draft_pass = []  # whether each awarded pass got a draft (pair or train) bonus
        self.train_pass = []  # whether each awarded pass got the draft-train bonus
        self.last_draft = np.full(TEAM, -np.inf)   # last time each car was drafting
        self.last_train = -np.inf                  # last time the three-car draft train held
        self.train_time = 0.0
        self.last_scored = {}  # (vehicle, traffic id) -> time of its last scored pass
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
        drafting = np.zeros(TEAM, bool)
        if m.draft_leader is not None:
            for i, j in enumerate(m.draft_leader):
                drafting[i] = j >= 0 and m.draft_gap[i] <= cfg.draft_headway*m.speeds[i]                     and min(m.speeds[i], m.speeds[j]) > cfg.draft_min_speed
        self.last_draft[drafting] = self.time
        train = bool(cfg.train_rate or cfg.train_pass_gain != 1.0) and m.train_gap is not None and \
            all(g <= cfg.train_headway*v for g, v in zip(m.train_gap, m.speeds) if np.isfinite(g)) and \
            min(m.speeds) > (cfg.train_min_speed or cfg.draft_min_speed)
        if train:
            self.last_train, self.train_time = self.time, self.train_time+m.dt
        for o in m.overtakes:
            points = close_pass_points(o.min_clearance, o.speed, cfg)
            key = (o.vehicle, o.traffic_id)
            if points and cfg.repass_cooldown and self.time-self.last_scored.get(key, -np.inf) < cfg.repass_cooldown-EPS:
                points = 0.0   # re-passing the same car too soon: no points, no combo
            if points:
                self.last_scored[key] = self.time
                team = team_pass_multiplier(o.vehicle, m.pair_distance, cfg)
                train_pass = cfg.train_pass_gain != 1.0 and self.time-self.last_train <= cfg.draft_pass_window+EPS
                draft_pass = self.time-self.last_draft[o.vehicle] <= cfg.draft_pass_window+EPS
                gain = cfg.train_pass_gain if train_pass else cfg.draft_pass_gain if draft_pass else 1.0
                points *= team*self.multiplier*gain
                self.team_pass.append(team); self.draft_pass.append(bool(train_pass or draft_pass))
                self.train_pass.append(bool(train_pass))
                inc['close_pass'][o.vehicle] += points
                self.awarded.append((o, points))
                self.passes.setdefault(o.vehicle, []).append(o.time)
                self.combo += 1; self.last_pass = self.time
                self.max_combo = max(self.max_combo, self.combo)
        for (i, j), distance in zip(PAIRS, m.pair_distance):
            if distance < cfg.proximity_distance and min(m.speeds[i], m.speeds[j]) > cfg.proximity_min_speed:
                weight = 1-distance/cfg.proximity_distance if cfg.proximity_falloff == 'linear' else 1.0
                inc['proximity'][[i, j]] += 0.5*cfg.proximity_rate*weight*m.dt*self.multiplier
        if train and cfg.train_rate:
            inc['draft'] += cfg.train_rate*m.dt*self.multiplier
        elif cfg.draft_rate:
            inc['draft'][drafting] += cfg.draft_rate*m.dt*self.multiplier
        if cfg.spread_penalty_rate:
            for (i, j), distance in zip(PAIRS, m.pair_distance):
                excess = max(0.0, distance-cfg.spread_free_distance)/cfg.spread_scale
                inc['spread'][[i, j]] -= 0.5*min(cfg.spread_penalty_rate*excess**2, cfg.spread_penalty_cap)*m.dt
        inc['speed'] += cfg.speed_rate*np.maximum(m.speeds-cfg.speed_floor, 0)*m.dt
        if cfg.top_speed_rate:
            inc['top_speed'] += cfg.top_speed_rate*(np.asarray(m.speeds) > cfg.top_speed_threshold)*m.dt*self.multiplier
        for c in m.collisions:
            inc['collision'][c.vehicle] -= cfg.collision_penalty
            if c.kind == 'team':
                inc['collision'][c.other] -= cfg.collision_penalty
        if cfg.collision_wipe and m.collisions:
            for i in range(TEAM):   # the whole team's run: banked points before this step's increments
                banked = sum(self.total[t][i] for t in TERMS)
                inc['collision'][i] -= cfg.collision_wipe*max(banked, 0.0)
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
