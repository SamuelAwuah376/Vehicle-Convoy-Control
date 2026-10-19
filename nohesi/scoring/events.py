"""Measurements and events taken from executed snapshots only. No scoring formula lives here."""
from dataclasses import dataclass
import numpy as np
from ..geometry import body_centre, footprint, clearance
from ..world import TEAM, traffic_states

PAIRS = ((0, 1), (0, 2), (1, 2))


@dataclass(frozen=True)
class Overtake:
    time: float
    vehicle: int
    traffic_id: int
    min_clearance: float   # minimum footprint clearance while longitudinally alongside
    speed: float           # team car speed when the overtake completed


@dataclass(frozen=True)
class LaneChange:
    time: float            # when the car settled in the new lane
    vehicle: int
    start_time: float      # last time it was settled in the old lane (left it)
    from_lane: int
    to_lane: int
    size: int              # abs(to_lane - from_lane)
    min_traffic_ds: float  # min |CG s difference| to traffic in from/to/between lanes over the move; inf if none
    min_speed: float       # min team-car speed over the move


@dataclass(frozen=True)
class Measurements:
    dt: float
    speeds: np.ndarray            # (3,)
    pair_distance: np.ndarray     # (3,) body-centre distances for PAIRS
    pair_clearance: np.ndarray    # (3,) footprint clearances for PAIRS
    traffic_clearance: np.ndarray # (3,) nearest clearance to traffic alongside (inf if none)
    overtakes: tuple
    collisions: tuple
    lane_changes: tuple = ()
    team_s: np.ndarray = None       # (3,) CG s
    team_d: np.ndarray = None       # (3,) CG d
    team_lane: np.ndarray = None    # (3,) int, road.lane_of(team_d)
    draft_leader: np.ndarray = None # (3,) int: nearest teammate ahead in the same lane, -1 if none
    draft_gap: np.ndarray = None    # (3,) bumper gap (m) to draft_leader, inf if none
    train_gap: np.ndarray = None    # (3,) bumper gap (m) along the road to the next teammate ahead in any lane
                                    # (negative when overlapping alongside), inf for the front car


def draft_leaders(team_s, team_lane, length, traffic_s=(), traffic_lane=()):
    """Teammate directly ahead in the same lane for each car: the nearest vehicle ahead (bumper gap >= 0) must be a
    teammate, not traffic. Returns (leader (3,) int, -1 if none; gap (3,), inf if none)."""
    leader, gap = np.full(TEAM, -1), np.full(TEAM, np.inf)
    traffic_s, traffic_lane = np.asarray(traffic_s, float), np.asarray(traffic_lane, int)
    for i in range(TEAM):
        for j in range(TEAM):
            g = team_s[j]-team_s[i]-length
            if j != i and team_lane[j] == team_lane[i] and 0 <= g < gap[i]:
                leader[i], gap[i] = j, g
        if leader[i] >= 0:
            between = (traffic_lane == team_lane[i]) & (traffic_s > team_s[i]) & (traffic_s < team_s[leader[i]])
            if np.any(between):   # a traffic car sits between: not drafting
                leader[i], gap[i] = -1, np.inf
    return leader, gap


def train_gaps(team_s, length):
    """Bumper gap along the road from each car to the next teammate ahead of it in s order, in any lane (negative when
    they overlap alongside); inf for the front car."""
    order = np.argsort(team_s, kind='stable')
    gap = np.full(TEAM, np.inf)
    for back, front in zip(order[:-1], order[1:]):
        gap[back] = team_s[front]-team_s[back]-length
    return gap


class EventDetector:
    """Tracks each team car's passes of traffic. A pass window opens when the cars overlap
    longitudinally (within `margin`) and laterally within `lateral_window`; an overtake is
    reported when the window closes with the team car ahead after entering from behind.
    A lane change is reported when a car's whole footprint has been inside a new lane for
    `dwell` seconds; brief passes through a middle lane do not split the move."""
    def __init__(self, vehicle, road, margin=1.0, lateral_window=6.0, dwell=0.5):
        self.vehicle, self.road = vehicle, road
        self.window = vehicle.length+margin
        self.lateral_window = lateral_window
        self.active = {}   # (vehicle, traffic id) -> [entered_behind, min_clearance]
        self.dwell = dwell
        self.inside_tol = (road.cfg.lane_width-vehicle.width)/2
        self.lane = {}     # vehicle -> dict(lane, left, cand, since, ds {lane: min |ds|}, v)

    def _lane_changes(self, snap):
        lanes = self.road.lane_of(snap.team_d)
        inside = np.abs(snap.team_d-np.array([self.road.lane_center(l) for l in lanes])) <= self.inside_tol+1e-9
        traffic_lanes = self.road.lane_of(snap.traffic_d)
        out = []
        for i in range(TEAM):
            lane, t = int(lanes[i]), snap.time
            st = self.lane.get(i)
            if st is None:
                self.lane[i] = dict(lane=lane, left=t, cand=None, since=t, ds={}, v=np.inf)
                continue
            if inside[i] and lane == st['lane']:
                st.update(left=t, cand=None, ds={}, v=np.inf)
                continue
            st['v'] = min(st['v'], float(snap.team[i, 3]))
            for l, ds in zip(traffic_lanes, np.abs(snap.team_s[i]-snap.traffic_s)):
                st['ds'][int(l)] = min(st['ds'].get(int(l), np.inf), float(ds))
            if not inside[i]:
                st['cand'] = None
                continue
            if st['cand'] != lane:
                st.update(cand=lane, since=t)
            if t-st['since'] >= self.dwell-1e-9:
                a, b = sorted((st['lane'], lane))
                ds = min((st['ds'].get(l, np.inf) for l in range(a, b+1)), default=np.inf)
                out.append(LaneChange(t, i, st['left'], st['lane'], lane, b-a, ds, st['v']))
                st.update(lane=lane, left=t, cand=None, ds={}, v=np.inf)
        return out

    def update(self, snap, dt):
        v = self.vehicle
        team_polys = [footprint(st, v) for st in snap.team]
        centres = body_centre(snap.team, v)
        pair_distance = np.array([np.linalg.norm(centres[i]-centres[j]) for i, j in PAIRS])
        pair_clearance = np.array([clearance(team_polys[i], team_polys[j]) for i, j in PAIRS])
        states = traffic_states(snap, self.road)
        ids = set(int(k) for k in snap.traffic_id)
        traffic_clearance = np.full(TEAM, np.inf)
        overtakes = []
        for i in range(TEAM):
            ds = snap.team_s[i]-snap.traffic_s
            inside = (np.abs(ds) < self.window) & (np.abs(snap.team_d[i]-snap.traffic_d) < self.lateral_window)
            for j in np.flatnonzero(inside):
                key = (i, int(snap.traffic_id[j]))
                c = clearance(team_polys[i], footprint(states[j], v))
                traffic_clearance[i] = min(traffic_clearance[i], c)
                entry = self.active.setdefault(key, [ds[j] < 0, c])
                entry[1] = min(entry[1], c)
            for j in np.flatnonzero(~inside):
                key = (i, int(snap.traffic_id[j]))
                if key in self.active:
                    entered_behind, c = self.active.pop(key)
                    if entered_behind and ds[j] > 0:
                        overtakes.append(Overtake(snap.time, i, key[1], c, float(snap.team[i, 3])))
        # Windows of recycled cars close without an overtake.
        for key in [k for k in self.active if k[1] not in ids]:
            del self.active[key]
        lanes = self.road.lane_of(snap.team_d)
        leader, gap = draft_leaders(snap.team_s, lanes, v.length, snap.traffic_s, self.road.lane_of(snap.traffic_d))
        return Measurements(dt, snap.team[:, 3].copy(), pair_distance, pair_clearance,
                            traffic_clearance, tuple(overtakes), snap.collisions,
                            tuple(self._lane_changes(snap)), np.array(snap.team_s, float), np.array(snap.team_d, float),
                            lanes, leader, gap, train_gaps(np.asarray(snap.team_s, float), v.length))
