from dataclasses import dataclass, replace
import numpy as np
from .dynamics import DynamicPlant, kinematic_body, make_plant
from .geometry import footprint, overlap
from .road import Road
from .traffic import Traffic

TEAM = 3


@dataclass(frozen=True)
class Collision:
    vehicle: int
    kind: str       # 'traffic', 'team' or 'edge'
    other: int      # traffic id, teammate index, or -1 for the road edge


@dataclass(frozen=True)
class Snapshot:
    """Executed world state at one time. Arrays are copies; treat them as read-only."""
    time: float
    team: np.ndarray          # (3, 4) CG [x, y, yaw, v]
    team_steer: np.ndarray    # (3,)
    team_s: np.ndarray
    team_d: np.ndarray
    traffic_s: np.ndarray
    traffic_d: np.ndarray
    traffic_v: np.ndarray
    traffic_id: np.ndarray
    collisions: tuple
    done: bool
    traffic_vd: np.ndarray = None           # lateral speed (m/s); None means zeros
    traffic_target_lane: np.ndarray = None  # lane each car is heading to (its lane when not changing); None means lane_of(traffic_d)
    team_vy: np.ndarray = None    # (3,) body lateral CG speed; kinematic plant: v sin(beta). None means zeros
    team_r: np.ndarray = None     # (3,) yaw rate; kinematic plant: v cos(beta) tan(steer)/L


def traffic_states(snapshot, road):
    xy = road.to_xy(snapshot.traffic_s, snapshot.traffic_d)
    heading = road.heading(snapshot.traffic_s)
    if snapshot.traffic_vd is not None:
        heading = heading+np.arctan2(snapshot.traffic_vd, np.maximum(snapshot.traffic_v, 1e-3))
    return np.column_stack((xy, heading, snapshot.traffic_v))


def _nearest(snapshot, road, s, lane, skip, direction, include_team=True):
    s_all = np.r_[snapshot.team_s, snapshot.traffic_s]
    lane_all = np.r_[road.lane_of(snapshot.team_d), road.lane_of(snapshot.traffic_d)]
    v_all = np.r_[snapshot.team[:, 3], snapshot.traffic_v]
    offset = direction*(s_all-s)
    mask = (lane_all == lane) & (offset > 0)
    mask[skip] = False
    if not include_team:
        mask[:len(snapshot.team_s)] = False
    if not np.any(mask):
        return None
    j = np.flatnonzero(mask)[np.argmin(offset[mask])]
    return offset[j], v_all[j]


def vehicle_ahead(snapshot, road, s, lane, skip, include_team=True):
    """Nearest vehicle ahead of s in a lane, excluding teammate `skip`: (CG gap, speed) or None."""
    return _nearest(snapshot, road, s, lane, skip, 1, include_team)


def vehicle_behind(snapshot, road, s, lane, skip):
    """Nearest vehicle behind s in a lane, excluding teammate `skip`: (CG gap, speed) or None."""
    return _nearest(snapshot, road, s, lane, skip, -1)


class World:
    """Three team cars plus background traffic, advanced synchronously from one snapshot."""
    def __init__(self, scenario, seed):
        self.scenario, self.seed = scenario, seed
        self.road = Road(scenario.road)
        self.plant = make_plant(scenario.vehicle, scenario.timing.dt)
        v = scenario.vehicle
        spacing = v.length+v.standstill_gap+(scenario.team_start_headway or v.headway)*scenario.team_speed
        s = scenario.team_start_s+spacing*np.arange(TEAM)[::-1]
        d = self.road.lane_center(scenario.team_lane)
        self.team = np.column_stack((self.road.to_xy(s, np.full(TEAM, d)), self.road.heading(s),
                                     np.full(TEAM, scenario.team_speed)))
        self.steer = np.zeros(TEAM)
        self.team_lat = np.zeros((TEAM, 2))   # body [vy, r]; DynamicPlant state, kinematic relations otherwise
        self.stability = None   # stability.StabilityLayer, installed by the planner (dynamic plant): inner loop
        self.traffic = Traffic(scenario.traffic, self.road, np.random.default_rng(seed), s, scenario.team_lane)
        self.steps = 0
        self.time = 0.0
        self.snapshot = self._snapshot()
        if self.snapshot.done:
            raise ValueError('initial state is in collision')

    def step(self, controls):
        """Apply (3, 2) [acceleration, steering] commands; return the new snapshot and applied controls."""
        if self.snapshot.done:
            raise RuntimeError('episode is over')
        prev, dt = self.snapshot, self.scenario.timing.dt
        applied = np.empty((TEAM, 2))
        for i in range(TEAM):
            if isinstance(self.plant, DynamicPlant):
                inner = None if self.stability is None else self.stability.inner(i, self.time)
                out, a, self.steer[i] = self.plant.step(np.r_[self.team[i], self.team_lat[i]], controls[i][0],
                                                        controls[i][1], self.steer[i], inner)
                self.team[i], self.team_lat[i] = out[:4], out[4:]
                if self.stability is not None:
                    self.stability.buffer.post(i, self.time+dt, out, self.steer[i], a)
            else:
                self.team[i], a, self.steer[i] = self.plant.step(self.team[i], controls[i][0], controls[i][1],
                                                                 self.steer[i])
                self.team_lat[i] = kinematic_body(self.team[i][3], self.steer[i], self.plant.params)[1:]
            applied[i] = a, self.steer[i]
        # Traffic reacts to the team positions from the same pre-step snapshot.
        self.traffic.step(dt, prev.team_s, self.road.lane_of(prev.team_d), prev.team[:, 3], self.scenario.vehicle.length,
                          prev.team_d, self.scenario.vehicle.width)
        self.traffic.recycle(prev.team_s)
        self.steps += 1
        self.time = self.steps*dt
        self.snapshot = self._snapshot()
        return self.snapshot, applied

    def _snapshot(self):
        team_s, team_d = self.road.to_frenet(self.team[:, :2])
        snap = Snapshot(self.time, self.team.copy(), self.steer.copy(), team_s, team_d,
                        self.traffic.s.copy(), self.traffic.d, self.traffic.v.copy(),
                        self.traffic.id.copy(), (), False, self.traffic.vd, self.traffic.target_lane.copy(),
                        self.team_lat[:, 0].copy(), self.team_lat[:, 1].copy())
        collisions = self.collisions(snap)
        return replace(snap, collisions=collisions, done=bool(collisions))

    def collisions(self, snap):
        cfg = self.plant.params
        reach = cfg.length+cfg.width
        team_polys = [footprint(st, cfg) for st in snap.team]
        states = traffic_states(snap, self.road)
        found = []
        for i, poly in enumerate(team_polys):
            _, corner_d = self.road.to_frenet(poly)
            if np.max(np.abs(corner_d)) >= self.road.width/2:
                found.append(Collision(i, 'edge', -1))
            near = np.flatnonzero((np.abs(snap.traffic_s-snap.team_s[i]) < reach) &
                                  (np.abs(snap.traffic_d-snap.team_d[i]) < reach))
            for j in near:
                if overlap(poly, footprint(states[j], cfg)):
                    found.append(Collision(i, 'traffic', int(snap.traffic_id[j])))
        for i in range(TEAM):
            for j in range(i+1, TEAM):
                if overlap(team_polys[i], team_polys[j]):
                    found.append(Collision(i, 'team', j))
        return tuple(found)
