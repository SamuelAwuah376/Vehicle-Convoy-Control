"""Maneuver intentions: turns policy actions into MPC targets with gap locking and commitment.

Action index = 9*(lane_cmd+2) + 3*(speed_cmd+1) + (lean_cmd+1) (45 actions), with lane_cmd -2/-1/0/+1/+2 =
two lanes right/right/keep/left/two lanes left (lane index increases to the left), speed_cmd -1/0/+1 =
slower/hold/faster and lean_cmd -1/0/+1 = offset the target lean_offset to the right/centre/left within the lane.
A two-lane change locks a destination-lane gap at double_gap_lookahead and needs a clear middle-lane corridor.
"""
from dataclasses import dataclass
import numpy as np
from .planning.base import Target
from .world import TEAM

ACTIONS = tuple((lane, speed, lean) for lane in (-2, -1, 0, 1, 2) for speed in (-1, 0, 1) for lean in (-1, 0, 1))


def encode(lane_cmd, speed_cmd, lean_cmd=0):
    return 9*(lane_cmd+2)+3*(speed_cmd+1)+(lean_cmd+1)


@dataclass
class Intention:
    lane: int                  # target lane; equals the current lane while keeping
    speed: float               # persistent target speed
    phase: str = 'keep'        # 'keep', 'pending' (not yet crossed) or 'committed'
    origin: int = None
    front: tuple = None        # locked gap's bounding cars, ('traffic', id) or ('team', index)
    rear: tuple = None
    fresh: bool = False        # accepted at this decision, not yet solved
    rejected: bool = False     # last lane request was rejected (observation flag)
    lean: int = 0              # -1/0/+1: target offset right/none/left of the lane centre; never masked


@dataclass(frozen=True)
class Record:
    time: float
    vehicle: int
    kind: str                  # accepted, rejected, aborted, masked, committed, completed, cancelled, relocked, unlocked
    reason: str = ''
    lane: int = -1
    gap_margin: float = float('nan')   # accepted/relocked: predicted bumper margin to the nearer gap car at the lookahead


class IntentionManager:
    def __init__(self, cfg, road, params, lane, speed):
        self.cfg, self.road, self.params = cfg, road, params
        self.intentions = [Intention(lane, float(np.clip(speed, cfg.speed_min, cfg.speed_max))) for _ in range(TEAM)]
        self.records = []

    def target(self, i):
        it = self.intentions[i]
        return Target(self.target_d(i), it.speed, it.front, it.rear)

    def target_d(self, i):
        it = self.intentions[i]
        return self.road.lane_center(it.lane)+it.lean*self.cfg.lean_offset

    @staticmethod
    def entered(origin, lane):
        """Lanes a change from `origin` to `lane` enters, in order."""
        step = int(np.sign(lane-origin))
        return list(range(origin+step, lane+step, step)) if step else []

    def _lane_vehicles(self, snap, plans, i, lane, horizon):
        """(key, s at `horizon`, s now, v now, v at `horizon`) of vehicles in, entering or crossing `lane`, excluding
        teammate i."""
        out = [(('traffic', int(snap.traffic_id[j])), snap.traffic_s[j]+snap.traffic_v[j]*horizon,
                snap.traffic_s[j], snap.traffic_v[j], snap.traffic_v[j])
               for j in np.flatnonzero(self.road.lane_of(snap.traffic_d) == lane)]
        for j in range(TEAM):
            if j == i:
                continue
            it = self.intentions[j]
            crossing = it.phase != 'keep' and lane in self.entered(it.origin, it.lane)
            if self.road.lane_of(snap.team_d[j]) == lane or crossing:
                plan, v = plans[j], snap.team[j, 3]
                z = plan.at(int(round(horizon/plan.step))) if plan is not None else (snap.team_s[j]+v*horizon, 0, 0, v)
                out.append((('team', j), z[0], snap.team_s[j], v, z[3]))
        return out

    def reach(self, v, speed, T, dt=0.1):
        """Distance and speed after T s tracking `speed` within the plant's accel (power-limited) and brake limits."""
        p, s = self.params, 0.0
        for _ in range(int(round(T/dt))):
            a = float(np.clip((speed-v)/dt, -p.brake_max, p.accel_max*min(1.0, p.power_speed/max(v, 1e-6))))
            s, v = s+v*dt+0.5*a*dt*dt, v+a*dt
        return s, v

    def gap_rule(self, snap, plans, i, lane, speed, middle=True):
        """Lock the gap containing the predicted end position: ((front, rear), '') or (None, reason).

        The end position uses the reachable speed profile. The gap must hold at the lookahead (double_gap_lookahead
        for a two-lane change); the rear gap car additionally needs rear_closing_time s of room per m/s it closes on
        the ego (now or at the lookahead). Gap cars may currently be alongside or on the other side (pass and cut
        in): the MPC keeps the car beside them until the reference reaches the gap. A two-lane change (with
        `middle`) also needs every vehicle in the middle lane clear of the ego's reachable position, bumper to
        bumper, throughout the crossing window.
        """
        if not 0 <= lane < self.road.cfg.lanes:
            return None, 'no_lane'
        me = self.intentions[i]
        origin = me.origin if me.phase != 'keep' else me.lane
        double, side = abs(lane-origin) == 2, int(np.sign(lane-origin))
        for j, it in enumerate(self.intentions):
            # A teammate entering a lane the ego enters (middle or destination) from the other side nearby: plans
            # would converge.
            if j == i or it.phase == 'keep' or abs(snap.team_s[j]-snap.team_s[i]) >= self.cfg.merge_window:
                continue
            shared = set(self.entered(origin, lane)) & set(self.entered(it.origin, it.lane))
            if shared and np.sign(it.lane-it.origin) != side:
                return None, 'teammate_merging'
        T, s, v = self.cfg.double_gap_lookahead if double else self.cfg.gap_lookahead, snap.team_s[i], snap.team[i, 3]
        run, v_end = self.reach(v, speed, T)
        end = s+run
        cars = self._lane_vehicles(snap, plans, i, lane, T)
        ahead = [c for c in cars if c[1] > end]
        behind = [c for c in cars if c[1] <= end]
        front = min(ahead, key=lambda c: c[1]) if ahead else None
        rear = max(behind, key=lambda c: c[1]) if behind else None
        L, m = self.params.length, self.cfg.margin_long
        self.last_gap_margin = min(front[1]-end if front else np.inf, end-rear[1] if rear else np.inf)-L
        if (front and front[1]-end < L+m) or (rear and end-rear[1] < L+m):
            return None, 'gap_too_short'
        if rear:
            closing = max(0.0, rear[3]-v, rear[4]-v_end)
            if end-rear[1]-L < m+closing*self.cfg.rear_closing_time:
                return None, 'rear_closing'
        if double and middle and self._middle_blocked(snap, plans, i, origin+side, speed):
            return None, 'middle_blocked'
        gap = (front[0] if front else None, rear[0] if rear else None)
        for j, it in enumerate(self.intentions):
            if j != i and it.phase != 'keep' and it.lane == lane and (it.front, it.rear) == gap:
                return None, 'gap_claimed'
        return gap, ''

    def _middle_blocked(self, snap, plans, i, lane, speed, dt=0.1):
        """Any vehicle in `lane` within length + margin_long of the ego (bumper overlap) during the crossing window."""
        c, s, v, far = self.cfg, snap.team_s[i], snap.team[i, 3], self.params.length+self.cfg.margin_long
        for t in np.arange(c.middle_window_start, c.middle_window_end+dt/2, dt):
            ego = s+self.reach(v, speed, t, dt)[0]
            if any(abs(car[1]-ego) < far for car in self._lane_vehicles(snap, plans, i, lane, t)):
                return True
        return False

    def decide(self, snap, actions, plans, order):
        for i in order:
            lane_cmd, speed_cmd, lean_cmd = ACTIONS[int(actions[i])]
            it = self.intentions[i]
            it.lean = lean_cmd
            it.speed = float(np.clip(it.speed+speed_cmd*self.cfg.speed_step, self.cfg.speed_min, self.cfg.speed_max))
            if it.phase == 'committed':
                if lane_cmd != it.lane-it.origin:
                    self.records.append(Record(snap.time, i, 'masked', '', it.lane))
                continue
            if it.phase == 'pending':
                if lane_cmd != it.lane-it.origin:
                    self.records.append(Record(snap.time, i, 'aborted', '', it.lane))
                    self._keep(i, it.origin)
                continue
            if lane_cmd == 0:
                continue
            lane = it.lane+lane_cmd
            gap, reason = self.gap_rule(snap, plans, i, lane, it.speed)
            it.rejected = gap is None
            if gap is None:
                self.records.append(Record(snap.time, i, 'rejected', reason, lane))
                continue
            it.origin, it.lane, it.phase, it.fresh = it.lane, lane, 'pending', True
            it.front, it.rear = gap
            self.records.append(Record(snap.time, i, 'accepted', '', lane, self.last_gap_margin))

    def update_phases(self, snap, plans=None):
        half = self.params.width/2
        for i, it in enumerate(self.intentions):
            d = snap.team_d[i]
            if it.phase == 'pending':
                side = int(np.sign(it.lane-it.origin))
                line = 0.5*(self.road.lane_center(it.origin)+self.road.lane_center(it.origin+side))   # first line crossed
                if (d-line)*side > -half:   # a footprint edge reached the lane line
                    it.phase = 'committed'
                    self.records.append(Record(snap.time, i, 'committed', '', it.lane))
            if it.phase == 'committed' and abs(d-self.target_d(i)) < self.cfg.done_tolerance:
                self.records.append(Record(snap.time, i, 'completed', '', it.lane))
                self._keep(i, it.lane)
            if it.phase != 'keep':
                self._check_lock(snap, plans, i)

    def _check_lock(self, snap, plans, i):
        """Re-lock, cancel (pending) or drop a stale gap car (committed) once a locked car passed the ego the wrong way."""
        it = self.intentions[i]
        pos = {('team', j): snap.team_s[j] for j in range(TEAM)}
        pos.update((('traffic', int(k)), s) for k, s in zip(snap.traffic_id, snap.traffic_s))
        it.front, it.rear = (k if k in pos else None for k in (it.front, it.rear))   # recycled traffic: unlocked
        far, s = self.params.length+self.cfg.margin_long, snap.team_s[i]
        stale_rear = it.rear is not None and pos[it.rear]-s > far
        stale_front = it.front is not None and s-pos[it.front] > far
        if not (stale_rear or stale_front):
            return
        gap, reason = self.gap_rule(snap, plans if plans is not None else [None]*TEAM, i, it.lane, it.speed,
                                    middle=it.phase == 'pending')
        if gap is not None:
            it.front, it.rear = gap
            self.records.append(Record(snap.time, i, 'relocked', '', it.lane, self.last_gap_margin))
        elif it.phase == 'pending':
            self.cancel(i, snap.time, 'stale_gap')
        else:
            it.rear, it.front = (None if stale_rear else it.rear), (None if stale_front else it.front)
            self.records.append(Record(snap.time, i, 'unlocked', reason, it.lane))

    def reject_fresh(self, i, time, reason):
        it = self.intentions[i]
        self.records.append(Record(time, i, 'rejected', reason, it.lane))
        self._keep(i, it.origin)
        it.rejected = True

    def cancel(self, i, time, reason):
        """Abandon a lane change: the vehicle returns to its origin lane."""
        it = self.intentions[i]
        self.records.append(Record(time, i, 'cancelled', reason, it.lane))
        self._keep(i, it.origin if it.phase != 'keep' else it.lane)

    def _keep(self, i, lane):
        it = self.intentions[i]
        it.lane, it.phase, it.origin, it.front, it.rear, it.fresh = lane, 'keep', None, None, None, False
