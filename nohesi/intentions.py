"""Maneuver intentions: turns policy actions into MPC targets with gap locking and commitment.

Action index = 9*(lane_cmd+2) + 3*(speed_cmd+1) + (lean_cmd+1) (45 actions), with lane_cmd -2/-1/0/+1/+2 =
two lanes right/right/keep/left/two lanes left (lane index increases to the left), speed_cmd -1/0/+1 =
slower/hold/faster and lean_cmd -1/0/+1 = offset the target lean_offset to the right/centre/left within the lane.
A two-lane change locks a destination-lane gap at double_gap_lookahead and needs a clear middle-lane corridor.

Overtakes (prototype; indices after the 45, so the RL env's action space is unchanged): OVERTAKES[n] = (side,
speed_cmd), encode_overtake. An overtake passes the nearest traffic car ahead in the lane (within overtake_look) on the
left (+1) or right (-1) at least overtake_margin*(2+speed_cmd) m/s faster (3/6/9 by default, above the car's current
target speed if that is higher). Stages: 'out' (a lane change into the passing lane: its gap rule, plus a passing
corridor clear until the ego is past the car; the MPC's lateral reference blends into the passing lane over
overtake_merge_time), 'pass' (in the passing lane; each step the gap rule back into the home
lane is tried, and accepted once its gap lies ahead of the passed car), 'return' (that lane change; the MPC keeps to
the passing lane until its reference clears the passed car: overtake_release). With no room to return within
overtake_pass_max s of reaching the passing lane, the overtake ends there ('stayed'): a normal outcome, not an abort. The policy repeats the overtake action to continue; any other action ends the overtake
(stays where the car is going). The RL env (e5) masks other actions during an overtake, so there it runs to completion.
"""
from dataclasses import dataclass
import numpy as np
from .planning.base import Target
from .prediction import traffic_lanes
from .world import TEAM

ACTIONS = tuple((lane, speed, lean) for lane in (-2, -1, 0, 1, 2) for speed in (-1, 0, 1) for lean in (-1, 0, 1))
OVERTAKES = tuple((side, speed) for side in (-1, 1) for speed in (-1, 0, 1))


def encode(lane_cmd, speed_cmd, lean_cmd=0):
    return 9*(lane_cmd+2)+3*(speed_cmd+1)+(lean_cmd+1)


def encode_overtake(side, speed_cmd=0):
    return len(ACTIONS)+3*((side+1)//2)+(speed_cmd+1)


@dataclass
class Overtake:
    car: tuple                 # ('traffic', id) being passed
    side: int                  # +1 left, -1 right
    home: int                  # lane the overtake started from (the return lane)
    margin: float              # m/s the pass is at least faster than the car
    stage: str = 'out'         # 'out', 'pass' or 'return'
    since: float = None        # overtake start
    passing: float = None      # time the car reached the passing lane
    merge_start: float = None  # return: when the reference clears the passed car (predicted; fixed once clear now)
    merge_fixed: bool = False
    out_from: float = None     # car's d when the overtake started (the pull-out's blend starts there)
    out_time: float = None     # s the pull-out's blend takes (shorter when closing fast on the car)


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
    since: float = None        # time the current phase ('pending' or 'committed') began
    best: float = np.inf       # committed: smallest |d - target lane centre| so far, updated by > commit_progress_eps
    best_time: float = None    # when `best` last improved
    overtake: Overtake = None  # active overtake (OVERTAKES actions), else None


@dataclass(frozen=True)
class Record:
    time: float
    vehicle: int
    kind: str                  # accepted, rejected, aborted, masked, committed, completed, cancelled, relocked, unlocked;
                               # overtakes: overtake (started), returned, stayed, overtake_ended
    reason: str = ''
    lane: int = -1
    gap_margin: float = float('nan')   # accepted/relocked: predicted bumper margin to the nearer gap car at the lookahead


class IntentionManager:
    def __init__(self, cfg, road, params, lane, speed):
        self.cfg, self.road, self.params = cfg, road, params
        self.intentions = [Intention(lane, float(np.clip(speed, cfg.speed_min, cfg.speed_max))) for _ in range(TEAM)]
        self.records = []
        self.cooldown = {}   # vehicle -> (lane, until): blocked re-requests after a forced cancel

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
        teammate i. Traffic changing lanes counts in its current and its target lane; a teammate counts in any lane
        its footprint reaches into (e.g. leaning over a lane line) and in lanes its lane change enters."""
        current, target = traffic_lanes(snap, self.road)
        reach = (self.road.cfg.lane_width+self.params.width)/2
        out = [(('traffic', int(snap.traffic_id[j])), snap.traffic_s[j]+snap.traffic_v[j]*horizon,
                snap.traffic_s[j], snap.traffic_v[j], snap.traffic_v[j])
               for j in np.flatnonzero((current == lane) | (target == lane))]
        for j in range(TEAM):
            if j == i:
                continue
            it = self.intentions[j]
            crossing = it.phase != 'keep' and lane in self.entered(it.origin, it.lane)
            if abs(snap.team_d[j]-self.road.lane_center(lane)) < reach or crossing:
                plan, v = plans[j], snap.team[j, 3]
                z = plan.at(int(round(horizon/plan.step))) if plan is not None else (snap.team_s[j]+v*horizon, 0, 0, v)
                out.append((('team', j), z[0], snap.team_s[j], v, z[3]))
        return out

    def reach(self, v, speed, T, dt=0.1):
        """Distance and speed after T s tracking `speed` within the plant's accel (power-limited) and brake limits."""
        p, s = self.params, 0.0
        v, speed = float(v), float(speed)
        for _ in range(int(round(T/dt))):
            a = min(max((speed-v)/dt, -p.brake_max), p.accel_max*min(1.0, p.power_speed/max(v, 1e-6)))
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
        blocked = self.cooldown.get(i)
        if blocked and blocked[0] == lane and snap.time < blocked[1]:
            return None, 'cooldown'
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
            a, it = int(actions[i]), self.intentions[i]
            if a >= len(ACTIONS):
                self._overtake(snap, plans, i, *OVERTAKES[a-len(ACTIONS)])
                continue
            if it.overtake is not None:   # in the passing lane (pass or return stage) the car stays: not an abort
                self._end_overtake(snap.time, i, 'overtake_ended' if it.overtake.stage == 'out' else 'stayed', 'policy')
            lane_cmd, speed_cmd, lean_cmd = ACTIONS[a]
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
            it.origin, it.lane, it.phase, it.fresh, it.since = it.lane, lane, 'pending', True, snap.time
            it.front, it.rear = gap
            self.records.append(Record(snap.time, i, 'accepted', '', lane, self.last_gap_margin))

    def update_phases(self, snap, plans=None):
        half = self.params.width/2
        for i, it in enumerate(self.intentions):
            d = snap.team_d[i]
            if it.phase != 'keep' and it.since is None:
                it.since = snap.time
            if it.phase == 'pending':
                side = int(np.sign(it.lane-it.origin))
                line = 0.5*(self.road.lane_center(it.origin)+self.road.lane_center(it.origin+side))   # first line crossed
                if (d-line)*side > -half:   # a footprint edge reached the lane line
                    it.phase, it.since, it.best, it.best_time = 'committed', snap.time, np.inf, snap.time
                    self.records.append(Record(snap.time, i, 'committed', '', it.lane))
            if it.phase == 'committed' and abs(d-self.target_d(i)) < self.cfg.done_tolerance:
                self.records.append(Record(snap.time, i, 'completed', '', it.lane))
                self._keep(i, it.lane)
            if it.phase != 'keep' and not self._timed_out(snap, i):
                self._check_lock(snap, plans, i)
            if it.overtake is not None:
                self._overtake_step(snap, plans if plans is not None else [None]*TEAM, i)

    # Overtakes (module docstring).
    def _overtake_keep(self, snap, i):
        """Lane a pending change cancelled by the intention layer keeps: during an overtake the lane the car's centre
        is in (a blended pull-out spends longer across the line, and sent back to its origin it could be pinned
        straddling the line beside a car there), otherwise None (the origin, as before)."""
        return int(self.road.lane_of(snap.team_d[i])) if self.intentions[i].overtake is not None else None

    def _lead(self, snap, i):
        """Nearest traffic car ahead in car i's lane within overtake_look: (key, CG gap, speed) or None."""
        lanes = self.road.lane_of(snap.traffic_d)
        ds = snap.traffic_s-snap.team_s[i]
        mask = (lanes == self.intentions[i].lane) & (ds > 0) & (ds < self.cfg.overtake_look)
        if not mask.any():
            return None
        k = np.flatnonzero(mask)[np.argmin(ds[mask])]
        return ('traffic', int(snap.traffic_id[k])), float(ds[k]), float(snap.traffic_v[k])

    def _car(self, snap, key):
        """(s, v) of a traffic car now, or None once it is gone (recycled)."""
        k = np.flatnonzero(snap.traffic_id == key[1])
        return (float(snap.traffic_s[k[0]]), float(snap.traffic_v[k[0]])) if len(k) else None

    def _pass_speed(self, i, v_car):
        it, c = self.intentions[i], self.cfg
        it.speed = float(np.clip(max(it.speed, v_car+it.overtake.margin), c.speed_min, c.speed_max))

    def _corridor(self, snap, plans, i, speed, lead, front):
        """The passing lane stays clear until the ego is past the car: the passing lane's front gap car (if any) is
        still length + margin_long ahead of the ego's reachable position when the ego is overtake_clear past the car
        (constant-speed car); and that happens within overtake_pass_max."""
        c, p = self.cfg, self.params
        _, gap, v_car = lead
        need = gap+p.length+c.margin_long+c.overtake_clear
        run, v, t, dt = 0.0, float(snap.team[i, 3]), 0.0, 0.1
        while run-v_car*t < need:
            if t >= c.overtake_pass_max:
                return False
            a = min(max((speed-v)/dt, -p.brake_max), p.accel_max*min(1.0, p.power_speed/max(v, 1e-6)))
            run, v, t = run+v*dt+0.5*a*dt*dt, v+a*dt, t+dt
        return front is None or self._predicted_s(snap, plans, front, t)-(snap.team_s[i]+run) >= p.length+c.margin_long

    def overtake_check(self, snap, plans, i, side, speed_cmd):
        """Would car i's overtake on `side` with passing margin speed_cmd start now? (reason ('' if so), lead (key, CG
        gap, speed), passing speed, passing-lane gap). No side effects beyond gap_rule's last_gap_margin; the env's
        action masks and observations use it as the intention layer does."""
        it, c = self.intentions[i], self.cfg
        lead = self._lead(snap, i) if it.phase == 'keep' and it.overtake is None else None
        settled = abs(snap.team_d[i]-self.target_d(i)) <= c.overtake_settled
        reason = 'overtake_busy' if it.phase != 'keep' or it.overtake is not None else \
            'no_lane' if not 0 <= it.lane+side < self.road.cfg.lanes else 'not_settled' if not settled else \
            'no_target' if lead is None else ''
        if reason:
            return reason, lead, None, None
        speed = float(np.clip(max(it.speed, lead[2]+c.overtake_margin*(2+speed_cmd)), c.speed_min, c.speed_max))
        gap, reason = self.gap_rule(snap, plans, i, it.lane+side, speed)
        if gap is not None and not self._corridor(snap, plans, i, speed, lead, gap[0]):
            reason = 'corridor'
        return reason, lead, speed, gap

    def _overtake(self, snap, plans, i, side, speed_cmd):
        """An overtake action: start one (passing lane change, gap rule plus corridor), or continue the active one."""
        it, c = self.intentions[i], self.cfg
        it.lean = 0
        if it.overtake is not None:
            it.overtake.margin = c.overtake_margin*(2+speed_cmd)
            car = self._car(snap, it.overtake.car)
            if car is not None:
                self._pass_speed(i, car[1])
            return
        lane = it.lane+side
        reason, lead, speed, gap = self.overtake_check(snap, plans, i, side, speed_cmd)
        it.rejected = bool(reason)
        if reason:
            self.records.append(Record(snap.time, i, 'rejected', reason, lane))
            return
        it.speed = speed
        it.origin, it.lane, it.phase, it.fresh, it.since = it.lane, lane, 'pending', True, snap.time
        it.front, it.rear = gap
        # Pull-out blend: overtake_merge_time, or half the time to contact with the car when closing on it faster
        # (at least overtake_out_min): a slow pull-out from close behind a much slower car leaves no way out.
        closing = snap.team[i, 3]-lead[2]
        ttc = (lead[1]-self.params.length)/closing if closing > 0 else np.inf
        out_time = float(np.clip(0.5*ttc, c.overtake_out_min, c.overtake_merge_time))
        it.overtake = Overtake(lead[0], side, it.origin, c.overtake_margin*(2+speed_cmd), 'out', snap.time,
                               out_from=float(snap.team_d[i]), out_time=out_time)
        self.records.append(Record(snap.time, i, 'accepted', '', lane, self.last_gap_margin))
        self.records.append(Record(snap.time, i, 'overtake', f'pass {lead[0][1]} {"left" if side > 0 else "right"}', lane))

    def _end_overtake(self, time, i, kind, reason=''):
        it = self.intentions[i]
        self.records.append(Record(time, i, kind, reason, it.lane))
        it.overtake = None

    def _overtake_step(self, snap, plans, i):
        """Advance car i's overtake after the lane-change phases: out -> pass when it reaches the passing lane;
        pass -> return once the gap rule accepts a home-lane gap ahead of the passed car; return -> done (returned),
        or back to pass if the return was cancelled. Stays in the passing lane after overtake_pass_max there."""
        it, c, t = self.intentions[i], self.cfg, snap.time
        ot = it.overtake
        car = self._car(snap, ot.car)
        if car is None:
            self._end_overtake(t, i, 'overtake_ended', 'car_gone')
            return
        self._pass_speed(i, car[1])
        if ot.stage == 'out':
            if it.phase == 'keep':
                if it.lane == ot.home+ot.side:
                    ot.stage, ot.passing = 'pass', t
                else:
                    self._end_overtake(t, i, 'overtake_ended', 'out_cancelled')
            return
        if ot.stage == 'return':
            if it.phase == 'keep':
                if it.lane == ot.home:
                    self._end_overtake(t, i, 'returned')
                else:
                    ot.stage = 'pass'
            return
        if it.phase != 'keep':   # the passing lane is being left some other way
            return
        if t-ot.passing > c.overtake_pass_max:
            self._end_overtake(t, i, 'stayed', 'no_room')
            return
        gap = self.return_gap(snap, plans, i)
        if gap is None:
            return
        it.origin, it.lane, it.phase, it.fresh, it.since = it.lane, ot.home, 'pending', True, t
        it.front, it.rear = gap
        ot.stage = 'return'
        self.records.append(Record(t, i, 'accepted', 'return', ot.home, self.last_gap_margin))

    def passed_car(self, snap, i):
        """(s, v) of the car car i's overtake is passing, or None."""
        ot = self.intentions[i].overtake
        return self._car(snap, ot.car) if ot is not None else None

    def return_gap(self, snap, plans, i):
        """In an overtake's pass stage: the home-lane gap the return would lock now, or None. Not before the car is
        about to clear the passed car (within overtake_return_lead s at the current relative speed: the gap rule's
        reachable-speed prediction is optimistic, and a return pending longer times out), and only a gap ahead of the
        passed car."""
        it, c = self.intentions[i], self.cfg
        ot = it.overtake
        car = self._car(snap, ot.car) if ot is not None and ot.stage == 'pass' and it.phase == 'keep' else None
        if car is None:
            return None
        short = self.params.length+c.margin_long+c.overtake_clear-(snap.team_s[i]-car[0])
        rel = snap.team[i, 3]-car[1]
        if short > 0 and (rel <= 0.1 or short/rel > c.overtake_return_lead):
            return None
        gap, _ = self.gap_rule(snap, plans, i, ot.home, it.speed)
        if gap is None:
            return None
        rear = gap[1]
        ahead_of_car = rear is None or rear == ot.car or (
            rear[0] == 'traffic' and self._car(snap, rear) is not None and self._car(snap, rear)[0] > car[0]) or (
            rear[0] == 'team' and snap.team_s[rear[1]] > car[0])
        return gap if ahead_of_car else None

    def overtake_release(self, i):
        """During an overtake's pull-out or return (a lane change in progress): (the Overtake, lateral position it
        starts from, target d) for the MPC's staged lateral reference (pull-out: a smooth blend from where the car
        was when it started; return: the passing lane until the reference clears the car, then a smooth merge),
        else None."""
        it = self.intentions[i]
        ot = it.overtake
        if ot is None or it.phase == 'keep' or ot.stage not in ('out', 'return'):
            return None
        start = ot.out_from if ot.stage == 'out' else float(self.road.lane_center(it.origin))
        return ot, start, float(self.target_d(i))

    def _timed_out(self, snap, i):
        """Cancel a pending change not committed within pending_max_time (back to the origin lane), or a committed one
        older than commit_max_time or without commit_progress_eps of lateral progress toward the target lane centre
        for commit_stall_time (into the lane the car's centre is in)."""
        it, c, t = self.intentions[i], self.cfg, snap.time
        if it.phase == 'pending':
            if t-it.since > c.pending_max_time:
                self.cancel(i, t, 'timeout', self._overtake_keep(snap, i))
                return True
            return False
        e = abs(snap.team_d[i]-self.road.lane_center(it.lane))
        if e < it.best-c.commit_progress_eps:
            it.best, it.best_time = e, t
        reason = 'timeout' if t-it.since > c.commit_max_time else 'stalled' if t-it.best_time >= c.commit_stall_time else ''
        if reason:
            self.cancel(i, t, reason, int(self.road.lane_of(snap.team_d[i])))
        return bool(reason)

    def _predicted_s(self, snap, plans, key, T):
        kind, j = key
        if kind == 'traffic':
            k = int(np.flatnonzero(snap.traffic_id == j)[0])
            return snap.traffic_s[k]+snap.traffic_v[k]*T
        plan = plans[j] if plans is not None else None
        return plan.at(int(round(T/plan.step)))[0] if plan is not None else snap.team_s[j]+snap.team[j, 3]*T

    def _check_lock(self, snap, plans, i):
        """Re-lock, cancel (pending) or drop a stale gap car (committed).

        A locked car is stale once it passed the ego the wrong way by length + margin_long now AND is on the wrong
        side of the ego's predicted end position at the gap rule's lookahead (the same prediction gap_rule uses).
        A rear car that is ahead now but that the ego is predicted to overtake by the lookahead is consistent: gap_rule
        would lock it again, so re-checking it every step only churns. Re-locking onto the same gap records nothing.
        """
        it = self.intentions[i]
        pos = {('team', j): snap.team_s[j] for j in range(TEAM)}
        pos.update((('traffic', int(k)), s) for k, s in zip(snap.traffic_id, snap.traffic_s))
        it.front, it.rear = (k if k in pos else None for k in (it.front, it.rear))   # recycled traffic: unlocked
        far, s = self.params.length+self.cfg.margin_long, snap.team_s[i]
        T = self.cfg.double_gap_lookahead if abs(it.lane-it.origin) == 2 else self.cfg.gap_lookahead
        end = s+self.reach(snap.team[i, 3], it.speed, T)[0]
        stale_rear = it.rear is not None and pos[it.rear]-s > far and self._predicted_s(snap, plans, it.rear, T) > end
        stale_front = it.front is not None and s-pos[it.front] > far and self._predicted_s(snap, plans, it.front, T) <= end
        if not (stale_rear or stale_front):
            return
        gap, reason = self.gap_rule(snap, plans if plans is not None else [None]*TEAM, i, it.lane, it.speed,
                                    middle=it.phase == 'pending')
        if gap is not None:
            if gap != (it.front, it.rear):
                it.front, it.rear = gap
                self.records.append(Record(snap.time, i, 'relocked', '', it.lane, self.last_gap_margin))
        elif it.phase == 'pending':
            self.cancel(i, snap.time, 'stale_gap', self._overtake_keep(snap, i))
        else:
            it.rear, it.front = (None if stale_rear else it.rear), (None if stale_front else it.front)
            self.records.append(Record(snap.time, i, 'unlocked', reason, it.lane))

    def reject_fresh(self, i, time, reason):
        it = self.intentions[i]
        self.records.append(Record(time, i, 'rejected', reason, it.lane))
        self._keep(i, it.origin)
        it.rejected = True

    def cancel(self, i, time, reason, lane=None):
        """Abandon a lane change: the vehicle keeps `lane` (default: returns to its origin lane). The record's
        `lane` is the lane the vehicle keeps. A cancel the policy didn't ask for blocks re-requesting the abandoned
        target lane for retry_cooldown s."""
        it = self.intentions[i]
        keep = lane if lane is not None else it.origin if it.phase != 'keep' else it.lane
        if it.phase != 'keep':
            self.cooldown[i] = (it.lane, time+self.cfg.retry_cooldown)
        self.records.append(Record(time, i, 'cancelled', reason, keep))
        self._keep(i, keep)

    def _keep(self, i, lane):
        it = self.intentions[i]
        it.lane, it.phase, it.origin, it.front, it.rear, it.fresh = lane, 'keep', None, None, None, False
        it.since, it.best, it.best_time = None, np.inf, None
