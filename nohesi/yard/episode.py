"""The yard episode loop: the leader scans, its driver assigns spots (each checked and planned), and the followers drive
to them under per-follower actions, the coordination shield and a safety supervisor.

Per decision (Planning.decision_interval) every follower gets one of ACTIONS:
    HOLD      stop (brake_comfort) and stay stopped
    CREEP     follow the planned path at Truck.v_creep
    GO        follow the planned path at the yard limits (v_forward_max forward, v_reverse_max reverse)
    BACK_OFF  reverse back along the ground just driven, up to BACK_OFF_DISTANCE (to clear a zone or open a gap);
              choosing another action ends it once the truck has stopped
    REPLAN    plan again from where the truck stands (stopped only), around the trucks where they are now; if the spot
              itself has become unsafe (a revealed obstacle), the driver assigns a new one
The path itself may reverse (Hybrid A* plans cusps); GO and CREEP drive it in whichever direction it goes.

Every dt: missed obstacles near a follower are revealed into the shared map (V2V), which can block planned paths;
every mpc_step the coordinator's shield gives each truck the last sample it may reach (pairwise conflict zones),
the tracker (MPC) turns the action, the shield and any block into commands, and the safety supervisor brakes a truck
whose path ahead (its stopping distance + 1 m) or footprint comes within SAFETY_MARGIN of an obstacle (all true ones in
near-field range), the leader or another follower. A collision ends the episode.
"""
from dataclasses import dataclass, field
import numpy as np
from ..geometry import clearances
from .truck import TruckPlant, corners, wrap
from .world import YardWorld

ACTIONS = ('HOLD', 'CREEP', 'GO', 'BACK_OFF', 'REPLAN')
HOLD, CREEP, GO, BACK_OFF, REPLAN = range(len(ACTIONS))
BACK_OFF_DISTANCE = 6.0   # m reversed per BACK_OFF manoeuvre
SAFETY_MARGIN = 0.3       # m: the supervisor stops a truck whose path ahead comes this close to anything
BLOCK_BUFFER = 4.0        # m: a truck whose path is blocked stops this far before the first blocked sample
MOVING = 0.05             # m/s: slower counts as standing still


@dataclass
class YardEpisode:
    scenario: object
    seed: int
    n: int
    time: np.ndarray
    states: np.ndarray
    steer: np.ndarray
    actions: np.ndarray
    safety_stop: np.ndarray
    obstacles: object
    missed: np.ndarray
    revealed_at: np.ndarray
    leader: np.ndarray
    grid0: object
    spots: list
    picks: list
    paths: list
    events: list
    end_reason: str


@dataclass
class Follower:
    tracker: object
    spot: np.ndarray = None       # (3,) assigned spot
    path: object = None           # the planned path to the spot (None: none yet / planning failed)
    progress: int = 0             # sample on `path`
    parked: bool = False
    blocked: int = None           # first sample of `path` a revealed obstacle blocks
    backing: dict = None          # BACK_OFF in progress: {'k0', 'target', 'stopping'}
    last_replan: float = -np.inf
    last_move: float = 0.0        # time the truck last moved
    safety: bool = False          # the supervisor is braking it
    action: int = HOLD
    status: str = ''              # the tracker's last status
    lateral_error: float = 0.0
    driven: float = 0.0           # m driven (both directions)


class YardRun:
    """One episode, advanced one decision at a time: step(actions) with a (n,) action per follower."""

    def __init__(self, scenario, seed, n_followers=None):
        from .coordination import Coordinator
        from .hybrid_astar import HybridAStar
        from .mpc import PathTracker
        from .sensors import initial_map
        from .spots import Driver
        self.scenario, self.seed = scenario, int(seed)
        self.truck, self.cfg = scenario.truck, scenario.planning
        self.world = YardWorld(scenario, seed, n_followers)
        self.n = self.world.n
        self.plant = TruckPlant(self.truck, self.cfg.dt)
        self.grid, self.missed = initial_map(self.world, scenario.sensors, np.random.default_rng([self.seed, 1]))
        self.grid0 = self.grid.copy()
        self.revealed = np.zeros(len(self.world.obstacles), bool)
        self.revealed_at = np.full(len(self.world.obstacles), np.inf)
        self.planner = HybridAStar(self.truck, self.cfg)
        self.coordinator = Coordinator(self.truck, self.cfg, self.n)
        self.driver = Driver(scenario, np.random.default_rng([self.seed, 2]))
        self.f = [Follower(PathTracker(self.truck, self.cfg)) for _ in range(self.n)]
        self.time, self.k = 0.0, 0
        self.picks, self.paths, self.events = [], [], []
        self.leader_foot = corners(self.truck, self.world.leader[None])
        self.end_reason = None
        self.totals = {'back_off_m': 0.0}
        self._record = {'time': [], 'states': [], 'steer': [], 'actions': [], 'safety': []}
        starts = self.world.footprints()
        for i in range(self.n):   # the driver assigns spots in queue order (prioritized planning)
            assigned = [g.spot for g in self.f[:i] if g.spot is not None]
            others = np.concatenate((self.leader_foot, starts[i+1:]))
            self._assign(i, assigned, others)
        self.stop_index = [None]*self.n
        self._snapshot()

    # ------------------------------------------------------------------ planning
    def _assign(self, i, assigned, others):
        """Driver picks for follower i until one is accepted (its path comes with it)."""
        pose, picks = self.driver.pick(i, self.world, self.grid, assigned, others, self.planner)
        for p in picks:
            self.picks.append({'time': self.time, 'follower': i, 'pose': np.asarray(p['pose'], float),
                               'accepted': bool(p['accepted']), 'reason': p['reason']})
        f = self.f[i]
        if pose is None:
            f.spot, f.path = None, None
            self.events.append({'time': self.time, 'follower': i, 'kind': 'no_spot', 'detail': ''})
            return False
        result = next(p.get('result') for p in picks if p['accepted'])
        f.spot = np.asarray(pose, float)
        self._set_path(i, result.path if result is not None else None)
        return f.path is not None

    def _set_path(self, i, path):
        f = self.f[i]
        f.path, f.progress, f.blocked, f.backing = path, 0, None, None
        if path is not None:
            f.tracker.set_path(path, 0)
            self.paths.append((self.time, i, path))
        self.coordinator.set_path(i, path, 0)

    def _obstacles_for(self, i):
        """Footprints follower i's replanned path must avoid: the leader, every other follower where it stands, and the
        spots of followers earlier in the coordinator's order that are not parked yet."""
        feet = self.world.footprints()
        rank = {j: r for r, j in enumerate(self.coordinator.order)}
        early = [corners(self.truck, g.spot[None])[0] for j, g in enumerate(self.f)
                 if j != i and g.spot is not None and not g.parked and rank[j] < rank[i]]
        parts = [self.leader_foot, feet[np.arange(self.n) != i]]+([np.array(early)] if early else [])
        return np.concatenate(parts)

    def replan(self, i):
        """REPLAN for follower i; returns True on a new path."""
        from .spots import validate
        f = self.f[i]
        f.last_replan = self.time
        start = self.world.states[i, :3].copy()
        if f.spot is not None:
            extra = self._obstacles_for(i)
            result = self.planner.plan(self.grid.clearance(extra=extra), start, f.spot)
            if result.path is not None:
                self._set_path(i, result.path)
                self.events.append({'time': self.time, 'follower': i, 'kind': 'replan', 'detail': ''})
                return True
            reason, _ = validate(self.scenario, self.grid, f.spot, [g.spot for j, g in enumerate(self.f)
                                                                     if j != i and g.spot is not None], self.leader_foot)
            if reason not in ('', 'unreachable', 'too_close'):   # the spot itself is no longer safe: a new one
                self.events.append({'time': self.time, 'follower': i, 'kind': 'reassigned', 'detail': reason})
            else:
                self.events.append({'time': self.time, 'follower': i, 'kind': 'replan_failed', 'detail': result.reason})
                return False
        assigned = [g.spot for j, g in enumerate(self.f) if j != i and g.spot is not None]
        ok = self._assign(i, assigned, self._obstacles_for(i))
        if ok:
            self.events.append({'time': self.time, 'follower': i, 'kind': 'replan', 'detail': 'new spot'})
        return ok

    def _check_blocks(self):
        """After a reveal: the first sample of each follower's remaining path that the shared map now blocks."""
        from .hybrid_astar import collision_free
        parked = [corners(self.truck, self.world.states[j][None])[0] for j, g in enumerate(self.f) if g.parked]
        clearance = self.grid.clearance(extra=np.concatenate([self.leader_foot]+([np.array(parked)] if parked else [])))
        for f in self.f:
            if f.path is None or f.parked:
                continue
            ok = collision_free(self.truck, self.cfg, clearance, f.path.states[f.progress:])
            bad = np.flatnonzero(~ok)
            f.blocked = f.progress+int(bad[0]) if bad.size else None

    # ------------------------------------------------------------------ masks
    def can_replan(self, i):
        f = self.f[i]
        return not f.parked and abs(self.world.states[i, 3]) < 0.2 and \
            self.time-f.last_replan >= self.cfg.replan_cooldown and f.backing is None

    def can_back_off(self, i):
        f = self.f[i]
        if f.parked or f.path is None or f.backing is not None:
            return f.backing is not None and not f.parked   # continuing a back-off is allowed
        if f.path.s[f.progress] < 1.0:
            return False
        target, c = self._back_target(f), self.coordinator
        # Never back into a zone (gate included) whose token another truck holds.
        held = [(z.side(i)[0]-c.gate, z.side(i)[1]) for z, o in zip(c.zones, c.owner) if i in (z.a, z.b) and o not in (None, i)]
        return not any(lo <= f.progress and hi >= target for lo, hi in held)

    def masks(self):
        """(n, 5) bool: absent or parked -> HOLD only; no path -> HOLD, REPLAN when allowed."""
        out = np.zeros((self.n, len(ACTIONS)), bool)
        for i, f in enumerate(self.f):
            out[i, HOLD] = True
            if f.parked:
                continue
            if f.path is not None:
                out[i, CREEP] = out[i, GO] = True
                out[i, BACK_OFF] = self.can_back_off(i)
            out[i, REPLAN] = self.can_replan(i)
        return out

    def _back_target(self, f):
        s = f.path.s
        return int(np.searchsorted(s, s[f.progress]-BACK_OFF_DISTANCE))

    # ------------------------------------------------------------------ stepping
    @property
    def done(self):
        return self.end_reason is not None

    def step(self, actions):
        """Apply one decision's actions (n,) and simulate decision_interval; returns this decision's tallies."""
        if self.done:
            raise RuntimeError('episode is over')
        masks = self.masks()
        tally = {'parked': 0, 'safety_stops': 0, 'replans': 0, 'replans_failed': 0, 'back_off_m': 0.0,
                 'collision': False, 'progress_m': 0.0}
        before = self.remaining()
        for i, a in enumerate(np.asarray(actions, int)[:self.n]):
            a = int(a) if masks[i, int(a)] else HOLD
            f = self.f[i]
            if f.backing is not None and a != BACK_OFF:
                f.backing['stopping'] = True
            if a == REPLAN:
                n0 = len(self.events)
                ok = self.replan(i)
                tally['replans' if ok else 'replans_failed'] += 1
                if not ok and self.events[n0:] and self.events[-1]['kind'] == 'reassigned':
                    tally['replans_failed'] -= 1
                a = HOLD
            elif a == BACK_OFF and f.backing is None:
                k0, target = f.progress, self._back_target(f)
                f.backing = {'k0': k0, 'target': target, 'stopping': False}
                f.tracker.set_path(f.path.retrace(k0, target), 0)
            elif a == BACK_OFF:
                f.backing['stopping'] = False
            f.action = a
        steps = int(round(self.cfg.decision_interval/self.cfg.dt))
        plan_every = int(round(self.cfg.mpc_step/self.cfg.dt))
        for _ in range(steps):
            x_before = self.world.states[:, :2].copy()
            if self.k % plan_every == 0:
                self._control()
            stops = self._advance()
            tally['safety_stops'] += stops
            moved = np.hypot(*(self.world.states[:, :2]-x_before).T)
            for i, f in enumerate(self.f):
                f.driven += moved[i]
                if f.backing is not None:
                    tally['back_off_m'] += moved[i]
                    self.totals['back_off_m'] += moved[i]
            tally['parked'] += self._update_parking()
            if self.world.collisions():
                c = self.world.collisions()[0]
                self.events.append({'time': self.time, 'follower': c[0], 'kind': 'collision', 'detail': f'{c[1]} {c[2]}'})
                tally['collision'] = True
                self.end_reason = 'collision'
            elif all(f.parked for f in self.f):
                self.end_reason = 'parked'
            elif self.time >= self.cfg.duration-1e-9:
                self.end_reason = 'timeout'
            elif self.time-max(f.last_move for f in self.f) >= self.cfg.stall_time:
                self.end_reason = 'stalled'
            self._snapshot()
            if self.done:
                break
        after = self.remaining()
        tally['progress_m'] = float(sum(b-a for b, a in zip(before, after) if b is not None and a is not None))
        return tally

    def remaining(self):
        """Per follower: path length left to its spot (m), 0 when parked, None without a path."""
        return [0.0 if f.parked else (f.path.remaining(f.progress) if f.path is not None else None) for f in self.f]

    def _control(self):
        """Shield, trackers: new commands for every follower (held until the next mpc_step)."""
        progress = np.array([f.progress for f in self.f])
        # A truck still rolling keeps wanting (the coordinator must not release its token while it brakes).
        wants = np.array([f.path is not None and not f.parked and f.backing is None and
                          (f.action in (GO, CREEP) or abs(self.world.states[i, 3]) > MOVING) for i, f in enumerate(self.f)])
        self.stop_index = self.coordinator.update(progress, wants)
        self._cmd = []
        for i, f in enumerate(self.f):
            st, steer, accel = self.world.states[i], self.world.steer[i], self.world.accel[i]
            if f.parked or f.path is None:
                self._cmd.append((steer, 0.0, int(np.sign(st[3])) if abs(st[3]) > MOVING else 0)); continue
            if f.backing is not None:
                limit = 0.0 if f.backing['stopping'] else self.truck.v_creep
                cmd = f.tracker.command(st, steer, accel, limit, None)
                f.progress = max(f.backing['target'], f.backing['k0']-f.tracker.progress)
                done = f.tracker.arrived or (f.backing['stopping'] and abs(st[3]) < MOVING)
                if done:   # back on the planned path at the sample reached
                    f.tracker.set_path(f.path, f.progress)
                    f.backing = None
                    if f.action == BACK_OFF:
                        f.action = HOLD
            else:
                limit = {HOLD: 0.0, CREEP: self.truck.v_creep, GO: np.inf}.get(f.action, 0.0)
                stop = self.stop_index[i]
                if f.blocked is not None:
                    buffer = int(np.searchsorted(f.path.s, f.path.s[f.blocked]-BLOCK_BUFFER))
                    stop = buffer if stop is None else min(stop, buffer)
                if stop is not None:
                    stop = max(int(stop), f.progress)
                cmd = f.tracker.command(st, steer, accel, limit, stop)
                f.progress = int(cmd[3].get('progress', f.tracker.progress))
            f.status = cmd[3].get('status', '')
            f.lateral_error = float(cmd[3].get('lateral_error', 0.0))
            self._cmd.append(cmd[:3])

    def _advance(self):
        """One dt: reveal, supervise, step the plant. Returns the number of new safety stops."""
        from .sensors import reveal
        new = reveal(self.grid, self.world, self.missed, self.revealed, self.scenario.sensors)
        if new:
            self.revealed_at[new] = self.time
            self._check_blocks()
        onsets = 0
        for i, f in enumerate(self.f):
            steer_cmd, accel_cmd, gear = self._cmd[i]
            stop = self._supervise(i, gear)
            if stop:
                v = self.world.states[i, 3]
                accel_cmd = -np.sign(v)*self.truck.brake_emergency if abs(v) > MOVING else 0.0
                if not f.safety:
                    onsets += 1
                    self.events.append({'time': self.time, 'follower': i, 'kind': 'safety_stop', 'detail': f.status})
            f.safety = stop
            st, steer, accel = self.plant.step(self.world.states[i], self.world.steer[i], self.world.accel[i],
                                               steer_cmd, accel_cmd, gear if not (stop and abs(self.world.states[i, 3]) <= MOVING) else 0)
            self.world.states[i], self.world.steer[i], self.world.accel[i] = st, steer, accel
            if abs(st[3]) > MOVING:
                f.last_move = self.time+self.cfg.dt
        self.k += 1
        self.time = round(self.k*self.cfg.dt, 9)
        return onsets

    def _supervise(self, i, gear):
        """Safety supervisor: brake follower i if its footprint, or its path ahead within its stopping distance + 1 m in
        the direction it moves (or is about to), comes within SAFETY_MARGIN of an obstacle in near-field range (missed
        ones too), the leader or another follower."""
        f, st = self.f[i], self.world.states[i]
        v = st[3]
        direction = int(np.sign(v)) if abs(v) > MOVING else int(gear)
        if direction == 0 or f.parked:
            return False
        reach = v*v/(2*self.truck.brake_max)+1.0
        poses = [st[:3]]
        path, k = (f.path, f.progress) if f.backing is None else (None, None)
        if path is not None:
            ahead = np.flatnonzero((path.s > path.s[k]) & (path.s <= path.s[k]+reach))
            same = ahead[path.direction[np.maximum(ahead-1, 0)] == direction]
            poses += [path.states[j] for j in same[::2]]
        else:   # backing off, or no path: straight-line projection of the current motion
            for d in np.arange(0.5, reach+0.5, 0.5):
                poses.append(st[:3]+direction*d*np.array([np.cos(st[2]), np.sin(st[2]), 0.0]))
        feet = corners(self.truck, np.array(poses))
        obs = self.world.obstacles.corners
        centre = st[:2]
        near = np.flatnonzero(np.min(np.hypot(obs[..., 0]-centre[0], obs[..., 1]-centre[1]), axis=1) <
                              self.scenario.sensors.near_range+self.truck.length+reach)
        others = [obs[near], self.leader_foot] + [self.world.footprints()[np.arange(self.n) != i]]
        targets = np.concatenate(others)
        if not len(targets):
            return False
        A = np.repeat(feet, len(targets), 0)
        B = np.tile(targets, (len(feet), 1, 1))
        gaps = clearances(A, B).reshape(len(feet), len(targets))
        # Ignore what the truck already touches only at its current pose with zero motion (a collision is detected
        # separately); stop if anything ahead is too close.
        return bool(np.any(gaps < SAFETY_MARGIN))

    def _update_parking(self):
        """Mark followers that reached their spot standing still; returns how many parked this step."""
        new = 0
        for i, f in enumerate(self.f):
            if f.parked or f.path is None or f.backing is not None or f.spot is None:
                continue
            st = self.world.states[i]
            err = np.hypot(*(st[:2]-f.spot[:2]))
            if abs(st[3]) < MOVING and err <= self.cfg.park_tolerance and abs(wrap(st[2]-f.spot[2])) <= 0.1 and \
                    f.progress >= len(f.path)-3:
                f.parked, new = True, new+1
                # A parked truck is a static obstacle from now on, out of the conflict zones: the zones' margin (disks,
                # zone_margin) is wider than the planner's clearance, so a path planned around it would stay blocked
                # by its zone for good. Paths it now blocks are flagged like a revealed obstacle (then REPLAN).
                self.coordinator.set_path(i, None, 0)
                self.events.append({'time': self.time, 'follower': i, 'kind': 'parked', 'detail': f'{err:.2f} m'})
        if new:
            self._check_blocks()
        return new

    # ------------------------------------------------------------------ record
    def _snapshot(self):
        r = self._record
        r['time'].append(self.time); r['states'].append(self.world.states.copy()); r['steer'].append(self.world.steer.copy())
        r['actions'].append(np.array([f.action for f in self.f])); r['safety'].append(np.array([f.safety for f in self.f]))

    def episode(self):
        r = self._record
        return YardEpisode(self.scenario, self.seed, self.n, np.array(r['time']), np.array(r['states']),
                           np.array(r['steer']), np.array(r['actions']), np.array(r['safety']), self.world.obstacles,
                           self.missed.copy(), self.revealed_at.copy(), self.world.leader.copy(), self.grid0,
                           [None if f.spot is None else f.spot.copy() for f in self.f], list(self.picks),
                           list(self.paths), list(self.events), self.end_reason or 'running')

    def metrics(self):
        ev = self.events
        count = lambda kind: sum(e['kind'] == kind for e in ev)
        return {'seed': self.seed, 'success': self.end_reason == 'parked', 'parked': int(sum(f.parked for f in self.f)),
                'followers': self.n, 'collision': self.end_reason == 'collision', 'end_reason': self.end_reason,
                'duration_s': float(self.time), 'replans': count('replan'), 'replans_failed': count('replan_failed'),
                'safety_stops': count('safety_stop'), 'back_off_m': float(self.totals['back_off_m']),
                'picks_rejected': sum(not p['accepted'] for p in self.picks),
                'reassigned': count('reassigned'),
                'path_length_m': float(sum(p.length for t, i, p in self.paths if t == 0.0)),
                'driven_m': float(sum(f.driven for f in self.f))}
