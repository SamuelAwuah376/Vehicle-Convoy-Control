"""The yard episode loop: the leader scans, its driver assigns spots (each checked and planned), and the followers drive
to them under per-follower actions, the coordination shield and a safety supervisor.

Per decision (Planning.decision_interval) every follower gets one of ACTIONS:
    HOLD      stop (brake_comfort) and stay stopped
    CREEP     follow the planned path at Truck.v_creep
    GO        follow the planned path at the yard limits (v_forward_max forward, v_reverse_max reverse)
    BACK_OFF  reverse back along the ground just driven, up to BACK_OFF_DISTANCE (to clear a zone or open a gap);
              choosing another action ends it once the truck has stopped
    REPLAN    plan again from where the truck stands (stopped, and only when a replan can change something: no path, a
              blocked path, off its path, or stuck while trying to move), around what stays put; if the spot
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
REASSIGN_AFTER = 2        # failed replans in a row after which the driver picks the follower a new spot
REPLAN_IDLE = 10.0        # s standing still after which REPLAN is offered (deadlock breaking) without a block


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
    failed_replans: int = 0       # in a row: after REASSIGN_AFTER the driver is asked for a new spot
    failed_key: tuple = None      # the inputs of the last failed replan (a repeat is not searched again)
    last_remaining: float = None  # remaining path when its last path was dropped (progress potential without a path)
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
        for f in self.f:
            if f.path is None:
                f.last_remaining = 0.0   # its first path's length is then charged by the potential when it gets one
        self._check_blocks()             # a follower left without a spot stands on the road: later paths through it
        self.initial_path_length = sum(f.path.length for f in self.f if f.path is not None)
        self.stop_index = [None]*self.n
        self._snapshot()

    # ------------------------------------------------------------------ planning
    def _assign(self, i, assigned, others, source='assign'):
        """Driver picks for follower i until one is accepted (its path comes with it). source: 'assign' (the start) or
        'replan' (a reassignment): recorded in a 'no_spot' event's detail."""
        pose, picks = self.driver.pick(i, self.world, self.grid, assigned, others, self.planner)
        for p in picks:
            self.picks.append({'time': self.time, 'follower': i, 'pose': np.asarray(p['pose'], float),
                               'accepted': bool(p['accepted']), 'reason': p['reason']})
        f = self.f[i]
        if pose is None:
            f.spot = None
            self._set_path(i, None)   # out of the coordinator: a standing pathless truck is a static obstacle
            self.events.append({'time': self.time, 'follower': i, 'kind': 'no_spot', 'detail': source})
            if self.time > 0:
                self._check_blocks()
            return False
        result = next(p.get('result') for p in picks if p['accepted'])
        f.spot = np.asarray(pose, float)
        self._set_path(i, result.path if result is not None else None)
        return f.path is not None

    def _set_path(self, i, path):
        f = self.f[i]
        if f.path is not None and not f.parked:
            f.last_remaining = f.path.remaining(f.progress)
        f.path, f.progress, f.blocked, f.backing = path, 0, None, None
        if path is not None:
            f.tracker.set_path(path, 0)
            self.paths.append((self.time, i, path))
        self.coordinator.set_path(i, path, 0)

    def _static_feet(self, exclude=None):
        """Footprints that stay put: the leader, parked followers and standing followers without a path."""
        feet = self.world.footprints()
        still = [feet[j] for j, g in enumerate(self.f) if j != exclude and (g.parked or g.path is None)]
        return np.concatenate([self.leader_foot]+([np.array(still)] if still else []))

    def _obstacles_for(self, i):
        """Footprints follower i's replanned path must avoid: what stays put, i.e. the leader, the parked followers, and
        the spots of followers earlier in the coordinator's order that are not parked yet (prioritized planning). The
        followers still on the move are not obstacles: the conflict zones of the new path keep the trucks apart (a
        follower queued on the access road would otherwise wall the road off for the ones behind it)."""
        rank = {j: r for r, j in enumerate(self.coordinator.order)}
        early = [corners(self.truck, g.spot[None])[0] for j, g in enumerate(self.f)
                 if j != i and g.spot is not None and not g.parked and rank[j] < rank[i]]
        return np.concatenate([self._static_feet(i)]+([np.array(early)] if early else []))

    def replan(self, i):
        """REPLAN for follower i; returns 'ok' (a new path), 'failed' (no path, same spot) or 'no_spot' (the spot was
        reassigned and the driver found none)."""
        from .spots import validate
        f = self.f[i]
        f.last_replan = self.time
        start = self.world.states[i, :3].copy()
        if f.spot is not None:
            extra = self._obstacles_for(i)
            # The planner is deterministic: from the same pose against the same map and obstacles it fails the same way,
            # so a repeat of the last failed replan is not searched again.
            key = (tuple(np.round(start, 1)), tuple(np.round(f.spot, 2)), int(self.revealed.sum()),
                   tuple(np.round(extra, 1).ravel()))
            if key == f.failed_key:
                result = type('Cached', (), {'path': None, 'reason': 'repeat'})()
            else:
                result = self.planner.plan(self.grid.clearance(extra=extra), start, f.spot)
                f.failed_key = None if result.path is not None else key
            if result.path is not None:
                self._set_path(i, result.path)
                f.failed_replans = 0
                self.events.append({'time': self.time, 'follower': i, 'kind': 'replan', 'detail': ''})
                return 'ok'
            f.failed_replans += 1
            reason, _ = validate(self.scenario, self.grid, f.spot, [g.spot for j, g in enumerate(self.f)
                                                                     if j != i and g.spot is not None], self.leader_foot)
            if reason not in ('', 'unreachable', 'too_close'):   # the spot itself is no longer safe: a new one
                self.events.append({'time': self.time, 'follower': i, 'kind': 'reassigned', 'detail': reason})
            elif f.failed_replans >= REASSIGN_AFTER:   # no way there from here: the driver picks another spot
                self.events.append({'time': self.time, 'follower': i, 'kind': 'reassigned',
                                    'detail': f'unreachable after {f.failed_replans} replans'})
            else:
                self.events.append({'time': self.time, 'follower': i, 'kind': 'replan_failed', 'detail': result.reason})
                return 'failed'
        assigned = [g.spot for j, g in enumerate(self.f) if j != i and g.spot is not None]
        ok = self._assign(i, assigned, self._obstacles_for(i), 'replan')
        f.failed_replans = 0
        if ok:
            self.events.append({'time': self.time, 'follower': i, 'kind': 'replan', 'detail': 'new spot'})
        return 'ok' if ok else 'no_spot'

    def _check_blocks(self):
        """After a reveal, a parking or a failed assignment: the first sample of each follower's remaining path that the
        shared map or a truck that stays put (parked, or standing without a path) now blocks."""
        from .hybrid_astar import collision_free
        for i, f in enumerate(self.f):
            if f.path is None or f.parked:
                continue
            clearance = self.grid.clearance(extra=self._static_feet(i))
            ok = collision_free(self.truck, self.cfg, clearance, f.path.states[f.progress:])
            bad = np.flatnonzero(~ok)
            f.blocked = f.progress+int(bad[0]) if bad.size else None

    # ------------------------------------------------------------------ masks
    def needs_replan(self, i):
        """A replan can change something: no path, the path is blocked by a revealed obstacle or a parked truck, the
        truck is off its path, or it has stood still for REPLAN_IDLE s (a deadlock to break). Otherwise the planner (with
        the same static obstacles) would return about the same path, so REPLAN is masked: a nuisance action. Standing
        still counts only while the truck is trying to move (GO or CREEP) and no other truck's zone is what stops it."""
        f = self.f[i]
        stuck = f.action in (GO, CREEP) and self.time-f.last_move >= REPLAN_IDLE and not self.coordinator.blocking(i)
        return f.path is None or f.blocked is not None or f.status == 'off_path' or stuck

    def can_replan(self, i):
        f = self.f[i]
        return not f.parked and abs(self.world.states[i, 3]) < 0.2 and \
            self.time-f.last_replan >= self.cfg.replan_cooldown and f.backing is None and self.needs_replan(i)

    def can_back_off(self, i):
        """BACK_OFF starts only from standstill (so the coordinator's progress stays true) and not toward a zone that
        another truck holds or is close enough to request; a back-off in progress may continue (_control re-checks
        it every mpc_step and stops it before such a zone)."""
        f = self.f[i]
        if f.parked or f.path is None or f.backing is not None:
            return f.backing is not None and not f.parked
        if f.path.s[f.progress] < 1.0 or abs(self.world.states[i, 3]) > MOVING:
            return False
        return not self._back_conflicts(i, f.progress, self._back_target(f))

    def _back_conflicts(self, i, k_from, k_to):
        """Does retracing samples k_from -> k_to (k_to < k_from) enter the hold range of a zone that another truck holds,
        or that its partner could request now (within gate + look of its entry)? A zone the truck stands in (k_from in
        its hold range) is its own already: backing out of it is what BACK_OFF is for (it frees the zone for the truck
        waiting at its gate)."""
        c = self.coordinator
        for z, o in zip(c.zones, c.owner):
            if i not in (z.a, z.b):
                continue
            lo, hi = z.side(i)[0]-c.gate, z.side(i)[1]
            if hi < k_to or lo > k_from or lo <= k_from <= hi:
                continue
            j = z.other(i)
            j_in, j_out = z.side(j)
            if o not in (None, i) or (j_in-c.gate-c.look <= c.progress[j] <= j_out):
                return True
        return False

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
                tally['replans' if self.replan(i) == 'ok' else 'replans_failed'] += 1
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
        for n in range(steps):
            x_before = self.world.states[:, :2].copy()
            if n % plan_every == 0:   # a decision's actions take effect at once, then every mpc_step
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
        """Per follower: path length left to its spot (m), 0 when parked; without a path the last remaining length
        (the progress potential carries over until a new path; None if it never had one)."""
        return [0.0 if f.parked else (f.path.remaining(f.progress) if f.path is not None else f.last_remaining)
                for f in self.f]

    def _control(self):
        """Shield, trackers: new commands for every follower (held until the next mpc_step)."""
        for i, f in enumerate(self.f):   # fresh progress for the shield (the trackers' is one mpc_step old)
            if f.path is not None and not f.parked and f.backing is None:
                end = f.path.segments()[f.path.segment_of(f.progress)][1]
                f.progress = f.path.nearest(self.world.states[i, :2], f.progress, end+1)
        progress = np.array([f.progress for f in self.f])
        # A truck still rolling keeps wanting (the coordinator must not release its token while it brakes).
        wants = np.array([f.path is not None and not f.parked and f.backing is None and
                          (f.action in (GO, CREEP) or abs(self.world.states[i, 3]) > MOVING) for i, f in enumerate(self.f)])
        self.stop_index = self.coordinator.update(progress, wants)
        self._cmd = []
        for i, f in enumerate(self.f):
            st, steer, accel = self.world.states[i], self.world.steer[i], self.world.accel[i]
            if f.parked or f.path is None:   # brake to a standstill and stay there
                moving = abs(st[3]) > MOVING
                self._cmd.append((steer, -np.sign(st[3])*self.truck.brake_comfort if moving else 0.0,
                                  int(np.sign(st[3])) if moving else 0)); continue
            if f.backing is not None:
                # Stop before re-entering a zone another truck holds or may request (stopping distance + 0.5 m ahead).
                v = abs(st[3])
                look = int(np.ceil((v*v/(2*self.truck.brake_max)+0.5)/self.cfg.path_sample))+1
                if self._back_conflicts(i, f.progress, max(f.backing['target'], f.progress-look)):
                    f.backing['stopping'] = True
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
        poses = []
        if f.backing is None and f.path is not None:   # the path ahead in the direction of motion
            path, k = f.path, f.progress
            ahead = np.flatnonzero((path.s > path.s[k]) & (path.s <= path.s[k]+reach))
            same = ahead[path.direction[np.maximum(ahead-1, 0)] == direction]
            poses += [path.states[j] for j in same[::2]]
        elif f.backing is not None:   # the retrace ahead (the planned path backwards from the current sample)
            path, k = f.path, f.progress
            behind = np.flatnonzero((path.s < path.s[k]) & (path.s >= path.s[k]-reach))[::-1]
            poses += [path.states[j] for j in behind[::2]]
        step = st[:3]+direction*0.25*np.array([np.cos(st[2]), np.sin(st[2]), 0.0])   # just ahead of the current pose
        feet = corners(self.truck, np.array([st[:3], step]+poses))
        obs = self.world.obstacles.corners
        # Obstacles in near-field range of the footprint (polygon distance: a berm is long and has corners only at its
        # ends), the leader and the other followers.
        here = np.repeat(feet[:1], len(obs), 0)
        near = np.flatnonzero(clearances(here, obs) < self.scenario.sensors.near_range+reach)
        targets = np.concatenate([obs[near], self.leader_foot, self.world.footprints()[np.arange(self.n) != i]])
        A = np.repeat(feet, len(targets), 0)
        B = np.tile(targets, (len(feet), 1, 1))
        gaps = clearances(A, B).reshape(len(feet), len(targets))
        # The current pose itself never triggers a stop: a truck already within the margin of something may move as long
        # as the step ahead does not bring it closer to it (else it would be frozen in both directions); everything
        # ahead must keep the margin.
        closer = (gaps[1] < SAFETY_MARGIN) & (gaps[1] < gaps[0]-1e-6)
        return bool(np.any(closer) or np.any(gaps[2:] < SAFETY_MARGIN))

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
                'duration_s': float(self.time), 'replans': count('replan'),
                'replans_failed': count('replan_failed')+sum(e['kind'] == 'no_spot' and e['detail'] == 'replan' for e in ev),
                'safety_stops': count('safety_stop'), 'back_off_m': float(self.totals['back_off_m']),
                'picks_rejected': sum(not p['accepted'] for p in self.picks),
                'reassigned': count('reassigned'),
                'path_length_m': float(self.initial_path_length),
                'driven_m': float(sum(f.driven for f in self.f))}
