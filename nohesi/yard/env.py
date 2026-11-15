"""Gymnasium environment for the yard coordination policy: one step = one decision (Planning.decision_interval, 0.5 s),
one of episode.ACTIONS per follower slot (3 slots; an absent or parked follower can only HOLD).

The planner, the coordination shield and the safety supervisor (episode.py) make every allowed action safe with
respect to the other trucks and the mapped obstacles, so what the policy learns is the order and the timing: who
goes first through a shared stretch, when to creep, when to back off and let another truck by, when to replan.

Observation (float32, D = 3*SLOT + 2 = 152), slot k = follower k in queue order, zeros for an absent follower:
  ego (16): present, parked, v/3, steer/steer_max, direction of the segment being driven (+1/-1, 0 none),
      remaining path/100, distance to the segment's end (a cusp or the spot)/20, distance to the spot/100, heading
      error to the spot/pi, lateral tracking error/1.5, time since it last moved/20, safety stop, blocked (no path,
      path blocked by a revealed obstacle, or off its path), can replan now, backing off, cusps left/4
  near field (8): free distance along 8 rays in the truck frame / near_range (radar + ultrasonic)
  per other follower (2 x 11, other slots in order): present, relative x/50 and y/50 in the ego frame, sin and cos of
      the relative heading, v/3, parked, shares a conflict zone ahead, ego's distance to that zone's gate/50, the
      other's distance to its gate/50 (negative inside), the other holds a zone that stops ego
  leader (4): relative x/50, y/50, sin, cos of the relative heading
  global (2): time/duration, followers/3
Reward per decision (YardEnvConfig weights): -w_time per unparked follower-second, +w_progress per metre of remaining
path removed (potential-based: replans that lengthen a path pay that back), +w_park per follower parked, -w_collision on a collision (ends the episode), -w_safety per safety stop,
-w_replan per replan, -w_replan_fail per failed replan, -w_back per metre backed off, -w_stall if the episode stalls.
"""
from dataclasses import dataclass
import numpy as np
import gymnasium
from gymnasium import spaces
from .episode import ACTIONS, GO, HOLD, REPLAN, YardRun
from .scenarios import YARD_SCENARIOS
from .truck import wrap

SLOTS = 3
EGO, NEAR, OTHER, LEADER, GLOBAL = 16, 8, 11, 4, 2
SLOT = EGO+NEAR+OTHER*(SLOTS-1)+LEADER


@dataclass(frozen=True)
class YardEnvConfig:
    """YardEnv settings. Bump version whenever the observation, the masks or the reward change."""
    version: str = 'ye1'
    scenario: str = 'yard1'
    followers: tuple = ()          # (min, max) followers per episode; () keeps the scenario's
    w_time: float = 0.02           # per unparked follower per second
    w_progress: float = 0.02       # per metre of remaining path removed (potential-based: a replan that lengthens the
                                   # path costs what driving the extra length earns back)
    w_park: float = 5.0
    w_collision: float = 50.0     # above the most a long episode can cost otherwise (time ~18 + forgone park bonuses)
    w_safety: float = 0.5
    w_replan: float = 0.2
    w_replan_fail: float = 0.5
    w_back: float = 0.02
    w_stall: float = 5.0


def obs_dim():
    return SLOTS*SLOT+GLOBAL


class YardEnv(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg=YardEnvConfig(), seed_pool=None, rng_seed=None):
        from dataclasses import replace
        self.cfg = cfg
        scenario = YARD_SCENARIOS[cfg.scenario]()
        self.scenario = replace(scenario, followers=tuple(cfg.followers)) if cfg.followers else scenario
        self.seed_pool, self.pool_index = (list(seed_pool) if seed_pool is not None else None), 0
        self.action_space = spaces.MultiDiscrete([len(ACTIONS)]*SLOTS)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim(),), np.float32)
        self.run = None
        if rng_seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(rng_seed)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            if self.seed_pool:
                seed = self.seed_pool[self.pool_index % len(self.seed_pool)]
                self.pool_index += 1
            else:
                seed = int(self.np_random.integers(2000, 2**31-1))
        self.run = YardRun(self.scenario, int(seed))
        return self._obs(), {'seed': int(seed)}

    def action_masks(self):
        out = np.zeros((SLOTS, len(ACTIONS)), bool)
        out[:, HOLD] = True
        if self.run is not None:
            out[:self.run.n] = self.run.masks()
        return out.ravel()

    step_started = None   # monotonic start of the step in progress (the training workers' watchdog reads it)

    def step(self, action):
        import time
        self.step_started = time.monotonic()
        try:
            return self._step(action)
        finally:
            self.step_started = None

    def _step(self, action):
        run, cfg = self.run, self.cfg
        if run is None or run.done:
            raise RuntimeError('call reset() first')
        action = np.asarray(action, int).reshape(SLOTS)
        unparked = sum(not f.parked for f in run.f)
        t0 = run.time
        tally = run.step(action[:run.n])
        dt = run.time-t0
        reward = (-cfg.w_time*unparked*dt+cfg.w_progress*tally['progress_m']+cfg.w_park*tally['parked']
                  - cfg.w_collision*tally['collision']-cfg.w_safety*tally['safety_stops']-cfg.w_replan*tally['replans']
                  - cfg.w_replan_fail*tally['replans_failed']-cfg.w_back*tally['back_off_m']
                  - cfg.w_stall*(run.end_reason == 'stalled'))
        terminated = run.end_reason in ('parked', 'collision')
        truncated = run.end_reason in ('timeout', 'stalled')
        info = {'tally': tally}
        if run.done:
            info['episode_metrics'] = run.metrics()
        return self._obs(), float(reward), terminated, truncated, info

    def episode(self):
        return self.run.episode()

    # ------------------------------------------------------------------ observation
    def _obs(self):
        from .sensors import near_field
        run = self.run
        truck, cfg, sensors = run.truck, run.cfg, run.scenario.sensors
        st = run.world.states
        out = np.zeros(obs_dim(), np.float32)
        zones = [run.coordinator.zones_of(i) for i in range(run.n)]
        for i, f in enumerate(run.f):
            ego = np.zeros(EGO)
            ego[0], ego[1] = 1.0, float(f.parked)
            ego[2], ego[3] = st[i, 3]/3.0, run.world.steer[i]/truck.steer_max
            if f.path is not None and not f.parked:
                path, k = f.path, f.progress
                seg = path.segments()[path.segment_of(k)]
                ego[4] = path.direction[k]
                ego[5] = path.remaining(k)/100
                ego[7:9] = np.hypot(*(st[i, :2]-f.spot[:2]))/100, wrap(f.spot[2]-st[i, 2])/np.pi
                ego[6] = (path.s[seg[1]]-path.s[k])/20
                ego[15] = np.sum(path.cusps > k)/4
            ego[9] = f.lateral_error/cfg.track_tolerance
            ego[10] = min(run.time-f.last_move, 20.0)/20
            ego[11] = float(f.safety)
            ego[12] = float(not f.parked and (f.path is None or f.blocked is not None or f.status == 'off_path'))
            ego[13] = float(run.can_replan(i))
            ego[14] = float(f.backing is not None)
            near = near_field(run.world, i, sensors, NEAR)/sensors.near_range
            others = []
            c, s = np.cos(st[i, 2]), np.sin(st[i, 2])
            rel = lambda p: ((p[0]-st[i, 0])*c+(p[1]-st[i, 1])*s, -(p[0]-st[i, 0])*s+(p[1]-st[i, 1])*c, p[2]-st[i, 2])
            for j in range(SLOTS):
                if j == i:
                    continue
                o = np.zeros(OTHER)
                if j < run.n:
                    dx, dy, dyaw = rel(st[j])
                    o[:7] = 1.0, dx/50, dy/50, np.sin(dyaw), np.cos(dyaw), st[j, 3]/3.0, float(run.f[j].parked)
                    z = zones[i][j]
                    if z is not None:
                        o[7:11] = 1.0, z[0]/50, z[1]/50, float(z[2])
                others.append(o)
            dx, dy, dyaw = rel(run.world.leader)
            lead = np.array([dx/50, dy/50, np.sin(dyaw), np.cos(dyaw)])
            out[i*SLOT:(i+1)*SLOT] = np.concatenate((ego, near, *others, lead))
        out[-2:] = run.time/cfg.duration, run.n/SLOTS
        return out


def scripted_actions(env):
    """The prioritized baseline: queue order is priority. A follower GOes along its path unless a higher-priority
    follower shares a zone ahead with it that it has not cleared yet and the follower is at that zone's gate (then it
    HOLDs); a follower whose path is blocked, that is off its path or has none REPLANs once it has stopped."""
    run = env.run
    out = np.full(SLOTS, HOLD)
    progress = np.array([f.progress for f in run.f])
    for i, f in enumerate(run.f):
        if f.parked:
            continue
        stuck = f.path is None or f.blocked is not None or f.status == 'off_path'
        if stuck:
            if f.blocked is not None and f.path is not None and f.progress < f.blocked-int(6/run.cfg.path_sample):
                out[i] = GO   # drive up to the block first
            else:
                out[i] = REPLAN if run.can_replan(i) else HOLD
            continue
        out[i] = GO
        if run.coordinator.priority_wait(i, progress):
            # Wait before the gate of a zone a higher-priority truck has not cleared, starting a full comfortable stop
            # (plus a decision's travel and 1 m) ahead; never hold once inside a zone: that would block the truck
            # waited for (it stops at its own gate for the zone this truck holds).
            rank = {j: r for r, j in enumerate(run.coordinator.order)}
            v = abs(run.world.states[i, 3])
            reach = v*v/(2*run.truck.brake_comfort)+v*run.cfg.decision_interval+1.0
            gates = [z[0] for j, z in enumerate(run.coordinator.zones_of(i)) if z is not None and rank[j] < rank[i]]
            if any(0 <= g <= reach for g in gates):
                out[i] = HOLD
    return out


def make_env(cfg=YardEnvConfig(), rank=0, base_seed=0, seed_pool=None):
    """Thunk for SubprocVecEnv: env RNG seeded with base_seed + rank (episode seeds drawn >= 2000). Worker processes
    run the step watchdog (nohesi.rl.watchdog) when the trainer has set NOHESI_HANG_DIR."""
    def thunk():
        import multiprocessing
        env = YardEnv(cfg, seed_pool, rng_seed=base_seed+rank)
        if multiprocessing.current_process().name != 'MainProcess':
            from ..rl.watchdog import watch
            watch(env)
        return env
    return thunk
