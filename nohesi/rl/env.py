"""Gymnasium environment: one step = one team decision (Planner.decision_interval, 10 sim steps at the defaults).

The action picks one of the 45 intention actions per car (nohesi.intentions.ACTIONS); the centralized coordinator
turns them into MPC targets exactly as for the scripted policies, so an Overtaker driving the env reproduces
run(..., 'overtaker'). Reward = summed team score increments / reward_scale + aux penalties (EnvConfig).

Observation (float32, D = 3*(20+7K+14)+3 = 273 at K=8 on a 3-lane road), car blocks in car-index order, then team:
  ego (17+lanes): speed/30, vy (m/s), yaw rate/0.5, steer/steer_max, applied accel/accel_max, (d - current lane
      centre)/lane_width, heading error/0.1, current lane one-hot, phase one-hot (keep/pending/committed),
      (target lane - current lane)/2, locked lane command (lane - origin; 0 keeping)/2, target speed/30, lean,
      rejected, on fallback (mode != qp), time in phase/5 s
  traffic (7 per car x K): nearest K by |ds| within [-behind, +ahead]: ds/100, dd/lane_width, dv/30,
      lateral speed (m/s), present, could merge toward -d / +d now (the lane exists and no car, traffic or team,
      is alongside there or within the traffic lane_change_gap; cooldowns are not observable); zero padded
  teammates (7 x 2, other cars in index order): ds/100, dd/lane_width, dv/30, phase one-hot, (target lane -
      current lane)/2
  team (3): combo multiplier/5, min(time since last scored pass/combo_window, 1), time/duration
"""
from dataclasses import replace
import time
import numpy as np
import gymnasium
from gymnasium import spaces
from ..config import SCENARIOS, SCORES, Planner
from ..coordination.centralized import CentralizedTeam
from ..evaluation.metrics import metrics
from ..evaluation.runner import EpisodeRun
from ..intentions import ACTIONS, encode
from ..world import TEAM
from .config import EnvConfig

N_ACTIONS = len(ACTIONS)
LANE_CMD = np.array([a[0] for a in ACTIONS])
SPEED_CMD = np.array([a[1] for a in ACTIONS])
PHASES = ('keep', 'pending', 'committed')
EGO, TRAFFIC, MATE, TEAM_FEATURES = 17, 7, 7, 3


def merge_possible(snap, road, params, traffic_cfg):
    """(n_traffic, 2) floats: whether each traffic car could start a lane change toward -d / +d right now under
    the traffic model's room rule (target lane exists; nobody there alongside or within lane_change_gap car
    lengths). A team car counts in its CG lane and in a lane its footprint crosses into."""
    lanes, w = road.cfg.lanes, road.cfg.lane_width
    t_lane = road.lane_of(snap.traffic_d).astype(int)
    m_lane = road.lane_of(snap.team_d).astype(int)
    off = snap.team_d-np.array([road.lane_center(l) for l in m_lane])
    m_lane2 = np.where(np.abs(off)+params.width/2 > w/2, np.clip(m_lane+np.sign(off).astype(int), 0, lanes-1), m_lane)
    tgt = snap.traffic_target_lane if snap.traffic_target_lane is not None else t_lane
    s = np.r_[snap.traffic_s, snap.team_s]
    reach = params.length*(1+traffic_cfg.lane_change_gap)
    out = np.zeros((len(t_lane), 2))
    for j in range(len(t_lane)):
        for n, new in enumerate((t_lane[j]-1, t_lane[j]+1)):
            if not 0 <= new < lanes:
                continue
            occupied = np.r_[(t_lane == new) | (tgt == new), (m_lane == new) | (m_lane2 == new)]
            occupied[j] = False
            out[j, n] = not np.any(occupied & (np.abs(s-snap.traffic_s[j]) < reach))
    return out


def obs_dim(cfg, lanes=3):
    return TEAM*(EGO+lanes+TRAFFIC*cfg.traffic_k+MATE*(TEAM-1))+TEAM_FEATURES


class External:
    """Coordinator policy returning the actions the env set for this decision."""
    name = 'rl'

    def __init__(self):
        self.actions = np.full(TEAM, encode(0, 0, 0))

    def __call__(self, snap, intentions):
        return self.actions


class NoHesiEnv(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg: EnvConfig = EnvConfig(), seed_pool=None, rng_seed=None):
        self.cfg, self.seed_pool, self.pool_index = cfg, list(seed_pool) if seed_pool is not None else None, 0
        s = SCENARIOS[cfg.scenario]()
        self.scenario = replace(s, score=SCORES[cfg.score](), vehicle=replace(s.vehicle, model=cfg.vehicle_model))
        self.planner = Planner(mpc=cfg.mpc, speed_max=self.scenario.team_speed_max)
        self.lanes = self.scenario.road.lanes
        self.action_space = spaces.MultiDiscrete([N_ACTIONS]*TEAM)
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim(cfg, self.lanes),), np.float32)
        self.run = None
        self.step_started = None
        if rng_seed is not None:
            self.np_random, _ = gymnasium.utils.seeding.np_random(rng_seed)

    # Coordinator state, for scripted policies driving the env: policy(env.snapshot, env.intentions).
    @property
    def snapshot(self):
        return self.run.snap

    @property
    def team(self):
        return self.run.policy

    @property
    def intentions(self):
        return self.team.intentions.intentions

    def gap_check(self, i, lane):
        """CentralizedTeam's bind_gap_check callable for the current decision."""
        team = self.team
        return team.intentions.gap_rule(self.snapshot, team.plans, i, lane, self.intentions[i].speed)

    def episode(self):
        return self.run.episode()

    def reset(self, *, seed=None, options=None):
        """seed given: the episode's scenario seed (and the env RNG's seed). None: next seed_pool entry (cycled),
        else a draw >= 2000 from the env RNG."""
        super().reset(seed=seed)
        if seed is None:
            if self.seed_pool:
                seed = self.seed_pool[self.pool_index % len(self.seed_pool)]
                self.pool_index += 1
            else:
                seed = int(self.np_random.integers(2000, 2**31-1))
        self.episode_seed = int(seed)
        self.policy = External()
        self.run = EpisodeRun(self.scenario, self.episode_seed,
                              lambda world: CentralizedTeam(self.scenario, world, self.policy, self.planner),
                              self.planner)
        self.team.prepare(self.snapshot)
        return self._obs(), {'seed': self.episode_seed, 'team_score': 0.0}

    def step(self, action):
        self.step_started = time.monotonic()   # read by the step watchdog (rl/watchdog.py) in training workers
        try:
            return self._step(action)
        finally:
            self.step_started = None

    def _step(self, action):
        run, team, cfg = self.run, self.team, self.cfg
        if run is None or run.done:
            raise RuntimeError('call reset() first')
        assert team.deciding
        self.policy.actions = np.asarray(action, int).reshape(TEAM)
        log, records = team.log, team.intentions.records
        marks = len(records), len(log.fallbacks), len(log.conflicts)
        terms = {}
        for _ in range(team.decide_every):
            for t, v in run.step().items():
                terms[t] = terms.get(t, 0.0)+float(v.sum())
            if run.done:
                break
        if not run.done:
            team.prepare(run.snap)
        fallbacks = [f[2] for f in log.fallbacks[marks[1]:]]
        aux = {'rejected': cfg.aux_rejected*sum(r.kind == 'rejected' for r in records[marks[0]:]),
               'fallback': cfg.aux_fallback*sum(f in ('shifted', 'lane_keep') for f in fallbacks),
               'conflict': cfg.aux_conflict*sum(c[4] == 'unresolved' for c in log.conflicts[marks[2]:]),
               'emergency': cfg.aux_emergency*(fallbacks.count('emergency')+fallbacks.count('escape')),
               'collision': cfg.collision_extra*terms.get('collision', 0.0)/cfg.reward_scale}
        reward = sum(terms.values())/cfg.reward_scale+sum(aux.values())
        terminated = bool(run.snap.done)
        truncated = run.done and not terminated
        info = {'score_terms': terms, 'aux': aux, 'team_score': run.scorer.team(), 'seed': self.episode_seed}
        if run.done:
            info['episode_metrics'] = metrics(run.episode())
        return self._obs(), float(reward), terminated, truncated, info

    def action_masks(self):
        """(3*45,) bool: lane commands into an off-road lane masked (pending: relative to the origin, any other
        valid command aborts); committed: only the locked command (lane - origin). With cfg.mask_gap_rule, a car
        keeping its lane can only request lane changes the gap rule accepts now, checked per lane x speed action
        at the target speed that action would set (a teammate deciding first can still cause a rare rejection). Speed and lean are
        never masked."""
        out = np.zeros((TEAM, N_ACTIONS), bool)
        for i, it in enumerate(self.intentions):
            if it.phase == 'committed':
                ok = LANE_CMD == it.lane-it.origin
            else:
                base = it.origin if it.phase == 'pending' else it.lane
                ok = (base+LANE_CMD >= 0) & (base+LANE_CMD < self.lanes)
                if it.phase == 'keep' and self.cfg.mask_gap_rule:
                    team, pc = self.team, self.team.cfg
                    for c in np.unique(LANE_CMD[ok & (LANE_CMD != 0)]):
                        for sc in (-1, 0, 1):
                            # The speed the decision would set, as IntentionManager.decide applies it before the gap rule.
                            speed = float(np.clip(it.speed+sc*pc.speed_step, pc.speed_min, pc.speed_max))
                            if team.intentions.gap_rule(self.snapshot, team.plans, i, base+int(c), speed)[0] is None:
                                ok &= ~((LANE_CMD == c) & (SPEED_CMD == sc))
            out[i] = ok if ok.any() else LANE_CMD == 0
        return out.ravel()

    def _obs(self):
        cfg, run, team, snap = self.cfg, self.run, self.team, self.snapshot
        road, p, w, K = team.road, team.params, self.scenario.road.lane_width, cfg.traffic_k
        lanes = road.lane_of(snap.team_d).astype(int)
        v = snap.team[:, 3]
        accel = run.controls[-1][:, 0] if run.controls else np.zeros(TEAM)
        vy = snap.team_vy if snap.team_vy is not None else np.zeros(TEAM)
        r = snap.team_r if snap.team_r is not None else np.zeros(TEAM)
        vd = snap.traffic_vd if snap.traffic_vd is not None else np.zeros_like(snap.traffic_d)
        heading = (snap.team[:, 2]-road.heading(snap.team_s)+np.pi) % (2*np.pi)-np.pi
        phase = np.zeros((TEAM, 3))
        for i, it in enumerate(self.intentions):
            phase[i, PHASES.index(it.phase)] = 1
        target = np.array([(it.lane-lanes[i])/2 for i, it in enumerate(self.intentions)])
        merge = merge_possible(snap, road, p, self.scenario.traffic)
        blocks = []
        for i, it in enumerate(self.intentions):
            ego = np.r_[v[i]/30, vy[i], r[i]/0.5, snap.team_steer[i]/p.steer_max, accel[i]/p.accel_max,
                        (snap.team_d[i]-road.lane_center(lanes[i]))/w, heading[i]/0.1, np.eye(self.lanes)[lanes[i]],
                        phase[i], target[i], (it.lane-it.origin)/2 if it.phase != 'keep' else 0.0, it.speed/30,
                        it.lean, float(it.rejected), float(team.mode[i] != 'qp'),
                        (snap.time-it.since)/5 if it.since is not None and it.phase != 'keep' else 0.0]
            traffic = np.zeros((K, TRAFFIC))
            ds = snap.traffic_s-snap.team_s[i]
            near = np.flatnonzero((ds >= -cfg.behind) & (ds <= cfg.ahead))
            near = near[np.argsort(np.abs(ds[near]), kind='stable')][:K]
            traffic[:len(near)] = np.column_stack((ds[near]/100, (snap.traffic_d[near]-snap.team_d[i])/w,
                                                   (snap.traffic_v[near]-v[i])/30, vd[near], np.ones(len(near)),
                                                   merge[near]))
            mates = [np.r_[(snap.team_s[j]-snap.team_s[i])/100, (snap.team_d[j]-snap.team_d[i])/w, (v[j]-v[i])/30,
                           phase[j], target[j]] for j in range(TEAM) if j != i]
            blocks += [ego, traffic.ravel(), *mates]
        sc = run.scorer
        blocks.append([sc.multiplier/5, min((sc.time-sc.last_pass)/self.scenario.score.combo_window, 1.0),
                       snap.time/self.scenario.timing.duration])
        return np.concatenate(blocks).astype(np.float32)


def make_env(cfg=EnvConfig(), rank=0, base_seed=0, seed_pool=None):
    """Thunk for SubprocVecEnv: env RNG seeded with base_seed + rank (episode seeds drawn >= 2000). In a worker process
    with the nonlinear MPC, the C-level stdout goes to os.devnull: acados prints a line per failed QP, which would
    flood the training log (the main process keeps its stdout). Worker processes also run the step watchdog
    (rl/watchdog.py) when the trainer has set NOHESI_HANG_DIR."""
    def thunk():
        import multiprocessing
        import os
        from .watchdog import watch
        worker = multiprocessing.current_process().name != 'MainProcess'
        if cfg.mpc == 'nonlinear' and worker:
            os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
        env = NoHesiEnv(cfg, seed_pool, rng_seed=base_seed+rank)
        if worker:
            watch(env)
        return env
    return thunk
