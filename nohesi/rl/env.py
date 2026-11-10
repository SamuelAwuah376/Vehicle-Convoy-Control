"""Gymnasium environment: one step = one team decision (EnvConfig.decision_interval, 10 sim steps at the default 0.5 s).

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

e5 (EnvConfig.overtakes): 51 actions per car, the 45 plus intentions.OVERTAKES (pass the traffic car ahead on the
right/left, at least 3/6/9 m/s faster). Masks: an overtake starts only if IntentionManager.overtake_check accepts it
now (entry gap and passing corridor, per passing margin); once started it runs to completion (returns when there is
room, else stays in the passing lane; the intention layer may still cancel it for safety), so during one only the
same-side overtakes are allowed, the speed command setting the passing margin; a plain lane change is how the policy
moves over and stays. Observation: the e4 vector, then 11 per car (appended so e4 inputs keep their places): overtake stage
one-hot (none/out/pass/return), side, passed car ds/100 and dv/30, time in the overtake/5 s, overtake left / right
possible now (margin +6), return possible now. Reward: aux_overtake_abort per overtake cancelled before the pass.

Follow (EnvConfig.follow, e5): action FOLLOW (= the overtake right while slowing, which then never starts) follows the
teammate directly ahead in the lane (Planner.follow); masked unless the car follows already or follow_check would start
one. During a follow it is the only action (the follower makes no decisions of its own): the follow ends only on the
intention layer's break-offs (its join refused for follow_join_time, its lane change cancelled, a fallback, the leader
beyond follow_break or overtaking), and the car is back on the policy. Observation: 2 per car appended
after the e5 features: following, follow possible now. Missed escapes: a follower whose leader is changing lanes or
starts a lane change at this decision takes the leader's way out with it (not charged); a dropped join of a follower
(record 'aborted', reason 'follow') is not charged as aux_lc_abort.
"""
from dataclasses import replace
import time
import numpy as np
import gymnasium
from gymnasium import spaces
from ..config import SCENARIOS, SCORES, Planner
from ..coordination.centralized import CentralizedTeam
from ..evaluation.metrics import metrics, stuck_mask
from ..evaluation.runner import EpisodeRun
from ..intentions import ACTIONS, FOLLOW, OVERTAKES, encode
from ..world import TEAM, World
from .config import EnvConfig

N_ACTIONS = len(ACTIONS)
LANE_CMD = np.array([a[0] for a in ACTIONS])
SPEED_CMD = np.array([a[1] for a in ACTIONS])
LEAN_CMD = np.array([a[2] for a in ACTIONS])
OT_SIDE = np.array([o[0] for o in OVERTAKES])
OT_SPEED = np.array([o[1] for o in OVERTAKES])
PHASES = ('keep', 'pending', 'committed')
OT_STAGES = (None, 'out', 'pass', 'return')
EGO, TRAFFIC, MATE, TEAM_FEATURES, OT_FEATURES, FOLLOW_FEATURES = 17, 7, 7, 3, 11, 2


def n_actions(cfg):
    """Actions per car: 45 (e4), plus the overtakes with cfg.overtakes (e5)."""
    return N_ACTIONS+(len(OVERTAKES) if cfg.overtakes else 0)


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
    return TEAM*(EGO+lanes+TRAFFIC*cfg.traffic_k+MATE*(TEAM-1))+TEAM_FEATURES+(TEAM*OT_FEATURES if cfg.overtakes else 0) + \
        (TEAM*FOLLOW_FEATURES if getattr(cfg, 'follow', False) else 0)


def clock_index(cfg, lanes=3):
    """Position of the episode clock (the team block's last feature, EnvConfig.obs_clock) in the observation."""
    return TEAM*(EGO+lanes+TRAFFIC*cfg.traffic_k+MATE*(TEAM-1))+TEAM_FEATURES-1


class External:
    """Coordinator policy returning the actions the env set for this decision."""
    name = 'rl'

    def __init__(self):
        self.actions = np.full(TEAM, encode(0, 0, 0))

    def __call__(self, snap, intentions):
        return self.actions


def planner_for(cfg, scenario):
    """The team's Planner for an EnvConfig: its MPC and decision timing, the scenario's top speed, then the run's
    planner_overrides and follow mode. The scripted baseline (rl.evaluate.scenario_for) uses the same one."""
    planner = Planner(mpc=cfg.mpc, speed_max=scenario.team_speed_max, decision_interval=cfg.decision_interval,
                      speed_step=cfg.speed_step)
    if cfg.planner_overrides:
        planner = replace(planner, **dict(cfg.planner_overrides))
    if cfg.follow:
        planner = replace(planner, follow=True)
    return planner


class NoHesiEnv(gymnasium.Env):
    metadata = {'render_modes': []}

    def __init__(self, cfg: EnvConfig = EnvConfig(), seed_pool=None, rng_seed=None, world=World):
        """world: (scenario, seed) -> world for each episode (EpisodeRun); a CarlaSession runs it in CARLA."""
        self.cfg, self.seed_pool, self.pool_index = cfg, list(seed_pool) if seed_pool is not None else None, 0
        self.world = world
        s = SCENARIOS[cfg.scenario]()
        self.scenario = replace(s, score=SCORES[cfg.score](), vehicle=replace(s.vehicle, model=cfg.vehicle_model))
        self.planner = planner_for(cfg, self.scenario)
        self.lanes = self.scenario.road.lanes
        self.action_space = spaces.MultiDiscrete([n_actions(cfg)]*TEAM)
        self._options = None   # (time, (3, len(OVERTAKES)) bool): overtakes that would start now, per decision
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
        return team.intentions.lane_gap(self.snapshot, team.plans, i, lane, self.intentions[i].speed)[:2]

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
                              self.planner, world=self.world)
        self.team.prepare(self.snapshot)
        self.stuck_for, self._stuck_t = np.zeros(TEAM), self.snapshot.time
        self.lean_changes = 0   # this episode's lean changes while holding the lane (aux_lean_change), all cars
        return self._obs(), {'seed': self.episode_seed, 'team_score': 0.0}

    def stuck(self, snap):
        """(3,) bool: cars stuck behind traffic now (EnvConfig.aux_stuck; evaluation.metrics.stuck_mask)."""
        return stuck_mask(self.team.intentions.road, snap.team_s[None], snap.team_d[None], snap.team[None, :, 3],
                          snap.traffic_s[None], snap.traffic_d[None], snap.traffic_v[None],
                          self.cfg.stuck_window, self.cfg.stuck_closing)[0]

    def _stuck_seconds(self, snap, charge=None):
        """Advance the per-car stuck timers to snap; car-seconds stuck beyond stuck_grace since the last call (only the
        cars in the (3,) bool charge, if given; the timers run for all)."""
        dt, before = snap.time-self._stuck_t, self.stuck_for
        self.stuck_for = np.where(self.stuck(snap), before+dt, 0.0)
        self._stuck_t = snap.time
        over = np.maximum(self.stuck_for-np.maximum(before, self.cfg.stuck_grace), 0.0)
        return float(np.sum(over if charge is None else over[charge]))

    def escapes(self, cars=None, actions=None):
        """(3,) bool: cars with a way out of their lane at this decision (EnvConfig.stuck_escape_only), among the
        (3,) bool cars (default all). A car keeping its lane with no overtake running has one if an overtake would start
        now, or if the mask allows a lane change into a lane whose nearest traffic car ahead within stuck_window is at
        least stuck_closing faster than the one in its own lane, or has none; with stuck_pass_lanes, also into a lane
        it can pass its lead traffic car through (_passable). A car changing lanes or overtaking: no. With actions (this
        decision's), a car whose action takes one of its ways out (that overtake or lane change) is left out: a move
        into a lane that is not a way out does not count."""
        cfg, snap, road = self.cfg, self.snapshot, self.team.intentions.road
        t_lane = road.lane_of(snap.traffic_d).astype(int)
        out = np.zeros(TEAM, bool)
        for i, it in enumerate(self.intentions):
            if (cars is not None and not cars[i]) or it.phase != 'keep' or getattr(it, 'overtake', None) is not None:
                continue
            ds = snap.traffic_s-snap.team_s[i]

            def lead_v(lane):
                sel = (t_lane == lane) & (ds > 0) & (ds < cfg.stuck_window)
                return snap.traffic_v[sel][np.argmin(ds[sel])] if sel.any() else np.inf
            own = lead_v(it.lane)
            ok = self._lane_mask(i, it)
            ways = [c for c in np.unique(LANE_CMD[ok & (LANE_CMD != 0)]) if lead_v(it.lane+int(c)) >= own+cfg.stuck_closing]
            if cfg.stuck_pass_lanes:
                ways += [c for c in np.unique(LANE_CMD[ok & (LANE_CMD != 0)])
                         if c not in ways and self._passable(i, it, int(c), ok)]
            overtake = bool(cfg.overtakes and self._overtake_options()[i].any())
            a = None if actions is None else int(actions[i])
            if cfg.follow and a == FOLLOW:   # following: the leader's way out is the follower's
                j = it.follow if it.follow is not None else self.team.intentions.follow_check(snap, i)[0]
                lead = self.intentions[j] if j is not None else None
                a_j = int(actions[j]) if j is not None else None
                taken = lead is not None and (lead.phase != 'keep' or (a_j < N_ACTIONS and LANE_CMD[a_j] != 0))
            else:
                taken = a is not None and ((a >= N_ACTIONS and overtake) or (a < N_ACTIONS and LANE_CMD[a] in ways))
            out[i] = (bool(ways) or overtake) and not taken
        return out

    def _passable(self, i, it, c, ok):
        """EnvConfig.stuck_pass_lanes: can car i (keeping its lane) get past its lead traffic car (within overtake_look)
        through the adjacent lane it.lane+c? The overtake corridor test (IntentionManager._corridor: overtake_clear past
        the lead within overtake_pass_max, the lane's front gap car still length + margin_long ahead then) at the team's
        top speed, on a gap the mask allows the change into (any of its allowed speed commands)."""
        im, snap, pc = self.team.intentions, self.snapshot, self.team.cfg
        lead = im._lead(snap, i)
        if lead is None:
            return False
        for sc in (-1, 0, 1):
            if not (ok & (LANE_CMD == c) & (SPEED_CMD == sc)).any():
                continue
            speed = float(np.clip(it.speed+sc*pc.speed_step, pc.speed_min, pc.speed_max))
            gap = im.lane_gap(snap, self.team.plans, i, it.lane+c, speed, sc)[0]
            if gap is not None and im._corridor(snap, self.team.plans, i, pc.speed_max, lead, gap[0]):
                return True
        return False

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
        terms, stuck, braked = {}, 0.0, 0.0
        # Missed escapes are judged at the decision (before the commands act) and charged over its interval, unless the
        # car takes one now; only cars that can pass stuck_grace in it are checked.
        charge = None
        if cfg.aux_stuck and cfg.stuck_escape_only:
            due = self.stuck_for+team.decide_every*self.scenario.timing.dt > cfg.stuck_grace
            charge = self.escapes(self.stuck(run.snap) & due, self.policy.actions)
        lean_changes = 0   # cars holding their lane (no lane change or overtake started or running) that change lean now
        for i, it in enumerate(self.intentions):
            a = int(self.policy.actions[i])
            lean_changes += bool(it.phase == 'keep' and getattr(it, 'overtake', None) is None and a < N_ACTIONS
                                 and LANE_CMD[a] == 0 and LEAN_CMD[a] != it.lean)
        self.lean_changes += lean_changes
        for _ in range(team.decide_every):
            for t, v in run.step().items():
                terms[t] = terms.get(t, 0.0)+float(v.sum())
            if cfg.aux_stuck:
                stuck += self._stuck_seconds(run.snap, charge)
            if cfg.aux_brake:
                braked += float(np.maximum(-run.controls[-1][:, 0], 0.0).sum())*self.scenario.timing.dt
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
        if cfg.crash_penalty:   # fixed, whatever the banked points
            aux['crash'] = cfg.crash_penalty*bool(run.snap.done)
        if cfg.overtakes:   # an overtake abandoned before the pass (ending in the passing lane is 'stayed', free)
            aux['overtake_abort'] = cfg.aux_overtake_abort*sum(r.kind == 'overtake_ended' and r.reason != 'car_gone'
                                                               for r in records[marks[0]:])
        if cfg.aux_stuck:
            aux['stuck'] = cfg.aux_stuck*stuck
        if cfg.aux_brake:
            aux['brake'] = cfg.aux_brake*braked
        if cfg.aux_lean_change:
            aux['lean_change'] = cfg.aux_lean_change*lean_changes
        if cfg.aux_lc_abort:
            aux['lc_abort'] = cfg.aux_lc_abort*sum(r.kind == 'aborted' and r.reason != 'follow' for r in records[marks[0]:])
        reward = sum(terms.values())/cfg.reward_scale+sum(aux.values())
        terminated = bool(run.snap.done)
        truncated = run.done and not terminated
        info = {'score_terms': terms, 'aux': aux, 'team_score': run.scorer.team(), 'seed': self.episode_seed}
        if run.done:
            info['episode_metrics'] = {**metrics(run.episode()), 'lean_changes': self.lean_changes}
        return self._obs(), float(reward), terminated, truncated, info

    def _overtake_options(self):
        """(3, len(OVERTAKES)) bool: the overtakes IntentionManager.overtake_check would start now (cached per
        decision: the masks and the observation both use them)."""
        snap = self.snapshot
        if self._options is None or self._options[0] != (id(self.run), snap.time):
            team = self.team
            ok = np.array([[team.intentions.overtake_check(snap, team.plans, i, side, sc)[0] == ''
                            for side, sc in OVERTAKES] for i in range(TEAM)])
            if self.cfg.follow:   # that slot is FOLLOW: its overtake never starts
                ok[:, FOLLOW-N_ACTIONS] = False
            self._options = ((id(self.run), snap.time), ok)
        return self._options[1]

    def action_masks(self):
        """(3*45,) bool (e4; (3*51,) with overtakes, see the module docstring): lane commands into an off-road lane
        masked (pending: relative to the origin, any other valid command aborts); committed: only the locked command
        (lane - origin). With cfg.mask_gap_rule, a car keeping its lane can only request lane changes the gap rule
        accepts now, checked per lane x speed action at the target speed that action would set (a teammate deciding
        first can still cause a rare rejection). Speed and lean are never masked."""
        if self.cfg.overtakes:
            return self._masks_with_overtakes()
        out = np.zeros((TEAM, N_ACTIONS), bool)
        for i, it in enumerate(self.intentions):
            ok = self._lane_mask(i, it)
            out[i] = ok if ok.any() else LANE_CMD == 0
        return out.ravel()

    def _lane_mask(self, i, it):
        """(45,) bool: car i's lane x speed x lean actions allowed (action_masks, before its keep fallback)."""
        if it.phase == 'committed':
            return LANE_CMD == it.lane-it.origin
        base = it.origin if it.phase == 'pending' else it.lane
        ok = (base+LANE_CMD >= 0) & (base+LANE_CMD < self.lanes)
        if it.phase == 'keep' and self.cfg.mask_gap_rule:
            team, pc = self.team, self.team.cfg
            for c in np.unique(LANE_CMD[ok & (LANE_CMD != 0)]):
                for sc in (-1, 0, 1):
                    # The speed the decision would set, as IntentionManager.decide applies it before the gap rule.
                    speed = float(np.clip(it.speed+sc*pc.speed_step, pc.speed_min, pc.speed_max))
                    if team.intentions.lane_gap(self.snapshot, team.plans, i, base+int(c), speed, sc)[0] is None:
                        ok &= ~((LANE_CMD == c) & (SPEED_CMD == sc))
        return ok

    def _masks_with_overtakes(self):
        """(3*51,) bool: without an overtake, the lane mask plus the overtakes that would start now; during one, only
        the same-side overtakes (it runs to completion, the speed command setting its passing margin; to move over and
        stay, the policy uses a plain lane change instead); a car with nothing allowed keeps its lane."""
        keep = np.r_[LANE_CMD == 0, np.zeros(len(OVERTAKES), bool)]
        options = self._overtake_options()
        out = np.zeros((TEAM, n_actions(self.cfg)), bool)
        for i, it in enumerate(self.intentions):
            ot = it.overtake
            row = np.r_[self._lane_mask(i, it), options[i]] if ot is None else \
                np.r_[np.zeros(N_ACTIONS, bool), OT_SIDE == ot.side]
            if self.cfg.follow:
                if it.follow is not None:   # following: no decisions of its own until a break-off ends it
                    row = np.zeros(n_actions(self.cfg), bool)
                    row[FOLLOW] = True
                else:
                    row[FOLLOW] = ot is None and self._can_follow(i)
            out[i] = row if row.any() else keep
        return out.ravel()

    def _can_follow(self, i):
        """EnvConfig.follow: car i follows now, or follow_check would start one."""
        return self.intentions[i].follow is not None or \
            self.team.intentions.follow_check(self.snapshot, i)[0] is not None

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
                       snap.time/self.scenario.timing.duration if cfg.obs_clock else 0.0])
        if cfg.overtakes:   # e5: appended, so the e4 inputs keep their places (a warm start copies their weights)
            options, ims = self._overtake_options(), team.intentions
            left, right = (int(np.flatnonzero((OT_SIDE == s) & (OT_SPEED == 0))[0]) for s in (1, -1))
            for i, it in enumerate(self.intentions):
                ot = it.overtake
                stage = np.zeros(len(OT_STAGES))
                stage[OT_STAGES.index(ot.stage if ot is not None else None)] = 1
                car = ims.passed_car(snap, i)
                rel = ((car[0]-snap.team_s[i])/100, (car[1]-v[i])/30) if car is not None else (0.0, 0.0)
                ready = ot is not None and ot.stage == 'pass' and ims.return_gap(snap, team.plans, i) is not None
                blocks.append(np.r_[stage, ot.side if ot is not None else 0, rel,
                                    (snap.time-ot.since)/5 if ot is not None else 0.0,
                                    float(options[i, left]), float(options[i, right]), float(ready)])
        if cfg.follow:   # appended after the e5 features (a warm start keeps the other inputs' places)
            for i, it in enumerate(self.intentions):
                blocks.append([float(it.follow is not None), float(self._can_follow(i))])
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
