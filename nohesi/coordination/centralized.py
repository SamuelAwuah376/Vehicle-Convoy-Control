"""Centralized team policy with one MPC per vehicle, solved sequentially in priority order."""
from dataclasses import dataclass, field, replace
import numpy as np
from ..dynamics import DynamicPlant, brake_limit
from ..intentions import IntentionManager
from ..planning.base import Obstacles, Target, constant_speed_plan
from ..planning.fallback import dense, first_conflict, lane_keep_plan, shifted_fallback
from ..planning.single import LaneChangeMPC
from ..prediction import traffic_prediction
from ..stability import StabilityLayer
from ..rl.scripted import lane_keep_control
from ..world import TEAM


def make_mpc(cfg, plant, road):
    """Per-vehicle MPC selected by cfg.mpc."""
    if cfg.mpc == 'linear':
        return LaneChangeMPC(cfg, plant, road)
    from ..planning.nonlinear import NonlinearMPC   # lazy: the linear path never imports acados
    return NonlinearMPC(cfg, plant, road)


@dataclass
class PlannerLog:
    solves: list = field(default_factory=list)       # (time, vehicle, status, iterations, solve_time_s, attempt)
    fallbacks: list = field(default_factory=list)    # (time, vehicle, 'shifted' | 'lane_keep' | 'emergency' | 'escape')
    conflicts: list = field(default_factory=list)    # (time, vehicle, other key, step, resolution)
    plans: list = field(default_factory=list)        # (time, vehicle, source, z) executed plan per replan
    predictions: list = field(default_factory=list)  # (due_time, vehicle, s, d) planned one MPC step ahead
    missed: np.ndarray = field(default_factory=lambda: np.zeros(TEAM))  # missed-opportunity seconds


class CentralizedTeam:
    """Calls the policy every decision interval, replans every MPC step, returns held controls.

    Priority: vehicles mid lane change first, then front to back. A vehicle avoids plans already
    computed this cycle and the shifted previous-cycle plans of teammates not yet planned. After
    the cycle, every plan is footprint-checked against predicted traffic and higher-priority
    teammates' final plans; on a conflict the lower-priority vehicle yields (abandon any lane change
    for the lane whose lane-keep rollout stays clear longest and re-solve, else the emergency escape). With the
    nonlinear MPC (Planner.nmpc_emergency) the escape is an MPC plan in emergency mode (_emergency): candidate
    manoeuvres (brake, hold or gain speed, in the car's lane or one either side), each solved with the contact rows
    only, the clear one (else the one with the least contact) executed like any plan; the car plans first next cycle.
    Otherwise, and as the last resort when no emergency solve succeeds ('escape'), the lane-keep law escape: full
    braking, holding or gaining speed in the nearby lane with the gentlest predicted impact.
    A car mid lane change backs out when its target slot closes: if, on its plan, the time to contact with a gap
    car or a vehicle merging into a lane it enters (at constant speed) drops below slot_margin wherever they would
    share the lane, it abandons the change for the clearest nearby lane.
    Plans are checked check_substeps times per step. The checks use predictions: not a guarantee.
    On the dynamic plant every braking command (MPC, lane-keep fallback, emergency escape) is capped at grip_brake mu g,
    so braking leaves grip to steer with, and the MPC starts from the acceleration last commanded (on any mode), not
    from its own last plan's. There the stability layer (Planner.stability) carries out every command at 100 Hz within
    grip: the MPC's with its planned motion, the fallbacks' with the car's steady-state response; the MPC replans
    from the state the layer last posted.
    """
    name = 'centralized'

    def __init__(self, scenario, world, policy, cfg):
        cfg = replace(cfg, speed_max=scenario.team_speed_max)   # the scenario owns the team speed cap
        dt = scenario.timing.dt
        self.decide_every = int(round(cfg.decision_interval/dt))
        self.plan_every = int(round(cfg.step/dt))
        if not np.isclose(self.plan_every*dt, cfg.step) or self.decide_every % self.plan_every:
            raise ValueError('planner step and decision interval must be multiples of dt and of each other')
        self.cfg, self.world, self.policy = cfg, world, policy
        self.road, self.plant, self.params = world.road, world.plant, world.plant.params
        self.mpc = make_mpc(cfg, world.plant, world.road)
        self.intentions = IntentionManager(cfg, world.road, self.params, scenario.team_lane, scenario.team_speed)
        self.plans = [None]*TEAM
        self.mode = ['qp']*TEAM
        self.escape = [None]*TEAM   # (lane, target speed, accel cap) of the lane_keep/emergency control law
        self.held = np.zeros((TEAM, 2))
        self.brake = brake_limit(world.plant, cfg.grip_brake)   # brake_max on the kinematic plant
        self.dynamic = isinstance(world.plant, DynamicPlant)
        self.emergency_mpc = cfg.mpc == 'nonlinear' and cfg.nmpc_emergency
        self.stability = StabilityLayer(world.plant, cfg) if self.dynamic and cfg.stability else None
        world.stability = self.stability
        self.last_accel = np.zeros(TEAM)   # acceleration last commanded, whatever the mode
        self.log = PlannerLog()
        self.k = 0
        self.prepared = None   # step whose phase update already ran (prepare)
        self.name = f'centralized/{getattr(policy, "name", type(policy).__name__)}'

    def prepare(self, snap):
        """Run the next call's phase update now, so an external policy sees the intentions that call decides with."""
        if self.prepared != self.k:
            self.intentions.update_phases(snap, self.plans)
            self.prepared = self.k

    @property
    def deciding(self):
        """The next call makes a policy decision."""
        return self.k % self.decide_every == 0

    def __call__(self, snap):
        self.prepare(snap)
        if self.deciding:
            self._decide(snap)
        if self.k % self.plan_every == 0:
            self._replan(snap)
        controls = self.held.copy()
        for i in range(TEAM):
            if not self._plan_driven(i):
                controls[i] = lane_keep_control(snap, self.road, self.params, i, *self.escape[i], self.brake)
        self.last_accel = controls[:, 0].copy()
        if self.stability is not None:
            for i in range(TEAM):   # the MPC's plan (planned motion) or None: steady-state reference of the command
                self.stability.post(i, self.plans[i] if self._plan_driven(i) else None)
        self.k += 1
        return controls

    def _plan_driven(self, i):
        """Car i follows its plan's commands (else a control law: lane keeping or the law escape)."""
        return self.mode[i] in ('qp', 'shifted') or (self.mode[i] == 'emergency' and self.emergency_mpc)

    def order(self, snap):
        """Planning order: cars in an MPC emergency escape first, then cars mid lane change, then front to back."""
        busy = [it.phase != 'keep' for it in self.intentions.intentions]
        escaping = [self.mode[i] == 'emergency' and self.emergency_mpc for i in range(TEAM)]
        return sorted(range(TEAM), key=lambda i: (not escaping[i], not busy[i], -snap.team_s[i]))

    def _decide(self, snap):
        intents = self.intentions.intentions
        if hasattr(self.policy, 'bind_gap_check'):
            # Read-only: lets a scripted policy ask whether a lane change would be accepted right now.
            self.policy.bind_gap_check(lambda i, lane: self.intentions.gap_rule(snap, self.plans, i, lane, intents[i].speed))
        actions = self.policy(snap, intents)
        stuck = [it.phase == 'keep' and hasattr(self.policy, 'stuck') and self.policy.stuck(snap, i, it) is not None
                 for i, it in enumerate(intents)]
        self.intentions.decide(snap, actions, self.plans, self.order(snap))
        for i, it in enumerate(intents):
            # Missed opportunity: stuck, still keeping lane, yet the gap rule would accept a change.
            if stuck[i] and it.phase == 'keep' and any(
                    self.intentions.gap_rule(snap, self.plans, i, it.lane+c, it.speed)[0] is not None for c in (-1, 1)):
                self.log.missed[i] += self.cfg.decision_interval

    def _obstacles(self, snap, i, fresh):
        cfg, N = self.cfg, self.cfg.horizon
        s, d, v, _ = traffic_prediction(snap, self.road, np.arange(N+1)*cfg.step)
        traffic = Obstacles(tuple(('traffic', int(k)) for k in snap.traffic_id), s, d, v)
        mates = []
        for j in range(TEAM):
            if j == i:
                continue
            plan = fresh.get(j)
            if plan is None:
                plan = self.plans[j].shifted(1) if self.plans[j] is not None else \
                    constant_speed_plan(snap.time, cfg.step, N, snap.team_s[j], snap.team_d[j], snap.team[j, 3])
            mates.append(Obstacles((('team', j),), plan.z[:, :1], plan.z[:, 1:2], plan.z[:, 3:4]))
        return Obstacles.stack([traffic, *mates], N+1)

    def _solve(self, snap, i, obstacles, reference, attempt, target=None, emergency=False):
        latest = self.stability.buffer.latest.get(i) if self.stability is not None else None
        if latest is not None:   # the state the stability layer last posted (end of the last sim step)
            _, (x, y, yaw, v, vy, r), _, _ = latest
            s, d = self.road.to_frenet(np.array([x, y]))
            z0, lateral = np.array([s, d, yaw-self.road.heading(s), v]), (float(vy), float(r))
        else:
            z0 = np.array([snap.team_s[i], snap.team_d[i], snap.team[i, 2]-self.road.heading(snap.team_s[i]), snap.team[i, 3]])
            lateral = tuple(0.0 if x is None else float(x[i]) for x in (snap.team_vy, snap.team_r))   # dynamic plant (vy, r)
        if target is None:
            target = replace(self.intentions.target(i), yield_to=self._yield_to(snap, i))
            release = self.intentions.overtake_release(i)
            if release is not None:
                target = self._staged(target, release, obstacles, reference)
        # lateral passed positionally (solve(..., reference, lateral=None)) so positional solve wrappers keep working.
        # Dynamic plant: start from what the car was last given (a fallback may have braked since the last plan).
        accel = self.last_accel[i] if self.dynamic else self.held[i, 0]
        if emergency:
            plan = self.mpc.solve(snap.time, z0, snap.team_steer[i], accel, target, obstacles, reference, lateral,
                                  emergency=True)
        else:
            plan = self.mpc.solve(snap.time, z0, snap.team_steer[i], accel, target, obstacles, reference, lateral)
        self.log.solves.append((snap.time, i, plan.status, plan.iterations, plan.solve_time, attempt))
        return plan

    def _staged(self, target, release, obstacles, reference):
        """An overtake's pull-out or return (IntentionManager.overtake_release): lateral target per horizon step, a
        quintic blend from `start` to the target d over overtake_merge_time. Pull-out: the blend starts when the
        overtake started. Return: the passing lane until the reference is overtake_clear past the passed car's
        predicted position (bumper to bumper), the blend from then on; that start is the predicted clearing time,
        held fixed once the car is clear now, so replanning continues the merge instead of restarting it. The gap
        constraints keep the car beside the passed car until it is clear anyway."""
        ot, start, end = release
        h = self.cfg.step
        t = reference.time+np.arange(len(reference.z))*h
        duration = self.cfg.overtake_merge_time
        if ot.stage == 'out':
            begin, duration = ot.since, ot.out_time or duration
        else:
            o = obstacles.index(ot.car)
            if o is None:
                return target
            clear = self.params.length+self.cfg.margin_long+self.cfg.overtake_clear
            past = reference.z[:, 0]-obstacles.s[:, o] >= clear
            if not ot.merge_fixed:
                ot.merge_start = float(t[np.argmax(past)]) if past.any() else np.inf
                ot.merge_fixed = bool(past[0])
            begin = ot.merge_start
        x = np.clip((t-begin)/duration, 0.0, 1.0)
        blend = x*x*x*(10.0-15.0*x+6.0*x*x)
        return replace(target, d_stage=tuple(start+(end-start)*blend))

    def _yield_to(self, snap, j):
        """Teammates changing into car j's lane (middle or destination) within yield_window beyond bumper overlap.
        Skipped when getting clear the way j already is would contradict the changer's locked gap (j is its front
        car yet level/behind, or its rear car yet ahead): the changer then sorts it out itself."""
        lane, near, me = self.road.lane_of(snap.team_d[j]), self.params.length+self.cfg.yield_window, ('team', j)
        out = []
        for i, it in enumerate(self.intentions.intentions):
            ds = snap.team_s[j]-snap.team_s[i]
            if i == j or it.phase == 'keep' or lane not in self.intentions.entered(it.origin, it.lane) or abs(ds) >= near \
                    or abs(self.road.lane_of(snap.team_d[i])-lane) > 1:   # changer next to (or in) the lane
                continue
            if (it.front == me and ds <= 0) or (it.rear == me and ds > 0):
                continue
            out.append(('team', i))
        return tuple(out)

    def _lanes(self, snap, i, first=()):
        """Fallback lanes in preference order: `first`, the intention lane, the origin, the CG's lane, the lane across
        the closer lane line, then the lane across the other one."""
        it, d = self.intentions.intentions[i], snap.team_d[i]
        here = int(self.road.lane_of(d))
        side = 1 if d >= self.road.lane_center(here) else -1
        out = []
        for lane in (*first, it.lane, it.origin, here, here+side, here-side):
            if lane is not None and 0 <= lane < self.road.cfg.lanes and int(lane) not in out:
                out.append(int(lane))
        return out

    def _mates(self, snap, plans, check):
        """`plans` plus, for `check` teammates without one this cycle, the shifted previous or a constant-speed plan."""
        plans = dict(plans)
        for j in check:
            if j not in plans:
                plans[j] = self.plans[j].shifted(1) if self.plans[j] is not None else constant_speed_plan(
                    snap.time, self.cfg.step, self.cfg.horizon, snap.team_s[j], snap.team_d[j], snap.team[j, 3])
        return plans

    def _impact_speed(self, snap, plan, hit, plans):
        """Speed difference (m/s) between the car on `plan` and the vehicle it first conflicts with, at that step."""
        (kind, j), k = hit
        k = min(k, len(plan.z)-1)
        if kind == 'team':
            other = plans[j].z[min(k, len(plans[j].z)-1), 3]
        else:
            _, _, v, _ = traffic_prediction(snap, self.road, np.array([k*self.cfg.step]))
            other = v[0, int(np.flatnonzero(snap.traffic_id == j)[0])]
        return abs(float(plan.z[k, 3]-other))

    def _rollout(self, snap, i, candidates, plans, check, source):
        """First candidate (lane, speed, cap) whose lane-keep rollout is clear of predicted traffic and the
        `check` teammates' plans (this cycle's, else the shifted previous or a constant-speed one); if none is,
        the one with the gentlest predicted impact (speed difference at the first conflict, 1 m/s bins), then
        the latest first conflict, then the earlier candidate."""
        plans, best = self._mates(snap, plans, check), None
        for n, (lane, speed, cap) in enumerate(candidates):
            plan = lane_keep_plan(snap, self.road, self.plant, i, lane, speed, self.cfg, cap, source)
            hit = self._conflict(snap, i, {**plans, i: plan}, list(check))
            key = (True, 0.0, 0, -n) if hit is None else \
                (False, -round(self._impact_speed(snap, plan, hit, plans)), hit[1], -n)
            if best is None or key > best[0]:
                best = (key, (lane, speed, cap), plan)
            if hit is None:
                break
        return best[1], best[2]

    def _slot_vehicles(self, snap, i):
        """Gap cars of car i's lane change plus vehicles merging into a lane it enters: keys."""
        it, ints = self.intentions.intentions[i], self.intentions.intentions
        lanes = set(self.intentions.entered(it.origin, it.lane))
        keys = {k for k in (it.front, it.rear) if k is not None}
        current = self.road.lane_of(snap.traffic_d)
        target = snap.traffic_target_lane if snap.traffic_target_lane is not None else current
        keys |= {('traffic', int(k)) for k, c, g in zip(snap.traffic_id, current, target) if c != g and int(g) in lanes}
        keys |= {('team', j) for j, jt in enumerate(ints)
                 if j != i and jt.phase != 'keep' and lanes & set(self.intentions.entered(jt.origin, jt.lane))}
        return keys

    def _slot_closing(self, snap, i, plan, plans):
        """Least time to contact (s) between car i's plan and the slot vehicles while the lane change executes (from
        the footprint first reaching a lane it enters until it is fully inside the target lane), where the two would
        overlap sideways (gap below slot_lateral); each slot vehicle at constant speed. inf when nothing closes."""
        cfg, it = self.cfg, self.intentions.intentions[i]
        if not cfg.slot_check or it.phase == 'keep':
            return np.inf
        keys = self._slot_vehicles(snap, i)
        if not keys:
            return np.inf
        sub, N = cfg.check_substeps, cfg.horizon
        t = dense(np.arange(N+1)*cfg.step, sub)
        es, ed, ev = (dense(plan.z[:, c], sub) for c in (0, 1, 3))
        lw, W = self.road.cfg.lane_width, self.params.width
        entering = np.zeros(len(t), bool)
        for lane in self.intentions.entered(it.origin, it.lane):
            entering |= np.abs(ed-self.road.lane_center(lane)) < (lw+W)/2
        inside = np.abs(ed-self.road.lane_center(it.lane)) < (lw-W)/2
        executing = entering & (np.cumsum(inside) == 0)   # until first fully inside the target lane
        if not executing.any():
            return np.inf
        _, d_pred, _, _ = traffic_prediction(snap, self.road, np.arange(N+1)*cfg.step)
        L = self.params.length
        worst = np.inf
        for kind, j in keys:
            if kind == 'traffic':
                k = np.flatnonzero(snap.traffic_id == j)
                if not k.size:
                    continue
                s0, v0, do = snap.traffic_s[k[0]], snap.traffic_v[k[0]], dense(d_pred[:, k[0]], sub)
            else:
                mate = plans.get(j) or (self.plans[j].shifted(1) if self.plans[j] is not None else None)
                s0, v0 = snap.team_s[j], snap.team[j, 3]
                do = dense(mate.z[:, 1], sub) if mate is not None else np.full_like(t, snap.team_d[j])
            shared = executing & (np.abs(ed-do) < W+cfg.slot_lateral)
            if not shared.any():
                continue
            ds = s0+v0*t-es
            gap = np.abs(ds)-L
            closing = np.where(ds > 0, ev-v0, v0-ev)
            margin = np.where(gap <= 0, 0.0, np.where(closing > 1e-6, gap/np.maximum(closing, 1e-6), np.inf))
            worst = min(worst, float(margin[shared].min()))
        return worst

    def _lane_keep(self, snap, i, mode, plans, check):
        """Lane-keep fallback in the clearest nearby lane. 'emergency' also tries full braking first, then
        holding the current speed, in each lane."""
        it, v = self.intentions.intentions[i], float(snap.team[i, 3])
        lanes = self._lanes(snap, i)
        if mode in ('emergency', 'escape'):
            brake = -min(self.cfg.emergency_decel, self.brake)
            candidates = [(l, it.speed, brake) for l in lanes]+[(l, v, np.inf) for l in lanes] + \
                [(l, v+self.cfg.escape_accel, np.inf) for l in lanes]
        else:
            candidates = [(l, it.speed, np.inf) for l in lanes]
        (lane, speed, cap), plan = self._rollout(snap, i, candidates, plans, check, mode)
        self._keep_lane(snap, i, lane, mode)
        self.mode[i], self.escape[i] = mode, (lane, speed, cap)
        self.log.fallbacks.append((snap.time, i, mode))
        return plan

    def _emergency(self, snap, i, plans, check):
        """MPC emergency escape for car i: candidate manoeuvres (brake / hold / gain speed in its lane or one either
        side), each the reference of an emergency solve; the first whose plan passes the footprint check, else the
        one with the least contact (largest obstacle slack), then the latest first conflict. No solve succeeds: the
        lane-keep law escape ('escape')."""
        it, v = self.intentions.intentions[i], float(snap.team[i, 3])
        here = int(self.road.lane_of(snap.team_d[i]))
        lanes = [l for l in (here, here-1, here+1) if 0 <= l < self.road.cfg.lanes]
        brake = -min(self.cfg.emergency_decel, self.brake)
        # (lane, MPC target speed, law speed, law accel cap): brake hard, hold the speed, gain speed (threat behind).
        candidates = [(l, 0.0, it.speed, brake) for l in lanes]+[(l, v, v, np.inf) for l in lanes] + \
            [(here, v+self.cfg.escape_accel, v+self.cfg.escape_accel, np.inf)]
        obstacles = self._obstacles(snap, i, plans)
        mates, best = self._mates(snap, plans, check), None
        for n, (lane, speed, law_speed, cap) in enumerate(candidates):
            reference = lane_keep_plan(snap, self.road, self.plant, i, lane, law_speed, self.cfg, cap, 'emergency')
            plan = self._solve(snap, i, obstacles, reference, 'emergency', Target(self.road.lane_center(lane), speed),
                               emergency=True)
            if plan.status != 'solved':
                continue
            plan = replace(plan, source='emergency')
            hit = self._conflict(snap, i, {**mates, i: plan}, list(check))
            key = (hit is None, -round(self.mpc.last_slack, 2), 0 if hit is None else hit[1], -n)
            if best is None or key > best[0]:
                best = (key, (lane, law_speed, cap), plan)
            if hit is None:
                break
        if best is None:
            return self._lane_keep(snap, i, 'escape', plans, check)
        (lane, law_speed, cap), plan = best[1], best[2]
        self._keep_lane(snap, i, lane, 'emergency')
        self.mode[i], self.escape[i] = 'emergency', (lane, law_speed, cap)
        self.log.fallbacks.append((snap.time, i, 'emergency'))
        return plan

    def _keep_lane(self, snap, i, lane, reason):
        it = self.intentions.intentions[i]
        if it.phase != 'keep':
            self.intentions.cancel(i, snap.time, reason, lane)
        elif it.lane != lane:
            self.intentions._keep(i, lane)

    def _plan_vehicle(self, snap, i, fresh):
        """QP, then shifted previous plan, then lane-keep rollout."""
        cfg, N = self.cfg, self.cfg.horizon
        obstacles = self._obstacles(snap, i, fresh)
        previous = self.plans[i]
        reference = previous.shifted(1) if previous is not None else \
            constant_speed_plan(snap.time, cfg.step, N, snap.team_s[i], snap.team_d[i], snap.team[i, 3])
        it = self.intentions.intentions[i]
        plan = self._solve(snap, i, obstacles, reference, 'first')
        if plan.status != 'solved' and it.fresh:
            self.intentions.reject_fresh(i, snap.time, f'mpc_{plan.status}')
            plan = self._solve(snap, i, obstacles, reference, 'retry_keep')
        it.fresh = False
        if plan.status == 'solved':
            self.mode[i] = 'qp'
            return plan
        if plan.status == 'grip' and self.emergency_mpc:   # no plan within grip: plan the escape (verified too)
            return self._emergency(snap, i, fresh, [j for j in range(TEAM) if j != i])
        plan = shifted_fallback(previous, obstacles, cfg, self.params)
        if plan is not None:
            self.mode[i] = 'shifted'
            self.log.fallbacks.append((snap.time, i, 'shifted'))
            return plan
        others = [j for j in range(TEAM) if j != i]
        return self._lane_keep(snap, i, 'lane_keep', fresh, others)

    def _conflict(self, snap, i, plans, higher):
        """First conflict of vehicle i's plan with predicted traffic or higher-priority teammates."""
        cfg, N = self.cfg, self.cfg.horizon
        s, d, _, psi = traffic_prediction(snap, self.road, np.arange(N+1)*cfg.step)
        others = [(('traffic', int(k)), s[:, j], d[:, j], psi[:, j]) for j, k in enumerate(snap.traffic_id)]
        others += [(('team', j), plans[j].z[:, 0], plans[j].z[:, 1], plans[j].z[:, 2]) for j in higher]
        return first_conflict(plans[i].z, others, self.params, cfg.check_margin, substeps=cfg.check_substeps)

    def _replan(self, snap):
        fresh, order = {}, self.order(snap)
        for n, i in enumerate(order):
            fresh[i] = self._plan_vehicle(snap, i, fresh)
            it = self.intentions.intentions[i]
            if it.phase != 'keep' and self._slot_closing(snap, i, fresh[i], fresh) < self.cfg.slot_margin:
                # The target slot is closing: back out now for the clearest of the CG lane and the origin lane.
                lanes = list(dict.fromkeys((int(self.road.lane_of(snap.team_d[i])), it.origin)))
                (lane, _, _), _ = self._rollout(snap, i, [(l, it.speed, np.inf) for l in lanes], fresh, order[:n],
                                                'lane_keep')
                self.intentions.cancel(i, snap.time, 'slot_margin', lane)
                self.plans[i] = None   # no shifted fallback onto the abandoned lane change
                fresh[i] = self._plan_vehicle(snap, i, fresh)
        for n, i in enumerate(order):
            hit = self._conflict(snap, i, fresh, order[:n])
            if hit is None:
                continue
            # Yield: abandon any lane change for the nearby lane (CG lane first: still reversible there) whose
            # lane-keep rollout stays clear longest, and re-plan against everyone's current plans.
            it = self.intentions.intentions[i]
            if it.phase != 'keep':
                lanes = [l for l in self._lanes(snap, i, (int(self.road.lane_of(snap.team_d[i])),)) if l != it.lane]                     or [it.origin]
                (lane, _, _), _ = self._rollout(snap, i, [(l, it.speed, np.inf) for l in lanes], fresh, order[:n],
                                                'lane_keep')
                self.intentions.cancel(i, snap.time, 'conflict', lane)
            self.plans[i] = None   # no shifted fallback onto the conflicting plan
            fresh[i] = self._plan_vehicle(snap, i, fresh)
            resolution = 'replanned'
            if self._conflict(snap, i, fresh, order[:n]) is not None:
                fresh[i] = self._emergency(snap, i, fresh, order[:n]) if self.emergency_mpc else \
                    self._lane_keep(snap, i, 'emergency', fresh, order[:n])
                resolution = 'emergency' if self._conflict(snap, i, fresh, order[:n]) is None else 'unresolved'
            self.log.conflicts.append((snap.time, i, hit[0], hit[1], resolution))
        for i in order:
            if self._plan_driven(i):
                self.held[i] = fresh[i].u[0]
                self.log.predictions.append((snap.time+self.cfg.step, i, *fresh[i].z[1, :2]))
        self.plans = [fresh[i] for i in range(TEAM)]
        self.log.plans.extend((snap.time, i, fresh[i].source, fresh[i].z.copy()) for i in range(TEAM))
