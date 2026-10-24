"""Stability layer: the inner loop between every command source and the dynamic plant.

Any command source posts to it: the MPC with its planned motion (Plan.lateral: body lateral speed vy and yaw rate r
per plan step), the fallbacks (lane keeping, emergency escape) without one. Every plant substep (100 Hz) it carries
out the command last applied within the tyres' grip and tracks the posted motion:
- reference: the posted plan's yaw rate and sideslip at that time; without a plan, the steady-state response of the
  car to the commanded steer (linear bicycle with its understeer gradient), the yaw rate capped at grip mu g / vx;
- steering: the command, for a source without a plan (fallbacks: their steering knows nothing of grip) capped at
  steer_cap times the steady-state steer of the grip-limited turn (more steer than grip can use only over-rotates the
  car; the MPC's plans are grip-limited already and need more in transients), plus minor corrections (within +-max_correction): k_yaw (r_ref - r)
  when the car yaws past the reference in its direction (over-rotation; a car lagging the reference is left alone)
  and k_slip * (sideslip beyond slip_free of the reference, toward the velocity: countersteer); finally capped so the
  front slip stays within steer_margin of its saturation slip (past it more steer adds no force: in a slide this
  points the front wheels along their velocity, the natural countersteer);
- longitudinal, lateral first (ABS and traction control): each axle's Fx at most sqrt((grip mu Fz)^2 - Fy^2) with Fy
  the lateral force the axle needs at that slip, braking never below min_brake mu Fz; the commanded acceleration is
  scaled to the tightest axle.
Steering, throttle and overall braking only: no per-wheel braking. `buffer` holds each car's latest state (and the
substeps of the last sim step), which the planner replans from.
"""
from dataclasses import dataclass, field
import math
import numpy as np
from .dynamics import G, fiala


@dataclass
class StateBuffer:
    """Latest state per car, posted every plant substep: (time, [x, y, yaw, v, vy, r], steer, accel); plus the
    substeps of the current sim step."""
    latest: dict = field(default_factory=dict)
    substeps: dict = field(default_factory=dict)

    def post(self, i, t, state, steer, accel, first=False):
        entry = (t, np.asarray(state, float).copy(), float(steer), float(accel))
        self.latest[i] = entry
        if first:
            self.substeps[i] = []
        self.substeps.setdefault(i, []).append(entry)


class StabilityLayer:
    def __init__(self, plant, cfg):
        self.plant, self.cfg, p = plant, cfg, plant.params
        self.l_f, self.l_r = p.wheelbase-p.cg_to_rear_axle, p.cg_to_rear_axle
        # Understeer gradient (rad per m/s^2 of lateral acceleration) of the linear bicycle, axle stiffnesses.
        self.k_us = p.mass/p.wheelbase*(self.l_r/p.cornering_stiffness_front-self.l_f/p.cornering_stiffness_rear)
        self.plans = {}        # car -> Plan posted with the current command (None: steady-state reference)
        self.buffer = StateBuffer()
        self.interventions = {'steer': 0, 'longitudinal': 0, 'substeps': 0}

    def post(self, i, plan=None):
        """Car i's command source for the next steps: a plan with planned motion (Plan.lateral), or None."""
        self.plans[i] = plan if plan is not None and getattr(plan, 'lateral', None) is not None else None

    def reference(self, i, t, vx, steer_cmd):
        """(yaw rate, sideslip) reference at time t."""
        p, c = self.plant.params, self.cfg
        u = max(vx, 1.0)
        plan = self.plans.get(i)
        if plan is not None:
            k = min(max((t-plan.time)/plan.step, 0.0), len(plan.lateral)-1)
            vy, r = (np.interp(k, np.arange(len(plan.lateral)), plan.lateral[:, j]) for j in (0, 1))
            return float(r), math.atan2(float(vy), u)
        r = u*steer_cmd/(p.wheelbase+self.k_us*u*u)
        r_max = c.stab_grip*p.friction*G/u
        r = min(max(r, -r_max), r_max)
        beta = r/u*(self.l_r-p.mass*self.l_f*u*u/(p.wheelbase*p.cornering_stiffness_rear))
        return r, beta

    def control(self, i, t, state, steer_now, accel_cmd, steer_cmd):
        """(acceleration, steer target) for one substep of car i at time t; state = [x, y, yaw, vx, vy, r]."""
        p, c = self.plant.params, self.cfg
        _, _, _, vx, vy, r = state
        u = max(vx, 1.0)
        self.interventions['substeps'] += 1
        # Steering: no more than the grip-limited steady-state turn needs, then minor corrections against
        # over-rotation and excess sideslip.
        r_ref, beta_ref = self.reference(i, t, vx, steer_cmd)
        steer = steer_cmd
        if self.plans.get(i) is None:
            cap = c.stab_steer_cap*c.stab_grip*p.friction*G/u*(p.wheelbase+self.k_us*u*u)/u
            steer = min(max(steer, -cap), cap)
        over = r-r_ref
        yaw_fix = -c.stab_k_yaw*over if over*(r_ref if abs(r_ref) > 1e-3 else r) > 0 else 0.0
        beta = math.atan2(vy, u)
        excess = beta-beta_ref
        excess = math.copysign(max(abs(excess)-c.stab_slip_free, 0.0), excess)
        correction = yaw_fix+c.stab_k_slip*excess
        steer += min(max(correction, -c.stab_max_correction), c.stab_max_correction)
        # ... within the front tyre's grip (its saturation slip with the longitudinal force of the command).
        fz_f, fz_r, fx_f, fx_r, _ = self.plant.axle_forces(accel_cmd)
        peak_f = math.sqrt(max((p.friction*fz_f)**2-fx_f**2, (0.05*p.friction*fz_f)**2))
        sat = math.atan(3*peak_f/p.cornering_stiffness_front)
        flow = math.atan((vy+self.l_f*r)/u)   # front wheel velocity direction relative to the body
        steer = min(max(steer, flow-c.stab_steer_margin*sat), flow+c.stab_steer_margin*sat)
        if abs(steer-steer_cmd) > 1e-3:
            self.interventions['steer'] += 1
        # Longitudinal: lateral first; each axle gets the friction circle left after the lateral force it needs.
        alpha_f = steer_now-flow
        alpha_r = -math.atan((vy-self.l_r*r)/u)
        fy_f = fiala(alpha_f, p.cornering_stiffness_front, p.friction*fz_f)
        fy_r = fiala(alpha_r, p.cornering_stiffness_rear, p.friction*fz_r)
        braking = accel_cmd < 0
        share_f = p.brake_bias_front if braking else p.drive_bias_front
        a = accel_cmd
        for fz, fy, share in ((fz_f, fy_f, share_f), (fz_r, fy_r, 1-share_f)):
            if share <= 0:
                continue
            room = math.sqrt(max((c.stab_grip*p.friction*fz)**2-fy**2, 0.0))
            if braking:
                room = max(room, c.stab_min_brake*p.friction*fz)
            limit = room/(share*p.mass)
            a = max(a, -limit) if braking else min(a, limit)
        if abs(a-accel_cmd) > 1e-3:
            self.interventions['longitudinal'] += 1
        return a, steer

    def inner(self, i, t0):
        """Callback for DynamicPlant.step: substep n -> (acceleration, steer target); posts the state at the start of
        each substep to the buffer (the world posts the state at the end of the step)."""
        h = self.plant.substep_dt

        def call(n, state, steer_now, accel_cmd, steer_cmd):
            t = t0+n*h
            a, steer = self.control(i, t, state, steer_now, accel_cmd, steer_cmd)
            x, y, yaw, vx, vy, r = state
            self.buffer.post(i, t, [x, y, yaw, math.hypot(vx, vy), vy, r], steer_now, a, first=n == 0)
            return a, steer
        return call
