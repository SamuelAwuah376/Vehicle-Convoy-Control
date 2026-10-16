"""Fallbacks and plan checks: shifted previous plan, lane-keep rollout, and sampled footprint checks."""
from dataclasses import replace
import numpy as np
from ..geometry import clearance, footprint, overlap
from ..rl.scripted import lane_keep_law
from ..prediction import vehicle_ahead_incl_merging
from .base import Plan


def plan_is_clear(plan, obstacles, params, start=1):
    """Sampled footprint check of a plan against predicted obstacles (touching counts as a hit)."""
    reach = params.length+params.width
    for k in range(start, len(plan.z)):
        s, d, psi, _ = plan.z[k]
        near = np.flatnonzero((np.abs(obstacles.s[k]-s) < reach) & (np.abs(obstacles.d[k]-d) < reach))
        if near.size:
            ego = footprint(np.array([s, d, psi, 0.0]), params)
            if any(overlap(ego, footprint(np.array([obstacles.s[k, j], obstacles.d[k, j], 0.0, 0.0]), params))
                   for j in near):
                return False
    return True


def shifted_fallback(previous, obstacles, cfg, params):
    """Previous plan advanced one step, if not used open loop for too long and still clear of predictions."""
    if previous is None or previous.source != 'qp' and previous.source != 'shifted':
        return None
    candidate = previous.shifted(1)
    if candidate.age > cfg.max_shifted_steps or not plan_is_clear(candidate, obstacles, params):
        return None
    return replace(candidate, source='shifted')


def lane_keep_plan(snap, road, plant, i, lane, speed, cfg, max_accel=np.inf, source='lane_keep'):
    """Roll the lane-keeping law forward on the plant (lead car, including traffic merging into the lane, at
    constant speed) so teammates see where a fallback vehicle will actually go."""
    p, dt = plant.params, plant.params.dt
    sub = int(round(cfg.step/dt))
    route = road.lane_route(lane)
    state, steer = snap.team[i].copy(), snap.team_steer[i]
    lead = vehicle_ahead_incl_merging(snap, road, snap.team_s[i], lane, skip=i)
    s0 = snap.team_s[i]
    z, u = [], []
    for k in range(cfg.horizon+1):
        s = road.to_frenet(state[:2])[0]
        z.append([s, road.to_frenet(state[:2])[1], state[2]-road.heading(s), state[3]])
        if k == cfg.horizon:
            break
        for n in range(sub):
            t = (k*sub+n)*dt
            ahead = None if lead is None else (lead[0]+lead[1]*t-(s-s0), lead[1])
            s = road.to_frenet(state[:2])[0]
            a, delta = lane_keep_law(state, s, route, p, speed, ahead, max_accel)
            state, a, steer = plant.predict_step(state, a, delta, steer)   # kinematic, whatever the plant
            if n == 0:
                u.append([a, steer])
    return Plan(snap.time, cfg.step, np.array(z, float), np.array(u, float), source)


def first_conflict(z, others, params, margin, start=1):
    """First (key, step) where plan z comes within `margin` of another vehicle's footprint, else None.

    others: list of (key, s, d, psi) arrays over the same plan times."""
    if not others:
        return None
    S, D, PSI = (np.array([o[n] for o in others]) for n in (1, 2, 3))
    # Conservative broad phase: a footprint's lateral half-extent about its CG is at most
    # width/2 + length*|sin(psi)|, so cars beyond this lateral distance cannot be within `margin`.
    lateral = params.width+margin+params.length*(np.abs(np.sin(PSI))+np.abs(np.sin(z[:, 2])))
    near = (np.abs(S-z[:, 0]) < params.length+params.width+margin) & (np.abs(D-z[:, 1]) < lateral)
    near[:, :start] = False
    for j, k in sorted(np.argwhere(near), key=lambda jk: (jk[1], jk[0])):
        ego = footprint(np.array([z[k, 0], z[k, 1], z[k, 2], 0.0]), params)
        if clearance(ego, footprint(np.array([S[j, k], D[j, k], PSI[j, k], 0.0]), params)) < margin:
            return others[j][0], int(k)
    return None
