"""Fallbacks and plan checks: shifted previous plan, lane-keep rollout, and sampled footprint checks."""
from dataclasses import replace
import numpy as np
from ..geometry import clearance, footprint, overlap
from ..rl.scripted import lane_keep_law
from ..prediction import vehicle_ahead_incl_merging
from .base import Plan


def dense(x, substeps, axis=-1):
    """Linear interpolation of samples along `axis` at `substeps` points per interval (1: unchanged)."""
    x = np.asarray(x, float)
    if substeps <= 1:
        return x
    x = np.moveaxis(x, axis, -1)
    t = np.arange((x.shape[-1]-1)*substeps+1)/substeps
    k = np.minimum(np.floor(t).astype(int), x.shape[-1]-2)
    f = t-k
    return np.moveaxis(x[..., k]*(1-f)+x[..., k+1]*f, -1, axis)


def plan_is_clear(plan, obstacles, params, start=1, substeps=1):
    """Footprint check of a plan against predicted obstacles at `substeps` points per step (touching counts as a hit)."""
    reach = params.length+params.width
    zs, zd, zpsi = (dense(plan.z[:, c], substeps) for c in range(3))
    os_, od = dense(obstacles.s, substeps, axis=0), dense(obstacles.d, substeps, axis=0)
    for k in range(start*substeps, len(zs)):
        s, d, psi = zs[k], zd[k], zpsi[k]
        near = np.flatnonzero((np.abs(os_[k]-s) < reach) & (np.abs(od[k]-d) < reach))
        if near.size:
            ego = footprint(np.array([s, d, psi, 0.0]), params)
            if any(overlap(ego, footprint(np.array([os_[k, j], od[k, j], 0.0, 0.0]), params)) for j in near):
                return False
    return True


def shifted_fallback(previous, obstacles, cfg, params):
    """Previous plan advanced one step, if not used open loop for too long and still clear of predictions."""
    if previous is None or previous.source != 'qp' and previous.source != 'shifted':
        return None
    candidate = previous.shifted(1)
    if candidate.age > cfg.max_shifted_steps or not plan_is_clear(candidate, obstacles, params,
                                                                   substeps=getattr(cfg, 'check_substeps', 1)):
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


def first_conflict(z, others, params, margin, start=1, substeps=1):
    """First (key, step) where plan z comes within `margin` of another vehicle's footprint, else None.

    others: list of (key, s, d, psi) arrays over the same plan times. With substeps > 1 both trajectories are
    interpolated and checked `substeps` times per step (fast relative motion can clip between nodes); the reported
    step is the plan step at or after the hit."""
    if not others:
        return None
    S, D, PSI = (dense(np.array([o[n] for o in others]), substeps) for n in (1, 2, 3))
    z = dense(z[:, :3], substeps, axis=0)
    start = start*substeps
    # Conservative broad phase: a footprint's lateral half-extent about its CG is at most
    # width/2 + length*|sin(psi)|, so cars beyond this lateral distance cannot be within `margin`.
    lateral = params.width+margin+params.length*(np.abs(np.sin(PSI))+np.abs(np.sin(z[:, 2])))
    near = (np.abs(S-z[:, 0]) < params.length+params.width+margin) & (np.abs(D-z[:, 1]) < lateral)
    near[:, :start] = False
    for j, k in sorted(np.argwhere(near), key=lambda jk: (jk[1], jk[0])):
        ego = footprint(np.array([z[k, 0], z[k, 1], z[k, 2], 0.0]), params)
        if clearance(ego, footprint(np.array([S[j, k], D[j, k], PSI[j, k], 0.0]), params)) < margin:
            return others[j][0], int(-(-k//substeps))
    return None
