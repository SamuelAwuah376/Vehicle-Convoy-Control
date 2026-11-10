"""Fallbacks and plan checks: shifted previous plan, lane-keep rollout, and sampled footprint checks."""
from dataclasses import replace
import numpy as np
from ..dynamics import brake_limit
from ..geometry import clearances, footprint, footprints, overlaps
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
    zs, zd, zpsi, os_, od = zs[start*substeps:], zd[start*substeps:], zpsi[start*substeps:], \
        os_[start*substeps:], od[start*substeps:]
    kk, jj = np.nonzero((np.abs(os_-zs[:, None]) < reach) & (np.abs(od-zd[:, None]) < reach))
    if not kk.size:
        return True
    zero = np.zeros(len(kk))
    ego = footprints(np.column_stack((zs[kk], zd[kk], zpsi[kk], zero)), params)
    return not overlaps(ego, footprints(np.column_stack((os_[kk, jj], od[kk, jj], zero, zero)), params)).any()


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
    constant speed) so teammates see where a fallback vehicle will actually go. Braking is capped as when it
    runs (dynamics.brake_limit)."""
    p, dt = plant.params, plant.params.dt
    brake = brake_limit(plant, cfg.grip_brake)
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
            a, delta = lane_keep_law(state, s, route, p, speed, ahead, max_accel, brake)
            state, a, steer = plant.predict_step(state, a, delta, steer)   # kinematic, whatever the plant
            if n == 0:
                u.append([a, steer])
    return Plan(snap.time, cfg.step, np.array(z, float), np.array(u, float), source)


def edge_grip_accel(snap, road, params, i, control, brake, zone):
    """Planner.escape_edge_grip: control-law acceleration of car i, braking capped by the friction circle once its
    body corner is within `zone` of the road edge with its CG moving toward that edge: |a| <= sqrt(brake^2 - a_lat^2),
    a_lat = v^2 tan(steer)/wheelbase the lateral demand of the law's steering command. Otherwise unchanged."""
    a, steer = float(control[0]), float(control[1])
    if a >= 0.0:
        return a
    _, corner = road.to_frenet(footprint(snap.team[i], params))
    out = corner[np.argmax(np.abs(corner))]
    if road.width/2-abs(out) > zone:
        return a
    v = float(snap.team[i, 3])
    vy = 0.0 if snap.team_vy is None else float(snap.team_vy[i])
    psi = float(snap.team[i, 2]-road.heading(snap.team_s[i]))
    d_rate = v*np.sin(psi+np.arcsin(np.clip(vy/max(v, 1e-3), -1.0, 1.0)))
    if np.sign(out)*d_rate <= 0.0:
        return a
    lateral = v*v*abs(np.tan(steer))/params.wheelbase
    return max(a, -float(np.sqrt(max(brake*brake-lateral*lateral, 0.0))))


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
    jj, kk = np.nonzero(near)
    if not jj.size:
        return None
    order = np.lexsort((jj, kk))   # by step, then vehicle: the first conflict in time
    jj, kk = jj[order], kk[order]
    zero = np.zeros(len(jj))
    ego = footprints(np.column_stack((z[kk, 0], z[kk, 1], z[kk, 2], zero)), params)
    hit = np.flatnonzero(clearances(ego, footprints(np.column_stack((S[jj, kk], D[jj, kk], PSI[jj, kk], zero)), params))
                         < margin)
    if not hit.size:
        return None
    j, k = jj[hit[0]], kk[hit[0]]
    return others[j][0], int(-(-k//substeps))
