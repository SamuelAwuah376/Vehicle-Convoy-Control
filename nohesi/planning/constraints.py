"""Solver-independent per-stage linear constraints of the lane-change MPC (the selection logic of LaneChangeMPC).

Row i constrains the road-frame state at horizon step k[i] in 1..N:
    lo[i] <= cs[i]*s + cd[i]*d + cv[i]*v <= hi[i]
with s RELATIVE TO s0 (the ego's current s): the functions subtract s0 from the reference and obstacle s, so pass
absolute inputs with s0 = z0[0], or inputs already shifted with s0 = 0 (the default). Hard rows have
slack_weight 0. A soft row (slack_weight w > 0) has exactly one finite bound, relaxed by its own slack sigma >= 0
(lo - sigma <= row, or row <= hi + sigma) costing w*(sigma^2 + sigma); in acados terms Z = 2w, z = w.
Rows come in the linear MPC's assembly order (hard rows first, then soft rows grouped as their slacks are).

Selection (from the reference zr, per step k and obstacle j): one separating half-plane, `left`/`right` (beside,
d bound), `behind`/`ahead` (s bound), or none when not relevant. Gap cars: "on the gap side" once the reference
reaches the gap, beside before; a gap car past the ego the wrong way is ordinary. Beside when laterally clear to
within lateral_tolerance; along the horizon a side flip or behind<->ahead jump keeps the previous step's choice.
Soft time headway (`headway_behind`/`headway_ahead`) to cars the reference shares a lane with, tau per obstacle kind;
soft `yield` get-clear rows (one per step) toward teammates in target.yield_to. `j` is the obstacle column (-1: none).
"""
from dataclasses import dataclass
import numpy as np

FIELDS = ('k', 'cs', 'cd', 'cv', 'lo', 'hi', 'slack_weight', 'kind', 'j')


@dataclass(frozen=True)
class StageConstraints:
    k: np.ndarray
    cs: np.ndarray
    cd: np.ndarray
    cv: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    slack_weight: np.ndarray
    kind: np.ndarray
    j: np.ndarray

    def __len__(self):
        return len(self.k)

    @property
    def soft(self):
        return self.slack_weight > 0

    def select(self, mask):
        return StageConstraints(*(getattr(self, f)[mask] for f in FIELDS))

    def per_stage(self, N):
        """(hard, soft) row counts per step k = 1..N, each shape (N,)."""
        count = lambda m: np.bincount(self.k[m], minlength=N+1)[1:]
        return count(~self.soft), count(self.soft)

    @staticmethod
    def concat(parts):
        if not parts:
            return StageConstraints(*(np.zeros(0, int if f in ('k', 'j') else object if f == 'kind' else float)
                                      for f in FIELDS))
        return StageConstraints(*(np.concatenate([getattr(p, f) for p in parts]) for f in FIELDS))


def rows(kind, k, lo, hi, cs=0.0, cd=0.0, cv=0.0, weight=0.0, j=-1):
    k = np.asarray(k, int)
    full = lambda x, t=float: np.broadcast_to(np.asarray(x, t), k.shape).copy()
    return StageConstraints(k, full(cs), full(cd), full(cv), full(lo), full(hi), full(weight), full(kind, object),
                            full(j, int))


def state_bounds(cfg, params, road):
    """Road edges on d (CG kept edge_margin plus half a width inside) and v >= 0, steps 1..N, hard."""
    ks, edge = np.arange(1, cfg.horizon+1), road.width/2-params.width/2-cfg.edge_margin
    return StageConstraints.concat([rows('edge', ks, -edge, edge, cd=1.0), rows('speed', ks, 0.0, np.inf, cv=1.0)])


def obstacle_constraints(cfg, params, road, zr, obstacles, target, s0=0.0):
    """Separating, headway and yield rows against `obstacles` (base.Obstacles, (N+1, K)) around the reference
    states zr (N+1, 4). Bounds on s are relative to s0 (see module docstring)."""
    c, p, N, keys = cfg, params, cfg.horizon, obstacles.keys
    if not keys:
        return StageConstraints.concat([])
    long_gap, lat_gap = p.length+c.margin_long, p.width+c.margin_lat
    obs_s = obstacles.s-s0
    os_, od, ov = obs_s[1:], obstacles.d[1:], obstacles.v[1:]
    ds, dd = (zr[1:, :1]-s0)-os_, zr[1:, 1:2]-od
    relevant = (np.abs(ds) <= c.relevance) & (np.abs(dd) <= 2*road.cfg.lane_width)
    front = np.array([key == target.front for key in keys])[None, :]
    rear = np.array([key == target.rear for key in keys])[None, :]
    tau = np.where([key[0] == 'team' for key in keys], c.teammate_headway, c.headway)[None, :]
    lateral_clear = np.abs(dd) >= lat_gap-c.lateral_tolerance
    # A gap car already past the ego the wrong way (rear car ahead, front car behind) is an ordinary obstacle.
    stale = (rear & (ds < -long_gap)) | (front & (ds > long_gap))
    front, rear = front & ~stale, rear & ~stale
    side = np.where(front, ds > -long_gap,
                    np.where(rear, ds < long_gap, lateral_clear & (np.abs(ds) < long_gap+c.side_zone)))
    behind = ~side & (front | (~rear & (ds < 0)))
    # Axis code per (step, obstacle): +1/-1 beside (left/right of it), 2 behind, -2 ahead, 0 not relevant.
    # Along the horizon a side flip or a behind<->ahead jump keeps the previous step's choice.
    code = np.where(side, np.where(dd >= 0, 1, -1), np.where(behind, 2, -2))*relevant
    for k in range(1, N):
        flip = (code[k] == -code[k-1]) & (code[k] != 0)
        code[k, flip] = code[k-1, flip]
    behind, ahead = code == 2, code == -2
    kk = np.broadcast_to(np.arange(1, N+1)[:, None], ds.shape)
    jj = np.broadcast_to(np.arange(len(keys))[None, :], ds.shape)
    full = lambda x: np.broadcast_to(x, ds.shape)
    parts = []
    for kind, mask, coef, lo, hi in (('left', code == 1, 'cd', od+lat_gap, np.inf),
                                     ('right', code == -1, 'cd', -np.inf, od-lat_gap),
                                     ('behind', behind, 'cs', -np.inf, os_-long_gap),
                                     ('ahead', ahead, 'cs', os_+long_gap, np.inf)):
        if mask.any():
            parts.append(rows(kind, kk[mask], full(lo)[mask], full(hi)[mask], j=jj[mask], **{coef: 1.0}))
    # Soft headway only for cars the reference shares a lane with, tau per obstacle:
    # behind: s + tau*v <= obs - gap (+slack);  ahead: s >= obs + gap + tau*v_obs (-slack).
    mask = behind & ~lateral_clear
    if mask.any():
        parts.append(rows('headway_behind', kk[mask], -np.inf, (os_-long_gap)[mask], cs=1.0, cv=full(tau)[mask],
                          weight=c.w_slack, j=jj[mask]))
    mask = ahead & ~lateral_clear
    if mask.any():
        parts.append(rows('headway_ahead', kk[mask], (os_+long_gap+tau*ov)[mask], np.inf, cs=1.0,
                          weight=c.w_slack, j=jj[mask]))
    # Soft yield to teammates changing into the ego's lane, any lateral offset: get clear along the road the way
    # the ego already is (behind or level: drop back; ahead: pull ahead), past the changer's stay-beside zone
    # (side_zone) so its MPC may enter. One row per step.
    ks, clear = np.arange(1, N+1), long_gap+c.side_zone+c.yield_margin
    for o in [keys.index(key) for key in target.yield_to if key in keys]:
        sign = 1.0 if obs_s[0, o] < 0 else -1.0
        bound = os_[:, o]+sign*clear
        parts.append(rows('yield', ks, *((bound, np.inf) if sign > 0 else (-np.inf, bound)), cs=1.0,
                          weight=c.w_yield, j=o))
    return StageConstraints.concat(parts)
