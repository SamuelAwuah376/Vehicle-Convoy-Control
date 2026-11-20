"""Tier A radar: a detection-level model of an automotive radar (design section 5).

Candidates are every traffic car and the carrying car's two teammates. A car is in the field of view when the point of
its box nearest the sensor (geometry.nearest_box_point) lies within range_max and within fov/2 of the boresight up to
near_range, fov_far/2 beyond it. It is then detected with probability

    pd = pd_range(r) * (occluded_pd + (1 - occluded_pd) * v)

where r is the range of that nearest point and v the visible fraction of the car's facing outline seen from the sensor
(geometry.visible_fraction, the own box skipped): radar reaches part of the way past an occluder (multipath under and
around it), so a fully hidden car keeps occluded_pd.

pd_range comes from the radar equation. The received SNR is proportional to rcs / r^4, so in dB
SNR(r) = rcs_dBsm - 40 log10(r) + K, and the detection curve is logistic in it, pd = 1 / (1 + exp(-(SNR - SNR_50) / w)).
Calibrating SNR_50 = SNR(range_max) (pd 0.5 at the range limit) and w so that pd(ref_range) = pd_ref gives

    pd_range(r) = min(pd_ref, 1 / (1 + (r_e / range_max)^n)),  n = ln(pd_ref / (1 - pd_ref)) / ln(range_max / ref_range)

with r_e = r 10^((spec.rcs_dbsm - rcs) / 40), the range at which a target of the spec's cross-section returns the same
power (every car has the spec's cross-section, so r_e = r). The min holds pd at pd_ref inside ref_range, where the
logistic would exceed it; pd_range is 0 beyond range_max, and flat at pd_ref when the calibration is impossible
(ref_range >= range_max, or pd_ref outside (0.5, 1)).

A detected car returns 1..points_max points (uniform): the nearest point, plus points along the outline facing the
sensor at arc-length offsets N(0, SPREAD) from it (reflected at the ends of the facing outline; dropped when outside
the field of view). Each point's range rate is the line-of-sight component of the car's CG velocity minus the sensor's
velocity (negative when closing). Clutter is Poisson(false_alarms) per scan, uniform in range (CLUTTER_RANGE_MIN to
range_max) and azimuth over the field of view, i.e. the same rate in every resolution cell: CLUTTER_STATIONARY of it on
fixed world points (the range rate of a stationary object seen from the moving sensor), the rest moving at a uniform
range rate in CLUTTER_RANGE_RATE. Points of one scan closer than res_range in range and res_azimuth in azimuth are
unresolved and merge (connected groups) into one at their mean; then every point gets Gaussian range, azimuth and
range-rate noise. A scan lists its points by measured range.

Simplifications: the scan is 2D (ground range, the mount height ignored); reflection points ignore partial occlusion of
the car (the visible fraction only sets whether it is detected); a target point moves with the car's CG velocity (no
yaw-rate term), as the tracker's measurement model assumes; no snr is reported.

Speed: every numpy call costs microseconds, so the model is vectorized over sensors as well as cars. simulate_radars
runs all of a step's scans (four radars on three cars) in one pass, several times faster than a loop over
simulate_radar.

With numba on (perception/_jit.py) the pass runs compiled (_scans_compiled: the kernels of _radar_sim_jit between
the random draws, which stay here with the same calls in the same order, so the same scans to rounding); the numpy
code (_scans_numpy) is its reference and the path with NOHESI_JIT=0.
"""
from math import cos, sin
import numpy as np
from .. import _jit
from ..geometry import car_boxes, mount_offset, wrap
from ..types import RadarScan
from . import _radar_sim_jit as _K

__all__ = ['simulate_radar', 'simulate_radars', 'detection_probability', 'sensor_velocity', 'nearest_points',
           'visible_fractions', 'SPREAD', 'CLUTTER_RANGE_MIN', 'CLUTTER_STATIONARY', 'CLUTTER_RANGE_RATE']

SPREAD = 1.0                         # m: sigma of an extra reflection point's offset from the nearest point
CLUTTER_RANGE_MIN = 5.0
CLUTTER_STATIONARY = 0.7             # share of clutter on fixed world points (guardrails, signs, the road)
CLUTTER_RANGE_RATE = (-40.0, 10.0)   # m/s: uniform range rate of the moving clutter
VIS_SAMPLES = 9                      # geometry.visible_fraction's default samples per edge
NEXT = [1, 2, 3, 0]                  # corner k -> k + 1: edge k of a box runs corner k -> NEXT[k] (counter-clockwise)
_SAMPLES = np.linspace(0, 1, VIS_SAMPLES, endpoint=False)    # _visibility's sample fractions along an edge


def _angle_gap(a):
    """|a| wrapped to [0, pi], for a in (-2 pi, 2 pi): the angle between two directions in [-pi, pi) (no modulo,
    which costs ten times more)."""
    m = np.abs(a)
    return np.minimum(m, 2*np.pi-m)


def _pd_range(pd_ref, ref_range, range_max, r):
    p = np.clip(pd_ref, 0.0, 1.0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n = np.log(p/(1-p))/np.log(range_max/ref_range)
        curve = np.minimum(p, 1/(1+(np.maximum(r, 0)/range_max)**n))
    return np.where((p > 0.5) & (p < 1) & (ref_range > 0) & (ref_range < range_max), curve, p)


def detection_probability(spec, r, rcs_dbsm=None):
    """pd_range of a fully visible target at range(s) r (module docstring); rcs_dbsm defaults to spec.rcs_dbsm."""
    r = np.asarray(r, float)
    r_e = r if rcs_dbsm is None else r*10**((spec.rcs_dbsm-rcs_dbsm)/40)
    return np.where(r <= spec.range_max, _pd_range(spec.pd_ref, spec.ref_range, spec.range_max, r_e), 0.0)


def sensor_velocity(mount, yaw, vx, vy, r, params):
    """World velocity (2,) of a sensor mounted on a car at heading yaw with body CG velocity (vx, vy) and yaw rate r:
    the CG velocity plus r x the mount's lever arm (geometry.mount_offset), rotated to the world."""
    ox, oy = mount_offset(mount, params)
    bx, by = vx-r*oy, vy+r*ox
    c, s = cos(yaw), sin(yaw)
    return np.array([c*bx-s*by, s*bx+c*by])


def _nearest(rel, e):
    """Nearest outline points of boxes from sensors: rel (..., 4, 2) the corners relative to the sensor, e (..., 4, 2)
    the edges. Returns the points relative to the sensor (..., 2) and their distances (...)."""
    t = np.clip(-(rel*e).sum(-1)/(e*e).sum(-1), 0, 1)
    q = rel+t[..., None]*e
    d = np.hypot(q[..., 0], q[..., 1])
    k = d.argmin(-1)[..., None]
    return np.take_along_axis(q, k[..., None], -2)[..., 0, :], np.take_along_axis(d, k, -1)[..., 0]


def nearest_points(origin, boxes):
    """geometry.nearest_box_point for every box (M, 4, 2) at once: points (M, 2) and their distances (M,)."""
    o, B = np.asarray(origin, float), np.asarray(boxes, float).reshape(-1, 4, 2)
    q, d = _nearest(B-o, B[:, NEXT]-B)
    return q+o, d


def visible_fractions(origin, boxes, targets, samples=VIS_SAMPLES, skip=()):
    """geometry.visible_fraction(origin, boxes, target, samples, skip) of every target at once (the same samples and
    ray test, so the same result)."""
    o, B = np.asarray(origin, float), np.asarray(boxes, float)
    targets = np.asarray(targets, int).reshape(-1)
    rel, e = (B-o)[None], B[:, NEXT]-B
    skipped = np.zeros((1, len(B)), bool)
    skipped[0, np.asarray(skip, int)] = True
    return _visibility(o[None], B, rel, e, _nearest(rel, e)[1], np.zeros(len(targets), int), targets, skipped,
                       samples)


def _visibility(O, B, rel, e, reach, si, ti, skipped, samples=VIS_SAMPLES):
    """Visible fraction of box ti[c] seen from sensor si[c], as geometry.visible_fraction: O (S, 2) the sensors, B
    (M, 4, 2) the boxes, rel (S, M, 4, 2) their corners from each sensor, e (M, 4, 2) their edges, reach (S, M) their
    nearest distances, skipped (S, M) the boxes each sensor ignores.

    A box can only block a sample whose bearing lies in the box's bearing span and that lies beyond the box's nearest
    point. A box entirely nearer than all of a target's samples blocks exactly its bearing span; the other boxes, and
    samples within 1e-9 rad of a span edge (grazing rays), get geometry.raycast's ray test."""
    C = len(si)
    if C == 0:
        return np.zeros(0)
    tb, oc = B[ti], O[si]
    f = np.linspace(0, 1, samples, endpoint=False)
    pts = (tb[:, :, None]+f[:, None]*e[ti][:, :, None]).reshape(C, -1, 2)
    dist = np.linalg.norm(pts-oc[:, None], axis=-1)
    near = dist <= np.linalg.norm(tb.mean(1)-oc, axis=-1)[:, None]
    ang = np.arctan2(pts[..., 1]-oc[:, 1:], pts[..., 0]-oc[:, :1])
    cx, cy = rel[..., 0].sum(-1, keepdims=True), rel[..., 1].sum(-1, keepdims=True)   # towards each box centre
    side = np.arctan2(cx*rel[..., 1]-cy*rel[..., 0], cx*rel[..., 0]+cy*rel[..., 1])     # corner bearings about it
    cx, cy = cx[..., 0], cy[..., 0]
    lo, hi = side.min(-1), side.max(-1)
    mid, half = wrap(np.arctan2(cy, cx)+(lo+hi)/2), (hi-lo)/2
    inside = (e[..., 0]*rel[..., 1]-e[..., 1]*rel[..., 0] < 0).all(-1)
    half = np.where(inside, np.pi, half)
    far = np.hypot(rel[..., 0], rel[..., 1]).max(-1)
    c = np.arange(C)
    mid_c, half_c = mid[si], half[si]
    pair = ((_angle_gap(mid_c-mid_c[c, ti][:, None]) <= half_c+half_c[c, ti][:, None]+1e-9)
            & (reach[si] < np.where(near, dist, 0).max(1)[:, None]) & ~skipped[si])
    pair[c, ti] = False
    ii, bb = np.nonzero(pair)
    count = near.sum(1)
    if len(ii) == 0:
        return np.where(count > 0, 1.0, 0.0)
    gap = _angle_gap(ang[ii]-mid_c[ii, bb][:, None])-half_c[ii, bb][:, None]
    hit = gap <= 0
    rays = (far[si[ii], bb] >= np.where(near, dist, np.inf).min(1)[ii]-1e-6) | (np.abs(gap) <= 1e-9).any(1)
    if rays.any():
        q = np.flatnonzero(rays)
        k, b = ii[q], bb[q]
        ea, w = e[b], B[b]-oc[k][:, None]
        d = np.stack((np.cos(ang[k]), np.sin(ang[k])), -1)
        denom = d[:, :, None, 0]*ea[:, None, :, 1]-d[:, :, None, 1]*ea[:, None, :, 0]
        with np.errstate(divide='ignore', invalid='ignore'):
            tr = (w[:, None, :, 0]*ea[:, None, :, 1]-w[:, None, :, 1]*ea[:, None, :, 0])/denom
            ue = (w[:, None, :, 0]*d[:, :, None, 1]-w[:, None, :, 1]*d[:, :, None, 0])/denom
        hit[q] = ((np.abs(denom) > 1e-12) & (tr > 0) & (ue >= 0) & (ue <= 1) & (tr < dist[k][:, :, None]-1e-6)).any(2)
    start = np.flatnonzero(np.r_[True, ii[1:] != ii[:-1]])
    blocked = np.zeros(near.shape, bool)
    blocked[ii[start]] = np.logical_or.reduceat(hit, start, axis=0)
    return np.where(count > 0, (near & ~blocked).sum(1)/np.maximum(count, 1), 0.0)


def _reflection_points(O, centre, heading, half, offsets):
    """(D, 1 + m, 2) world points on the outline of car boxes (body centre (D, 2), heading (D,), half size (hx, hy))
    facing sensors O (D, 2): the nearest point, then one at each arc-length offset (D, m) from it along the facing
    outline, reflected at its ends. In the box frame the facing outline is the end face (|x| > hx) and/or the long side
    (|y| > hy) through the corner (sx hx, sy hy) nearest the sensor; arc runs from -2 hy along the end face, through
    0 at that corner, to 2 hx along the side."""
    hx, hy = half
    c, s = np.cos(heading), np.sin(heading)
    dx, dy = O[:, 0]-centre[:, 0], O[:, 1]-centre[:, 1]
    lx, ly = c*dx+s*dy, c*dy-s*dx
    end, side = np.abs(lx) > hx, np.abs(ly) > hy
    lo, hi = np.where(end, -2*hy, 0.0)[:, None], np.where(side, 2*hx, 0.0)[:, None]
    q = np.where(end, np.where(side, 0.0, np.abs(ly)-hy), hx-np.abs(lx))[:, None]
    arc = np.concatenate((q, q+offsets), axis=1)
    arc = np.clip(hi-np.abs(hi-lo-np.abs(arc-lo)), lo, hi)
    px = np.where(arc < 0, hx, hx-arc)*np.where(lx < 0, -1.0, 1.0)[:, None]
    py = np.where(arc < 0, hy+arc, hy)*np.where(ly < 0, -1.0, 1.0)[:, None]
    c, s = c[:, None], s[:, None]
    return np.stack((centre[:, :1]+c*px-s*py, centre[:, 1:]+s*px+c*py), -1)


def _merge(r, az, rr, tid, sensor, res_range, res_azimuth):
    """Merge the connected groups of one sensor's points closer than res_range in range and res_azimuth in azimuth
    (per point, its sensor's) into one at their mean. A group keeps the truth id of its nearest car point (clutter,
    -1, only if it has none)."""
    n = len(r)
    if n < 2:
        return r, az, rr, tid, sensor
    adj = ((sensor[:, None] == sensor[None]) & (np.abs(r[:, None]-r[None]) < res_range[:, None])
           & (_angle_gap(az[:, None]-az[None]) < res_azimuth[:, None]))
    np.fill_diagonal(adj, True)
    if adj.sum() == n:
        return r, az, rr, tid, sensor
    lab = np.arange(n)
    while True:
        new = np.where(adj, lab[None], n).min(1)
        if np.array_equal(new, lab):
            break
        lab = new
    _, lab, count = np.unique(lab, return_inverse=True, return_counts=True)
    order = np.lexsort((r, tid == -1))
    _, first = np.unique(lab[order], return_index=True)
    rep = order[first]
    mean = lambda x: np.bincount(lab, x)/count
    return mean(r), wrap(az[rep]+mean(wrap(az-az[rep][lab]))), mean(rr), tid[rep], sensor[rep]


def _scene(truth, t, params):
    """What every scan at time t shares: the true state; every car's (traffic, then team) box, body centre, heading,
    CG velocity in the world and truth id; and the team's body forward speeds."""
    state = truth.at(t)
    team, vy = state.team, state.team_vy
    vx = np.sqrt(np.maximum(team[:, 3]**2-vy**2, 0.0))      # team v is the CG speed (DynamicPlant)
    c, s = np.cos(team[:, 2]), np.sin(team[:, 2])
    states = np.vstack((state.traffic[:, :3], team[:, :3]))
    boxes = car_boxes(states, params)
    vel = np.vstack((state.traffic[:, 3:5], np.column_stack((c*vx-s*vy, s*vx+c*vy))))
    ids = np.concatenate((np.asarray(state.traffic_id, int), -2-np.arange(len(team))))
    return state, boxes, boxes.mean(1), states[:, 2], vel, ids, vx


def _scans(jobs, scene, t, params, rng):
    """The scans of jobs [(spec, car), ...]: compiled (_radar_sim_jit) when numba is on, else in numpy."""
    return (_scans_compiled if _jit.ENABLED else _scans_numpy)(jobs, scene, t, params, rng)


_TABLES = {}        # (ids of the specs, id of params) -> ((specs, params), kept alive so the ids stay theirs; table)


def _spec_table(specs, params):
    """What _scans_compiled needs of the specs, cached per spec sequence (specs and params are frozen): the TAB rows
    (S, 15), the point counts, the clutter rates and whether a scan's clutter area is empty (as _scans_numpy)."""
    key = tuple(map(id, specs))+(id(params),)
    hit = _TABLES.get(key)
    if hit is not None:
        return hit[1]
    rows = []
    for spec in specs:
        ox, oy = mount_offset(spec.mount, params)
        rows.append([float(getattr(spec, name)) for name in _K.TAB_FIELDS[:12]]+[ox, oy, spec.mount.yaw])
    tab = np.array(rows, float)
    col = lambda name: np.array([getattr(spec, name) for spec in specs], float)
    counts = np.maximum(col('points_max').astype(int), 1)
    rmax, near_r, fov, fov_far = col('range_max'), col('near_range'), col('fov'), col('fov_far')
    r0, near_c = CLUTTER_RANGE_MIN, np.minimum(near_r, rmax)
    empty = np.maximum(near_c-r0, 0.0)*fov+np.maximum(rmax-np.maximum(near_r, r0), 0.0)*fov_far <= 0
    out = (tab, counts, int(counts.max()), np.maximum(col('false_alarms'), 0.0), empty)
    if len(_TABLES) > 256:
        _TABLES.clear()
    _TABLES[key] = ((tuple(specs), params), out)
    return out


def _arg(x, dtype):
    """x as a writable C-contiguous array of dtype, a kernel argument: numba compiles a separate specialization for
    each layout and for read-only arrays (TruthHistory.at may return cached read-only ones), so they are copied."""
    x = np.ascontiguousarray(x, dtype)
    return x if x.flags.writeable else x.copy()


def _scans_compiled(jobs, scene, t, params, rng):
    """_scans_numpy by the kernels of _radar_sim_jit, the random draws the same calls in the same order (the same
    stream, so the same scans to rounding)."""
    state, boxes, centre, heading, vel, ids, vx = scene
    specs = [spec for spec, _ in jobs]
    cars = np.array([car for _, car in jobs], np.int64)
    S = len(jobs)
    tab, counts, m, lam, empty = _spec_table(specs, params)
    f64 = lambda x: _arg(x, np.float64)
    O, yaw, VS, si, ti, pd = _K.view_pd(f64(state.team), f64(state.team_vy), f64(state.team_r), f64(vx), cars,
                                        len(state.traffic), tab, f64(boxes), _SAMPLES)
    det = rng.random(len(si)) < pd
    ds, dt = si[det], ti[det]
    n_pts = rng.integers(1, counts[ds]+1)
    offsets = rng.normal(0.0, SPREAD, size=(len(ds), m-1))
    nc = rng.poisson(lam)
    nc[empty] = 0
    cs = np.repeat(np.arange(S), nc)
    u = rng.random((4, len(cs)))
    uni = rng.uniform(*CLUTTER_RANGE_RATE, size=len(cs))
    r, az, rr, tid, sensor = _K.points_merge(O, yaw, VS, tab, ds, dt, f64(centre), f64(heading), f64(vel),
                                             _arg(ids, np.int64), float(params.length/2), float(params.width/2),
                                             _arg(n_pts, np.int64), offsets, cs, u, uni)
    z = rng.standard_normal((3, len(r)))
    r, az, rr, tid, edge = _K.noise_sort(r, az, rr, tid, sensor, z, tab, S)
    return [RadarScan(spec.name, int(car), float(t), r[a:b], az[a:b], rr[a:b], truth_id=tid[a:b])
            for spec, car, a, b in zip(specs, cars, edge[:-1], edge[1:])]


def _scans_numpy(jobs, scene, t, params, rng):
    """The scans of jobs [(spec, car), ...], vectorized over the sensors: the reference for _scans_compiled."""
    state, boxes, centre, heading, vel, ids, vx = scene
    specs = [spec for spec, _ in jobs]
    cars = np.array([car for _, car in jobs], int)
    S = len(jobs)
    col = lambda name: np.array([getattr(spec, name) for spec in specs], float)
    rmax, near_r, fov, fov_far = col('range_max'), col('near_range'), col('fov'), col('fov_far')
    half_fov = lambda k, r: np.where(r <= near_r[k], fov[k], fov_far[k])/2
    team = state.team
    off = np.array([mount_offset(spec.mount, params) for spec in specs], float)
    c, s = np.cos(team[cars, 2]), np.sin(team[cars, 2])
    O = np.column_stack((team[cars, 0]+c*off[:, 0]-s*off[:, 1], team[cars, 1]+s*off[:, 0]+c*off[:, 1]))
    yaw = wrap(team[cars, 2]+np.array([spec.mount.yaw for spec in specs]))
    yr = state.team_r[cars]
    bx, by = vx[cars]-yr*off[:, 1], state.team_vy[cars]+yr*off[:, 0]
    VS = np.column_stack((c*bx-s*by, s*bx+c*by))
    own = np.zeros((S, len(boxes)), bool)
    own[np.arange(S), len(state.traffic)+cars] = True

    e = boxes[:, NEXT]-boxes
    rel = boxes[None]-O[:, None, None]
    q, reach = _nearest(rel, e)
    az = wrap(np.arctan2(q[..., 1], q[..., 0])-yaw[:, None])
    S_ = np.arange(S)[:, None]
    view = (reach <= rmax[:, None]) & (np.abs(az) <= half_fov(S_, reach)) & ~own
    si, ti = np.nonzero(view)
    pd = _pd_range(col('pd_ref')[si], col('ref_range')[si], rmax[si], reach[si, ti])
    occ = col('occluded_pd')[si]
    hidden = np.flatnonzero(occ < 1)
    if len(hidden):
        v = _visibility(O, boxes, rel, e, reach, si[hidden], ti[hidden], own)
        pd[hidden] *= occ[hidden]+(1-occ[hidden])*v
    det = rng.random(len(si)) < pd
    ds, dt = si[det], ti[det]

    counts = np.maximum(col('points_max').astype(int), 1)
    m = int(counts.max())
    n_pts = rng.integers(1, counts[ds]+1)
    pts = _reflection_points(O[ds], centre[dt], heading[dt], (params.length/2, params.width/2),
                             rng.normal(0.0, SPREAD, size=(len(ds), m-1))) - O[ds][:, None]
    pr = np.hypot(pts[..., 0], pts[..., 1])
    paz = wrap(np.arctan2(pts[..., 1], pts[..., 0])-yaw[ds][:, None])
    keep = (np.arange(m) < n_pts[:, None]) & (pr <= rmax[ds][:, None]) & (np.abs(paz) <= half_fov(ds[:, None], pr))
    keep[:, 0] = True
    prr = (pts*(vel[dt]-VS[ds])[:, None]).sum(-1)/np.maximum(pr, 1e-9)

    nc = rng.poisson(np.maximum(col('false_alarms'), 0.0))
    r0, near_c = CLUTTER_RANGE_MIN, np.minimum(near_r, rmax)
    a1 = np.maximum(near_c-r0, 0.0)*fov
    a2 = np.maximum(rmax-np.maximum(near_r, r0), 0.0)*fov_far
    nc[a1+a2 <= 0] = 0
    cs = np.repeat(np.arange(S), nc)
    u = rng.random((4, len(cs)))
    outer = u[0]*(a1+a2)[cs] < a2[cs]
    cr = np.where(outer, np.maximum(near_r, r0)[cs], r0)
    cr = cr+(np.where(outer, rmax[cs], near_c[cs])-cr)*u[1]
    caz = np.where(outer, fov_far[cs], fov[cs])*(u[2]-0.5)
    crr = np.where(u[3] < CLUTTER_STATIONARY, -(np.cos(yaw[cs]+caz)*VS[cs, 0]+np.sin(yaw[cs]+caz)*VS[cs, 1]),
                   rng.uniform(*CLUTTER_RANGE_RATE, size=len(cs)))

    sensor = np.concatenate((np.broadcast_to(ds[:, None], keep.shape)[keep], cs))
    r, az, rr, tid, sensor = _merge(np.concatenate((pr[keep], cr)), np.concatenate((paz[keep], caz)),
                                    np.concatenate((prr[keep], crr)),
                                    np.concatenate((np.broadcast_to(ids[dt][:, None], keep.shape)[keep],
                                                    np.full(len(cs), -1))),
                                    sensor, col('res_range')[sensor], col('res_azimuth')[sensor])
    z = rng.standard_normal((3, len(r)))
    r = np.abs(r+col('sigma_range')[sensor]*z[0])
    az = wrap(az+col('sigma_azimuth')[sensor]*z[1])
    rr = rr+col('sigma_range_rate')[sensor]*z[2]
    order = np.lexsort((r, sensor))
    r, az, rr, tid = r[order], az[order], rr[order], np.asarray(tid, int)[order]
    edge = np.searchsorted(sensor[order], np.arange(S+1))
    return [RadarScan(spec.name, int(car), float(t), r[a:b], az[a:b], rr[a:b], truth_id=tid[a:b])
            for spec, car, a, b in zip(specs, cars, edge[:-1], edge[1:])]


def simulate_radar(spec, car, truth, t, params, rng):
    """One scan of radar `spec` on team car `car` at time t from the true state (types.TruthHistory): detections in
    the sensor frame (range, azimuth, range rate) with truth ids (traffic id, -1 clutter, -2 - j team car j).
    params: the vehicle parameters (every car's size); rng: the perception numpy Generator."""
    return _scans([(spec, car)], _scene(truth, t, params), t, params, rng)[0]


def simulate_radars(specs, cars, truth, t, params, rng):
    """simulate_radar for every car in `cars` and spec in `specs` (car-major order) in one vectorized pass (the same
    model; the random draws are batched, so the numbers differ from a loop over simulate_radar)."""
    jobs = [(spec, car) for car in cars for spec in specs]
    return _scans(jobs, _scene(truth, t, params), t, params, rng) if jobs else []
