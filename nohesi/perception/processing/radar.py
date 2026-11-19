"""Radar processing (design section 6): one scan's point detections -> one 'radar' Detection per car-sized cluster.

1. The sensor pose and velocity from the ego state: the body velocity rotated to the world plus yaw rate x the mount's
   lever arm (geometry.mount_offset), so a turn's swing of a bumper or mirror radar is included.
2. Stationary rejection (the Doppler test): a fixed world point at azimuth a has range rate -(v_sensor . los(a)).
   Detections within STATIONARY_SIGMAS sigma of it are not vehicles (clutter; in CARLA guardrails and signs) and are
   dropped; sigma combines sigma_range_rate with the azimuth noise's effect on that range rate (|v| sigma_azimuth
   across the line of sight: 0.5 m/s for a side radar's 1 deg at 30 m/s, three times its sigma_range_rate). A car
   moving across the line of sight (abeam of a side radar) passes the same test; only its points within a few degrees
   of abeam are lost.
3. Detections within MATE_MARGIN of a teammate's box (its V2V state at the scan time), or within MATE_SIGMAS of
   their own position noise when that is wider, are dropped.
4. DBSCAN (dbscan) on (x, y, range_rate) scaled by SCALE: one car gives 1-3 points on its near outline, the tracker
   assigns one measurement per track, and a slow and a fast car side by side stay apart by their speed difference.
   The range-rate difference of two points is taken net of what one rigid car can show between their lines of sight
   (MAX_REL_SPEED times the angle between them): a car beside a side radar spans tens of degrees, and its points'
   range rates differ by a few m/s although it is one car. At range the allowance vanishes.
5. One Detection per cluster: [range, azimuth, range_rate] of the cluster centre (the mean point in the sensor frame,
   the mean range rate), with R = diag(sigma^2)/n_points (the mean of n independently noisy points) plus the cluster's
   spread about its centre (the points lie along the car's outline, not at one spot, and the tracker predicts the box
   point nearest the sensor).

ego_at and sensor_velocity are shared with camera processing.
"""
from dataclasses import replace
import numpy as np
from ..geometry import car_boxes, mount_offset, rot, sensor_pose, to_world, wrap
from ..types import Detection

__all__ = ['STATIONARY_SIGMAS', 'MATE_MARGIN', 'MATE_SIGMAS', 'SCALE', 'MAX_REL_SPEED', 'ego_at', 'sensor_velocity',
           'stationary_range_rate', 'stationary', 'box_distance', 'cluster_distance', 'dbscan', 'process_radar',
           'process_radars']

STATIONARY_SIGMAS = 3.0
MATE_MARGIN = 1.0                    # m: points this near a teammate's box are its own (at least; see MATE_SIGMAS)
MATE_SIGMAS = 3.0                    # ... or within this many sigma of the point's position noise (range, and range
                                     # x azimuth across): 0.8 m across at 150 m for the front radar, so a fixed 1 m let
                                     # 1-2 % of the scans pass teammate points on as cars
SCALE = (1.5, 1.5, 1.0)              # DBSCAN scales: world x (along the road), y (m), range rate (m/s); eps 1
MAX_REL_SPEED = 15.0                 # m/s: the largest speed of a car relative to the sensor (v9 lanes 55-80 mph)
SIGMA_FLOOR = (1e-3, 1e-5, 1e-3)     # keeps R positive definite for noise-free ('ideal') sensors


def ego_at(ego, t):
    """The EgoState moved to time t at constant body velocity and yaw rate (the SE(2) exponential of the twist);
    ego itself when it is already at t."""
    tau = float(t)-ego.time
    if abs(tau) < 1e-9:
        return ego
    dpsi = ego.r*tau
    if abs(ego.r) < 1e-4:
        dx, dy = ego.vx*tau, ego.vy*tau
    else:
        dx = (ego.vx*np.sin(dpsi)-ego.vy*(1-np.cos(dpsi)))/ego.r
        dy = (ego.vx*(1-np.cos(dpsi))+ego.vy*np.sin(dpsi))/ego.r
    c, s = np.cos(ego.yaw), np.sin(ego.yaw)
    return replace(ego, time=float(t), x=ego.x+c*dx-s*dy, y=ego.y+s*dx+c*dy, yaw=float(wrap(ego.yaw+dpsi)))


def sensor_velocity(mount, ego, params):
    """World velocity (2,) of a sensor at `mount` on the car in EgoState ego: (vx, vy) plus r x the lever arm."""
    ox, oy = mount_offset(mount, params)
    return rot(float(ego.yaw)) @ np.array([ego.vx-ego.r*oy, ego.vy+ego.r*ox])


def stationary_range_rate(azimuth, pose, v_sensor):
    """Range rate (n,) of fixed world points at sensor azimuths (n,) seen from a sensor at pose moving at v_sensor."""
    a = pose[2]+np.asarray(azimuth, float)
    return -(np.cos(a)*v_sensor[0]+np.sin(a)*v_sensor[1])


def stationary(azimuth, range_rate, pose, v_sensor, spec, sigmas=STATIONARY_SIGMAS):
    """(n,) True for detections whose range rate is within `sigmas` sigma of a fixed world point's at their azimuth,
    sigma^2 = sigma_range_rate^2 + (d stationary_range_rate / d azimuth * sigma_azimuth)^2."""
    a = pose[2]+np.asarray(azimuth, float)
    slope = np.sin(a)*v_sensor[0]-np.cos(a)*v_sensor[1]
    sigma = np.sqrt(spec.sigma_range_rate**2+(slope*spec.sigma_azimuth)**2)
    with np.errstate(invalid='ignore'):
        return np.abs(np.asarray(range_rate, float)-stationary_range_rate(azimuth, pose, v_sensor)) < sigmas*sigma


def box_distance(points, boxes):
    """Distance (n, k) from world points (n, 2) to box polygons (k, 4, 2) (counter-clockwise, as car_boxes); 0 inside."""
    p, b = np.asarray(points, float), np.asarray(boxes, float)
    if len(p) == 0 or len(b) == 0:
        return np.full((len(p), len(b)), np.inf)
    ab = np.roll(b, -1, axis=1)-b                                    # (k, 4, 2)
    ap = p[:, None, None, :]-b[None]                                 # (n, k, 4, 2)
    t = np.clip(np.einsum('nkej,kej->nke', ap, ab)/np.einsum('kej,kej->ke', ab, ab), 0, 1)
    d = np.linalg.norm(ap-t[..., None]*ab, axis=-1).min(-1)
    inside = (ab[None, ..., 0]*ap[..., 1]-ab[None, ..., 1]*ap[..., 0] >= 0).all(-1)
    return np.where(inside, 0.0, d)


def cluster_distance(pts, azimuth, range_rate, scale=SCALE, rel_speed=MAX_REL_SPEED):
    """Squared scaled distances (n, n) for clustering: position (pts (n, 2)) over scale[:2], and the range-rate
    difference less rel_speed times the angle between the two lines of sight (what one rigid car shows), over
    scale[2]."""
    p = np.asarray(pts, float)/np.asarray(scale[:2], float)
    rr = np.abs(np.subtract.outer(range_rate, range_rate))
    if rel_speed > 0:
        rr = np.maximum(rr-rel_speed*np.abs(wrap(np.subtract.outer(azimuth, azimuth))), 0.0)
    return ((p[:, None]-p[None])**2).sum(-1)+(rr/float(scale[2]))**2


def dbscan(X, eps=1.0, min_samples=1, d2=None):
    """DBSCAN labels (n,) of the rows of X (n, m) under the Euclidean metric (or of the points behind a precomputed
    squared-distance matrix d2 (n, n)): 0..K-1 by cluster, -1 for noise. Core points have at least min_samples points
    (themselves included) within eps; clusters are the connected groups of core points, and a border point joins its
    nearest core neighbour's cluster. min_samples 1 makes every point core (single-linkage groups at distance eps).
    Dense pairwise form, for the tens of points of a radar scan."""
    if d2 is None:
        X = np.asarray(X, float)
        X = X.reshape(len(X), -1) if len(X) else X.reshape(0, 1)
        d2 = ((X[:, None]-X[None])**2).sum(-1)
    n = len(d2)
    if n == 0:
        return np.zeros(0, int)
    adj = d2 <= eps*eps
    core = adj.sum(1) >= min_samples
    link = adj & core[:, None] & core[None]
    lab = np.where(core, np.arange(n), n)
    while True:                                   # each core point takes the least index of its connected group
        new = np.where(core, np.where(link, lab[None], n).min(1), n)
        if np.array_equal(new, lab):
            break
        lab = new
    border = ~core & (adj & core[None]).any(1)
    if border.any():
        lab[border] = lab[np.argmin(np.where(adj & core[None], d2, np.inf)[border], axis=1)]
    out = np.full(n, -1)
    ok = lab < n
    out[ok] = np.unique(lab[ok], return_inverse=True)[1]
    return out


def process_radar(scan, spec, ego, mates, params, scale=SCALE, eps=1.0, min_samples=1, rel_speed=MAX_REL_SPEED):
    """'radar' Detections of RadarScan `scan` from radar `spec` on the car in EgoState ego (moved to scan.time if it is
    at another time); mates (k, >= 3) the teammates' CG states [x, y, yaw, ...] at scan.time (or None); params the
    vehicle parameters (sensor mount, every car's size). rel_speed 0 clusters on the plain scaled (x, y, range_rate).
    meta: n_points, scan (the scan time)."""
    return process_radars([scan], [spec], [ego], [mates], params, scale, eps, min_samples, rel_speed)[0]


def process_radars(scans, specs, egos, mates, params, scale=SCALE, eps=1.0, min_samples=1, rel_speed=MAX_REL_SPEED):
    """process_radar of several scans at once (scans[k] from specs[k] on the car in egos[k], its teammates mates[k]):
    one list of Detections per scan, the same as scan by scan. Every step runs over all points of all scans in one
    batch (points of different scans never cluster together), so twelve scans cost about what one did."""
    egos = [ego_at(e, sc.time) for e, sc in zip(egos, scans)]
    poses = [sensor_pose(sp.mount, (e.x, e.y, e.yaw), params) for sp, e in zip(specs, egos)]
    flat = lambda name: [np.asarray(getattr(sc, name), float).reshape(-1) for sc in scans]
    sizes = [len(x) for x in flat('range')]
    out = [[] for _ in scans]
    if sum(sizes) == 0:
        return out
    r, az, rr = (np.concatenate(flat(k)) for k in ('range', 'azimuth', 'range_rate'))
    which = np.repeat(np.arange(len(scans)), sizes)
    P = np.array(poses, float)[which]
    noise = np.array([np.maximum((sp.sigma_range, sp.sigma_azimuth, sp.sigma_range_rate), SIGMA_FLOOR)
                      for sp in specs])
    V = np.array([sensor_velocity(sp.mount, e, params) for sp, e in zip(specs, egos)])[which]
    ok = np.isfinite(r) & np.isfinite(az) & np.isfinite(rr) & (r > 0)
    a = P[:, 2]+az                                                    # stationary(): the Doppler test per point
    slope = np.sin(a)*V[:, 0]-np.cos(a)*V[:, 1]
    rr_rate = np.array([sp.sigma_range_rate for sp in specs])[which]
    sigma = np.sqrt(rr_rate**2+(slope*np.array([sp.sigma_azimuth for sp in specs])[which])**2)
    with np.errstate(invalid='ignore'):
        ok &= ~(np.abs(rr+(np.cos(a)*V[:, 0]+np.sin(a)*V[:, 1])) < STATIONARY_SIGMAS*sigma)
    pts = np.column_stack((r*np.cos(az), r*np.sin(az)))
    c, s = np.cos(P[:, 2]), np.sin(P[:, 2])
    world = np.column_stack((c*pts[:, 0]-s*pts[:, 1]+P[:, 0], s*pts[:, 0]+c*pts[:, 1]+P[:, 1]))
    groups = {}
    for k, m in enumerate(mates):
        if m is not None and len(m):
            groups.setdefault(id(m), (m, []))[1].append(k)
    sr, sa = noise[which, 0], noise[which, 1]
    margin = np.maximum(MATE_MARGIN, MATE_SIGMAS*np.sqrt(sr**2+(r*sa)**2))    # a far point's own noise reaches past 1 m
    for m, ks in groups.values():
        sel = np.flatnonzero(ok & np.isin(which, ks))
        if sel.size:
            boxes = car_boxes(np.asarray(m, float).reshape(len(m), -1)[:, :3], params)
            ok[sel[(box_distance(world[sel], boxes) <= margin[sel, None]).any(1)]] = False
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return out
    pts, r, az, rr, which, world = pts[idx], r[idx], az[idx], rr[idx], which[idx], world[idx]
    d2 = cluster_distance(world, az, rr, scale, rel_speed)
    d2[which[:, None] != which[None, :]] = np.inf
    lab = dbscan(None, eps, min_samples, d2=d2)
    keep = lab >= 0
    pts, r, az, rr, lab, which = pts[keep], r[keep], az[keep], rr[keep], lab[keep], which[keep]
    if len(lab) == 0:
        return out
    count = np.bincount(lab)
    mean = lambda x: np.bincount(lab, x)/count
    cx, cy, c_rr = mean(pts[:, 0]), mean(pts[:, 1]), mean(rr)
    c_r, c_az = np.hypot(cx, cy), np.arctan2(cy, cx)
    dev = np.column_stack((r-c_r[lab], wrap(az-c_az[lab]), rr-c_rr[lab]))
    spread = np.zeros((len(count), 3, 3))
    np.add.at(spread, lab, dev[:, :, None]*dev[:, None, :])
    owner = np.zeros(len(count), int)
    owner[lab] = which
    sig = noise[owner]
    R = (sig[:, :, None]*np.eye(3)*sig[:, None, :])/count[:, None, None]+spread/count[:, None, None]
    for k in np.argsort(owner, kind='stable'):
        q = owner[k]
        sc = scans[q]
        out[q].append(Detection('radar', sc.sensor, sc.car, float(sc.time), np.array([c_r[k], c_az[k], c_rr[k]]), R[k],
                                poses[q], egos[q], {'n_points': int(count[k]), 'scan': float(sc.time)}))
    return out
