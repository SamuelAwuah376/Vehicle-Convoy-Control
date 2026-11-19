"""Track existence for the team tracker (design section 6, IPDA style): each sensor's expected detection probability
of a track, its occlusion-aware visibility, the sensors' clutter densities and the existence update.

A track in a sensor's coverage that the scan does not detect loses existence by the detection probability that sensor
was expected to have for it, so a car hidden behind another one, or outside every field of view, is not penalized.

API (the tracker builds on it):
- visibility(origins, boxes, targets, skip=None, want=None, samples=VIS_SAMPLES) -> (C, T): geometry.visible_fraction
  of every target box from every origin (C, 2) in one pass (the same sample points and the same ray test, each ray
  against only the boxes that can block it); skip (C, B) bool per origin, want (C, T) the pairs to evaluate.
  visible_fractions(origin, boxes, targets, skip=(), samples) -> (len(targets),): the same for one origin.
- radar_pd_range(spec, r): the radar's nominal detection curve of a fully visible car (radar equation, logistic in
  SNR, calibrated to pd_ref at ref_range and 0.5 at range_max; 0 beyond range_max).
- lidar_pd(spec, r, extent, vis): probability that at least LIDAR_MIN_POINTS points return from a car spanning
  `extent` rad, `vis` of it visible: beams spaced spec.beam_spacing, spec.layers points each, each returning with
  1 - dropout - far_dropout r / range_max (binomial tail).
- camera_pd(spec, height_px, vis): spec.pd when the box is at least min_height_px tall and min_visible of the car is
  seen, else 0.
- doppler_pass(spec, closing, slope): probability that a radar point of a car closing at `closing` along the line of
  sight passes the processing's stationary (Doppler) test; the tracker multiplies a radar's pd by it for the point
  nearest the sensor, the one its measurement model predicts (0 for a car abeam of a side radar).
- expected_pd(spec, rng, az, los, vis, params, car_height, through=True) -> (pd, covered): the expected detection
  probability and the coverage (field of view and range) of tracks whose CG lies at range rng and azimuth az in the
  sensor frame, seen along a line of sight at angle los to the road with visible fraction vis (any broadcastable
  shapes), for every spec type. spec may be a SpecRows (spec_rows(specs), cached) to evaluate (k, n) for k scans of
  one sensor type. through False leaves out a radar's detections of hidden cars (occluded_pd): the tracker's misses
  use that, since such detections are rare and persist or fail together from scan to scan (the multipath geometry
  changes slowly), so a run of them missing says little about whether the hidden car is there.
- reach(spec, car_height) -> (half angle, range): the widest field of view and the farthest detection of a sensor.
- clutter_density(spec): false detections per unit measurement volume of one scan (radar: per m rad m/s of
  [range, azimuth, range rate]; camera: per px^4 of [u_left, u_right, v_top, v_bottom]; lidar: per m^2 rad of
  [x, y, heading]) = clutter_rate(spec) (the spec's false-alarm rate; lidar LIDAR_FALSE per scan) over the product
  of the spans of the dimensions, clutter_log_volumes(spec) (their logs): radar range_max, fov and RANGE_RATE_SPAN;
  camera the image width, CAMERA_SIZE_SHARE of it (the box's width), the height and CAMERA_SIZE_SHARE of it; lidar
  the field of view's area (split evenly over x and y) and LIDAR_HEADINGS.
- miss_log_factor(pd, gate), hit_log_factor(pd, gate, log_lr): log(1 - delta) of the IPDA update for a missed and a
  detected track, delta = pd (gate - lr) with lr = N(z; zhat, S) / clutter density (0 when missed). The update is
  multiplicative in the odds, o+ = (1 - delta) o, so a cycle's scans sum in log-odds in any order.
- survive(p, cfg, dt): p x exist_survive^dt. apply_log_odds(p, log_factor, p_max): p after the summed factors,
  capped at p_max (EXIST_MAX by default), so a long-tracked car still dies after a few confident misses.
"""
from dataclasses import fields
from functools import lru_cache
import numpy as np
from scipy.special import ndtr
from ..config import CameraSpec, LidarSpec, RadarSpec
from ..geometry import focal
from ..processing.radar import STATIONARY_SIGMAS

__all__ = ['SpecRows', 'spec_rows', 'VIS_SAMPLES', 'LIDAR_MIN_POINTS', 'LIDAR_FALSE', 'RANGE_RATE_SPAN', 'EXIST_MAX',
           'EXIST_MIN', 'PD_DETECTED_MIN', 'visibility', 'visible_fractions', 'radar_pd_range', 'lidar_pd', 'camera_pd',
           'doppler_pass', 'expected_pd', 'reach', 'clutter_rate', 'clutter_log_volumes', 'clutter_density',
           'miss_log_factor', 'hit_log_factor', 'survive', 'apply_log_odds', 'kind_of']

VIS_SAMPLES = 2           # samples per box edge: its corners and edge midpoints (geometry.visible_fraction's outline)
LIDAR_MIN_POINTS = 3      # lidar processing fits no box to fewer points
LIDAR_FALSE = 0.1         # false lidar boxes per scan (the spec has no false-alarm field)
LIDAR_HEADINGS = np.radians(40.0)   # heading span of a false lidar box (the box fit searches +-20 deg)
RANGE_RATE_SPAN = 50.0    # m/s spread of radar clutter range rates
CAMERA_SIZE_SHARE = 0.25  # a false camera box's width and height spread over this share of the image
EXIST_MAX = 0.999
EXIST_MIN = 1e-9
PD_DETECTED_MIN = 0.3     # a detected track whose expected pd was lower is credited with this one


class SpecRows:
    """Several specs of one type as arrays (k, 1) per numeric field, so the pd models below evaluate every scan of a
    fusion cycle (k scans x n tracks) in one call. kind: their Detection kind."""
    def __init__(self, specs):
        specs = tuple(specs)
        self.kind = kind_of(specs[0])
        if any(kind_of(s) != self.kind for s in specs):
            raise TypeError('SpecRows needs specs of one type')
        for f in fields(specs[0]):
            v = [getattr(s, f.name) for s in specs]
            if all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in v):
                setattr(self, f.name, np.array(v, float)[:, None])


@lru_cache(maxsize=256)
def spec_rows(specs):
    """SpecRows of a tuple of specs (cached: a cycle's scans repeat)."""
    return SpecRows(specs)


def kind_of(spec):
    """The Detection kind a spec produces (SpecRows: theirs)."""
    if isinstance(spec, SpecRows):
        return spec.kind
    if isinstance(spec, LidarSpec):
        return 'lidar_box'
    if isinstance(spec, RadarSpec):
        return 'radar'
    if isinstance(spec, CameraSpec):
        return 'camera_box'
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')


def _angle_gap(a, b):
    """|wrap(a - b)| for angles a, b in [-pi, pi]."""
    d = np.abs(a-b)
    return np.minimum(d, 2*np.pi-d)


def visibility(origins, boxes, targets, skip=None, want=None, samples=VIS_SAMPLES):
    """(C, T) geometry.visible_fraction(origins[c], boxes, targets[t], samples, skip) for every origin (C, 2) and
    target box index (T,) at once. skip (C, B) bool: the boxes each origin ignores (its own car's body); want (C, T)
    bool: the pairs to evaluate (the rest are 0). A target that no other box can block (none overlaps its bearing
    interval nearer than its centre) is wholly visible without a test; the rest test each sample ray against only the
    boxes that can block it (a slab test in the box's own frame: the boxes are rectangles), so the cost follows the
    occlusions, not the number of boxes."""
    o = np.asarray(origins, float).reshape(-1, 2)
    B = np.asarray(boxes, float)
    targets = np.asarray(targets, int).reshape(-1)
    C, T, nb = len(o), len(targets), len(B)
    out = np.zeros((C, T))
    if C == 0 or T == 0 or nb == 0:
        return out
    active = np.ones((C, nb), bool) if skip is None else ~np.asarray(skip, bool).reshape(C, nb)
    want = np.ones((C, T), bool) if want is None else np.asarray(want, bool).reshape(C, T)
    centre = B.mean(1)
    ex, ey = B[:, 1]-B[:, 0], B[:, 3]-B[:, 0]                              # the box's own axes
    hl, hw = np.hypot(ex[:, 0], ex[:, 1])/2, np.hypot(ey[:, 0], ey[:, 1])/2
    ux, uy = ex/(2*hl[:, None]), ey/(2*hw[:, None])
    rel = centre[None]-o[:, None, :]                                       # (C, B, 2)
    dc = np.hypot(rel[..., 0], rel[..., 1])
    bearing = np.arctan2(rel[..., 1], rel[..., 0])
    corner = B[None]-o[:, None, None, :]
    half = _angle_gap(np.arctan2(corner[..., 1], corner[..., 0]), bearing[..., None]).max(-1)
    reach = dc-np.hypot(hl, hw)                                            # no point of the box is nearer
    ox = -(rel[..., 0]*ux[:, 0]+rel[..., 1]*ux[:, 1])                      # each origin in each box's frame
    oy = -(rel[..., 0]*uy[:, 0]+rel[..., 1]*uy[:, 1])
    inside = (np.abs(ox) <= hl) & (np.abs(oy) <= hw)
    block = (_angle_gap(bearing[:, targets, None], bearing[:, None, :]) <= half[:, targets, None]+half[:, None, :]+1e-9)
    block &= reach[:, None, :] < dc[:, targets, None]
    block |= (inside & active).any(1)[:, None, None]                     # rays start inside a box: no shortcut
    block &= active[:, None, :]
    block[:, np.arange(T), targets] = False
    cast = (block.any(2) | inside[:, targets]) & want
    out[want & ~cast] = 1.0
    if not cast.any():
        return out
    pc, pt = np.nonzero(cast)
    tb = B[targets]
    steps = np.arange(samples)/samples
    pts = (tb[:, :, None, :]+steps[None, None, :, None]*(np.roll(tb, -1, axis=1)-tb)[:, :, None, :]).reshape(T, -1, 2)
    v = pts[pt]-o[pc][:, None, :]                                          # rays to the sample points
    dist = np.hypot(v[..., 0], v[..., 1])
    ray_pair, ray_k = np.nonzero(dist <= dc[pc, targets[pt]][:, None])
    count = np.bincount(ray_pair, minlength=len(pc))
    hit = np.zeros(len(ray_pair), bool)
    q, b = np.nonzero(block[pc, pt])                                       # (pair, possible blocker)
    reps = count[q]
    if reps.sum():
        first = np.cumsum(count)-count                                     # each pair's first ray
        b, q = np.repeat(b, reps), np.repeat(q, reps)
        r = first[q]+np.arange(len(q))-np.repeat(np.cumsum(reps)-reps, reps)    # (ray, possible blocker)
        vr, dr, c = v[q, ray_k[r]], dist[q, ray_k[r]], pc[q]
        dx = vr[:, 0]*ux[b, 0]+vr[:, 1]*ux[b, 1]
        dy = vr[:, 0]*uy[b, 0]+vr[:, 1]*uy[b, 1]
        with np.errstate(divide='ignore', invalid='ignore'):
            tx1, tx2 = (-hl[b]-ox[c, b])/dx, (hl[b]-ox[c, b])/dx               # along the ray, 1 at the sample
            ty1, ty2 = (-hw[b]-oy[c, b])/dy, (hw[b]-oy[c, b])/dy
        t_in = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        t_out = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
        t_hit = np.where(t_in > 0, t_in, t_out)                            # the exit from inside a box
        crosses = (t_in <= t_out) & (t_hit > 0) & (t_hit*dr < dr-1e-6)
        hit = np.bincount(r, crosses, minlength=len(ray_pair)) > 0
    seen = np.bincount(ray_pair, ~hit, minlength=len(pc))
    out[pc, pt] = np.where(count > 0, seen/np.maximum(count, 1), 0.0)
    return out


def visible_fractions(origin, boxes, targets, skip=(), samples=VIS_SAMPLES):
    """geometry.visible_fraction(origin, boxes, t, samples, skip) of every box t in targets (visibility for one
    origin; skip a list of box indices)."""
    mask = np.zeros(len(boxes), bool)
    mask[np.asarray(skip, int)] = True
    return visibility(np.asarray(origin, float)[None], boxes, targets, mask[None], samples=samples)[0]


def radar_pd_range(spec, r):
    """Nominal detection probability of a fully visible car at range(s) r (module docstring)."""
    r = np.asarray(r, float)
    p, ref, rmax = np.clip(spec.pd_ref, 0.0, 1.0), spec.ref_range, spec.range_max
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        n = np.log(p/(1-p))/np.log(rmax/ref)
        curve = np.minimum(p, 1/(1+(np.maximum(r, 0)/rmax)**n))
    curve = np.where((p > 0.5) & (p < 1) & (ref > 0) & (ref < rmax), curve, p)
    return np.where(r <= rmax, curve, 0.0)


def lidar_pd(spec, r, extent, vis):
    """P(at least LIDAR_MIN_POINTS of the beams on the visible part of a car return) (module docstring)."""
    r, extent, vis = np.broadcast_arrays(*(np.asarray(a, float) for a in (r, extent, vis)))
    n = np.floor(np.maximum(extent*vis, 0)/spec.beam_spacing)*np.maximum(np.floor(spec.layers), 1)
    p = np.clip(1-spec.dropout-spec.far_dropout*np.clip(r/spec.range_max, 0, 1), 0, 1)
    q = 1-p
    tail = np.zeros(n.shape)
    for k in range(LIDAR_MIN_POINTS):
        coef = np.ones(n.shape)
        for i in range(k):
            coef = coef*(n-i)/(i+1)
        with np.errstate(invalid='ignore', divide='ignore'):
            tail = tail+np.where(n >= k, coef*p**k*q**np.maximum(n-k, 0), 0.0)
    return np.where(n >= LIDAR_MIN_POINTS, np.clip(1-tail, 0, 1), 0.0)


def camera_pd(spec, height_px, vis):
    """spec.pd when the box is tall enough and enough of the car is seen, else 0."""
    ok = (np.asarray(height_px, float) >= spec.min_height_px) & (np.asarray(vis, float) >= spec.min_visible)
    return np.where(ok, spec.pd, 0.0)


def doppler_pass(spec, closing, slope, sigmas=STATIONARY_SIGMAS):
    """Probability that a radar point survives the processing's stationary test (processing.radar.stationary): the
    test rejects a point whose range rate is within `sigmas` sigma of a fixed world point's at its azimuth, and the
    difference between the two is the car's own world velocity along the line of sight, `closing` (m/s), plus noise
    of sigma^2 = sigma_range_rate^2 + (slope sigma_azimuth)^2, slope the stationary range rate's change with azimuth
    (the sensor's speed across the line of sight). A car abeam of a side radar at any speed has closing 0: its point
    nearest the sensor, the one the measurement model predicts, never passes."""
    closing, slope = np.broadcast_arrays(np.asarray(closing, float), np.asarray(slope, float))
    sigma = np.sqrt(spec.sigma_range_rate**2+(slope*spec.sigma_azimuth)**2)
    z = closing/np.maximum(sigma, 1e-9)
    return 1-(ndtr(sigmas-z)-ndtr(-sigmas-z))


def _near_offset(los, params):
    """Distance from a car's centre to its near face along a line of sight at angle los to the road."""
    return np.abs(np.cos(los))*params.length/2+np.abs(np.sin(los))*params.width/2


def expected_pd(spec, rng, az, los, vis, params, car_height, through=True):
    """(pd, covered) for tracks at range rng and sensor azimuth az (of the CG), line of sight at los to the road and
    visible fraction vis (module docstring). The field of view is tested on the CG's azimuth, so a car half out of
    view counts as unseen (conservative: its misses cost nothing). through False leaves out the radar's detections of
    the hidden part of a car (occluded_pd: multipath under and around the occluder)."""
    rng, az, los, vis = np.broadcast_arrays(*(np.asarray(a, float) for a in (rng, az, los, vis)))
    near = np.maximum(rng-_near_offset(los, params), 0.1)
    a = np.abs(az)
    kind = kind_of(spec)
    if kind == 'lidar_box':
        covered = (a <= spec.fov/2) & (near <= spec.range_max)
        extent = (np.abs(np.cos(los))*params.width+np.abs(np.sin(los))*params.length)/np.maximum(rng, 1.0)
        pd = lidar_pd(spec, near, extent, vis)
    elif kind == 'radar':
        covered = (near <= spec.range_max) & (a <= np.where(near <= spec.near_range, spec.fov, spec.fov_far)/2)
        pd = radar_pd_range(spec, near)*((spec.occluded_pd+(1-spec.occluded_pd)*vis) if through else vis)
    else:
        depth = near*np.cos(np.minimum(a, np.pi/2))
        covered = (a <= spec.hfov/2) & (depth > 1.0)
        pd = camera_pd(spec, focal(spec)*car_height/np.maximum(depth, 1e-3), vis)
    return np.where(covered, pd, 0.0), covered


def reach(spec, car_height):
    """(half angle, range) of the sensor's widest field of view and farthest detection (a camera's: where a car's
    image is min_height_px tall)."""
    if isinstance(spec, LidarSpec):
        return spec.fov/2, spec.range_max
    if isinstance(spec, RadarSpec):
        return max(spec.fov, spec.fov_far)/2, spec.range_max
    if isinstance(spec, CameraSpec):
        return spec.hfov/2, focal(spec)*car_height/spec.min_height_px
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')


def clutter_rate(spec):
    """Mean false detections per scan (lidar: LIDAR_FALSE, the spec has no false-alarm field)."""
    if isinstance(spec, RadarSpec):
        return max(spec.false_alarms, 1e-3)
    if isinstance(spec, CameraSpec):
        return max(spec.false_rate, 1e-3)
    if isinstance(spec, LidarSpec):
        return LIDAR_FALSE
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')


def clutter_log_volumes(spec):
    """Log of the span of each measured dimension that false detections spread over (module docstring), so a
    detection with dimensions that carry no information (a camera edge cut by the image border) can be compared with
    clutter on the others alone."""
    if isinstance(spec, RadarSpec):
        return np.log([spec.range_max, spec.fov, RANGE_RATE_SPAN])
    if isinstance(spec, CameraSpec):
        W, H = float(spec.width_px), float(spec.height_px)
        return np.log([W, CAMERA_SIZE_SHARE*W, H, CAMERA_SIZE_SHARE*H])
    if isinstance(spec, LidarSpec):
        area = 0.5*spec.fov*spec.range_max**2
        return np.log([np.sqrt(area), np.sqrt(area), LIDAR_HEADINGS])
    raise TypeError(f'unknown sensor spec {type(spec).__name__}')


def clutter_density(spec):
    """False detections of one scan per unit measurement volume (module docstring)."""
    return clutter_rate(spec)/float(np.exp(clutter_log_volumes(spec).sum()))


def miss_log_factor(pd, gate):
    """log(1 - pd gate): the log-odds change of a track the scan was expected to see with pd and did not."""
    return np.log1p(-np.clip(np.asarray(pd, float)*gate, 0.0, 1-1e-12))


def hit_log_factor(pd, gate, log_lr):
    """log(1 - pd gate + pd lr): the log-odds change of a detected track, lr the measurement's likelihood ratio
    against clutter (log_lr its log)."""
    pd = np.clip(np.maximum(np.asarray(pd, float), PD_DETECTED_MIN), 1e-12, 1.0)
    return np.logaddexp(np.log1p(-np.clip(pd*gate, 0.0, 1-1e-12)), np.log(pd)+np.asarray(log_lr, float))


def survive(p, cfg, dt):
    """Existence after dt seconds without information: p x exist_survive^dt."""
    return np.asarray(p, float)*float(cfg.exist_survive)**np.maximum(np.asarray(dt, float), 0.0)


def apply_log_odds(p, log_factor, p_max=EXIST_MAX):
    """p after adding log_factor to its log-odds, clipped to [EXIST_MIN, p_max]."""
    p = np.clip(np.asarray(p, float), EXIST_MIN, 1-1e-15)
    L = np.log(p)-np.log1p(-p)+np.asarray(log_factor, float)
    return np.clip(0.5*(1+np.tanh(0.5*L)), EXIST_MIN, p_max)
