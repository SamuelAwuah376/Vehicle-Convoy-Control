"""Tier A camera simulator, detection level (design section 5): the true cars' 3D boxes projected through a pinhole
camera on a pitching body (geometry.camera_tilt: a front camera pitches with the body, a rear one against it, a side
one rolls), with truncation, occlusion, a size cut-off, missed detections, pixel noise and false boxes.

What a 2D detector on real images would report, without rendering anything: the box of each car is the pixel extent
of its 8 projected corners, clipped to the image. Tier B (CARLA) replaces this with a real detector on RGB frames.
In the simulation loop, simulate_cameras runs a car's four cameras together (one occlusion pass instead of four)."""
from functools import lru_cache
import numpy as np
from ..geometry import box_corners_3d, focal, project, sensor_pose
from ..types import CameraFrame

__all__ = ['MIN_DEPTH', 'project_boxes', 'clip_boxes', 'visible_fractions', 'simulate_camera', 'simulate_cameras']

MIN_DEPTH = 0.5          # m: a car with a corner nearer the image plane than this is skipped (partly behind the camera)
KEEP_AREA = 0.5          # a box survives truncation if at least this share of its area is inside the image
SCORE_HALF_PX = 40.0     # box height at which a detection's score is halfway from 0.6 to 1.0
FALSE_NEAR = 0.15        # false boxes sit at FALSE_NEAR..1 of the range where a car is min_height_px tall
_TOL = 1e-9              # rad: bearing margin that keeps the interval shortcuts conservative


def _boxes(spec, pose, corners, pitch):
    u, v, depth = project(spec, pose, corners, pitch)
    with np.errstate(invalid='ignore'):
        boxes = np.stack((u.min(1), u.max(1), v.min(1), v.max(1)), -1)
    return boxes, (depth > MIN_DEPTH).all(1)


def project_boxes(spec, pose, states, params, height, pitch=0.0):
    """Unclipped image boxes (M, 4) [u_left, u_right, v_top, v_bottom] of cars at CG states (M, >= 3) seen by camera
    `spec` at pose (x, y, yaw, z) with body pitch, and a mask (M,) of the boxes whose every corner is deeper than
    MIN_DEPTH (only those boxes are meaningful)."""
    return _boxes(spec, pose, box_corners_3d(states, params, height), pitch)


def _image_limits(spec):
    return np.array([spec.width_px, spec.width_px, spec.height_px, spec.height_px], float)


def _area(boxes):
    return (boxes[:, 1]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 2])


def clip_boxes(boxes, spec):
    """Boxes (M, 4) clipped to the image, and the share (M,) of each box's area left inside (0 for an empty box)."""
    clipped = np.clip(boxes, 0.0, _image_limits(spec))
    full = _area(boxes)
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(full > 0, _area(clipped)/full, 0.0)
    return clipped, share


def _wrap(a):
    return (a+np.pi) % (2*np.pi)-np.pi


@lru_cache(maxsize=8)
def _steps(samples):
    """geometry.visible_fraction's sample positions along an edge (cached: linspace is slow for its size)."""
    w = np.linspace(0, 1, samples, endpoint=False)
    w.flags.writeable = False
    return w


def visible_fractions(origin, boxes, targets, samples=9):
    """geometry.visible_fraction of each box in `targets` (indices into boxes (M, 4, 2)) from world point `origin`,
    for all targets at once and with the same result."""
    targets = np.asarray(targets, int)
    return _visibility(np.asarray(origin, float)[None], np.asarray(boxes, float), np.zeros(len(targets), int), targets,
                       samples)


def _visibility(origins, B, cams, targets, samples):
    """visible_fractions of boxes `targets` (K,), each seen from origins[cams] (origins (C, 2)): every camera of a
    car in one pass.

    Bearing intervals and distance bounds of the boxes sort the targets first: one with no box that could block it
    is fully seen; one whose bearing interval lies strictly inside that of a single box wholly nearer is hidden (a ray
    strictly inside a convex box's interval crosses it). Only the rest cast their sample rays (_ray_fractions)."""
    K = len(targets)
    if K == 0:
        return np.zeros(0)
    q = B[None]-origins[:, None, None]
    qx, qy = q[..., 0], q[..., 1]
    cx, cy = (qx[..., 0]+qx[..., 2])*0.5, (qy[..., 0]+qy[..., 2])*0.5
    C = len(origins)
    ref = np.arctan2(np.bincount(cams, cy[cams, targets], C), np.bincount(cams, cx[cams, targets], C))
    bearing = _wrap(np.arctan2(qy, qx)-ref[:, None, None])   # from ref: each camera's targets clear of the +-pi cut
    lo, hi = bearing.min(-1), bearing.max(-1)
    wide = hi-lo >= np.pi-_TOL               # across the cut, or holding the origin: no usable interval
    nearest = np.sqrt(cx*cx+cy*cy)-0.5*np.sqrt((qx[..., 0]-qx[..., 2])**2+(qy[..., 0]-qy[..., 2])**2)
    farthest = np.sqrt(qx*qx+qy*qy).max(-1)

    L, H, W = lo[cams], hi[cams], wide[cams]
    tl, th, tw = lo[cams, targets, None], hi[cams, targets, None], wide[cams, targets, None]
    occ = (nearest[cams] < farthest[cams, targets, None]) & (W | tw | ((L-_TOL <= th) & (H+_TOL >= tl)))
    occ[np.arange(K), targets] = False
    covered = (occ & ~(W | tw) & (L+_TOL < tl) & (th < H-_TOL)
               & (farthest[cams] < nearest[cams, targets, None]-1e-6)).any(1)
    frac = np.where(covered, 0.0, 1.0)
    part = np.flatnonzero(occ.any(1) & ~covered)
    if len(part):
        frac[part] = _ray_fractions(origins, B, q, cams[part], targets[part], occ[part], ref, lo, hi, wide, nearest,
                                    farthest, samples)
    return frac


def _ray_fractions(origins, B, q, cams, targets, occ, ref, lo, hi, wide, nearest, farthest, samples):
    """_visibility by sample rays (geometry.visible_fraction's), each tested against the boxes in its target's row
    of occ: skipped unless the ray lies in the box's bearing interval and the box starts nearer than the sample,
    hidden outright when strictly inside the interval of a box wholly nearer, intersected exactly (as
    geometry.raycast) otherwise."""
    K = len(targets)
    e = B[:, (1, 2, 3, 0)]-B
    T = B[targets]
    o = origins[cams][:, None]
    pts = (T[:, :, None]+_steps(samples)[None, None, :, None]*e[targets][:, :, None]).reshape(K, -1, 2)
    px, py = pts[..., 0]-o[..., 0], pts[..., 1]-o[..., 1]
    dist = np.sqrt(px*px+py*py)
    c = T.mean(1)-o[:, 0]
    near = dist <= np.sqrt(c[:, 0]*c[:, 0]+c[:, 1]*c[:, 1])[:, None]
    owner = np.nonzero(near)[0]
    if len(owner) == 0:
        return np.zeros(K)
    d = dist[near]
    ang = np.arctan2(py[near], px[near])
    ray, box = np.nonzero(occ[owner])
    cam = cams[owner[ray]]
    A, D = _wrap(ang[ray]-ref[cam]), d[ray]
    keep = (nearest[cam, box] < D) & (wide[cam, box] | ((lo[cam, box]-_TOL <= A) & (A <= hi[cam, box]+_TOL)))
    ray, box, cam, A, D = ray[keep], box[keep], cam[keep], A[keep], D[keep]
    sure = ~wide[cam, box] & (lo[cam, box]+_TOL < A) & (A < hi[cam, box]-_TOL) & (farthest[cam, box] < D-1e-6)
    hidden = np.zeros(len(owner), bool)
    hidden[ray[sure]] = True
    rest = ~hidden[ray]
    ray, box, cam = ray[rest], box[rest], cam[rest]
    if len(ray):
        ex, ey, qx, qy = e[box, :, 0], e[box, :, 1], q[cam, box, :, 0], q[cam, box, :, 1]
        rx, ry = np.cos(ang[ray])[:, None], np.sin(ang[ray])[:, None]
        denom = rx*ey-ry*ex
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (qx*ey-qy*ex)/denom
            s = (qx*ry-qy*rx)/denom
        cut = (np.abs(denom) > 1e-12) & (t > 0) & (s >= 0) & (s <= 1) & (t < d[ray, None]-1e-6)
        hidden[ray[cut.any(1)]] = True
    return np.bincount(owner, weights=~hidden, minlength=K)/np.maximum(np.bincount(owner, minlength=K), 1)


def _false_boxes(spec, pose, params, rng, car_height, pitch):
    """Poisson(false_rate) car-sized boxes standing on the road below the horizon at random ranges and columns."""
    k = rng.poisson(spec.false_rate)
    if k == 0:
        return np.zeros((0, 4))
    f = focal(spec)
    r = rng.uniform(FALSE_NEAR, 1.0, k)*f*car_height/max(spec.min_height_px, 1.0)
    width = f*rng.uniform(params.width, params.length, k)/r
    u = rng.uniform(0.0, spec.width_px, k)
    bottom = spec.height_px/2-f*np.tan(pitch*np.cos(spec.mount.yaw))+f*pose[3]/r   # the horizon (the camera's pitch)
    boxes, _ = clip_boxes(np.stack((u-width/2, u+width/2, bottom-f*car_height/r, bottom), -1), spec)
    return boxes[_area(boxes) > 0]


def _candidates(spec, car, state, corners, params, rng):
    """Pitch, pose, and the boxes that pass truncation, the size cut-off and the pd draw (before occlusion)."""
    pitch = spec.pitch_per_accel*(-float(state.team_ax[car]))+float(rng.normal(0.0, spec.pitch_sigma))
    pose = sensor_pose(spec.mount, state.team[car], params)
    boxes, ok = _boxes(spec, pose, corners, pitch)
    cand = np.flatnonzero(ok)
    clipped, share = clip_boxes(boxes[cand], spec)
    keep = (share >= KEEP_AREA) & (clipped[:, 3]-clipped[:, 2] >= spec.min_height_px)
    cand, clipped = cand[keep], clipped[keep]
    hit = rng.random(len(cand)) < spec.pd
    return pitch, pose, cand[hit], clipped[hit]


def _frame(spec, car, t, pitch, pose, cand, clipped, ids, params, rng, car_height):
    """The CameraFrame of the detected boxes: edge noise, false boxes, scores and a shuffled order."""
    noisy = clipped+rng.normal(0.0, spec.sigma_px, clipped.shape)
    det = np.clip(np.sort(noisy.reshape(-1, 2, 2), -1).reshape(-1, 4), 0.0, _image_limits(spec))
    h = clipped[:, 3]-clipped[:, 2]
    fake = _false_boxes(spec, pose, params, rng, car_height, pitch)
    order = rng.permutation(len(det)+len(fake))
    score = np.concatenate((0.6+0.4*h/(h+SCORE_HALF_PX), rng.uniform(0.3, 0.7, len(fake))))
    truth_id = np.concatenate((ids[cand], np.full(len(fake), -1, int)))
    return CameraFrame(spec.name, car, float(t), np.concatenate((det, fake))[order], score[order], pitch,
                       truth_id[order])


def simulate_cameras(specs, car, truth, t, params, rng, car_height):
    """simulate_camera for several cameras on one car at the same time, sharing the truth lookup, the 3D boxes and
    one occlusion pass (_visibility) for all of them. Same model as simulate_camera; the random draws come in a
    different order (every camera's pitch and pd draws first), so the frames differ from successive simulate_camera
    calls except where nothing random matters."""
    state = truth.at(t)
    others = np.array([k for k in range(len(state.team)) if k != car], int)
    states = np.concatenate((state.traffic[:, :3], state.team[others, :3]))
    ids = np.concatenate((np.asarray(state.traffic_id, int), -2-others))
    corners = box_corners_3d(states, params, car_height)
    pre = [_candidates(spec, car, state, corners, params, rng) for spec in specs]
    occl = [k for k, (spec, p) in enumerate(zip(specs, pre)) if spec.min_visible > 0 and len(p[2])]
    if occl:
        cams = np.concatenate([np.full(len(pre[k][2]), n) for n, k in enumerate(occl)])
        frac = _visibility(np.array([pre[k][1][:2] for k in occl]), corners[:, :4, :2], cams,
                           np.concatenate([pre[k][2] for k in occl]), 9)
        for n, k in enumerate(occl):
            pitch, pose, cand, clipped = pre[k]
            seen = frac[cams == n] >= specs[k].min_visible
            pre[k] = pitch, pose, cand[seen], clipped[seen]
    return tuple(_frame(spec, car, t, *p, ids, params, rng, car_height) for spec, p in zip(specs, pre))


def simulate_camera(spec, car, truth, t, params, rng, car_height):
    """One frame of camera `spec` on team car `car` at time t, from TruthHistory `truth`.

    Every traffic car and the two teammates is projected (project_boxes) with the body pitch at t (pitch_per_accel
    times the braking deceleration plus N(0, pitch_sigma)). A box is kept when at most half of it is cut off by the
    image border; it is detected with probability pd when its clipped height is at least min_height_px and at least
    min_visible of it is seen past the other cars (visible_fractions from the camera, ignoring the carrying car's
    own body). Detected boxes get N(0, sigma_px) on each edge (re-ordered should the noise cross two edges) and are
    clipped to the image again; false boxes (_false_boxes) are added and the order shuffled. Scores: 0.6..1.0
    growing with box height for detections, uniform 0.3..0.7 for false boxes. truth_id as in RadarScan: the traffic
    id, -2 - team index for a teammate, -1 for a false box."""
    return simulate_cameras((spec,), car, truth, t, params, rng, car_height)[0]
