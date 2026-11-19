"""Camera processing (design section 6, camera): one frame's 2D boxes -> 'camera_box' Detections.

The tracker updates in the image itself: z is the box [u_left, u_right, v_top, v_bottom] with R = sigma_px^2 per edge
plus the frame's pitch noise, which moves the four edges together (pitch_jacobian). The body pitch reaches each camera
through its mount yaw (geometry.camera_tilt): a front camera pitches with it, a rear one against it, a side one rolls.
This module drops the teammates' boxes (IoU above IOU_MATE with their projected 3D boxes from V2V), flags the edges
cut by the image border (meta['truncated']: left, right, top, bottom within TRUNC_PX of it), and gives every box a
single-frame position of the car's CG with its covariance, meta['init'] = (xy, P), for births and gating:

- cue_init (design step 3): two range cues along the ray through the box, fused by inverse variance. The ground cue
  back-projects the bottom-centre pixel onto the road from the camera height at pitch ego.pitch (its error grows with
  r^2 and with the pitch error); the size cue is the known car height over the box height (a relative error, nearly
  free of pitch). The bearing comes from the box centre. The box bottom is the car's near face, so the CG is reached by
  stepping |cos phi| length/2 + |sin phi| width/2 along the ray (phi the ray's angle to the road) to the body centre,
  then the CG's offset from the body centre along the road: for a car ahead that is the rear bumper to CG distance,
  for a car behind the front bumper to CG.
- refine_init (refine=True, the default): Gauss-Newton from cue_init on the full projected-box model, i.e. the CG of a
  road-aligned car of the known size whose projected box best explains the usable edges. The pitch error (sigma
  spec.pitch_sigma + PITCH_EXTRA) and the car's heading (HEADING_SIGMA, from lane changes) are nuisance parameters:
  their effect on the edges, from three-point evaluations, shifts the predicted box and joins the pixel noise in the
  weights, so the fit itself finds that the box height is pitch-robust and the bottom edge is not (the inverse-variance
  fusion of the two cues, with the exact geometry). It removes the cue method's across-ray bias for a car off the
  ray's centre (the visible flank widens the box to one side: about 0.2 m at 20 m in the next lane, several sigma of
  the bearing) and uses the known size when an edge is truncated.

P is first order in both: the Jacobian of the cue estimate with respect to the edges and the pitch (cue_init), or the
inverse of the fit's information matrix (refine_init); long along the ray and narrow across it. Edges within
FIT_MARGIN_SIGMAS sigma_px of the border count as cut (a simulator or detector clips a box before its noise, so a cut
edge can land several pixels inside the border, and a cut edge read as a real one wrecks the fit): the cue estimate
leaves them out, and the fit takes them as bounds (censored measurements: the car reaches at least that far, so a
hypothesis whose box ends inside a cut edge is pulled out to it). A bound steers the estimate but adds nothing to P.
Without the bounds a car close ahead cut at the bottom and one side keeps only its top edge for range, which says next
to nothing for a camera near the roof height. Even with them that case is the weak one: the top edge's range cue is
strongly nonlinear (dr ~ r^2 dpitch / (car height - camera height)), and P has a heavier tail than a Gaussian there.

process_cameras runs the arithmetic compiled (_camera_jit.process_boxes, one call per sim step) when numba is on
(perception/_jit.py), and as this module's numpy code otherwise (NOHESI_JIT=0): the numpy functions below are the
reference the kernels follow operation by operation, so the two paths agree to rounding.
"""
from math import radians, sqrt
from typing import NamedTuple
import numpy as np
from ...geometry import cg_offsets
from .._jit import ENABLED
from ..geometry import box_corners_3d, camera_tilt, focal, sensor_pose
from ..types import Detection
from . import _camera_jit
from .radar import ego_at

__all__ = ['TRUNC_PX', 'FIT_MARGIN_SIGMAS', 'IOU_MATE', 'PITCH_EXTRA', 'HEADING_SIGMA', 'HEADING_SPREAD',
           'PITCH_SPREAD', 'NEAR', 'iou', 'box_model', 'mate_boxes', 'cue_init', 'refine_init', 'pitch_jacobian',
           'process_camera', 'process_cameras']

TRUNC_PX = 2.0
FIT_MARGIN_SIGMAS = 5.0
IOU_MATE = 0.5
PITCH_EXTRA = radians(0.25)     # pitch-estimate error beyond spec.pitch_sigma (the braking model is approximate)
HEADING_SIGMA = radians(2.0)    # a traffic car's heading to the road: 0 keeping its lane, up to ~5 deg changing
HEADING_SPREAD = 2.5            # heading points at +-2.5 sigma (5 deg, mid lane change), weights 1/(2 2.5^2): most
                                # cars keep their lane, and the box only widens beyond a few degrees
PITCH_SPREAD = sqrt(3.0)        # pitch points at +-sqrt(3) sigma, weights 1/6 (the unscented points of a Gaussian)
NEAR = 0.1                      # m: near plane; the part of a box nearer the camera is cut off
ITERS = 4                       # Gauss-Newton updates at most; a frame usually stops after two
CONVERGED = 1e-2                # squared Mahalanobis length (0.1 sigma) of a Gauss-Newton step that ends the iterations
PRIOR_ALONG = 1e3               # m: the refinement's (vanishing) prior along the ray; across it, the car length
TRUST = 0.5                     # a Gauss-Newton step moves the CG at most this share of its distance from the camera
SIGMA_FLOOR = 1e-3              # px: keeps R and the fits positive definite for noise-free ('ideal') cameras
_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
_STEPS = np.array([0.5, 0.5, 0.5, 0.5, 1e-4, 1e-3])   # finite-difference steps: px for the edges, rad for the pitch
                                                      # and the roll's spread
_CUT_SIDE = np.array([1.0, -1.0, 1.0, -1.0])     # a cut left / top edge bounds the box from above, right / bottom below
_JIT = ENABLED                  # process_cameras' path: the compiled kernel, or the numpy code (tests switch it)
_CORNERS = {}                   # _local_corners' cache


def _local_corners(params, height):
    """(3, 8) x, y, z of a car's 3D box corners about its CG at heading 0 (geometry.box_corners_3d), read-only. Cached
    on what they depend on (the body edges of cg_offsets, the width, the height): an episode's params, a
    SimpleNamespace, is not hashable."""
    key = (*cg_offsets(params), float(params.width), float(height))
    corners = _CORNERS.get(key)
    if corners is None:
        if len(_CORNERS) >= 16:
            _CORNERS.clear()
        corners = _CORNERS[key] = box_corners_3d(np.zeros((1, 3)), params, height)[0].T.copy()
        corners.flags.writeable = False
    return corners


def iou(a, b):
    """Intersection over union (n, k) of boxes a (n, 4) and b (k, 4), [u_left, u_right, v_top, v_bottom]."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    w = np.clip(np.minimum(a[:, None, 1], b[None, :, 1])-np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3])-np.maximum(a[:, None, 2], b[None, :, 2]), 0, None)
    area = lambda x: np.clip(x[:, 1]-x[:, 0], 0, None)*np.clip(x[:, 3]-x[:, 2], 0, None)
    inter = w*h
    union = area(a)[:, None]+area(b)[None]-inter
    return inter/np.where(union > 0, union, np.inf)


class _Cam(NamedTuple):
    """Camera parameters per box (arrays (n,), or scalars for one camera): focal length, principal point, image size,
    pixel sigma (floored), mount yaw, the per-frame pitch noise, and the sensor pose x, y, yaw, z."""
    f: object
    cu: object
    cv: object
    W: object
    H: object
    sig: object
    yaw: object
    pitch_sigma: object
    x: object
    y: object
    psi: object
    z: object

    def cols(self, k=1):
        """The fields with k trailing unit axes, to broadcast against per-box grids (n, ...)."""
        return _Cam(*(np.reshape(a, np.shape(a)+(1,)*k) for a in self))


def _cam(spec, pose):
    """The _Cam of one camera at pose (x, y, yaw, z)."""
    return _Cam(float(focal(spec)), spec.width_px/2, spec.height_px/2, float(spec.width_px), float(spec.height_px),
                max(spec.sigma_px, SIGMA_FLOOR), float(spec.mount.yaw), float(spec.pitch_sigma),
                *(float(v) for v in pose))


def _cams(specs, poses, counts):
    """The _Cam of counts[k] boxes seen by specs[k] at poses[k]."""
    rows = np.array([_cam(s, p) for s, p in zip(specs, poses)], float).reshape(-1, len(_Cam._fields))
    return _Cam(*np.repeat(rows, np.asarray(counts, int), axis=0).T)


def box_model(spec, pose, xy, heading, pitch, params, height):
    """Unclipped image boxes (..., 4) of cars with CG xy (..., 2) and world heading (broadcast to xy's leading shape)
    seen by camera spec at pose with body pitch (broadcast likewise): the pixel extent of the part of the 3D box at
    least NEAR in front of the camera (a box reaching behind the camera runs past the image edge, as a detector sees
    it); nan where nothing is in front. geometry.project's pinhole model, evaluated in the camera frame directly."""
    return _evaluate(_prepare(_cam(spec, pose), heading, pitch, params, height), xy)


def _prepare(cam, heading, pitch, params, height):
    """The part of box_model that does not depend on the CG position: the corners rotated into the sensor frame and
    the camera's tilt for the body pitch (geometry.camera_tilt), once per frame for the refinement's iterations. cam's
    fields broadcast against heading and pitch."""
    lx, ly, lz = _local_corners(params, height)
    rel = np.asarray(heading, float)-cam.psi
    cr, sr = np.cos(rel)[..., None], np.sin(rel)[..., None]
    R = camera_tilt(cam.yaw, pitch)[..., None, :, :]
    zr = lz-np.asarray(cam.z, float)[..., None]
    rows = tuple(R[..., 0, j] for j in range(3))+tuple(R[..., 1, j] for j in range(3))
    heights = tuple(R[..., 2, j]*zr for j in range(3))
    return cr*lx-sr*ly, sr*lx+cr*ly, rows, heights, cam


def _evaluate(prep, xy):
    lxs, lys, (a0, a1, a2, b0, b1, b2), (k0, k1, k2), cam = prep
    xy = np.asarray(xy, float)
    c0, s0 = np.cos(cam.psi), np.sin(cam.psi)
    dx, dy = xy[..., 0]-cam.x, xy[..., 1]-cam.y
    xs = (c0*dx+s0*dy)[..., None]+lxs
    ys = (c0*dy-s0*dx)[..., None]+lys
    depth = a0*xs+b0*ys+k0               # R^T p, as geometry.project
    yc = a1*xs+b1*ys+k1
    zc = a2*xs+b2*ys+k2
    f, cu, cv = (np.asarray(v, float)[..., None] for v in (cam.f, cam.cu, cam.cv))
    front = depth >= NEAR
    if front.all():
        u, v = cu-f*yc/depth, cv-f*zc/depth
        return np.stack((u.min(-1), u.max(-1), v.min(-1), v.max(-1)), -1)
    i, j = _EDGES.T
    di, dj = depth[..., i], depth[..., j]
    cut = front[..., i] != front[..., j]
    t = np.where(cut, (NEAR-di)/np.where(cut, dj-di, 1.0), 0.0)
    safe = np.where(front, depth, 1.0)
    u = np.concatenate((np.where(front, cu-f*yc/safe, np.nan),
                        np.where(cut, cu-f*(yc[..., i]+t*(yc[..., j]-yc[..., i]))/NEAR, np.nan)), -1)
    v = np.concatenate((np.where(front, cv-f*zc/safe, np.nan),
                        np.where(cut, cv-f*(zc[..., i]+t*(zc[..., j]-zc[..., i]))/NEAR, np.nan)), -1)
    return np.stack((np.fmin.reduce(u, -1), np.fmax.reduce(u, -1), np.fmin.reduce(v, -1), np.fmax.reduce(v, -1)), -1)


def mate_boxes(spec, pose, mates, params, height, pitch=0.0):
    """The image boxes (j, 4) of the teammates (CG states (k, >= 3)) in view, clipped to the image."""
    m = np.asarray(mates, float).reshape(len(mates), -1)
    if len(m) == 0:
        return np.zeros((0, 4))
    b = box_model(spec, pose, m[:, :2], m[:, 2], pitch, params, height)
    c = np.clip(b, 0.0, (spec.width_px, spec.width_px, spec.height_px, spec.height_px))
    ok = np.isfinite(b).all(1) & (c[:, 1] > c[:, 0]) & (c[:, 3] > c[:, 2])
    return c[ok]


def _cue_depths(cam, height, ul, ur, vt, vb, pitch):
    """Depths along the boresight (horizontal) of the car's near face from the ground cue (the ray through the
    bottom-centre pixel meets the road the camera's height cam.z below it, with the camera's whole tilt for the body
    pitch, geometry.camera_tilt) and from the size cue (box height, car height; exact in the camera's own pitch, pitch
    cos(mount yaw): the roll a side camera gets changes a box's height only by its width times the roll); nan where a
    cue has no solution. Also the lateral position of the road point at a depth x in the plane of the box's centre
    column, y = (-n_x x + n_z h) / n_y: the bearing at whichever depth the cues give. cam's fields broadcast against
    the boxes' edges."""
    f, cu, cy, h = cam.f, cam.cu, cam.cv, cam.z
    R = camera_tilt(cam.yaw, pitch)
    a, b = (cu-(ul+ur)/2)/f, (cy-vb)/f
    dx = R[..., 0, 0]+R[..., 0, 1]*a+R[..., 0, 2]*b      # the bottom-centre ray in the level frame
    dz = R[..., 2, 0]+R[..., 2, 1]*a+R[..., 2, 2]*b
    ground = np.where(dz < -1e-9, h*dx/np.where(dz < -1e-9, -dz, 1.0), np.nan)
    nx, ny, nz = (R[..., k, 1]-a*R[..., k, 0] for k in range(3))   # normal of the centre column's plane
    with np.errstate(divide='ignore', invalid='ignore'):
        lateral = lambda x: (nz*h-nx*x)/ny
    th = pitch*np.cos(cam.yaw)
    c, s = np.cos(th), np.sin(th)
    above, below = height-h, h            # the car's top above the camera, the road below it
    hp = vb-vt
    beta = c*s*(below-above)-f*height/np.where(hp > 0, hp, np.nan)   # (c x - s above)(c x + s below) hp = f height x
    with np.errstate(invalid='ignore'):
        size = (-beta+np.sqrt(beta**2+4*c**2*s**2*above*below))/(2*c**2)
    return ground, size, lateral


def cue_init(boxes, fit, spec, pose, pitch, params, height, sigma_pitch, heading_sigma=HEADING_SIGMA):
    """Two-cue CG estimates xy (n, 2) and covariances P (n, 2, 2) of boxes (n, 4) with fit (n, 4) the edges usable
    (not cut by the border), from a camera at pose with estimated pitch (module docstring). P adds to the first-order
    pixel and pitch terms the heading's swing of the CG about the near face across the ray, (L/2)^2 across it when a
    side edge is cut, and the squared depth along it when neither depth cue is usable (the fallback depth then comes
    from the box width)."""
    b = np.asarray(boxes, float).reshape(-1, 4)
    n = len(b)
    return _cue_init(b, fit, _cams([spec], [pose], [n]), np.full(n, float(pitch)), params, height,
                     np.full(n, float(sigma_pitch)), heading_sigma)


def _cue_init(b, fit, cam, pitch, params, height, sigma_pitch, heading_sigma):
    """cue_init with every box's own camera (cam fields (n,)), pitch (n,) and sigma_pitch (n,)."""
    fit = np.asarray(fit, bool).reshape(-1, 4)
    n = len(b)
    # Nuisances: the four edges, the pitch and the roll's spread rho = |roll| (a side camera's share of the pitch),
    # which grows a box by rho times its height on either side and rho times its width above and below (the box of a
    # rolled rectangle); taken as zero mean with E rho^2 = (sin(yaw) sigma_pitch)^2.
    sig = np.column_stack((np.broadcast_to(np.reshape(cam.sig, (-1, 1)), (n, 4)), np.maximum(sigma_pitch, 1e-9),
                           np.abs(np.sin(cam.yaw))*sigma_pitch))                                   # (n, k)
    k = len(_STEPS)
    Z = np.column_stack((b, pitch, np.zeros(n)))[:, None]+np.vstack((np.zeros(k), np.diag(_STEPS),
                                                                     -np.diag(_STEPS)))[None]
    hb, wb = (b[:, 3]-b[:, 2])[:, None]/2, (b[:, 1]-b[:, 0])[:, None]/2
    ul, ur, vt, vb, th, rho = np.moveaxis(Z, -1, 0)                                          # (n, 1 + 2k)
    ul, ur, vt, vb = ul-rho*hb, ur+rho*hb, vt-rho*wb, vb+rho*wb
    c1 = cam.cols(1)
    ground, size, lateral = _cue_depths(c1, height, ul, ur, vt, vb, th)
    ground = np.where(fit[:, 3:4], ground, np.nan)
    size = np.where(fit[:, 2:3] & fit[:, 3:4], size, np.nan)
    inv_var = []
    for x in (ground, size):
        var = ((sig*(x[:, 1:1+k]-x[:, 1+k:])/(2*_STEPS))**2).sum(1)
        inv_var.append(np.where(np.isfinite(x[:, 0]) & np.isfinite(var) & (var > 0), 1/np.where(var > 0, var, 1.0), 0.0))
    total = inv_var[0]+inv_var[1]
    w = np.where(total > 0, inv_var[0]/np.where(total > 0, total, 1.0), np.nan)[:, None]
    fallback = c1.f*params.width/np.maximum(b[:, 1]-b[:, 0], 1.0)[:, None]
    x = np.where(w >= 1, ground, np.where(w <= 0, size, w*ground+(1-w)*size))
    x = np.where(np.isfinite(w), x, fallback)                                                # (n, 1 + 2k) depths
    y = lateral(x)
    phi = c1.psi+np.arctan2(y, x)
    cph, sph = np.cos(phi), np.sin(phi)
    rear, front = cg_offsets(params)
    reach = np.hypot(x, y)+np.abs(cph)*params.length/2+np.abs(sph)*params.width/2
    xy = np.stack((c1.x+reach*cph-(rear+front)/2, c1.y+reach*sph), -1)                      # (n, 1 + 2k, 2)
    J = (xy[:, 1:1+k]-xy[:, 1+k:])/(2*_STEPS)[None, :, None]                                  # (n, k, 2)
    P = np.einsum('nki,nk,nkj->nij', J, sig**2, J)
    xy = xy[:, 0]
    o = _origin(cam, n)
    ray, perp = _ray(xy, o)
    swing = (np.abs(ray[:, 0])*-rear+np.abs(ray[:, 1])*params.width/2)*heading_sigma
    across = swing**2+np.where(fit[:, 0] & fit[:, 1], 0.0, (params.length/2)**2)
    depth = np.linalg.norm(xy-o, axis=1)
    along = np.where(np.isfinite(w[:, 0]), 0.0, depth**2)
    P = P+across[:, None, None]*perp[:, :, None]*perp[:, None, :]+along[:, None, None]*ray[:, :, None]*ray[:, None, :]
    bad = ~np.isfinite(P).all((1, 2))
    P[bad] = (depth[bad]**2+params.length**2)[:, None, None]*np.eye(2)
    return xy, (P+np.swapaxes(P, 1, 2))/2


def _origin(cam, n):
    """(n, 2) camera positions of n boxes."""
    return np.column_stack((np.broadcast_to(cam.x, (n,)), np.broadcast_to(cam.y, (n,))))


def _ray(xy, o):
    rel = xy-o
    ray = rel/np.maximum(np.linalg.norm(rel, axis=-1, keepdims=True), 1e-9)
    return ray, np.stack((-ray[:, 1], ray[:, 0]), -1)


def _inv2(A):
    det = A[:, 0, 0]*A[:, 1, 1]-A[:, 0, 1]*A[:, 1, 0]
    return np.stack((np.stack((A[:, 1, 1], -A[:, 0, 1]), -1), np.stack((-A[:, 1, 0], A[:, 0, 0]), -1)), -2)/det[:, None, None]


def _normal(use, J, S, res, prior):
    """Normal equations of the edges in `use` (n, 4): information A (n, 2, 2) with the prior, gradient J' S^-1 res
    (n, 2) and the residual cost (n,)."""
    J = np.where(use[..., None], J, 0.0)
    S = np.where(use[:, :, None] & use[:, None, :], S, np.eye(4)*~use[:, None, :])
    res = np.where(use, res, 0.0)
    X = np.linalg.solve(S, np.concatenate((J, res[..., None]), -1))           # S^-1 [J, res]
    Jt = np.swapaxes(J, 1, 2)
    return Jt@X[..., :2]+prior, (Jt@X[..., 2:])[..., 0], (res*X[..., 2]).sum(1)


def refine_init(boxes, fit, xy0, spec, pose, pitch, params, height, sigma_pitch, heading_sigma=HEADING_SIGMA,
                iters=ITERS):
    """Gauss-Newton refinement of CG estimates xy0 (n, 2) on the projected-box model (module docstring). Returns
    xy (n, 2), P (n, 2, 2) = the inverse information at the lowest-cost iterate, and ok (n,) False where no iterate
    could be evaluated (the caller keeps xy0 there)."""
    z = np.asarray(boxes, float).reshape(-1, 4)
    n = len(z)
    return _refine_init(z, fit, xy0, _cams([spec], [pose], [n]), np.full(n, float(pitch)), params, height,
                        np.full(n, float(sigma_pitch)), heading_sigma, iters)


def _refine_init(z, fit, xy0, cam, pitch, params, height, sigma_pitch, heading_sigma, iters):
    """refine_init with every box's own camera (cam fields (n,)), pitch (n,) and sigma_pitch (n,)."""
    m = np.asarray(fit, bool).reshape(-1, 4)
    n = len(z)
    xy0 = np.asarray(xy0, float)
    sig2 = np.broadcast_to(np.asarray(cam.sig, float)**2, (n,))
    o = _origin(cam, n)
    ray, perp = _ray(xy0, o)
    prior = perp[:, :, None]*perp[:, None, :]/params.length**2+ray[:, :, None]*ray[:, None, :]/PRIOR_ALONG**2
    step = 1e-3*np.maximum(np.linalg.norm(xy0-o, axis=1), 1.0)
    offs = np.zeros((9, 2))
    offs[1:5] = ((1, 0), (-1, 0), (0, 1), (0, -1))
    a = np.array([HEADING_SPREAD, PITCH_SPREAD])
    w = 1/(2*a**2)
    head = np.array([0, 0, 0, 0, 0, a[0]*heading_sigma, -a[0]*heading_sigma, 0, 0], float)
    pit = pitch[:, None]+np.array([0, 0, 0, 0, 0, 0, 0, a[1], -a[1]], float)*sigma_pitch[:, None]      # (n, 9)
    root = np.r_[np.repeat(np.sqrt(w), 2), np.sqrt(1-2*w)][None, :, None]     # weights of the 6 deviation rows
    prep = _prepare(cam.cols(1), head, pit, params, height)
    p = xy0.copy()
    best_p, best_cost, best_A = p.copy(), np.full(n, np.inf), np.tile(np.eye(2), (n, 1, 1))
    done = np.zeros(n, bool)             # converged boxes stop, so a box's result does not depend on its batch
    for k in range(iters+1):
        hb = _evaluate(prep, p[:, None]+offs*step[:, None, None])                                   # (n, 9, 4)
        good = np.isfinite(hb).all((1, 2))
        hb[~good] = 0.0
        h0 = hb[:, 0]
        # Each nuisance q by three points (0, +-a_q sigma_q; weights 1 - 2 w_q, w_q, w_q): the mean box and the
        # edges' covariance about it (the box widens with |heading| either way, a kink at 0, so the mean moves too).
        D = (hb[:, 5:]-h0[:, None]).reshape(n, 2, 2, 4)                         # (n, nuisance, +-, edge)
        mean = w[None, :, None]*D.sum(2)
        rows = np.concatenate(((D-mean[:, :, None]).reshape(n, 4, 4), mean), 1)*root   # (n, 6, 4)
        pred = h0+mean.sum(1)
        J = np.stack((hb[:, 1]-hb[:, 2], hb[:, 3]-hb[:, 4]), -1)/(2*step)[:, None, None]
        S = sig2[:, None, None]*np.eye(4)+np.swapaxes(rows, 1, 2)@rows
        use = m | (_CUT_SIDE*(pred-z) > 0)                                    # usable edges and violated bounds
        A, g, cost = _normal(use, J, S, z-pred, prior)
        # A violated bound steers the estimate, but holds no information: the car reaches past the cut edge by an
        # unknown amount. The covariance comes from the usable edges.
        info = A if (use == m).all() else _normal(m, J, S, z-pred, prior)[0]
        dp0 = p-xy0
        pd = (prior@dp0[..., None])[..., 0]
        cost = cost+(dp0*pd).sum(1)
        better = good & (cost < best_cost) & ~done
        best_p[better], best_cost[better], best_A[better] = p[better], cost[better], info[better]
        if k == iters:
            break
        g = g-pd
        det = A[:, 0, 0]*A[:, 1, 1]-A[:, 0, 1]*A[:, 1, 0]
        dp = np.stack((A[:, 1, 1]*g[:, 0]-A[:, 0, 1]*g[:, 1], A[:, 0, 0]*g[:, 1]-A[:, 1, 0]*g[:, 0]), -1)
        dp = np.where(good[:, None], dp/det[:, None], 0.0)
        small = (dp*g).sum(1) < CONVERGED                                     # dp' A dp
        length = np.linalg.norm(dp, axis=1)
        dp *= np.minimum(1.0, TRUST*np.linalg.norm(p-o, axis=1)/np.maximum(length, 1e-12))[:, None]
        new = small & ~done              # a step under 0.1 sigma: taken without another evaluation, and the box stops
        fin = new & better
        best_p[fin] += dp[fin]
        done |= new
        if done.all():
            break
        p = np.where(done[:, None], p, p+dp)
    P = _inv2(best_A)
    return best_p, (P+np.swapaxes(P, 1, 2))/2, np.isfinite(best_cost)


def pitch_jacobian(boxes, spec):
    """(n, 4) d [u_left, u_right, v_top, v_bottom] / d body pitch of boxes (n, 4), small-angle and from the pixels
    alone: the camera pitches by cos(mount yaw) and rolls by sin(mount yaw) per unit of body pitch
    (geometry.camera_tilt). Pitch moves a pixel by (du, dv) = (-(u - cu)(v - cv) / f, -f (1 + ((v - cv) / f)^2)),
    roll by (v - cv, -(u - cu)); a side edge is taken at the box's middle row, a top or bottom edge at its middle
    column."""
    return _pitch_jacobian(np.asarray(boxes, float).reshape(-1, 4), _cam(spec, (0.0, 0.0, 0.0, 0.0)))


def _pitch_jacobian(b, cam):
    f, cu, cv = cam.f, cam.cu, cam.cv
    a, r = np.cos(cam.yaw), np.sin(cam.yaw)
    um, vm = (b[:, 0]+b[:, 1])/2-cu, (b[:, 2]+b[:, 3])/2-cv
    du = lambda u: -a*(u-cu)*vm/f+r*vm
    dv = lambda v: -a*f*(1+((v-cv)/f)**2)-r*um
    return np.column_stack((du(b[:, 0]), du(b[:, 1]), dv(b[:, 2]), dv(b[:, 3])))


def process_camera(frame, spec, ego, mates, params, car_height, refine=True, pitch_extra=PITCH_EXTRA,
                   heading_sigma=HEADING_SIGMA, iters=ITERS, pitch_noise=None):
    """'camera_box' Detections of CameraFrame `frame` from camera `spec` on the car in EgoState ego (moved to
    frame.time if it is at another time; its pitch is the camera's pitch estimate); mates (k, >= 3) the teammates' CG
    states [x, y, yaw, ...] at frame.time (or None); params the vehicle parameters; car_height every car's height.
    R = sigma_px^2 I + pitch_noise^2 (j j^T + sin(yaw)^2 e e^T), j = pitch_jacobian and e the growth of the box with
    the roll's magnitude (cue_init's rho): the frame's own pitch error (None: spec.pitch_sigma, the per-frame residual)
    moves all four edges together. The braking model's error (pitch_extra) is not white from frame to frame, so it
    widens only the single-frame position (meta['init']).
    meta: init (xy (2,), P (2, 2)), truncated (4,) bool [left, right, top, bottom], score, scan (the frame time)."""
    return process_cameras([frame], [spec], [ego], [mates], params, car_height, refine, pitch_extra, heading_sigma,
                           iters, pitch_noise)[0]


def process_cameras(frames, specs, egos, mates, params, car_height, refine=True, pitch_extra=PITCH_EXTRA,
                    heading_sigma=HEADING_SIGMA, iters=ITERS, pitch_noise=None):
    """process_camera of several frames at once (frames[k] from specs[k] on the car in egos[k], its teammates
    mates[k]): one list of Detections per frame, as frame by frame. The geometric init and its refinement run over
    every box of every frame in one batch: compiled (_camera_jit.process_boxes) when numba is on, else in numpy
    (whose cost per call, not per box, dominates: a sim step's twelve frames cost about what one did)."""
    args = (frames, specs, [ego_at(e, fr.time) for e, fr in zip(egos, frames)], mates, params, car_height, refine,
            pitch_extra, heading_sigma, iters, pitch_noise)
    return _process_compiled(*args) if _JIT else _process_numpy(*args)


def _detections(frames, egos, poses, frame_of, local, b, xy, P, truncated, R):
    """process_cameras' Detections, one list per frame: row r of the kept boxes b (n, 4) is box local[r] of frame
    frame_of[r] (rows in frame order)."""
    out = [[] for _ in frames]
    scores = [None]*len(frames)
    for r, (f, k) in enumerate(zip(frame_of, local)):
        frame = frames[f]
        if scores[f] is None:
            scores[f] = np.ones(len(frame.boxes)) if frame.score is None else np.asarray(frame.score, float).reshape(-1)
        t = float(frame.time)
        out[f].append(Detection('camera_box', frame.sensor, frame.car, t, b[r], R[r], poses[f], egos[f],
                                {'init': (xy[r], P[r]), 'truncated': truncated[r], 'score': float(scores[f][k]),
                                 'scan': t}))
    return out


def _process_compiled(frames, specs, egos, mates, params, car_height, refine, pitch_extra, heading_sigma, iters,
                      pitch_noise):
    """process_cameras by _camera_jit.process_boxes (egos already at the frame times)."""
    poses = [sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), params) for spec, ego in zip(specs, egos)]
    boxes = [np.asarray(fr.boxes, float).reshape(-1, 4) for fr in frames]
    counts = [len(b) for b in boxes]
    if sum(counts) == 0:
        return [[] for _ in frames]
    cams = np.array([_cam(s, p) for s, p in zip(specs, poses)], float).reshape(-1, len(_Cam._fields))
    mt = [np.asarray(m, float).reshape(len(m), -1)[:, :3] if m is not None and len(m) else np.zeros((0, 3))
          for m in mates]
    mate_start = np.cumsum([0]+[len(m) for m in mt]).astype(np.int64)
    pitch = np.array([float(e.pitch) for e in egos])
    sp = cams[:, _camera_jit.PSIG].copy() if pitch_noise is None else np.full(len(frames), float(pitch_noise))
    rear, front = cg_offsets(params)
    car = np.array([params.length, params.width, rear, front], float)
    tune = np.array([globals()[name] for name in _camera_jit.TUNE_FIELDS], float)   # this module's, read now
    b = np.concatenate(boxes)
    frame = np.repeat(np.arange(len(frames), dtype=np.int64), counts)
    keep, truncated, xy, P, R = _camera_jit.process_boxes(
        b, frame, cams, pitch, sp, np.ascontiguousarray(np.concatenate(mt)), mate_start,
        _local_corners(params, car_height), car, float(car_height), float(heading_sigma), float(pitch_extra),
        bool(refine), int(iters), _EDGES, _STEPS, _CUT_SIDE, tune)
    idx = np.flatnonzero(keep)
    start = np.cumsum(counts)-counts
    return _detections(frames, egos, poses, frame[idx].tolist(), (idx-start[frame[idx]]).tolist(), b[idx], xy[idx],
                       P[idx], truncated[idx], R[idx])


def _process_numpy(frames, specs, egos, mates, params, car_height, refine, pitch_extra, heading_sigma, iters,
                   pitch_noise):
    """process_cameras in numpy (egos already at the frame times)."""
    poses, boxes, keeps = [], [], []
    for frame, spec, ego, mt in zip(frames, specs, egos, mates):
        pose = sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), params)
        b = np.asarray(frame.boxes, float).reshape(-1, 4)
        keep = np.isfinite(b).all(1) & (b[:, 1] > b[:, 0]) & (b[:, 3] > b[:, 2])
        if mt is not None and len(mt) and keep.any():
            mb = mate_boxes(spec, pose, mt, params, car_height, ego.pitch)
            if len(mb):
                keep &= ~(iou(b, mb) > IOU_MATE).any(1)
        poses.append(pose)
        boxes.append(b)
        keeps.append(np.flatnonzero(keep))
    counts = [len(k) for k in keeps]
    if sum(counts) == 0:
        return [[] for _ in frames]
    b = np.concatenate([bx[k] for bx, k in zip(boxes, keeps)])
    cam = _cams(specs, poses, counts)
    pitch = np.repeat([float(e.pitch) for e in egos], counts)
    cut = lambda px: np.column_stack((b[:, 0] <= px, b[:, 1] >= cam.W-px, b[:, 2] <= px, b[:, 3] >= cam.H-px))
    truncated = cut(TRUNC_PX)
    fit = ~cut(np.maximum(TRUNC_PX, FIT_MARGIN_SIGMAS*cam.sig))
    sigma_pitch = cam.pitch_sigma+pitch_extra
    xy, P = _cue_init(b, fit, cam, pitch, params, car_height, sigma_pitch, heading_sigma)
    if refine:
        xr, Pr, ok = _refine_init(b, fit, xy, cam, pitch, params, car_height, sigma_pitch, heading_sigma, iters)
        xy[ok], P[ok] = xr[ok], Pr[ok]
    # A detector box is the projected box's extent clipped to the image, so the pitch moves an uncut edge with the
    # extent of the whole projected box: past a cut edge, that extent is the model box at the init position.
    full = b.copy()
    if not fit.all():
        model = _evaluate(_prepare(cam, 0.0, pitch, params, car_height), xy)
        far = ~fit & np.isfinite(model) & (_CUT_SIDE*(model-b) < 0)
        full[far] = model[far]
    j = _pitch_jacobian(full, cam)
    sp = cam.pitch_sigma if pitch_noise is None else np.full(len(b), float(pitch_noise))
    spread = np.column_stack((full[:, 2]-full[:, 3], full[:, 3]-full[:, 2], full[:, 0]-full[:, 1],
                              full[:, 1]-full[:, 0]))/2                                         # cue_init's rho
    R = (cam.sig[:, None, None]**2*np.eye(4)+(sp**2)[:, None, None]*j[:, :, None]*j[:, None, :]
         + ((np.sin(cam.yaw)*sp)**2)[:, None, None]*spread[:, :, None]*spread[:, None, :])
    return _detections(frames, egos, poses, np.repeat(np.arange(len(frames)), counts).tolist(),
                       np.concatenate(keeps).tolist(), b, xy, P, truncated, R)
