"""Measurement-to-track association for the team tracker (design section 6): the per-detection table of a fusion
cycle, the measurement models evaluated for many (track, detection) pairs at once, the IMM gate of each pair, a coarse
position pre-gate and global nearest neighbour per scan.

One numpy call costs a fraction of a millisecond whatever its size, so the pairs of every sensor and kind of a fusion
cycle go through each step in one call. Every measurement is padded to M = 4 columns (lidar and radar have 3): a padded
column has z = 0, h = 0 and R = 1, so it adds nothing to an update or a NIS and a known -log(2 pi)/2 to a
log-likelihood (DetTable.ll_pad takes it back out).

API (the tracker builds on it):
- LIDAR, RADAR, CAMERA: kind codes; KIND_CODE {Detection kind: code}; M = 4.
- gate_threshold(gate, dof): chi-square quantile (inf for dof 0). wrap_angle(a): geometry.wrap, faster.
- h_radar_rows(X, origin, yaw, v_sensor, params) and h_camera_rows(X, pose, cam, params, car_height): models.h_radar
  and models.h_camera with their own sensor per row (cam = [width_px, height_px, focal, body pitch, mount yaw]), so
  the pairs of many scans share one call. Equal to the models functions row by row.
- DetTable.build(scans, params): per-detection arrays of a fusion cycle, scans a list of (detections, spec, log clutter
  density), with a dummy last row (index len(table)): time, kind, scan (index into scans), z (D, 4), R (D, 4, 4),
  R_fit (R where a lidar box carries its velocity model, meta 'R_fit': what the gate and update use), the lidar
  velocity model v_prior (D, 2) and dz_dv (D, 3, 2) (zeros without it), m (real dims), dof (dims that carry
  information: a camera edge cut by the image border does not), wrap (angle columns), sensor pose (D, 4), radar sensor
  velocity vs (D, 2), camera cam (D, 5),
  the coarse CG position xy (D, 2) with covariance C (D, 2, 2) and has_xy, n_points, log_clutter (the clutter density
  of the informative dims: the scan's density times the span of each cut edge, existence.clutter_log_volumes) and
  ll_pad (added to a pair's log-likelihood, it leaves the informative dims: the padded columns' log(2 pi)/2 and the
  cut edges' log(2 pi TRUNCATED_VAR)/2); dets and per scan its spec. Camera R is models.camera_R's: each cut edge
  decoupled with variance TRUNCATED_VAR (meta['truncated'] as camera_R reads it).
- evaluate(table, dj, X, params, car_height, offsets=None, motions=None) -> Z (P, J, K, M): the measurement model of
  detection dj[p] on sigma points X (P, J, K, 4) (J IMM models of K points each); a lidar box is models.h_lidar plus
  dz_dv (v_prior - the point's velocity), the box fit's first-order dependence on the velocity its points were moved
  with (processing.lidar.fit_box), so a car's relative speed is modelled, not taken as white noise. offsets (P,): each
  point is first
  moved that far in time by its own model's motion (motions, MODELS order), for a detection stamped off the time its
  track was predicted to (a lidar box at its mean beam time).
- moments(table, dj, Z, wm, wc) -> Moments(zhat, nu, Pzz, D), each (P, J, ...): the unscented predicted measurement
  (taken relative to the centre point, angle columns wrapped, as ukf.ukf_update), the innovation, its spread and
  D (n, M), the statistical linearization of the measurement in the sigma points' whitened coordinates: with
  x = mean + L xi (L the sigma points' Cholesky factor), z ~ zhat + D^T xi + e, Pxz = L D and cov(e) = Pzz - D^T D
  (positive semi-definite). The tracker's joint update builds on D.
- imm_gate(mo, R, mu, table, dj) -> (loglik, nis, logdet_s): per pair, the IMM mixture's log-likelihood
  log sum_j mu_j N(nu_j; 0, Pzz_j + R) and the NIS and log det S of the moment-matched predicted measurement (mean
  sum_j mu_j zhat_j, covariance sum_j mu_j (S_j + spread of the means)): the usual IMM gate.
- coarse_candidates(table, mean, cov, t, cfg=None) -> (track, det) index arrays: pairs whose rough positions agree
  (CV-predicted track CG at each detection's time, with cfg's process noise over the offset, against the
  detection's rough CG, 2D Mahalanobis with margins),
  radar pairs also on range rate (the track's velocity relative to the sensor along the measured line of sight);
  camera detections only for tracks in front of the camera (its measurement model is meaningless behind it).
- gnn(cost) -> (rows, cols): the assignment with the most finite (gated) pairs, then the least total cost
  (scipy.optimize.linear_sum_assignment); inf entries are never assigned.
- cholesky(S), forward(L, v), logdet(L): helpers for stacks of many small matrices (vectorized over the stack, several
  times faster than numpy.linalg's per-matrix calls): the Cholesky factor (ukf.robust_cholesky for the rare matrix
  with a non-positive pivot), L^-1 v and log det(L L^T).
"""
from functools import lru_cache
from typing import NamedTuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2
from ...geometry import cg_offsets
from ..geometry import camera_tilt, focal
from .existence import clutter_log_volumes
from .imm import MU_FLOOR
from .models import BORDER_PX, NEAR, TRUNCATED_VAR, h_lidar, sensor_velocity, V_MIN
from .ukf import robust_cholesky

__all__ = ['LIDAR', 'RADAR', 'CAMERA', 'KIND_CODE', 'M', 'COARSE_GATE', 'MARGIN', 'wrap_angle', 'gate_threshold',
           'h_radar_rows', 'h_camera_rows', 'DetTable', 'Moments', 'evaluate', 'moments', 'imm_gate',
           'coarse_candidates', 'gnn', 'cholesky', 'forward', 'logdet']

LIDAR, RADAR, CAMERA = 0, 1, 2
KIND_CODE = {'lidar_box': LIDAR, 'radar': RADAR, 'camera_box': CAMERA}
KIND_DIMS = np.array([3, 3, 4, 0])    # by code; the dummy row (code -1) has none
M = 4
LOG2PI = float(np.log(2*np.pi))
COARSE_GATE = 18.42                   # chi-square(2) 99.99 %: the coarse gate only prunes
COARSE_GATE_1D = 15.14                # chi-square(1) 99.99 %: the radar range-rate test
MARGIN = (0.5, 1.0, 1.0)              # m added (as sigma) to the coarse gate per kind: box fit, near-surface
                                      # points (radar), the single-frame camera position
RANGE_RATE_MARGIN = 1.0               # m/s added (as sigma) to the radar range-rate test
CAMERA_MIN_DEPTH = 1.0                # m in front of a camera for a pair to be gated
CAMERA_FOV_MARGIN = np.radians(15.0)  # camera detections without a position gate tracks this far past the view
_BIG = 1e12
_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6),
                   (3, 7)])           # the 3D box's edges (geometry.box_corners_3d order)
_SIGNS = np.array([(-1, -1), (1, -1), (1, 1), (-1, 1)], float)   # footprint corners (geometry.car_boxes order)


def wrap_angle(a):
    """geometry.wrap, [-pi, pi), with floor instead of the float modulo (several times faster on large arrays)."""
    a = np.asarray(a, float)
    return a-2*np.pi*np.floor((a+np.pi)/(2*np.pi))


@lru_cache(maxsize=64)
def gate_threshold(gate, dof):
    """chi-square quantile of probability `gate` with dof degrees of freedom (inf for dof <= 0: no information)."""
    return float(chi2.ppf(gate, dof)) if dof > 0 else np.inf


def _body(X, params):
    """Body centre (k, 2), heading (cos, sin) (k,) of track states X (k, 4) (models.heading, nohesi body_centre)."""
    vs = np.maximum(X[:, 2], V_MIN)
    n = np.hypot(vs, X[:, 3])
    c, s = vs/n, X[:, 3]/n
    shift = sum(cg_offsets(params))/2
    return X[:, 0]+shift*c, X[:, 1]+shift*s, c, s


def h_radar_rows(X, origin, yaw, v_sensor, params):
    """[range, azimuth, range_rate] (k, 3) of sigma points X (k, 4), each seen from its own radar: origin (k, 2),
    boresight yaw (k,), sensor world velocity (k, 2) (models.h_radar per row). The box point nearest the sensor is the
    sensor's position in the box frame clamped to the box (or moved to the nearest side from inside it)."""
    X = np.asarray(X, float)
    bx, by, c, s = _body(X, params)
    ex, ey = origin[:, 0]-bx, origin[:, 1]-by
    lx, ly = c*ex+s*ey, -s*ex+c*ey
    hl, hw = params.length/2, params.width/2
    px, py = np.clip(lx, -hl, hl), np.clip(ly, -hw, hw)
    inside = (np.abs(lx) < hl) & (np.abs(ly) < hw)
    if inside.any():
        side = hl-np.abs(lx[inside]) <= hw-np.abs(ly[inside])
        px[inside] = np.where(side, np.copysign(hl, lx[inside]), lx[inside])
        py[inside] = np.where(side, ly[inside], np.copysign(hw, ly[inside]))
    rx, ry = bx+c*px-s*py-origin[:, 0], by+s*px+c*py-origin[:, 1]
    rng = np.hypot(rx, ry)
    az = wrap_angle(np.arctan2(ry, rx)-yaw)
    rr = ((X[:, 2]-v_sensor[:, 0])*rx+(X[:, 3]-v_sensor[:, 1])*ry)/np.maximum(rng, 1e-9)
    return np.column_stack((rng, az, rr))


def h_camera_rows(X, pose, cam, params, car_height):
    """[u_left, u_right, v_top, v_bottom] (k, 4) of sigma points X (k, 4), each seen by its own camera: pose (k, 4)
    (x, y, yaw, z), cam (k, 5) [width_px, height_px, focal, body pitch, mount yaw] (models.h_camera per row, the
    pitch through geometry.camera_tilt; the near-plane cut is
    computed only for rows with a corner behind it). Corner-major arrays (8, k), so the extents reduce over a short
    leading axis."""
    X = np.asarray(X, float)
    bx, by, c, s = _body(X, params)
    cy, sy = np.cos(pose[:, 2]), np.sin(pose[:, 2])
    ex, ey = bx-pose[:, 0], by-pose[:, 1]
    x0, y0 = cy*ex+sy*ey, -sy*ex+cy*ey                      # body centre in the camera's level frame
    cr, sr = c*cy+s*sy, s*cy-c*sy                           # heading relative to the camera
    hl, hw = params.length/2, params.width/2
    sx, sw = _SIGNS[:, :1], _SIGNS[:, 1:]
    fx = x0+sx*(hl*cr)-sw*(hw*sr)                            # footprint (4, k)
    fy = y0+sx*(hl*sr)+sw*(hw*cr)
    T = camera_tilt(cam[:, 4], cam[:, 3])                    # (k, 3, 3), geometry.project's R^T p
    z0, z1 = -pose[:, 3], car_height-pose[:, 3]
    col = lambda j: np.concatenate((T[:, 0, j]*fx+T[:, 1, j]*fy+T[:, 2, j]*z0,
                                    T[:, 0, j]*fx+T[:, 1, j]*fy+T[:, 2, j]*z1))   # (8, k): floor, then roof
    xc, yc, zc = col(0), col(1), col(2)
    W, Hh, f = cam[:, 0], cam[:, 1], cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        inv = f/xc
        U, V = W/2-yc*inv, Hh/2-zc*inv
    box = np.column_stack((U.min(0), U.max(0), V.min(0), V.max(0)))
    ok = xc > NEAR
    cut = np.flatnonzero(~ok.all(0))
    if cut.size:
        a, b = _EDGES.T
        xk, yk, zk = xc[:, cut], yc[:, cut], zc[:, cut]
        xa, xb = xk[a], xk[b]
        cross = (xa-NEAR)*(xb-NEAR) < 0
        t = np.where(cross, (NEAR-xa)/np.where(cross, xb-xa, 1.0), 0.0)
        ye, ze, xe = yk[a]+t*(yk[b]-yk[a]), zk[a]+t*(zk[b]-zk[a]), xa+t*(xb-xa)
        with np.errstate(divide='ignore', invalid='ignore'):
            ue, ve = W[cut]/2-f[cut]*ye/xe, Hh[cut]/2-f[cut]*ze/xe
        okk = np.concatenate((ok[:, cut], cross))
        Uk, Vk = np.concatenate((U[:, cut], ue)), np.concatenate((V[:, cut], ve))
        box[cut] = np.column_stack((np.where(okk, Uk, np.inf).min(0), np.where(okk, Uk, -np.inf).max(0),
                                    np.where(okk, Vk, np.inf).min(0), np.where(okk, Vk, -np.inf).max(0)))
        none = cut[~okk.any(0)]
        box[none] = np.column_stack((W[none]/2, W[none]/2, Hh[none]/2, Hh[none]/2))
    return np.minimum(np.maximum(box, 0.0), np.column_stack((W, W, Hh, Hh)))


def _truncation(near, dets):
    """(k, 4) bool truncated edges of camera detections, as models.camera_R reads them: meta['truncated'] (4 flags, or
    one bool meaning "touches the border"), else near, the edges within BORDER_PX of the border."""
    flags = [d.meta.get('truncated') if d.meta else None for d in dets]
    if all(f is not None and np.size(f) == 4 for f in flags):
        return np.array(flags, bool).reshape(len(dets), 4)
    out = near.copy()
    for n, f in enumerate(flags):
        if f is not None:
            out[n] = (near[n] if bool(f) else False) if np.ndim(f) == 0 else np.asarray(f, bool).reshape(4)
    return out


class DetTable:
    """Per-detection arrays of a fusion cycle (module docstring)."""

    def __init__(self, **arrays):
        self.__dict__.update(arrays)

    def __len__(self):
        return len(self.dets)

    @classmethod
    def build(cls, scans, params):
        """The table of a cycle's scans [(detections, spec, log clutter density), ...]."""
        dets = [d for ds, _, _ in scans for d in ds]
        D = len(dets)
        sizes = [len(ds) for ds, _, _ in scans]
        scan = np.repeat(np.arange(len(scans)), sizes)
        kind = np.append(np.repeat([KIND_CODE[ds[0].kind] if ds else LIDAR for ds, _, _ in scans], sizes), -1)
        m = KIND_DIMS[kind]
        z, R = np.zeros((D+1, M)), np.tile(np.eye(M), (D+1, 1, 1))
        pose, vs, cam = np.zeros((D+1, 4)), np.zeros((D+1, 2)), np.zeros((D+1, 5))
        xy, C, has = np.zeros((D+1, 2)), np.tile(np.eye(2), (D+1, 1, 1)), np.zeros(D+1, bool)
        wrapc = np.zeros((D+1, M), bool)
        clut = np.zeros(D+1)
        if D:
            for code, dim in ((LIDAR, 3), (RADAR, 3), (CAMERA, 4)):
                sel = np.flatnonzero(kind[:D] == code)
                if sel.size:
                    z[sel, :dim] = np.array([dets[j].z for j in sel], float)[:, :dim]
                    R[sel, :dim, :dim] = np.array([dets[j].R for j in sel], float)
            pose[:D] = np.array([d.sensor_pose for d in dets], float)
            has[:D] = True
            clut[:D] = np.repeat([lc for _, _, lc in scans], sizes)
        rear, front = cg_offsets(params)
        shift = (rear+front)/2
        lid, rad, cmr = (np.flatnonzero(kind == code) for code in (LIDAR, RADAR, CAMERA))
        vp, dzdv, lin = np.zeros((D+1, 2)), np.zeros((D+1, 3, 2)), []
        if lid.size:
            wrapc[lid, 2] = True
            h = z[lid, 2]
            xy[lid] = z[lid, :2]-shift*np.column_stack((np.cos(h), np.sin(h)))
            C[lid] = R[lid, :2, :2]+MARGIN[LIDAR]**2*np.eye(2)
            lin = [j for j in lid if dets[j].meta and 'dz_dv' in dets[j].meta]   # the dummy row is no lidar
            if lin:
                vp[lin] = np.array([dets[j].meta['v_prior'] for j in lin], float)
                dzdv[lin] = np.array([dets[j].meta['dz_dv'] for j in lin], float)
        if rad.size:
            wrapc[rad, 1] = True
            vel = {s: sensor_velocity(scans[s][0][0]) for s in np.unique(scan[rad])}
            vs[rad] = np.array([vel[s] for s in scan[rad]])
            r, phi = z[rad, 0], pose[rad, 2]+z[rad, 1]
            u = np.column_stack((np.cos(phi), np.sin(phi)))
            ext = np.abs(u[:, 0])*params.length/2+np.abs(u[:, 1])*params.width/2       # near face -> centre
            xy[rad] = pose[rad, :2]+u*(r+ext)[:, None]
            xy[rad, 0] -= shift
            J = np.stack((u, np.column_stack((-r*u[:, 1], r*u[:, 0]))), -1)          # d xy / d (range, azimuth)
            C[rad] = J @ R[rad, :2, :2] @ np.swapaxes(J, -1, -2)+MARGIN[RADAR]**2*np.eye(2)
        dof = m.copy()
        ll_pad = 0.5*(M-m)*LOG2PI
        if cmr.size:
            cs = np.unique(scan[cmr])
            per = np.searchsorted(cs, scan[cmr])
            sp = [scans[k][1] for k in cs]
            ego = [scans[k][0][0].ego for k in cs]
            cam[cmr] = np.array([(c.width_px, c.height_px, focal(c), 0.0 if e is None else float(e.pitch), c.mount.yaw)
                                 for c, e in zip(sp, ego)])[per]
            W, Hh = cam[cmr, 0], cam[cmr, 1]
            near = np.column_stack((z[cmr, 0] <= BORDER_PX, z[cmr, 1] >= W-BORDER_PX, z[cmr, 2] <= BORDER_PX,
                                    z[cmr, 3] >= Hh-BORDER_PX))
            cdets = [dets[j] for j in cmr]
            cut = _truncation(near, cdets)
            keep = ~cut
            R[cmr] *= keep[:, :, None] & keep[:, None, :]
            R[cmr] += np.eye(4)*np.where(cut, TRUNCATED_VAR, 0.0)[:, None, :]
            dof[cmr] = keep.sum(1)
            ll_pad[cmr] += 0.5*cut.sum(1)*np.log(2*np.pi*TRUNCATED_VAR)
            clut[cmr] += (cut*np.array([clutter_log_volumes(c) for c in sp])[per]).sum(1)
            inits = [d.meta.get('init') if d.meta else None for d in cdets]
            got = np.array([i is not None for i in inits])
            has[cmr[~got]] = False
            if got.any():
                xy[cmr[got]] = np.array([i[0] for i in inits if i is not None], float)
                C[cmr[got]] = np.array([i[1] for i in inits if i is not None], float)+MARGIN[CAMERA]**2*np.eye(2)
        Rf = R.copy()                 # what the gate and the update use: a lidar box's R without its velocity prior
        if lin:
            Rf[lin, :3, :3] = np.array([dets[j].meta['R_fit'] for j in lin], float)
        npts = np.array([int(d.meta.get('n_points', 0)) if d.meta else 0 for d in dets]+[0], int)
        return cls(time=np.append(np.array([d.time for d in dets], float), 0.0), kind=kind, scan=np.append(scan, -1),
                   z=z, R=R, R_fit=Rf, v_prior=vp, dz_dv=dzdv, m=m, dof=dof, wrap=wrapc, ll_pad=ll_pad, pose=pose,
                   vs=vs, cam=cam, xy=xy, C=C, has_xy=has, n_points=npts, log_clutter=clut, dets=dets,
                   spec=[sp for _, sp, _ in scans])


def evaluate(table, dj, X, params, car_height, offsets=None, motions=None):
    """Z (P, J, K, M): detection dj[p]'s measurement model on sigma points X[p] (P, J, K, 4), padded columns 0;
    with offsets (P,), model j's points first moved by motions[j] over that time."""
    dj = np.asarray(dj, int).reshape(-1)
    X = np.asarray(X, float)
    P, J, K, _ = X.shape
    Z = np.zeros((P, J, K, M))
    if offsets is not None:
        off = np.asarray(offsets, float).reshape(-1)
        move = np.flatnonzero(off != 0)
        if move.size:
            X = X.copy()
            dt = np.repeat(off[move], K)
            for j, f in enumerate(motions):
                X[move, j] = f(X[move, j].reshape(-1, 4), dt).reshape(len(move), K, 4)
    kind = table.kind[dj]
    for code in (LIDAR, RADAR, CAMERA):
        p = np.flatnonzero(kind == code)
        if p.size == 0:
            continue
        d, n, dim = dj[p], J*K, KIND_DIMS[code]
        Xs = X[p].reshape(-1, 4)
        if code == LIDAR:
            Y = h_lidar(Xs, params)+np.einsum('kij,kj->ki', np.repeat(table.dz_dv[d], n, 0),
                                              np.repeat(table.v_prior[d], n, 0)-Xs[:, 2:4])
        elif code == RADAR:
            Y = h_radar_rows(Xs, np.repeat(table.pose[d, :2], n, 0), np.repeat(table.pose[d, 2], n),
                             np.repeat(table.vs[d], n, 0), params)
        else:
            Y = h_camera_rows(Xs, np.repeat(table.pose[d], n, 0), np.repeat(table.cam[d], n, 0), params, car_height)
        Z[p, ..., :dim] = Y.reshape(len(p), J, K, dim)
    return Z


class Moments(NamedTuple):
    """Per pair and IMM model: predicted measurement zhat (P, J, M), innovation nu (P, J, M), spread Pzz
    (P, J, M, M) and whitened linearization D (P, J, n, M) (module docstring)."""
    zhat: np.ndarray
    nu: np.ndarray
    Pzz: np.ndarray
    D: np.ndarray


def _res(a, b, mask):
    """a - b with the columns in mask (broadcast against the result) wrapped where they leave [-pi, pi)."""
    r = np.subtract(a, b)
    out = (r < -np.pi) | (r >= np.pi)
    if out.any():
        r = np.where(mask & out, wrap_angle(r), r)
    return r


def cholesky(S):
    """Lower Cholesky factors (..., n, n) of symmetric positive definite matrices S, column by column in vectorized
    numpy: for many small matrices several times faster than LAPACK's one call per matrix. Matrices with a pivot that
    is not positive (round-off on a nearly singular one) are redone by ukf.robust_cholesky."""
    S = np.asarray(S, float)
    n = S.shape[-1]
    c = [[None]*n for _ in range(n)]
    bad = np.zeros(S.shape[:-2], bool)
    for j in range(n):
        d = S[..., j, j]
        for k in range(j):
            d = d-c[j][k]*c[j][k]
        bad |= ~(d > 0)
        c[j][j] = np.sqrt(np.where(d > 0, d, 1.0))
        inv = 1/c[j][j]
        for i in range(j+1, n):
            v = S[..., i, j]
            for k in range(j):
                v = v-c[i][k]*c[j][k]
            c[i][j] = v*inv
    L = np.zeros(S.shape)
    for i in range(n):
        for j in range(i+1):
            L[..., i, j] = c[i][j]
    if bad.any():
        L[bad] = robust_cholesky(S[bad])
    return L


def forward(L, v):
    """L^-1 v for lower triangular L (..., n, n): v (..., n), or (..., n, r) for r columns at once."""
    n = L.shape[-1]
    v = np.asarray(v, float)
    cols = v.ndim == L.ndim
    batch = np.broadcast_shapes(L.shape[:-2], v.shape[:-2] if cols else v.shape[:-1])
    out = np.empty(batch+(v.shape[-2:] if cols else v.shape[-1:]))
    for i in range(n):
        if cols:
            acc = v[..., i, :]
            for k in range(i):
                acc = acc-L[..., i, k, None]*out[..., k, :]
            out[..., i, :] = acc/L[..., i, i, None]
        else:
            acc = v[..., i]
            for k in range(i):
                acc = acc-L[..., i, k]*out[..., k]
            out[..., i] = acc/L[..., i, i]
    return out


def logdet(L):
    """log det of L L^T from Cholesky factors L (..., n, n)."""
    return 2*np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)


def moments(table, dj, Z, wm, wc):
    """Moments of the images Z (P, J, K, M) of each pair's sigma points (weights wm, wc of ukf.sigma_points)."""
    dj = np.asarray(dj, int).reshape(-1)
    w4, w3 = table.wrap[dj][:, None, None, :], table.wrap[dj][:, None, :]
    ref = Z[..., :1, :]
    zhat = ref[..., 0, :]+wm @ _res(Z, ref, w4)
    dZ = _res(Z, zhat[..., None, :], w4)
    Pzz = np.swapaxes(dZ*wc[:, None], -1, -2) @ dZ
    n = (Z.shape[-2]-1)//2
    D = (dZ[..., 1:1+n, :]-dZ[..., 1+n:, :])*np.sqrt(wm[1]/2)     # (a_i - b_i) / (2 sqrt(c)), wm_i = 1 / (2 c)
    return Moments(zhat, _res(table.z[dj][:, None, :], zhat, w3), Pzz, D)


def imm_gate(mo, R, mu, table, dj):
    """(mixture log-likelihood, NIS, log det S) of every pair (module docstring); R (P, M, M), mu (P, J)."""
    dj = np.asarray(dj, int).reshape(-1)
    w2 = table.wrap[dj]
    S = mo.Pzz+R[:, None]
    L = cholesky(S)
    w = forward(L, mo.nu)
    a = np.log(np.maximum(mu, MU_FLOOR))-0.5*((w*w).sum(-1)+logdet(L)+M*LOG2PI)
    top = a.max(-1)
    ll = top+np.log(np.exp(a-top[:, None]).sum(-1))
    zc = mo.zhat[:, 0]+(mu[..., None]*_res(mo.zhat, mo.zhat[:, :1], w2[:, None])).sum(1)
    dz = _res(mo.zhat, zc[:, None], w2[:, None])
    Sc = (mu[..., None, None]*S).sum(1)+np.swapaxes(dz*mu[..., None], 1, 2) @ dz
    Lc = cholesky(Sc)
    wz = forward(Lc, _res(table.z[dj], zc, w2))
    return ll, (wz*wz).sum(-1), logdet(Lc)


def coarse_candidates(table, mean, cov, t, cfg=None):
    """(track, det) index arrays of the pairs worth gating (module docstring). mean (n, 4), cov (n, 4, 4) the tracks'
    combined states at times t (n,); cfg the TrackerConfig whose white-acceleration noise (q_long, q_lat) is added
    over each pair's time offset, as the full gate's prediction adds it (without it a track that went unseen keeps
    its last posterior here and its car, found again, never reaches the full gate)."""
    D, n = len(table), len(mean)
    if D == 0 or n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    dt = table.time[None, :D]-t[:, None]                                           # (n, D)
    ex = mean[:, None, 0]+mean[:, None, 2]*dt-table.xy[None, :D, 0]
    ey = mean[:, None, 1]+mean[:, None, 3]*dt-table.xy[None, :D, 1]
    c = cov[:, None]
    dt2 = dt*dt
    ql, qd = (0.0, 0.0) if cfg is None else (float(cfg.q_long)**2, float(cfg.q_lat)**2)
    adt = np.abs(dt)
    a = c[..., 0, 0]+2*dt*c[..., 0, 2]+dt2*c[..., 2, 2]+table.C[None, :D, 0, 0]+ql*adt**3/3
    b = c[..., 0, 1]+dt*(c[..., 0, 3]+c[..., 2, 1])+dt2*c[..., 2, 3]+table.C[None, :D, 0, 1]
    d = c[..., 1, 1]+2*dt*c[..., 1, 3]+dt2*c[..., 3, 3]+table.C[None, :D, 1, 1]+qd*adt**3/3
    m2 = (d*ex*ex-2*b*ex*ey+a*ey*ey)/(a*d-b*b)
    ok = np.where(table.has_xy[None, :D], m2 <= COARSE_GATE, True)
    rad = np.flatnonzero(table.kind[:D] == RADAR)
    if rad.size:
        phi = table.pose[rad, 2]+table.z[rad, 1]
        u = np.column_stack((np.cos(phi), np.sin(phi)))
        rr = mean[:, 2:4] @ u.T-np.einsum('rj,rj->r', table.vs[rad], u)[None, :]
        Pv = cov[:, 2:, 2:]
        var = (Pv[:, None, 0, 0]*u[None, :, 0]**2+2*Pv[:, None, 0, 1]*u[None, :, 0]*u[None, :, 1]
               + Pv[:, None, 1, 1]*u[None, :, 1]**2+table.R[rad, 2, 2][None, :]+RANGE_RATE_MARGIN**2
               + adt[:, rad]*(ql*u[None, :, 0]**2+qd*u[None, :, 1]**2))
        ok[:, rad] &= (rr-table.z[rad, 2][None, :])**2 <= COARSE_GATE_1D*var
    cam = np.flatnonzero(table.kind[:D] == CAMERA)
    if cam.size:
        pose = table.pose[cam]
        px, py = ex[:, cam]+table.xy[None, cam, 0]-pose[None, :, 0], ey[:, cam]+table.xy[None, cam, 1]-pose[None, :, 1]
        c_, s_ = np.cos(pose[:, 2]), np.sin(pose[:, 2])
        depth = px*c_+py*s_
        lat = -px*s_+py*c_
        half = np.array([table.spec[k].hfov/2 for k in table.scan[cam]])+CAMERA_FOV_MARGIN
        ok[:, cam] &= (depth > CAMERA_MIN_DEPTH) & (np.abs(np.arctan2(lat, depth)) <= half)
    ok &= table.dof[None, :D] > 0
    return np.nonzero(ok)


def gnn(cost):
    """Rows and columns of the assignment with the most finite entries, then the least total cost."""
    cost = np.asarray(cost, float)
    finite = np.isfinite(cost)
    if not finite.any():
        return np.zeros(0, int), np.zeros(0, int)
    r, c = linear_sum_assignment(np.where(finite, cost, _BIG))
    keep = finite[r, c]
    return r[keep], c[keep]
