"""Lidar processing: one sweep -> 'lidar_box' detections for the tracker (design section 6).

1. De-skew: each beam's point moved into the sensor frame at the scan end for the carrying car's own motion (constant
   twist, or the start, mid and end poses interpolated when the motion is far from constant).
2. Into the world frame with the sensor pose at the scan end; points inside a teammate's V2V box and points beyond the
   road edges are dropped.
3. Segmentation in scan order with adaptive breakpoints (Borges and Aldon 2004), plus a free-space rule for runs of
   empty beams that dropout alone would rarely give. Segments split by a nearer occluder, or by a grazing view of one
   face, are merged when together they fit one car.
4. Box fit per cluster (3 or more points) with the known car size: L-shape search (Zhang et al. 2017, variance
   criterion) over headings within 20 deg of the road, anchored on the faces and the free ends that are seen;
   optionally refined by point-to-line ICP (Gauss-Newton in x, y, heading, with a 5 deg road-heading prior) after RANSAC
   line fits. The covariance comes from the same residual model: point count, range noise projected on each face,
   which faces are seen, and the free ends' beam quantization and dropout.

Other cars move during the sweep: the de-skewed points are where each surface was at its own beam time. Each cluster
is moved to its mean beam time with a velocity prior (by default the carrying car's own speed along the road, near
every car's on a highway), stamped with that time, and the prior's error goes into R. The prior's error is the car's
own relative speed, fixed from sweep to sweep, so meta also carries what a tracker needs to model it instead:
'v_prior', 'dz_dv' (the fit's change per m/s of true velocity above the prior: a few mm and a tenth of a degree per
m/s for a car ahead, from the skew across its face) and 'R_fit' (R without the prior's share); see fit_box.
"""
import numpy as np
from ...geometry import cg_offsets
from ..geometry import mount_offset, sensor_pose, to_world, wrap
from ..types import Detection
from .. import _jit
from ._lidar_jit import fit_box_kernel

__all__ = ['deskew', 'process_lidar', 'fit_box', 'lshape_heading', 'refit']

BREAK_ANGLE = np.radians(10.0)     # Borges-Aldon lambda
HEADINGS = np.radians(np.arange(-20.0, 20.5, 1.0))   # L-shape search grid around the road heading
LSHAPE_MIN = 8                     # smaller clusters start from the road heading (an L of a few points is ambiguous)
REFINE_MIN = LSHAPE_MIN            # process_lidar refines only clusters this large: ICP on a few points of a far car
                                   # cost as much as on a near one (a third of all clusters on v9) and the heading
                                   # prior holds the fit there anyway
LSHAPE_POINTS = 32                 # clusters thinned to this many points for the L-shape search
MATE_MARGIN = 0.3                  # teammate boxes inflated by this (m)
EDGE_MARGIN = 0.5                  # points this far beyond the road edges are dropped
FIT_SLACK = (0.5, 0.25)            # a cluster fits one car within length + 0.5 (smear) by width + 0.25 (noise)
OCCLUDER_MARGIN = 0.5              # a neighbouring return this much nearer occludes a segment end
GAP_ARC = 0.5                      # empty beams spanning at least this arc (m) ...
GAP_P = 1e-4                       # ... and this unlikely as dropouts split a segment
END_LOOK = 3                       # beams looked past a segment end for its neighbour
STRAY_MAX = 0.5                    # RANSAC drops points off a face line by less than this (mirrors, noise)
VIS_ANGLE = np.radians(3.0)        # a face counts as possibly visible within this much heading of facing the sensor
TAN_VIS = np.tan(VIS_ANGLE)
GATE = 0.3                         # ICP correspondence gate (m)
POSE_SIGMA = 0.05                  # pose error allowed for when assigning points to faces (m)
STEP_MAX = (0.5, np.radians(3.0))  # Gauss-Newton step limits: centre (m), heading (rad)
CORNER_MIN = 0.15                  # a face counts as seen when one of its points lies this far from its corners
HEADING_PRIOR = np.radians(5.0)    # prior: cars point along the road (highway lane changes stay within ~3 deg)
STEADY_AX, STEADY_DR = 3.0, 0.05   # beyond these the de-skew interpolates the car's poses
ICP_ITERS = 5                      # Gauss-Newton steps per round of fixed correspondences
ICP_ROUNDS = 2                     # rounds of re-assigning the points to faces
NONE, KEEP, MATE, OFF = 0, 1, 2, 3


# ---------------------------------------------------------------------------------------------------------- de-skew
def _twist(vx, vy, r, tau):
    """SE(2) exponential of the body twist [vx, vy, r] over tau: heading change and displacement in the starting car
    frame. The sinc forms are exact and reach the straight-line limit [vx, vy] tau smoothly as r -> 0."""
    th = r*tau
    a = np.sinc(th/np.pi)                      # sin(th)/th
    b = 0.5*th*np.sinc(th/(2*np.pi))**2        # (1 - cos th)/th
    return th, tau*(vx*a-vy*b), tau*(vx*b+vy*a)


def _steady(e0, em, e1):
    if e0 is None or e1 is None or not e0.time < em.time < e1.time:
        return True
    r = (e0.r, em.r, e1.r)
    return max(abs(e0.ax), abs(em.ax), abs(e1.ax)) <= STEADY_AX and max(r)-min(r) <= STEADY_DR


def _lagrange(states, t):
    """(len(t), 3) car poses [x, y, yaw], quadratic in time through the three states (x, y relative to states[1])."""
    ts = np.array([e.time for e in states])
    X = np.array([[e.x, e.y, e.yaw] for e in states])
    X[:, 2] = np.unwrap(X[:, 2])
    X[:, :2] -= X[1, :2]
    out = np.zeros((len(t), 3))
    for k in range(3):
        i, j = (m for m in range(3) if m != k)
        out += np.multiply.outer((t-ts[i])*(t-ts[j])/((ts[k]-ts[i])*(ts[k]-ts[j])), X[k])
    return out


def deskew(scan, spec, ego_mid, params, ego_start=None, ego_end=None):
    """(n, 2) sensor-frame points of a sweep at scan.t_end, compensated for the carrying car's motion; nan rows where a
    beam returned nothing. Each point goes to the car frame through the mount (lever arm), is moved by the car's motion
    from its beam time to t_end (p_e = R(dpsi)^T (p_i - dp)), and comes back to the sensor frame. The motion is the
    constant twist [vx, vy, r] of ego_mid; with ego_start and ego_end given and the motion far from constant
    (|ax| > 3 m/s^2, or r changing by more than 0.05 rad/s), the three poses are interpolated instead."""
    rng, az = np.asarray(scan.range, float), np.asarray(scan.azimuth, float)
    t = np.asarray(scan.time, float)
    mx, my = mount_offset(spec.mount, params)
    cm, sm = np.cos(spec.mount.yaw), np.sin(spec.mount.yaw)
    sx, sy = rng*np.cos(az), rng*np.sin(az)
    px, py = cm*sx-sm*sy+mx, sm*sx+cm*sy+my
    if _steady(ego_start, ego_mid, ego_end):
        dpsi, dx, dy = _twist(ego_mid.vx, ego_mid.vy, ego_mid.r, scan.t_end-t)
    else:
        X = _lagrange((ego_start, ego_mid, ego_end), np.append(t, scan.t_end))
        Xi, Xe = X[:-1], X[-1]
        dpsi = Xe[2]-Xi[:, 2]
        ex, ey = Xe[0]-Xi[:, 0], Xe[1]-Xi[:, 1]
        ci, si = np.cos(Xi[:, 2]), np.sin(Xi[:, 2])
        dx, dy = ci*ex+si*ey, -si*ex+ci*ey
    c, s = np.cos(dpsi), np.sin(dpsi)
    qx, qy = px-dx, py-dy
    ex, ey = c*qx+s*qy-mx, -s*qx+c*qy-my
    return np.column_stack((cm*ex+sm*ey, -sm*ex+cm*ey))


# -------------------------------------------------------------------------------------------------------- box fitting
def lshape_heading(points, sensor, headings=HEADINGS):
    """Heading of an L-shape fit (Zhang et al. 2017, search-based, variance criterion) over `headings`, with a
    parabolic step between grid points. Each axis is measured from the boundary that faces the sensor (the seen faces
    always face it), and each point counts toward the nearer of the two boundaries. Large clusters are thinned evenly
    to LSHAPE_POINTS for the search."""
    P = np.asarray(points, float)
    if len(P) > LSHAPE_POINTS:
        P = P[np.linspace(0, len(P)-1, LSHAPE_POINTS).astype(int)]
    m = P.mean(0)
    P, o = P-m, np.asarray(sensor, float)[:2]-m
    cost = _lshape_cost(P, o, headings)
    k = int(np.argmin(cost))
    if 0 < k < len(headings)-1:
        y0, y1, y2 = cost[k-1:k+2]
        curv = y0-2*y1+y2
        if curv > 0:
            return float(headings[k]+np.clip(0.5*(y0-y2)/curv, -0.5, 0.5)*(headings[k+1]-headings[k]))
    return float(headings[k])


def _lshape_cost(P, o, headings):
    """Zhang et al.'s variance criterion of centred points P seen from o, per heading (lower is better)."""
    c, s = np.cos(headings), np.sin(headings)
    C1 = np.multiply.outer(c, P[:, 0])+np.multiply.outer(s, P[:, 1])
    C2 = np.multiply.outer(c, P[:, 1])-np.multiply.outer(s, P[:, 0])
    o1, o2 = c*o[0]+s*o[1], c*o[1]-s*o[0]
    D = []
    for C, oc in ((C1, o1), (C2, o2)):
        lo, hi = C.min(1), C.max(1)
        D.append(np.abs(C-np.where(oc < (lo+hi)/2, lo, hi)[:, None]))
    near = D[0] < D[1]
    cost = 0.0
    for d, sel in ((D[0], near), (D[1], ~near)):
        k = np.maximum(sel.sum(1), 1)
        ds = d*sel
        mean = ds.sum(1)/k
        cost = cost+(ds*d).sum(1)/k-mean*mean
    return cost


_FACE_AX = np.array([0, 0, 1, 1])               # faces: rear, front, right, left
_FACE_SG = np.array([-1.0, 1.0, -1.0, 1.0])


class _Cluster:
    """One cluster's points for the box fit, with what does not depend on the pose: unit rays from the sensor, time
    offsets tau to the reference time, the car's half size and noise, and its free ends (index, inward neighbour,
    dropout probability there)."""
    def __init__(self, P, o, tau, size, sr, sm, ends):
        self.P, self.o, self.tau, self.ends = P, o, tau, ends
        self.size, self.half = size, 0.5*np.asarray(size)
        self.sr, self.sm = sr, sm
        ray = P-o
        self.dist = np.maximum(np.hypot(ray[:, 0], ray[:, 1]), 1e-9)
        self.ray = ray/self.dist[:, None]
        self.rx, self.ry = self.ray[:, 0], self.ray[:, 1]


def _assign(cl, use, pose):
    """Faces of a box pose for each point. Candidate faces are those facing the sensor, give or take VIS_ANGLE of
    heading (a face seen at grazing incidence may face either way within the heading's uncertainty). Each point goes to
    the candidate face it is likeliest on: the smallest distance to the face's outline over its sigma there (range
    noise along the ray projected on the face normal, sm, and POSE_SIGMA for the pose's own error, so that a few cm of
    pose error does not move a grazing face's points onto the other face). A face is seen when a used point on it lies
    CORNER_MIN or more from its corners. Near a corner between candidate faces (within 3 sigma of both, pose error
    included) a point could be on either, so its variance there grows by that zone squared: it still holds the face
    weakly, but cannot pin a heading on the wrong face. Returns c, s of the heading, body coordinates B, each point's
    face (index into _FACE_AX and _FACE_SG; -1 for none), its variance, and which points are on a seen face."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    P, (hx, hy) = cl.P, cl.half
    d = P-pose[:2]
    B = d @ np.array([[c, -s], [s, c]])
    ox, oy = cl.o[0]-pose[0], cl.o[1]-pose[1]
    obx, oby = c*ox+s*oy, c*oy-s*ox
    tx, ty = TAN_VIS*abs(oby), TAN_VIS*abs(obx)
    cand = np.array([-obx > hx-tx, obx > hx-tx, -oby > hy-ty, oby > hy-ty])
    n = len(P)
    if not cand.any():
        return c, s, B, np.full(n, -1), np.ones(n), np.zeros(n, bool)
    A = np.abs(B)
    over = np.maximum(A-cl.half, 0.0)**2
    D = (B[:, _FACE_AX]-_FACE_SG*cl.half[_FACE_AX])**2+over[:, _FACE_AX^1]
    V = (cl.sr*(cl.ray @ np.array([[c, -s], [s, c]])))**2+cl.sm**2
    V = V[:, _FACE_AX]
    j = np.argmin(np.where(cand, D/(V+POSE_SIGMA**2), np.inf), 1)
    rows = np.arange(n)
    clear = (cl.half[::-1]-A[:, ::-1])[rows, _FACE_AX[j]]     # along the face, from its corners
    best = np.full(4, -1.0)
    np.maximum.at(best, j[use], clear[use])
    best = np.where(best[[1, 0, 3, 2]] > best, -1.0, best)     # a box never shows two opposite faces: keep the stronger
    on = (best >= CORNER_MIN)[j]
    var = V[rows, j]
    zone = 3*np.sqrt(cl.sr**2+cl.sm**2+POSE_SIGMA**2)
    for f0 in (0, 1):
        for f1 in (2, 3):
            if cand[f0] and cand[f1]:
                corner = (np.abs(B[:, 0]-_FACE_SG[f0]*hx) < zone) & (np.abs(B[:, 1]-_FACE_SG[f1]*hy) < zone)
                var = np.where(corner, var+zone*zone, var)
    return c, s, B, j, var, on


def _rows(cl, use, pose):
    """Residual rows r = sg (R^T (p - c))[ax] - h of a box pose: the used points on its seen faces within GATE of them;
    along an axis with no such face, the free ends of the cluster, else the middle of the seen extent with the hidden
    length as a uniform spread. The car's true end lies past a free end's point by the beam quantization (uniform over
    one gap) plus the beams lost to dropout there (geometric with the end's dropout probability pd): moved out by
    gap (1/2 + pd/(1-pd)), variance gap^2 (1/12 + pd/(1-pd)^2). Returns the rows (pts, ax, sg, h, var, tau) and
    (E, ax, faces, src): the box axes, each point's face axis, the seen faces (indices into _FACE_AX) and each row's
    point (-1 for the extent prior)."""
    c, s, B, j, var, on = _assign(cl, use, pose)
    half, P = cl.half, cl.P
    E = np.array([[c, s], [-s, c]])
    idx = np.flatnonzero(use & on)
    jx = j[idx]
    ax, sg = _FACE_AX[jx], _FACE_SG[jx]
    keep = np.abs(sg*B[idx, ax]-half[ax]) <= GATE
    idx, jx, ax, sg = idx[keep], jx[keep], ax[keep], sg[keep]
    faces = np.unique(jx)
    out = [[P[idx]], [ax], [sg], [half[ax]], [var[idx]], [cl.tau[idx]], [idx]]

    def add(p, *row):
        out[0].append(p[None])
        for col, v in zip(out[1:], row):
            col.append([v])

    for k in (0, 1):
        if (_FACE_AX[faces] == k).any():
            continue
        col = B[:, k]
        lo, hi, mid = col.min(), col.max(), col.mean()
        cand = []
        for i, nb, pd in cl.ends:
            gap = min(abs(col[i]-col[nb]), cl.size[k])
            sign = 1.0 if col[i] > mid else -1.0
            if abs(col[i]-(hi if sign > 0 else lo)) <= gap+0.1:
                cand.append((cl.dist[i], i, sign, gap, pd))
        if len(cand) == 2 and (cand[0][2] == cand[1][2] or hi-lo < cl.size[k]-0.3):
            cand = [min(cand)]
        for _, i, sign, gap, pd in cand:
            ck = abs(E[k, 0]*cl.rx[i]+E[k, 1]*cl.ry[i])
            lost = pd/(1-pd)
            add(P[i]+(0.5+lost)*gap*sign*E[k], k, sign, half[k],
                gap*gap*(1/12+lost/(1-pd))+(cl.sr*ck)**2+cl.sm**2, cl.tau[i], i)
        if not cand:
            add(0.5*(P[np.argmin(col)]+P[np.argmax(col)]), k, 1.0, 0.0, max(cl.size[k]-(hi-lo), 0.1)**2/12+cl.sm**2,
                0.0, -1)
    rows = tuple(np.concatenate(a) for a in out)
    return rows[:6], (E, np.where(j >= 0, _FACE_AX[j], -1), faces, rows[6])


def _linearize(rows, pose):
    """Residuals r, Jacobian J (m, 3) in (x, y, heading), weights w and the sensitivity Bv (m, 2) of r to the
    velocity prior's error."""
    pts, ax, sg, h, var, tau = rows
    c, s = np.cos(pose[2]), np.sin(pose[2])
    Bd = (pts-pose[:2]) @ np.array([[c, -s], [s, c]])
    rows = np.arange(len(ax))
    r = sg*Bd[rows, ax]-h
    e = np.array([[c, s], [-s, c]])[ax]*sg[:, None]          # outward face normals in the world
    J = np.empty((len(ax), 3))
    J[:, :2] = -e
    J[:, 2] = sg*np.where(ax == 0, Bd[:, 1], -Bd[:, 0])
    return r, J, 1.0/var, e*tau[:, None]


def _normal(r, J, w, pose):
    A = J.T @ (w[:, None]*J)
    g = J.T @ (w*r)
    A[2, 2] += 1/HEADING_PRIOR**2
    g[2] += wrap(pose[2])/HEADING_PRIOR**2
    return A, g


def _anchor(cl, use, th, passes):
    """Box pose with the L-shape heading held fixed: the centre solves the rows by weighted least squares, i.e. the
    seen faces' planes and the free ends, moved inward by half the known size. Starts from the points' extent on the
    side facing the sensor. Returns the pose, the last rows and their face assignment."""
    c, s = np.cos(th), np.sin(th)
    P, o, half = cl.P, cl.o, cl.half
    B = np.column_stack((c*P[:, 0]+s*P[:, 1], c*P[:, 1]-s*P[:, 0]))
    ob = np.array([c*o[0]+s*o[1], c*o[1]-s*o[0]])
    lo, hi = B.min(0), B.max(0)
    cb = np.where(ob < lo, lo+half, np.where(ob > hi, hi-half, (lo+hi)/2))
    for _ in range(passes):
        rows, info = _rows(cl, use, np.array([c*cb[0]-s*cb[1], s*cb[0]+c*cb[1], th]))
        pts, ax, sg, h, var, _ = rows
        target = np.where(ax == 0, c*pts[:, 0]+s*pts[:, 1], c*pts[:, 1]-s*pts[:, 0])-sg*h
        cb = np.bincount(ax, target/var, 2)/np.bincount(ax, 1/var, 2)
    return np.array([c*cb[0]-s*cb[1], s*cb[0]+c*cb[1], th]), rows, info


def _consensus(Q, direction, tol):
    """Mask of the points to keep from the best line through Q (deterministic RANSAC: hypotheses through pairs of up to
    12 evenly spaced points, within 10 deg of `direction`): all but those beyond tol and within STRAY_MAX of it; all
    when no line holds half the points."""
    n = len(Q)
    idx = np.unique(np.round(np.linspace(0, n-1, min(n, 12))).astype(int))
    i, j = np.triu_indices(len(idx), 1)
    i, j = idx[i], idx[j]
    d = Q[j]-Q[i]
    norm = np.hypot(d[:, 0], d[:, 1])
    d = d/np.maximum(norm, 1e-9)[:, None]
    ok = (norm > 0.05) & (np.abs(d @ direction) >= np.cos(np.radians(10.0)))
    if not ok.any():
        return np.ones(n, bool)
    i, d = i[ok], d[ok]
    dist = np.abs((Q[None, :, 0]-Q[i, None, 0])*d[:, None, 1]-(Q[None, :, 1]-Q[i, None, 1])*d[:, None, 0])
    inl = dist < tol
    score = inl.sum(1)-(dist*inl).sum(1)/(tol*n)
    best = int(np.argmax(score))
    return inl[best] | (dist[best] > STRAY_MAX) if inl[best].sum() >= 0.5*n else np.ones(n, bool)


def _ransac(cl, r, info, tol):
    """Points kept for ICP: on each seen face with 4 or more points, all but the strays near its best line (beyond tol
    but within STRAY_MAX of it, such as mirrors); points farther off belong to other structure, for ICP to assign.
    Skipped when every row's residual r at the anchor lies within tol (the best line then keeps them all)."""
    keep = np.ones(len(cl.P), bool)
    if (np.abs(r) <= tol).all():
        return keep
    E, ax, faces, _ = info
    for k in set(_FACE_AX[faces].tolist()):
        sel = np.flatnonzero(ax == k)
        if len(sel) >= 4:
            keep[sel] = _consensus(cl.P[sel], E[1-k], tol)
    return keep


def fit_box(points, times, sensor, params, spec, free=(True, True), velocity=(0.0, 0.0), velocity_sigma=(5.0, 1.0),
            refine=True, sigma_model=0.003, out=None):
    """Known-size box fitted to one cluster of world points in scan order, each taken at `times` from the sensor at
    world position `sensor` (2,). free: whether the cluster's first and last points are true ends of the car's outline
    (not occluded, not cut by the field of view). The points are moved to their mean time t_ref with `velocity` (world,
    m/s); velocity_sigma (along, across the road) is that prior's error. refine: RANSAC line fits, then point-to-line
    ICP. The heading starts from the L-shape search, or from the road's for under LSHAPE_MIN points. Returns
    (z, R, sides, t_ref): z = [x, y, heading] of the body centre at t_ref, R (3, 3), sides seen (0-2). With a dict
    `out`, also out['dz_dv'] (3, 2) = d z / d velocity (to first order, the fit with the car's true velocity v is
    z + dz_dv (v - velocity)) and out['R_fit'], R without the prior's share: a tracker that knows v models
    z = box(x) + dz_dv (velocity - v) with R_fit, instead of taking the prior's error as white noise. Runs as the
    compiled _lidar_jit.fit_box_kernel when _jit.ENABLED, else as _fit_box_py."""
    if _jit.ENABLED:
        z, cov, sides, t_ref, dz_dv, R_fit = fit_box_kernel(
            np.ascontiguousarray(points, float), np.ascontiguousarray(times, float),
            np.ascontiguousarray(np.asarray(sensor, float)[:2]), np.array([params.length, params.width], float),
            float(spec.range_sigma), float(sigma_model), bool(free[0]), bool(free[1]), float(spec.dropout),
            float(spec.far_dropout), float(spec.range_max), np.ascontiguousarray(velocity, float),
            np.ascontiguousarray(velocity_sigma, float), bool(refine), HEADINGS)
        if out is not None:
            out['dz_dv'], out['R_fit'] = dz_dv, R_fit
        return z, cov, int(sides), float(t_ref)
    return _fit_box_py(points, times, sensor, params, spec, free, velocity, velocity_sigma, refine, sigma_model, out)


def _fit_box_py(points, times, sensor, params, spec, free, velocity, velocity_sigma, refine, sigma_model, out):
    """fit_box's numpy path (NOHESI_JIT=0, or numba missing): the reference for _lidar_jit.fit_box_kernel."""
    tt = np.asarray(times, float)
    t_ref = float(tt.mean())
    tau = t_ref-tt
    P = np.asarray(points, float)+tau[:, None]*np.asarray(velocity, float)
    o = np.asarray(sensor, float)[:2]
    n = len(P)
    ends = [(i, nb, float(_dropout(spec, np.hypot(*(P[i]-o))))) for i, nb, f in
            ((0, min(1, n-1), free[0]), (n-1, max(n-2, 0), free[1])) if f and n > 1]
    cl = _Cluster(P, o, tau, (float(params.length), float(params.width)), float(spec.range_sigma), sigma_model, ends)
    use = np.ones(n, bool)
    th = lshape_heading(P, o) if n >= LSHAPE_MIN else 0.0
    pose, rows, info = _anchor(cl, use, th, 1 if refine else 2)
    if refine:
        lin = _linearize(rows, pose)
        use = _ransac(cl, lin[0], info, 3*np.hypot(cl.sr, cl.sm)+0.02)
        if not use.all():
            rows, info = _rows(cl, use, pose)
            lin = _linearize(rows, pose)
        for _ in range(ICP_ROUNDS):            # Gauss-Newton on fixed correspondences, then re-assign
            for _ in range(ICP_ITERS):
                r, J, w, _ = lin
                A, g = _normal(r, J, w, pose)
                step = -np.linalg.solve(A, g)
                step *= min(1.0, STEP_MAX[0]/max(np.hypot(step[0], step[1]), 1e-12),
                            STEP_MAX[1]/max(abs(step[2]), 1e-12))
                pose = pose+step
                lin = _linearize(rows, pose)
                if abs(step[0])+abs(step[1]) < 5e-4 and abs(step[2]) < 2e-5:
                    break
            before = (info[3], rows[1], rows[2])
            rows, info = _rows(cl, use, pose)
            if all(np.array_equal(a, b) for a, b in zip(before, (info[3], rows[1], rows[2]))):
                break
            lin = _linearize(rows, pose)
    r, J, w, Bv = _linearize(rows, pose)
    A, _ = _normal(r, J, w, pose)
    core = np.linalg.inv(A)
    S = core @ (J.T @ (w[:, None]*Bv))            # d pose / d velocity = -S: the residuals move by Bv per m/s
    if not refine:
        G = np.append(-np.linalg.solve(A[:2, :2], A[:2, 2]), 1.0)   # the centre follows the grid's heading error
        core = core+np.outer(G, G)*(HEADINGS[1]-HEADINGS[0])**2/12
    cov = core+S @ np.diag(np.square(velocity_sigma)) @ S.T
    if out is not None:
        out['dz_dv'], out['R_fit'] = -S, 0.5*(core+core.T)
    pose[2] = wrap(pose[2])
    return pose, 0.5*(cov+cov.T), len(info[2]), t_ref


# ------------------------------------------------------------------------------------------------------ segmentation
def _dropout(spec, r):
    """Beam dropout probability at range r: spec.dropout plus far_dropout reached linearly at range_max."""
    return np.clip(spec.dropout+spec.far_dropout*np.minimum(np.asarray(r)/spec.range_max, 1.0), 1e-3, 0.95)


def _free_gap(miss, r, spec):
    """True where `miss` empty beams at range r are free space: an arc of at least GAP_ARC, and a run that dropout
    would give with probability below GAP_P."""
    pd = _dropout(spec, r)
    return (miss >= np.maximum(2, np.ceil(np.log(GAP_P)/np.log(pd)))) & (miss*r*spec.beam_spacing >= GAP_ARC)


def _segments(P, rd, az, label, spec):
    """KEEP row indices K and the runs of them (starts, ends: inclusive indices into K) split at adaptive breakpoints
    (Borges-Aldon threshold r sin(dphi)/sin(lambda - dphi) + 3 sigma_r), at teammate and off-road rows, and at
    free-space gaps."""
    K = np.flatnonzero(label == KEEP)
    a, b = K[:-1], K[1:]
    dphi = az[b]-az[a]
    de = np.maximum(dphi, spec.beam_spacing)
    dist = np.hypot(*(P[b]-P[a]).T)
    with np.errstate(divide='ignore', invalid='ignore'):
        thr = rd[a]*np.sin(de)/np.sin(BREAK_ANGLE-de)+3*spec.range_sigma
    other = np.cumsum((label == MATE) | (label == OFF))
    miss = np.maximum(np.round(dphi/spec.beam_spacing)-1, 0)
    brk = (de >= BREAK_ANGLE) | (dist > thr) | (other[b] > other[a]) | _free_gap(miss, rd[a], spec)
    cut = np.flatnonzero(brk)
    return K, np.r_[0, cut+1], np.r_[cut, len(K)-1]


def _box_fits(ext, size):
    """Whether a road-aligned extent (2,) can hold one car: True (it fits at heading 0), False (it cannot at any
    heading within the search range), None (undecided)."""
    L, W = size[0]+FIT_SLACK[0], size[1]+FIT_SLACK[1]
    if ext[0] <= L and ext[1] <= W:
        return True
    sn = np.sin(HEADINGS[-1])
    if ext[0] > L+W*sn or ext[1] > L*sn+W:
        return False
    return None


def _fits(Q, o, size, main=None):
    """Whether points fit one car: extent within length + slack by width + slack at the L-shape heading of `main`
    (the larger piece of a merge, whose heading is the better determined; all of Q when it has under 5 points)."""
    ok = _box_fits(np.ptp(Q, 0), size)
    if ok is not None:
        return ok
    th = lshape_heading(Q if main is None or len(main) < 5 else main, o)
    c, s = np.cos(th), np.sin(th)
    return bool(np.ptp(c*Q[:, 0]+s*Q[:, 1]) <= size[0]+FIT_SLACK[0] and
                np.ptp(c*Q[:, 1]-s*Q[:, 0]) <= size[1]+FIT_SLACK[1])


def _merge(K, starts, ends, Wc, rd, az, label, o, size, spec):
    """Row-index arrays of the clusters (Wc: world points moved to one time with the velocity prior). Consecutive
    segments (up to 3 apart) are joined when every return between them is nearer than both facing ends (an occluder,
    a teammate included) or nothing lies between them (a grazing view), no off-road return or free-space gap lies
    between, and together they fit one car."""
    S = len(starts)
    first, last = K[starts], K[ends]
    rows = np.split(K, starts[1:])
    if S < 2:
        return rows
    lo, hi = np.minimum.reduceat(Wc[K], starts), np.maximum.reduceat(Wc[K], starts)
    rmax = np.maximum.reduceat(rd[K], starts)
    gmate = np.full(S-1, -np.inf)                       # farthest teammate return in each gap
    mates = np.flatnonzero(label == MATE)
    if len(mates):
        g = np.searchsorted(first, mates)-1
        inside = (g >= 0) & (g < S-1)
        np.maximum.at(gmate, g[inside], rd[mates[inside]])
    miss = np.maximum(np.round((az[first[1:]]-az[last[:-1]])/spec.beam_spacing)-1, 0)
    gbad = (_free_gap(miss, rd[last[:-1]], spec) & ~np.isfinite(gmate)) | \
        (np.cumsum(label == OFF)[first[1:]] > np.cumsum(label == OFF)[last[:-1]])
    root = list(range(S))
    members = {i: [i] for i in range(S)}
    nbad = np.r_[0, np.cumsum(gbad)]
    for d in (1, 2, 3):
        i = np.arange(S-d)
        near = np.minimum(rd[last[i]], rd[first[i+d]])-OCCLUDER_MARGIN
        ok = nbad[i+d] == nbad[i]
        for m in range(d):
            ok &= gmate[i+m] < near
            if m:
                ok &= rmax[i+m] < near
        for i in np.flatnonzero(ok):
            j = i+d
            ci, cj = root[i], root[j]
            if ci == cj:
                continue
            blo, bhi = np.minimum(lo[ci], lo[cj]), np.maximum(hi[ci], hi[cj])
            ok = _box_fits(bhi-blo, size)
            if ok is None:
                pi, pj = (np.concatenate([rows[m] for m in members[c]]) for c in (ci, cj))
                ok = _fits(Wc[np.r_[pi, pj]], o, size, Wc[pi if len(pi) >= len(pj) else pj])
            if ok:
                for m in members[cj]:
                    root[m] = ci
                members[ci] += members.pop(cj)
                lo[ci], hi[ci] = blo, bhi
    return [np.sort(np.concatenate([rows[m] for m in mem])) for mem in members.values()]


def _split(idx, Wc, o, size, depth=0):
    """Clusters that fit one car: an oversize cluster is cut at its widest gap between consecutive points,
    recursively."""
    Q = Wc[idx]
    if len(idx) < 3 or depth >= 4 or _fits(Q, o, size):
        return [idx]
    k = int(np.argmax(np.hypot(*np.diff(Q, axis=0).T)))+1
    return _split(idx[:k], Wc, o, size, depth+1)+_split(idx[k:], Wc, o, size, depth+1)


def _free_end(row, step, rd, label):
    """Whether the cluster end at `row` is a true end of the car's outline: the nearest return within END_LOOK beams
    past it is clearly farther or off the road, or there is none (empty beams). Not when that return is nearer or
    about as near (possibly occluded), or the end is the field of view's edge."""
    n = len(rd)
    k = row+step
    if not 0 <= k < n:
        return False
    for _ in range(END_LOOK):
        if not 0 <= k < n or label[k] == OFF:
            return True
        if label[k] != NONE:
            return bool(rd[k] > rd[row]+OCCLUDER_MARGIN)
        k += step
    return True


def _mate_mask(Wp, t, ok, mates, t0, t1, params):
    """Rows inside a teammate's box inflated by MATE_MARGIN, teammates linearly interpolated to each beam time."""
    mask = np.zeros(len(Wp), bool)
    if mates is None:
        return mask
    M0, M1 = np.atleast_2d(np.asarray(mates(t0), float)), np.atleast_2d(np.asarray(mates(t1), float))
    if M0.size == 0 or M0.shape != M1.shape:
        return mask
    rows = np.flatnonzero(ok)
    w = ((t[rows]-t0)/(t1-t0) if t1 > t0 else np.zeros(len(rows)))[:, None]
    x = M0[None, :, 0]+w*(M1[None, :, 0]-M0[None, :, 0])
    y = M0[None, :, 1]+w*(M1[None, :, 1]-M0[None, :, 1])
    yaw = M0[None, :, 2]+w*wrap(M1[None, :, 2]-M0[None, :, 2])
    c, s = np.cos(yaw), np.sin(yaw)
    shift = sum(cg_offsets(params))/2
    dx, dy = Wp[rows, None, 0]-(x+shift*c), Wp[rows, None, 1]-(y+shift*s)
    inside = (np.abs(c*dx+s*dy) <= params.length/2+MATE_MARGIN) & (np.abs(c*dy-s*dx) <= params.width/2+MATE_MARGIN)
    mask[rows] = inside.any(1)
    return mask


# ---------------------------------------------------------------------------------------------------------- pipeline
def process_lidar(scan, spec, ego_at, mates, params, road, refine=True, *, target_v=None, target_sigma=(5.0, 1.0),
                  sigma_model=0.003):
    """'lidar_box' detections of one sweep. ego_at(t) -> EgoState of the carrying car (exact); mates(t) -> (k, 3) CG
    states [x, y, yaw] of the other team cars (V2V; None for none); params the vehicle (every car's size); road has
    .width. target_v: world velocity (2,) the cars are assumed to move at during the sweep (None: the carrying car's own
    speed along the road), with prior sigma target_sigma (along, across the road) added to R. sigma_model: box-model
    error per point (m). refine: fit_box's refinement for clusters of at least REFINE_MIN points. Each detection's z is
    [x, y, heading] of the body centre at the cluster's mean beam time."""
    if len(scan.range) < 3:
        return []
    t0, t1 = float(scan.t_start), float(scan.t_end)
    e0, em, e1 = ego_at(t0), ego_at(0.5*(t0+t1)), ego_at(t1)
    order = np.argsort(np.asarray(scan.azimuth, float), kind='stable')
    az, t = np.asarray(scan.azimuth, float)[order], np.asarray(scan.time, float)[order]
    P = deskew(scan, spec, em, params, e0, e1)[order]
    rd = np.hypot(P[:, 0], P[:, 1])
    pose_e = sensor_pose(spec.mount, (e1.x, e1.y, e1.yaw), params)
    Wp = to_world(P, pose_e)
    ok = np.isfinite(rd)
    label = np.where(ok, KEEP, NONE).astype(np.int8)
    label[_mate_mask(Wp, t, ok, mates, t0, t1, params)] = MATE
    with np.errstate(invalid='ignore'):
        label[(label == KEEP) & (np.abs(Wp[:, 1]) > road.width/2+EDGE_MARGIN)] = OFF
    if np.count_nonzero(label == KEEP) < 3:
        return []
    if target_v is None:
        v = np.array([em.vx*np.cos(em.yaw)-em.vy*np.sin(em.yaw), 0.0])
    else:
        v = np.asarray(target_v, float)
    size = (float(params.length), float(params.width))
    o = np.array(pose_e[:2])
    Wc = Wp+(0.5*(t0+t1)-t)[:, None]*v       # shapes as at one time; a cluster's own time only shifts it
    K, starts, ends = _segments(P, rd, az, label, spec)
    merged = _merge(K, starts, ends, Wc, rd, az, label, o, size, spec)
    clusters = [c for idx in merged for c in _split(idx, Wc, o, size)]
    out = []
    for idx in sorted(clusters, key=lambda c: c[0]):
        if len(idx) < 3:
            continue
        free = (_free_end(idx[0], -1, rd, label), _free_end(idx[-1], 1, rd, label))
        t_ref = float(t[idx].mean())
        ego = ego_at(t_ref)
        pose = sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), params)
        fit = {}
        z, R, sides, _ = fit_box(Wp[idx], t[idx], pose[:2], params, spec, free, v, target_sigma,
                                 refine and len(idx) >= REFINE_MIN, sigma_model, out=fit)
        out.append(Detection('lidar_box', spec.name, scan.car, t_ref, z, R, pose, ego,
                             {'n_points': int(len(idx)), 'sides': min(max(sides, 1), 2), 'scan': t1, 'points': Wp[idx],
                              'times': t[idx], 'free': free, 'v_prior': v.copy(), **fit}))
    return out


def refit(det, spec, params, velocity, velocity_sigma=(0.5, 0.3), refine=True, sigma_model=0.003):
    """The detection refitted with a better velocity for its car (the design's optional second pass: a track's
    predicted world velocity after association). Same time stamp, sensor pose and points. The tracker does not need
    it: it models a box's dependence on the velocity prior to first order (meta 'v_prior', 'dz_dv', 'R_fit')."""
    m = det.meta
    fit = {}
    z, R, sides, _ = fit_box(m['points'], m['times'], det.sensor_pose[:2], params, spec, m['free'], velocity,
                             velocity_sigma, refine, sigma_model, out=fit)
    return Detection(det.kind, det.sensor, det.car, det.time, z, R, det.sensor_pose, det.ego,
                     {**m, 'sides': min(max(sides, 1), 2), 'v_prior': np.asarray(velocity, float).copy(), **fit})
