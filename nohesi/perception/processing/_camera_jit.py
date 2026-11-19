"""Compiled kernels of camera processing (processing/camera.py): one sim step's frames, every box of every frame, in
one call (process_boxes), instead of numpy's many small array operations per step.

Each kernel is camera.py's numpy code written out for one box, in the same order of operations, so the two paths agree
to rounding (the math library's last bit, summation order, and an LU solve of the 4 x 4 normal equations in place of
LAPACK's): the teammate filter (mate_boxes, iou), cue_init, refine_init with its batch logic per box (a converged box
stops; the best iterate wins), the projected box past a cut edge, pitch_jacobian and R. camera.py's numpy functions
stay the reference and the path with NOHESI_JIT=0 (_jit.py); there these kernels would run as plain Python, correct
but slow, and camera.py does not call them.

The tuning constants come in as arguments (the TUNE layout below, STEPS, EDGES, CUT_SIDE), not as module globals:
numba's on-disk cache keys on this file only, so a global read from camera.py would keep its old value after an edit
there. The camera of a box is a row of camera.py's _Cam fields (the CAM layout).
"""
from math import atan2, cos, hypot, inf, isfinite, nan, sin, sqrt
import numpy as np
from .._jit import njit

__all__ = ['CAM_FIELDS', 'TUNE_FIELDS', 'process_boxes']

CAM_FIELDS = ('f', 'cu', 'cv', 'W', 'H', 'sig', 'yaw', 'pitch_sigma', 'x', 'y', 'psi', 'z')   # camera.py's _Cam
F, CU, CV, W, H, SIG, YAW, PSIG, X, Y, PSI, Z = range(12)
TUNE_FIELDS = ('TRUNC_PX', 'FIT_MARGIN_SIGMAS', 'IOU_MATE', 'NEAR', 'CONVERGED', 'PRIOR_ALONG', 'TRUST',
               'HEADING_SPREAD', 'PITCH_SPREAD')
T_TRUNC, T_MARGIN, T_IOU, T_NEAR, T_CONV, T_PRIOR, T_TRUST, T_HSPREAD, T_PSPREAD = range(9)


@njit(error_model='numpy')
def _tilt(yaw, pitch, R):
    """R (3, 3) <- geometry.camera_tilt(yaw, pitch), entry by entry as geometry._tilt."""
    sa, ca = sin(yaw), cos(yaw)
    s, k = sin(pitch), 1-cos(pitch)
    R[0, 0], R[0, 1], R[0, 2] = 1-k*ca*ca, k*sa*ca, s*ca
    R[1, 0], R[1, 1], R[1, 2] = k*sa*ca, 1-k*sa*sa, -s*sa
    R[2, 0], R[2, 1], R[2, 2] = -s*ca, s*sa, 1-k


@njit(error_model='numpy')
def _prepare(cam, heading, pitch, corners, lxs, lys, R, kz):
    """camera._prepare for one car: its corners (3, 8) rotated into the sensor frame (lxs, lys (8,)), the camera's
    tilt R (3, 3) and the corners' heights in the camera's rows kz (3, 8) = R[2, j] (z - camera height)."""
    rel = heading-cam[PSI]
    cr, sr = cos(rel), sin(rel)
    for c in range(8):
        lxs[c] = cr*corners[0, c]-sr*corners[1, c]
        lys[c] = sr*corners[0, c]+cr*corners[1, c]
    _tilt(cam[YAW], pitch, R)
    for c in range(8):
        zr = corners[2, c]-cam[Z]
        for j in range(3):
            kz[j, c] = R[2, j]*zr


@njit(error_model='numpy')
def _evaluate(cam, c0, s0, lxs, lys, R, kz, x, y, edges, near, out, depth, yc, zc):
    """out (4,) <- camera._evaluate of one prepared car at CG (x, y): the box of the part of its 3D box at least
    `near` in front of the camera, nan where none is; c0, s0 the cosine and sine of the sensor yaw cam[PSI].
    depth, yc, zc (8,) are scratch."""
    dx, dy = x-cam[X], y-cam[Y]
    ex, ey = c0*dx+s0*dy, c0*dy-s0*dx
    front = True
    for c in range(8):
        xs, ys = ex+lxs[c], ey+lys[c]
        depth[c] = R[0, 0]*xs+R[1, 0]*ys+kz[0, c]
        yc[c] = R[0, 1]*xs+R[1, 1]*ys+kz[1, c]
        zc[c] = R[0, 2]*xs+R[1, 2]*ys+kz[2, c]
        front = front and depth[c] >= near
    f, cu, cv = cam[F], cam[CU], cam[CV]
    u0, u1, v0, v1 = inf, -inf, inf, -inf
    if front:
        for c in range(8):
            u, v = cu-f*yc[c]/depth[c], cv-f*zc[c]/depth[c]
            u0, u1, v0, v1 = min(u0, u), max(u1, u), min(v0, v), max(v1, v)
        out[0], out[1], out[2], out[3] = u0, u1, v0, v1
        return
    seen = False                         # the corners in front and the crossings of the near plane (np.fmin, fmax)
    for c in range(8):
        if depth[c] >= near:
            u, v = cu-f*yc[c]/depth[c], cv-f*zc[c]/depth[c]
            u0, u1, v0, v1 = min(u0, u), max(u1, u), min(v0, v), max(v1, v)
            seen = True
    for e in range(edges.shape[0]):
        i, j = edges[e, 0], edges[e, 1]
        if (depth[i] >= near) != (depth[j] >= near):
            t = (near-depth[i])/(depth[j]-depth[i])
            u, v = cu-f*(yc[i]+t*(yc[j]-yc[i]))/near, cv-f*(zc[i]+t*(zc[j]-zc[i]))/near
            u0, u1, v0, v1 = min(u0, u), max(u1, u), min(v0, v), max(v1, v)
            seen = True
    if seen:
        out[0], out[1], out[2], out[3] = u0, u1, v0, v1
    else:
        out[0], out[1], out[2], out[3] = nan, nan, nan, nan


@njit(error_model='numpy')
def _drop_mates(boxes, keep, lo, hi, cam, pitch, mates, corners, edges, tune):
    """keep[lo:hi] &= no IoU above IOU_MATE with a teammate's clipped box (camera.mate_boxes, camera.iou) for the
    boxes lo:hi of one frame from camera cam with body pitch `pitch`, teammates mates (k, 3) [x, y, yaw]."""
    k = mates.shape[0]
    mb = np.empty((k, 4))
    ok = np.zeros(k, np.bool_)
    lxs, lys, d, yc, zc = np.empty(8), np.empty(8), np.empty(8), np.empty(8), np.empty(8)
    R, kz, b = np.empty((3, 3)), np.empty((3, 8)), np.empty(4)
    lim = (cam[W], cam[W], cam[H], cam[H])
    c0, s0 = cos(cam[PSI]), sin(cam[PSI])
    for m in range(k):
        _prepare(cam, mates[m, 2], pitch, corners, lxs, lys, R, kz)
        _evaluate(cam, c0, s0, lxs, lys, R, kz, mates[m, 0], mates[m, 1], edges, tune[T_NEAR], b, d, yc, zc)
        fin = True
        for e in range(4):
            fin = fin and isfinite(b[e])
            mb[m, e] = min(max(b[e], 0.0), lim[e])
        ok[m] = fin and mb[m, 1] > mb[m, 0] and mb[m, 3] > mb[m, 2]
    for i in range(lo, hi):
        if not keep[i]:
            continue
        a = boxes[i]
        for m in range(k):
            if not ok[m]:
                continue
            q = mb[m]
            w = max(min(a[1], q[1])-max(a[0], q[0]), 0.0)
            h = max(min(a[3], q[3])-max(a[2], q[2]), 0.0)
            inter = w*h
            union = max(a[1]-a[0], 0.0)*max(a[3]-a[2], 0.0)+max(q[1]-q[0], 0.0)*max(q[3]-q[2], 0.0)-inter
            if inter/(union if union > 0 else inf) > tune[T_IOU]:
                keep[i] = False
                break


@njit(error_model='numpy')
def _inv_var(x, sig, steps):
    """The inverse variance of depth cue x (1 + 2 k,) [value, +steps, -steps] by central differences (0: unusable)."""
    k = steps.shape[0]
    var = 0.0
    for q in range(k):
        var += (sig[q]*(x[1+q]-x[1+k+q])/(2*steps[q]))**2
    return 1/var if isfinite(x[0]) and isfinite(var) and var > 0 else 0.0


@njit(error_model='numpy')
def _cue_work(k):
    """_cue's scratch arrays for k nuisances (1 + 2 k evaluation points)."""
    n = 1+2*k
    return (np.empty(k), np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n), np.empty(n),
            np.empty((3, 3)), np.empty(6))


@njit(error_model='numpy')
def _cue(b, fit, cam, pitch, sigma_pitch, car, height, heading_sigma, steps, xy, P, work):
    """xy (2,), P (2, 2) <- camera._cue_init of one box b (4,) with usable edges fit (4,) from camera cam with
    estimated pitch and its sigma; car = (length, width, rear, front) (geometry.cg_offsets); work from _cue_work."""
    length, width, rear, front = car[0], car[1], car[2], car[3]
    f, cu, cv, h = cam[F], cam[CU], cam[CV], cam[Z]
    k = steps.shape[0]
    n = 1+2*k
    sig, ground, size, nx, ny, nz, px, py, R, z = work
    for q in range(4):
        sig[q] = cam[SIG]
    sig[4], sig[5] = max(sigma_pitch, 1e-9), abs(sin(cam[YAW]))*sigma_pitch
    hb, wb = (b[3]-b[2])/2, (b[1]-b[0])/2
    for r in range(n):
        z[0], z[1], z[2], z[3], z[4], z[5] = b[0], b[1], b[2], b[3], pitch, 0.0
        if 1 <= r <= k:
            z[r-1] += steps[r-1]
        elif r > k:
            z[r-1-k] -= steps[r-1-k]
        th, rho = z[4], z[5]
        ul, ur, vt, vb = z[0]-rho*hb, z[1]+rho*hb, z[2]-rho*wb, z[3]+rho*wb
        # camera._cue_depths
        _tilt(cam[YAW], th, R)
        a, bb = (cu-(ul+ur)/2)/f, (cv-vb)/f
        dx = R[0, 0]+R[0, 1]*a+R[0, 2]*bb
        dz = R[2, 0]+R[2, 1]*a+R[2, 2]*bb
        ground[r] = h*dx/(-dz) if dz < -1e-9 and fit[3] else nan
        nx[r], ny[r], nz[r] = R[0, 1]-a*R[0, 0], R[1, 1]-a*R[1, 0], R[2, 1]-a*R[2, 0]
        tp = th*cos(cam[YAW])
        c, s = cos(tp), sin(tp)
        above, below = height-h, h
        hp = vb-vt
        beta = c*s*(below-above)-f*height/(hp if hp > 0 else nan)
        size[r] = (-beta+np.sqrt(beta**2+4*c**2*s**2*above*below))/(2*c**2) if fit[2] and fit[3] else nan
    inv_ground, inv_size = _inv_var(ground, sig, steps), _inv_var(size, sig, steps)
    total = inv_ground+inv_size
    w = inv_ground/total if total > 0 else nan
    fallback = f*width/max(b[1]-b[0], 1.0)
    for r in range(n):
        if not isfinite(w):
            x = fallback
        elif w >= 1:
            x = ground[r]
        elif w <= 0:
            x = size[r]
        else:
            x = w*ground[r]+(1-w)*size[r]
        y = (nz[r]*h-nx[r]*x)/ny[r]
        phi = cam[PSI]+atan2(y, x)
        cph, sph = cos(phi), sin(phi)
        reach = hypot(x, y)+abs(cph)*length/2+abs(sph)*width/2
        px[r], py[r] = cam[X]+reach*cph-(rear+front)/2, cam[Y]+reach*sph
    P[0, 0] = P[0, 1] = P[1, 0] = P[1, 1] = 0.0
    for q in range(k):
        j0, j1 = (px[1+q]-px[1+k+q])/(2*steps[q]), (py[1+q]-py[1+k+q])/(2*steps[q])
        s2 = sig[q]**2
        P[0, 0] += j0*s2*j0
        P[0, 1] += j0*s2*j1
        P[1, 0] += j1*s2*j0
        P[1, 1] += j1*s2*j1
    xy[0], xy[1] = px[0], py[0]
    rx, ry = xy[0]-cam[X], xy[1]-cam[Y]
    depth = sqrt(rx*rx+ry*ry)
    nr = max(depth, 1e-9)
    ray0, ray1 = rx/nr, ry/nr
    perp0, perp1 = -ray1, ray0
    swing = (abs(ray0)*-rear+abs(ray1)*width/2)*heading_sigma
    across = swing**2+(0.0 if fit[0] and fit[1] else (length/2)**2)
    along = 0.0 if isfinite(w) else depth**2
    P[0, 0] = P[0, 0]+across*perp0*perp0+along*ray0*ray0
    P[0, 1] = P[0, 1]+across*perp0*perp1+along*ray0*ray1
    P[1, 0] = P[1, 0]+across*perp1*perp0+along*ray1*ray0
    P[1, 1] = P[1, 1]+across*perp1*perp1+along*ray1*ray1
    if not (isfinite(P[0, 0]) and isfinite(P[0, 1]) and isfinite(P[1, 0]) and isfinite(P[1, 1])):
        v = depth**2+length**2
        P[0, 0], P[0, 1], P[1, 0], P[1, 1] = v*1.0, v*0.0, v*0.0, v*1.0
    off = (P[0, 1]+P[1, 0])/2
    P[0, 0], P[1, 1] = (P[0, 0]+P[0, 0])/2, (P[1, 1]+P[1, 1])/2
    P[0, 1] = P[1, 0] = off


@njit(error_model='numpy')
def _solve(S, B, X, M):
    """X (4, 3) <- S^-1 B for S (4, 4), B (4, 3): Gaussian elimination with partial pivoting (as LAPACK's LU).
    M (4, 7) is scratch."""
    n, m = 4, B.shape[1]
    for i in range(n):
        for j in range(n):
            M[i, j] = S[i, j]
        for j in range(m):
            M[i, n+j] = B[i, j]
    for c in range(n):
        p = c
        for i in range(c+1, n):
            if abs(M[i, c]) > abs(M[p, c]):
                p = i
        if p != c:
            for j in range(n+m):
                M[c, j], M[p, j] = M[p, j], M[c, j]
        for i in range(c+1, n):
            l = M[i, c]/M[c, c]
            for j in range(c, n+m):
                M[i, j] -= l*M[c, j]
    for j in range(m):
        for i in range(n-1, -1, -1):
            s = M[i, n+j]
            for t in range(i+1, n):
                s -= M[i, t]*X[t, j]
            X[i, j] = s/M[i, i]


@njit(error_model='numpy')
def _normal(use, J, S, res, prior, A, g, Sm, B, Xs, M):
    """camera._normal for one box: A (2, 2) <- J' S^-1 J + prior and g (2,) <- J' S^-1 res over the edges in use
    (4,); returns the residual cost res' S^-1 res. Sm (4, 4), B, Xs (4, 3) and M (4, 7) are scratch."""
    for i in range(4):
        for j in range(4):
            Sm[i, j] = S[i, j] if use[i] and use[j] else (1.0 if i == j and not use[j] else 0.0)
        B[i, 0] = J[i, 0] if use[i] else 0.0
        B[i, 1] = J[i, 1] if use[i] else 0.0
        B[i, 2] = res[i] if use[i] else 0.0
    _solve(Sm, B, Xs, M)
    cost = 0.0
    for a in range(2):
        g[a] = 0.0
        for b in range(2):
            A[a, b] = 0.0
            for i in range(4):
                A[a, b] += B[i, a]*Xs[i, b]
            A[a, b] += prior[a, b]
        for i in range(4):
            g[a] += B[i, a]*Xs[i, 2]
    for i in range(4):
        cost += B[i, 2]*Xs[i, 2]
    return cost


@njit(error_model='numpy')
def _refine_work():
    """_refine's scratch arrays."""
    return ((np.empty((9, 8)), np.empty((9, 8)), np.empty((9, 3, 3)), np.empty((9, 3, 8))),
            (np.empty(8), np.empty(8), np.empty(8), np.empty((9, 4)), np.empty((6, 4)), np.empty((2, 4))),
            (np.empty(4), np.empty(4), np.empty((4, 2)), np.empty((4, 4)), np.empty(4, np.bool_)),
            (np.empty((2, 2)), np.empty((2, 2)), np.empty((2, 2)), np.empty((2, 2)), np.empty(2), np.empty(2)),
            (np.empty((4, 4)), np.empty((4, 3)), np.empty((4, 3)), np.empty((4, 7))))


@njit(error_model='numpy')
def _refine(z, fit, xy0, cam, pitch, sigma_pitch, corners, length, heading_sigma, iters, edges, steps, tune,
            cut_side, xy, P, work):
    """camera._refine_init for one box z (4,) from its cue estimate xy0 (2,): on success xy (2,) and P (2, 2) <- the
    lowest-cost iterate and its inverse information, returning True; False (xy, P untouched) where no iterate could
    be evaluated. The batch's rules hold per box: a step under CONVERGED ends it, and so does the last evaluation.
    work from _refine_work."""
    (lxs, lys, Rs, kz), (d, yc, zc, hb, rows, mean), (pred, res, J, S, use), (prior, A, info, best_A, g, g_info), \
        (Sm, B, Xs, M) = work
    near = tune[T_NEAR]
    sig2 = cam[SIG]**2
    rx, ry = xy0[0]-cam[X], xy0[1]-cam[Y]
    nrm = sqrt(rx*rx+ry*ry)
    nr = max(nrm, 1e-9)
    ray0, ray1 = rx/nr, ry/nr
    perp0, perp1 = -ray1, ray0
    pa2, L2 = tune[T_PRIOR]**2, length**2
    prior[0, 0] = perp0*perp0/L2+ray0*ray0/pa2
    prior[0, 1] = perp0*perp1/L2+ray0*ray1/pa2
    prior[1, 0] = perp1*perp0/L2+ray1*ray0/pa2
    prior[1, 1] = perp1*perp1/L2+ray1*ray1/pa2
    step = 1e-3*max(nrm, 1.0)
    ox = (0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    oy = (0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0)
    a0, a1 = tune[T_HSPREAD], tune[T_PSPREAD]
    w0, w1 = 1/(2*a0**2), 1/(2*a1**2)
    hs = a0*heading_sigma
    head = (0.0, 0.0, 0.0, 0.0, 0.0, hs, -hs, 0.0, 0.0)
    ps = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a1*sigma_pitch, -a1*sigma_pitch)
    root = (sqrt(w0), sqrt(w0), sqrt(w1), sqrt(w1), sqrt(1-2*w0), sqrt(1-2*w1))
    wq = (w0, w1)
    for i in range(9):
        _prepare(cam, head[i], pitch+ps[i], corners, lxs[i], lys[i], Rs[i], kz[i])
    best_A[0, 0], best_A[0, 1], best_A[1, 0], best_A[1, 1] = 1.0, 0.0, 0.0, 1.0
    px, py = xy0[0], xy0[1]
    bx, by, best_cost = px, py, inf
    cpsi, spsi = cos(cam[PSI]), sin(cam[PSI])
    for it in range(iters+1):
        good = True
        for i in range(9):
            _evaluate(cam, cpsi, spsi, lxs[i], lys[i], Rs[i], kz[i], px+ox[i]*step, py+oy[i]*step, edges, near,
                      hb[i], d, yc, zc)
            for e in range(4):
                good = good and isfinite(hb[i, e])
        if not good:
            hb[:] = 0.0
        for e in range(4):
            for q in range(2):
                dp_, dm_ = hb[5+2*q, e]-hb[0, e], hb[6+2*q, e]-hb[0, e]
                mean[q, e] = wq[q]*(dp_+dm_)
                rows[2*q, e] = (dp_-mean[q, e])*root[2*q]
                rows[2*q+1, e] = (dm_-mean[q, e])*root[2*q+1]
                rows[4+q, e] = mean[q, e]*root[4+q]
            pred[e] = hb[0, e]+(mean[0, e]+mean[1, e])
            J[e, 0] = (hb[1, e]-hb[2, e])/(2*step)
            J[e, 1] = (hb[3, e]-hb[4, e])/(2*step)
            res[e] = z[e]-pred[e]
            use[e] = fit[e] or cut_side[e]*(pred[e]-z[e]) > 0
        same = True
        for e in range(4):
            for e2 in range(4):
                s = 0.0
                for r in range(6):
                    s += rows[r, e]*rows[r, e2]
                S[e, e2] = (sig2 if e == e2 else 0.0)+s
            same = same and use[e] == fit[e]
        cost = _normal(use, J, S, res, prior, A, g, Sm, B, Xs, M)
        # A violated bound steers the estimate but holds no information: the covariance comes from the usable edges
        if same:
            info[:] = A
        else:
            _normal(fit, J, S, res, prior, info, g_info, Sm, B, Xs, M)
        d0, d1 = px-xy0[0], py-xy0[1]
        pd0, pd1 = prior[0, 0]*d0+prior[0, 1]*d1, prior[1, 0]*d0+prior[1, 1]*d1
        cost = cost+(d0*pd0+d1*pd1)
        better = good and cost < best_cost
        if better:
            bx, by, best_cost = px, py, cost
            best_A[:] = info
        if it == iters:
            break
        g0, g1 = g[0]-pd0, g[1]-pd1
        det = A[0, 0]*A[1, 1]-A[0, 1]*A[1, 0]
        if good:
            s0, s1 = (A[1, 1]*g0-A[0, 1]*g1)/det, (A[0, 0]*g1-A[1, 0]*g0)/det
        else:
            s0, s1 = 0.0, 0.0
        small = s0*g0+s1*g1 < tune[T_CONV]                    # dp' A dp: a step under 0.1 sigma ends the box
        length_ = sqrt(s0*s0+s1*s1)
        qx, qy = px-cam[X], py-cam[Y]
        lim = tune[T_TRUST]*sqrt(qx*qx+qy*qy)/(length_ if length_ > 1e-12 or length_ != length_ else 1e-12)
        scale = lim if lim < 1.0 or lim != lim else 1.0        # np.minimum: nan propagates
        s0, s1 = s0*scale, s1*scale
        if small:
            if better:
                bx, by = bx+s0, by+s1
            break
        px, py = px+s0, py+s1
    if not isfinite(best_cost):
        return False
    xy[0], xy[1] = bx, by
    det = best_A[0, 0]*best_A[1, 1]-best_A[0, 1]*best_A[1, 0]
    i00, i01, i10, i11 = best_A[1, 1]/det, -best_A[0, 1]/det, -best_A[1, 0]/det, best_A[0, 0]/det
    P[0, 0], P[1, 1] = (i00+i00)/2, (i11+i11)/2
    P[0, 1] = P[1, 0] = (i01+i10)/2
    return True


@njit(error_model='numpy')
def _border(b, cam, px, flags):
    """flags (4,) <- the edges of box b within px of the image border: left, right, top, bottom."""
    flags[0], flags[1] = b[0] <= px, b[1] >= cam[W]-px
    flags[2], flags[3] = b[2] <= px, b[3] >= cam[H]-px


@njit(error_model='numpy')
def process_boxes(boxes, frame, cams, pitch, pitch_noise, mates, mate_start, corners, car, height, heading_sigma,
                  pitch_extra, refine, iters, edges, steps, cut_side, tune):
    """The arithmetic of camera.process_cameras for one step's boxes (n, 4) (every frame's, concatenated; frame (n,)
    the frame of each) seen by cameras cams (frames, 12) (the CAM layout) with the cars' estimated pitch (frames,),
    the R pitch noise (frames,), teammates mates[mate_start[k]:mate_start[k + 1]] (., 3) [x, y, yaw] of frame k,
    corners (3, 8) (camera._local_corners) and car = (length, width, rear, front). Returns keep (n,) (a valid box,
    not a teammate's), and for the kept boxes truncated (n, 4), the init xy (n, 2) and P (n, 2, 2), and R (n, 4, 4)
    (zero rows elsewhere)."""
    n, nf = boxes.shape[0], cams.shape[0]
    keep = np.zeros(n, np.bool_)
    truncated = np.zeros((n, 4), np.bool_)
    xy, P, R = np.zeros((n, 2)), np.zeros((n, 2, 2)), np.zeros((n, 4, 4))
    lo = 0
    for k in range(nf):                  # boxes of a frame are contiguous: the teammate filter, frame by frame
        hi = lo
        while hi < n and frame[hi] == k:
            hi += 1
        anyk = False
        for i in range(lo, hi):
            b = boxes[i]
            keep[i] = (isfinite(b[0]) and isfinite(b[1]) and isfinite(b[2]) and isfinite(b[3]) and b[1] > b[0]
                       and b[3] > b[2])
            anyk = anyk or keep[i]
        if anyk and mate_start[k+1] > mate_start[k]:
            _drop_mates(boxes, keep, lo, hi, cams[k], pitch[k], mates[mate_start[k]:mate_start[k+1]], corners, edges,
                        tune)
        lo = hi
    fit = np.empty(4, np.bool_)
    xr, Pr, full, jac, spread = np.empty(2), np.empty((2, 2)), np.empty(4), np.empty(4), np.empty(4)
    lxs, lys, d, yc, zc = np.empty(8), np.empty(8), np.empty(8), np.empty(8), np.empty(8)
    Rt, kz, model = np.empty((3, 3)), np.empty((3, 8)), np.empty(4)
    cue_work, refine_work = _cue_work(steps.shape[0]), _refine_work()
    for i in range(n):
        if not keep[i]:
            continue
        cam, b, th = cams[frame[i]], boxes[i], pitch[frame[i]]
        margin = max(tune[T_TRUNC], tune[T_MARGIN]*cam[SIG])
        _border(b, cam, tune[T_TRUNC], truncated[i])
        _border(b, cam, margin, fit)
        cut = fit[0] or fit[1] or fit[2] or fit[3]
        for e in range(4):
            fit[e] = not fit[e]
        sigma_pitch = cam[PSIG]+pitch_extra
        _cue(b, fit, cam, th, sigma_pitch, car, height, heading_sigma, steps, xy[i], P[i], cue_work)
        if refine and _refine(b, fit, xy[i], cam, th, sigma_pitch, corners, car[0], heading_sigma, iters, edges,
                              steps, tune, cut_side, xr, Pr, refine_work):
            xy[i] = xr
            P[i] = Pr
        # past a cut edge, the pitch moves an uncut edge with the extent of the model box at the init position
        full[:] = b
        if cut:
            _prepare(cam, 0.0, th, corners, lxs, lys, Rt, kz)
            _evaluate(cam, cos(cam[PSI]), sin(cam[PSI]), lxs, lys, Rt, kz, xy[i, 0], xy[i, 1], edges, tune[T_NEAR],
                      model, d, yc, zc)
            for e in range(4):
                if not fit[e] and isfinite(model[e]) and cut_side[e]*(model[e]-b[e]) < 0:
                    full[e] = model[e]
        # camera._pitch_jacobian and R
        f, cu, cv = cam[F], cam[CU], cam[CV]
        a, r = cos(cam[YAW]), sin(cam[YAW])
        um, vm = (full[0]+full[1])/2-cu, (full[2]+full[3])/2-cv
        for e in range(2):
            jac[e] = -a*(full[e]-cu)*vm/f+r*vm
            jac[2+e] = -a*f*(1+((full[2+e]-cv)/f)**2)-r*um
        spread[0], spread[1] = (full[2]-full[3])/2, (full[3]-full[2])/2
        spread[2], spread[3] = (full[0]-full[1])/2, (full[1]-full[0])/2
        sp = pitch_noise[frame[i]]
        s2, sr2, sig2 = sp**2, (r*sp)**2, cam[SIG]**2
        for e in range(4):
            for e2 in range(4):
                R[i, e, e2] = (sig2 if e == e2 else 0.0)+s2*jac[e]*jac[e2]+sr2*spread[e]*spread[e2]
    return keep, truncated, xy, P, R
