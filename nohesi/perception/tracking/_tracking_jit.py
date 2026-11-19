"""Compiled kernels of the team tracker (tracker.py, association.py, existence.py, imm.py): the per-track, per-pair
and per-ray loops of a fusion cycle, each in one call, instead of numpy's many small array operations per cycle.

- predict_rows(means, covs, mu, dt, ...): tracker.predict_rows (IMM mixing, each model's unscented prediction).
- sigma_rows(means, covs, sc): ukf.sigma_points of stacked filters.
- gate_pairs(...): the measurement part of TeamTracker._gate, association.evaluate (each sigma point moved to its
  detection's time, then the lidar, radar or camera model), association.moments and association.imm_gate.
- joint_update(...): tracker.joint_update.
- coarse_ok(...): association.coarse_candidates' (track, detection) test.
- coverage(...): TeamTracker._coverage (with _visibility and _doppler), existence.expected_pd and doppler_pass;
  visibility(...): existence.visibility.
- existence(...): TeamTracker._existence (existence.miss_log_factor, hit_log_factor, survive, apply_log_odds).
- birth_rows(...): the lidar and radar rows of TeamTracker._birth_states; occupied(...): TeamTracker._occupied.
- assign_parts(...): the cost matrices of TeamTracker._assign's per-scan problems (scipy's linear_sum_assignment
  still solves them).
- cv_rows(...): tracker._cv; combine(...), lane_rows(...): imm.imm_combine and imm.lane_probability.

Each kernel is the numpy code written out for one track, pair or ray, in the same order of operations, so the two
paths agree to rounding (the math library's last bit, matrix-product summation order, and a hand-written Cholesky
factor in place of LAPACK's for the sigma points). The numpy functions stay the reference and the path with
NOHESI_JIT=0 (perception/_jit.py); there these kernels would run as plain Python, correct but slow, and the callers do
not use them. A kernel that meets a matrix the plain factorization cannot take (a pivot that is not positive, where
the numpy code turns to ukf.robust_cholesky or ukf.floor_eigenvalues) returns ok False, and its caller redoes the
call in numpy, so the safeguards stay in one place.

The constants come in as arguments (the rows MOTION, GEOM, SPEC, COVER and BIRTH laid out by the *_FIELDS tuples
below, the sigma-point weights), not as module globals: numba's on-disk cache keys on this file only, so a global read
from models.py would keep its old value after an edit there (the structural ones kept here, the kind codes and the
box's corner signs and edges, are checked against association's by the tests). Arguments are float64 / int64 / bool
arrays in C order (the callers make them so), so each kernel compiles once; the first use on a machine compiles all
of them (about half a minute), later processes load them from the cache.
"""
from math import (atan2, copysign, cos, erf, erfc, exp, expm1, floor, hypot, inf, log, log1p, nan, pi, sin,
                  sqrt, tanh)
import numpy as np
from .._jit import njit

__all__ = ['MOTION_FIELDS', 'GEOM_FIELDS', 'SPEC_FIELDS', 'COVER_FIELDS', 'BIRTH_FIELDS', 'predict_rows',
           'sigma_rows', 'gate_pairs', 'joint_update', 'coarse_ok', 'coverage', 'visibility', 'existence',
           'birth_rows', 'occupied', 'assign_parts', 'cv_rows', 'combine', 'lane_rows']

MOTION_FIELDS = ('lane_keep_rate', 'lc_speed', 'lanes', 'lane_width', 'half_width', 'imm_switch', 'q_long2',
                 'q_lat2', 'mu_floor')
K_RATE, K_SPEED, K_LANES, K_W, K_HALF, K_SWITCH, K_QL, K_QD, K_FLOOR = range(9)
GEOM_FIELDS = ('shift', 'half_length', 'half_width', 'v_min', 'near', 'car_height')
G_SHIFT, G_HL, G_HW, G_VMIN, G_NEAR, G_HEIGHT = range(6)
TWO_PI = 2*pi
LIDAR, RADAR, CAMERA = 0, 1, 2          # association's kind codes
SIGNS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))     # footprint corners (association._SIGNS)
EDGES = ((0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7))


# ------------------------------------------------------------------------------------------------ scalar helpers
@njit(error_model='numpy')
def _nmin(a, b):
    """np.minimum of two floats (NaN propagates)."""
    if a != a or b != b:
        return nan
    return a if a < b else b


@njit(error_model='numpy')
def _nmax(a, b):
    """np.maximum of two floats (NaN propagates)."""
    if a != a or b != b:
        return nan
    return a if a > b else b


@njit(error_model='numpy')
def _clip(a, lo, hi):
    """np.clip of a float (NaN stays NaN)."""
    if a < lo:
        return lo
    if a > hi:
        return hi
    return a


@njit(error_model='numpy')
def _wrap(a):
    """association.wrap_angle: [-pi, pi)."""
    return a-TWO_PI*floor((a+pi)/TWO_PI)


@njit(error_model='numpy')
def _res(a, b, wrap):
    """association._res of two floats: a - b, wrapped when wrap and it leaves [-pi, pi)."""
    r = a-b
    if wrap and (r < -pi or r >= pi):
        r = _wrap(r)
    return r


@njit(error_model='numpy')
def _chol(S, L):
    """L (n, n) <- the lower Cholesky factor of S, column by column in association.cholesky's order of operations.
    False (L unfinished) at a pivot that is not positive."""
    n = S.shape[0]
    for j in range(n):
        d = S[j, j]
        for k in range(j):
            d = d-L[j, k]*L[j, k]
        if not d > 0:
            return False
        c = sqrt(d)
        L[j, j] = c
        inv = 1/c
        for i in range(j+1, n):
            v = S[i, j]
            for k in range(j):
                v = v-L[i, k]*L[j, k]
            L[i, j] = v*inv
        for i in range(j):
            L[i, j] = 0.0
    return True


@njit(error_model='numpy')
def _forward(L, v, out):
    """out <- L^-1 v for lower triangular L (n, n) and v (n,) (association.forward)."""
    n = L.shape[0]
    for i in range(n):
        acc = v[i]
        for k in range(i):
            acc = acc-L[i, k]*out[k]
        out[i] = acc/L[i, i]


@njit(error_model='numpy')
def _logdet(L):
    """log det of L L^T (association.logdet)."""
    s = 0.0
    for i in range(L.shape[0]):
        s += log(L[i, i])
    return 2*s


@njit(error_model='numpy')
def _decay(dt, mp):
    """(E, g) of models.motion over dt: E = exp(-k dt) and g = (1 - E) / k (dt for k = 0), the same for every point
    moved by dt."""
    k = mp[K_RATE]
    return exp(-k*dt), (-expm1(-k*dt)/k if k > 0 else dt)


@njit(error_model='numpy')
def _move(x, dt, step, mp, E, g, out):
    """out (4,) <- the models.motion step of state x (4,) over dt for the model with lateral step (0 keep, +1 left,
    -1 right); mp the MOTION row, (E, g) = _decay(dt, mp)."""
    k, speed, lanes, w, half = mp[K_RATE], mp[K_SPEED], mp[K_LANES], mp[K_W], mp[K_HALF]
    s, d, vs, vd = x[0], x[1], x[2], x[3]
    lane = _clip(floor((d+half)/w), 0.0, lanes-1)
    c = _clip((lane+step+0.5)*w-half, -half, half)
    out[0] = s+vs*dt
    out[2] = vs
    if step == 0:
        e = d-c
        out[1] = c+E*(e*(1+k*dt)+vd*dt)
        out[3] = E*(vd*(1-k*dt)-k*k*dt*e)
        return
    r = c-d
    sgn = 1.0 if r > 0 else (-1.0 if r < 0 else (0.0 if r == 0 else nan))
    vt = sgn*speed
    dn = d+vt*dt+(vd-vt)*g
    if sgn*(dn-c) >= 0:
        out[1] = c
        out[3] = 0.0
    else:
        out[1] = dn
        out[3] = vt+(vd-vt)*E


@njit(error_model='numpy')
def _points(m, P, sc, X, L, Ps):
    """X (2n+1, n) <- ukf.sigma_points of N(m, P) with scale sqrt(c) = sc; L, Ps (n, n) scratch. False when the
    symmetrized P has no plain Cholesky factor."""
    n = m.shape[0]
    for a in range(n):
        for b in range(n):
            Ps[a, b] = 0.5*(P[a, b]+P[b, a])
    if not _chol(Ps, L):
        return False
    for a in range(n):
        X[0, a] = m[a]
    for i in range(n):
        for a in range(n):
            g = L[a, i]*sc
            X[1+i, a] = m[a]+g
            X[1+n+i, a] = m[a]-g
    return True


# ------------------------------------------------------------------------------------------------ prediction
@njit(error_model='numpy')
def _transition(rate, dt, Pi):
    """Pi (3, 3) <- tracker.transition_rows(rate, |dt|) of one row."""
    if rate <= 0:
        for i in range(3):
            for j in range(3):
                Pi[i, j] = 1.0 if i == j else 0.0
        return
    a, b = rate, 10*rate
    lam = 2*a+b
    pk, pl = b/lam, a/lam
    e2, e1 = exp(-lam*dt), exp(-b*dt)
    Pi[0, 0] = pk+(1-pk)*e2
    Pi[0, 1] = Pi[0, 2] = pl*(1-e2)
    back = pk*(1-e2)
    Pi[1, 0] = Pi[2, 0] = back
    Pi[1, 1] = Pi[2, 2] = (1-back+e1)/2
    Pi[1, 2] = Pi[2, 1] = (1-back-e1)/2


@njit(error_model='numpy')
def predict_rows(means, covs, mu, dt, steps, mp, wm, wc, sc):
    """tracker.predict_rows of tracks (P, J, n), (P, J, n, n), (P, J) by their own dt (P,): (means, covs, mu, ok).
    steps (J,) the models' lateral steps, mp the MOTION row, wm, wc, sc the sigma-point weights and scale. Rows with
    dt = 0 come back unchanged."""
    P, J, n = means.shape
    om, oc, ou = means.copy(), covs.copy(), mu.copy()
    K = 2*n+1
    Pi = np.empty((3, 3))
    cbar = np.empty(J)
    w = np.empty((J, J))
    m0 = np.empty((J, n))
    P0 = np.empty((n, n))
    L, Ps = np.empty((n, n)), np.empty((n, n))
    X = np.empty((K, n))
    Y = np.empty((K, n))
    m = np.empty(n)
    for p in range(P):
        t = dt[p]
        if t == 0:
            continue
        a = abs(t)
        E, g = _decay(t, mp)
        _transition(mp[K_SWITCH], a, Pi)
        for j in range(J):
            acc = mu[p, 0]*Pi[0, j]
            for i in range(1, J):
                acc = acc+mu[p, i]*Pi[i, j]
            cbar[j] = max(acc, mp[K_FLOOR])
        for i in range(J):
            for j in range(J):
                w[i, j] = mu[p, i]*Pi[i, j]/cbar[j]
        for j in range(J):
            for q in range(n):
                acc = 0.0
                for i in range(J):
                    acc += w[i, j]*means[p, i, q]
                m0[j, q] = acc
        q3, q2 = a**3/3, a**2/2
        for j in range(J):
            for r in range(n):
                for c in range(n):
                    acc, spread = 0.0, 0.0
                    for i in range(J):
                        acc += w[i, j]*covs[p, i, r, c]
                        spread += (means[p, i, r]-m0[j, r])*w[i, j]*(means[p, i, c]-m0[j, c])
                    P0[r, c] = acc+spread
            if not _points(m0[j], P0, sc, X, L, Ps):
                return om, oc, ou, False
            for k in range(K):
                _move(X[k], t, steps[j], mp, E, g, Y[k])
            for q in range(n):
                acc = 0.0
                for k in range(K):
                    acc += wm[k]*Y[k, q]
                m[q] = acc
                om[p, j, q] = acc
            for r in range(n):
                for c in range(n):
                    acc = 0.0
                    for k in range(K):
                        acc += (Y[k, r]-m[r])*wc[k]*(Y[k, c]-m[c])
                    P0[r, c] = acc
            # noise_rows: q_long on (s, v_s) = (0, 2), q_lat on (d, v_d) = (1, 3)
            P0[0, 0] += mp[K_QL]*q3
            P0[0, 2] += mp[K_QL]*q2
            P0[2, 0] += mp[K_QL]*q2
            P0[2, 2] += mp[K_QL]*a
            P0[1, 1] += mp[K_QD]*q3
            P0[1, 3] += mp[K_QD]*q2
            P0[3, 1] += mp[K_QD]*q2
            P0[3, 3] += mp[K_QD]*a
            for r in range(n):
                for c in range(n):
                    oc[p, j, r, c] = 0.5*(P0[r, c]+P0[c, r])
        tot = cbar[0]
        for j in range(1, J):
            tot = tot+cbar[j]
        for j in range(J):
            ou[p, j] = cbar[j]/tot
    return om, oc, ou, True


@njit(error_model='numpy')
def sigma_rows(means, covs, sc):
    """ukf.sigma_points of stacked filters (..., n), (..., n, n) given as (B, n), (B, n, n): (X (B, 2n+1, n), ok)."""
    B, n = means.shape
    X = np.empty((B, 2*n+1, n))
    L, Ps = np.empty((n, n)), np.empty((n, n))
    for b in range(B):
        if not _points(means[b], covs[b], sc, X[b], L, Ps):
            return X, False
    return X, True


# ------------------------------------------------------------------------------------------------ gating
@njit(error_model='numpy')
def _body(x, gp):
    """association._body of one state: body centre (bx, by) and heading (c, s)."""
    vs = max(x[2], gp[G_VMIN])
    nn = hypot(vs, x[3])
    c, s = vs/nn, x[3]/nn
    return x[0]+gp[G_SHIFT]*c, x[1]+gp[G_SHIFT]*s, c, s


@njit(error_model='numpy')
def _h_lidar(x, dzdv, vp, gp, out):
    """models.h_lidar plus the box fit's velocity term dz_dv (v_prior - v) (association.evaluate)."""
    h = atan2(x[3], max(x[2], gp[G_VMIN]))
    e0, e1 = vp[0]-x[2], vp[1]-x[3]
    out[0] = x[0]+gp[G_SHIFT]*cos(h)+(dzdv[0, 0]*e0+dzdv[0, 1]*e1)
    out[1] = x[1]+gp[G_SHIFT]*sin(h)+(dzdv[1, 0]*e0+dzdv[1, 1]*e1)
    out[2] = h+(dzdv[2, 0]*e0+dzdv[2, 1]*e1)
    out[3] = 0.0


@njit(error_model='numpy')
def _h_radar(x, pose, vsen, gp, out):
    """association.h_radar_rows of one state."""
    bx, by, c, s = _body(x, gp)
    hl, hw = gp[G_HL], gp[G_HW]
    ex, ey = pose[0]-bx, pose[1]-by
    lx, ly = c*ex+s*ey, -s*ex+c*ey
    px, py = _clip(lx, -hl, hl), _clip(ly, -hw, hw)
    if abs(lx) < hl and abs(ly) < hw:
        if hl-abs(lx) <= hw-abs(ly):
            px, py = copysign(hl, lx), ly
        else:
            px, py = lx, copysign(hw, ly)
    rx, ry = bx+c*px-s*py-pose[0], by+s*px+c*py-pose[1]
    rng = hypot(rx, ry)
    out[0] = rng
    out[1] = _wrap(atan2(ry, rx)-pose[2])
    out[2] = ((x[2]-vsen[0])*rx+(x[3]-vsen[1])*ry)/max(rng, 1e-9)
    out[3] = 0.0


@njit(error_model='numpy')
def _h_camera(x, pose, cam, gp, T, cy, sy, xc, yc, zc, out):
    """association.h_camera_rows of one state; T (3, 3) the camera's tilt (geometry.camera_tilt), cy, sy the cosine
    and sine of its yaw, xc, yc, zc (8,) scratch."""
    bx, by, c, s = _body(x, gp)
    ex, ey = bx-pose[0], by-pose[1]
    x0, y0 = cy*ex+sy*ey, -sy*ex+cy*ey
    cr, sr = c*cy+s*sy, s*cy-c*sy
    hl, hw = gp[G_HL], gp[G_HW]
    z0, z1 = -pose[3], gp[G_HEIGHT]-pose[3]
    for q in range(4):
        sx, sw = SIGNS[q]
        fx = x0+sx*(hl*cr)-sw*(hw*sr)
        fy = y0+sx*(hl*sr)+sw*(hw*cr)
        for j in range(3):
            v = T[0, j]*fx+T[1, j]*fy
            if j == 0:
                xc[q], xc[4+q] = v+T[2, j]*z0, v+T[2, j]*z1
            elif j == 1:
                yc[q], yc[4+q] = v+T[2, j]*z0, v+T[2, j]*z1
            else:
                zc[q], zc[4+q] = v+T[2, j]*z0, v+T[2, j]*z1
    W, Hh, f = cam[0], cam[1], cam[2]
    near = gp[G_NEAR]
    u0, u1, v0, v1 = inf, -inf, inf, -inf
    allok = True
    for q in range(8):
        inv = f/xc[q]
        U, V = W/2-yc[q]*inv, Hh/2-zc[q]*inv
        u0, u1, v0, v1 = _nmin(u0, U), _nmax(u1, U), _nmin(v0, V), _nmax(v1, V)
        if not xc[q] > near:
            allok = False
    if not allok:
        u0, u1, v0, v1 = inf, -inf, inf, -inf
        anyok = False
        for q in range(8):
            if xc[q] > near:
                inv = f/xc[q]
                U, V = W/2-yc[q]*inv, Hh/2-zc[q]*inv
                u0, u1, v0, v1 = _nmin(u0, U), _nmax(u1, U), _nmin(v0, V), _nmax(v1, V)
                anyok = True
        for e in range(12):
            a, b = EDGES[e]
            xa, xb = xc[a], xc[b]
            if (xa-near)*(xb-near) < 0:
                t = (near-xa)/(xb-xa)
                ye, ze, xe = yc[a]+t*(yc[b]-yc[a]), zc[a]+t*(zc[b]-zc[a]), xa+t*(xb-xa)
                U, V = W/2-f*ye/xe, Hh/2-f*ze/xe
                u0, u1, v0, v1 = _nmin(u0, U), _nmax(u1, U), _nmin(v0, V), _nmax(v1, V)
                anyok = True
        if not anyok:
            u0, u1, v0, v1 = W/2, W/2, Hh/2, Hh/2
    out[0] = _nmin(_nmax(u0, 0.0), W)
    out[1] = _nmin(_nmax(u1, 0.0), W)
    out[2] = _nmin(_nmax(v0, 0.0), Hh)
    out[3] = _nmin(_nmax(v1, 0.0), Hh)


@njit(error_model='numpy')
def gate_pairs(X, row, dj, off, mu, kind, z, R_fit, dz_dv, v_prior, pose, vs, cam, wrap, steps, mp, gp, wm, wc):
    """The measurement part of TeamTracker._gate for pairs p (track row row[p] of the predicted sigma points X
    (U, J, K, n), detection dj[p] of the DetTable arrays kind ... wrap, time offset off[p] from the cycle time, the
    track's model probabilities mu (P, J)): every point moved by its model's motion over the offset, the detection's
    measurement model, association.moments and association.imm_gate. Returns (zhat (P, J, M), nu, Pzz (P, J, M, M),
    D (P, J, n, M), R (P, M, M) (R_fit with a lidar offset's position noise), loglik (P,), nis (P,), ok)."""
    P = len(dj)
    U, J, K, n = X.shape
    M = z.shape[1]
    zhat = np.zeros((P, J, M))
    nu = np.zeros((P, J, M))
    Pzz = np.zeros((P, J, M, M))
    D = np.zeros((P, J, n, M))
    R = np.zeros((P, M, M))
    ll, nis = np.zeros(P), np.zeros(P)
    Z = np.empty((K, M))
    dZ = np.empty((K, M))
    xm = np.empty(n)
    T = np.empty((3, 3))
    xc, yc, zc = np.empty(8), np.empty(8), np.empty(8)
    S = np.empty((J, M, M))
    L, Sc = np.empty((M, M)), np.empty((M, M))
    wv, r_, zcen = np.empty(M), np.empty(M), np.empty(M)
    a = np.empty(J)
    dk = sqrt(wm[1]/2)
    for p in range(P):
        d, o, kd = dj[p], off[p], kind[dj[p]]
        E, g = _decay(o, mp)
        if kd == CAMERA:
            ya, pt = cam[d, 4], cam[d, 3]         # geometry._tilt
            sa, ca = sin(ya), cos(ya)
            s_, k_ = sin(pt), 1-cos(pt)
            T[0, 0], T[0, 1], T[0, 2] = 1-k_*ca*ca, k_*sa*ca, s_*ca
            T[1, 0], T[1, 1], T[1, 2] = k_*sa*ca, 1-k_*sa*sa, -s_*sa
            T[2, 0], T[2, 1], T[2, 2] = -s_*ca, s_*sa, 1-k_
            cy, sy = cos(pose[d, 2]), sin(pose[d, 2])
        else:
            cy, sy = 1.0, 0.0
        for j in range(J):
            for k in range(K):
                x = X[row[p], j, k]
                if o != 0:
                    _move(x, o, steps[j], mp, E, g, xm)
                    x = xm
                if kd == LIDAR:
                    _h_lidar(x, dz_dv[d], v_prior[d], gp, Z[k])
                elif kd == RADAR:
                    _h_radar(x, pose[d], vs[d], gp, Z[k])
                else:
                    _h_camera(x, pose[d], cam[d], gp, T, cy, sy, xc, yc, zc, Z[k])
            # association.moments
            for c in range(M):
                acc = 0.0
                for k in range(K):
                    acc += wm[k]*_res(Z[k, c], Z[0, c], wrap[d, c])
                zh = Z[0, c]+acc
                zhat[p, j, c] = zh
                for k in range(K):
                    dZ[k, c] = _res(Z[k, c], zh, wrap[d, c])
                nu[p, j, c] = _res(z[d, c], zh, wrap[d, c])
            for r in range(M):
                for c in range(M):
                    acc = 0.0
                    for k in range(K):
                        acc += dZ[k, r]*wc[k]*dZ[k, c]
                    Pzz[p, j, r, c] = acc
            for i in range(n):
                for c in range(M):
                    D[p, j, i, c] = (dZ[1+i, c]-dZ[1+n+i, c])*dk
        for r in range(M):
            for c in range(M):
                R[p, r, c] = R_fit[d, r, c]
        if kd == LIDAR and o != 0:
            a3 = abs(o)**3
            R[p, 0, 0] += mp[K_QL]*a3/3
            R[p, 1, 1] += mp[K_QD]*a3/3
        # association.imm_gate
        for j in range(J):
            for r in range(M):
                for c in range(M):
                    S[j, r, c] = Pzz[p, j, r, c]+R[p, r, c]
            if not _chol(S[j], L):
                return zhat, nu, Pzz, D, R, ll, nis, False
            _forward(L, nu[p, j], wv)
            ss = 0.0
            for c in range(M):
                ss += wv[c]*wv[c]
            a[j] = log(max(mu[p, j], mp[K_FLOOR]))-0.5*(ss+_logdet(L)+M*log(TWO_PI))
        top = a[0]
        for j in range(1, J):
            top = max(top, a[j])
        se = 0.0
        for j in range(J):
            se += exp(a[j]-top)
        ll[p] = top+log(se)
        for c in range(M):
            acc = 0.0
            for j in range(J):
                acc += mu[p, j]*_res(zhat[p, j, c], zhat[p, 0, c], wrap[d, c])
            zcen[c] = zhat[p, 0, c]+acc
        for r in range(M):
            for c in range(M):
                acc, spread = 0.0, 0.0
                for j in range(J):
                    acc += mu[p, j]*S[j, r, c]
                    spread += _res(zhat[p, j, r], zcen[r], wrap[d, r])*mu[p, j]*_res(zhat[p, j, c], zcen[c], wrap[d, c])
                Sc[r, c] = acc+spread
        if not _chol(Sc, L):
            return zhat, nu, Pzz, D, R, ll, nis, False
        for c in range(M):
            r_[c] = _res(z[d, c], zcen[c], wrap[d, c])
        _forward(L, r_, wv)
        ss = 0.0
        for c in range(M):
            ss += wv[c]*wv[c]
        nis[p] = ss
    return zhat, nu, Pzz, D, R, ll, nis, True


# ------------------------------------------------------------------------------------------------ update
@njit(error_model='numpy')
def joint_update(means, covs, mu, X, wm1, start, nu, Pzz, D, R, mu_floor):
    """tracker.joint_update of tracks (U, J, n) ..., sigma points X (U, J, 2n+1, n), detections sorted by track with
    start (U,) the first of each track's: (means, covs (symmetrized), mu, ok, healthy). ok False: some R* had no
    plain Cholesky factor; healthy False: some posterior covariance is not positive definite (the caller floors its
    eigenvalues, tracker._healthy)."""
    U, J, n = means.shape
    Pn, _, M = nu.shape
    om, oc, ou = np.empty((U, J, n)), np.empty((U, J, n, n)), np.empty((U, J))
    Rs, L = np.empty((M, M)), np.empty((M, M))
    Aw = np.empty((M, n))
    col, cw = np.empty(M), np.empty(M)
    wv = np.empty(M)
    Y, Lc, Lci, inv, G = np.empty((n, n)), np.empty((n, n)), np.empty((n, n)), np.empty((n, n)), np.empty((n, n))
    eye_col, tmp = np.empty(n), np.empty(n)
    A2, Hm = np.empty((n, n)), np.empty((n, n))
    b, xi = np.empty(n), np.empty(n)
    a = np.empty(J)
    gs = sqrt(2*wm1)
    healthy = True
    for u in range(U):
        p0 = start[u]
        p1 = start[u+1] if u+1 < U else Pn
        for j in range(J):
            for r in range(n):
                b[r] = 0.0
                for c in range(n):
                    Y[r, c] = 0.0
            q = 0.0
            for p in range(p0, p1):
                for r in range(M):
                    for c in range(M):
                        dd = 0.0
                        for i in range(n):
                            dd += D[p, j, i, r]*D[p, j, i, c]
                        Rs[r, c] = R[p, r, c]+Pzz[p, j, r, c]-dd
                if not _chol(Rs, L):
                    return om, oc, ou, False, healthy
                for i in range(n):                          # Aw = L^-1 D^T, column by column
                    for r in range(M):
                        col[r] = D[p, j, i, r]
                    _forward(L, col, cw)
                    for r in range(M):
                        Aw[r, i] = cw[r]
                _forward(L, nu[p, j], wv)
                for r in range(n):
                    for c in range(n):
                        acc = 0.0
                        for k in range(M):
                            acc += Aw[k, r]*Aw[k, c]
                        Y[r, c] += acc
                    acc = 0.0
                    for k in range(M):
                        acc += Aw[k, r]*wv[k]
                    b[r] += acc
                ss = 0.0
                for k in range(M):
                    ss += wv[k]*wv[k]
                q += ss+_logdet(L)
            for r in range(n):
                for c in range(n):
                    Hm[r, c] = Y[r, c]+(1.0 if r == c else 0.0)
            if not _chol(Hm, Lc):
                return om, oc, ou, False, healthy
            for c in range(n):                              # Lci = Lc^-1 I
                for r in range(n):
                    eye_col[r] = 1.0 if r == c else 0.0
                _forward(Lc, eye_col, tmp)
                for r in range(n):
                    Lci[r, c] = tmp[r]
            for r in range(n):
                for c in range(n):
                    acc = 0.0
                    for k in range(n):
                        acc += Lci[k, r]*Lci[k, c]
                    inv[r, c] = acc
            bx = 0.0
            for r in range(n):
                acc = 0.0
                for c in range(n):
                    acc += inv[r, c]*b[c]
                xi[r] = acc
                bx += b[r]*acc
            a[j] = log(max(mu[u, j], mu_floor))-0.5*(q-bx+_logdet(Lc))
            for i in range(n):
                for c in range(n):
                    G[i, c] = (X[u, j, 1+i, c]-X[u, j, 0, c])*gs
            for c in range(n):
                acc = 0.0
                for i in range(n):
                    acc += G[i, c]*xi[i]
                om[u, j, c] = means[u, j, c]+acc
            for r in range(n):                              # A2 = G^T (I - inv)
                for c in range(n):
                    acc = 0.0
                    for k in range(n):
                        acc += G[k, r]*((1.0 if k == c else 0.0)-inv[k, c])
                    A2[r, c] = acc
            for r in range(n):
                for c in range(n):
                    acc = 0.0
                    for k in range(n):
                        acc += A2[r, k]*G[k, c]
                    Hm[r, c] = covs[u, j, r, c]-acc
            for r in range(n):
                for c in range(n):
                    oc[u, j, r, c] = 0.5*(Hm[r, c]+Hm[c, r])
            if healthy and not _chol(oc[u, j], Lc):
                healthy = False
        top = a[0]
        for j in range(1, J):
            top = max(top, a[j])
        tot = 0.0
        for j in range(J):
            a[j] = exp(a[j]-top)
            tot += a[j]
        tot2 = 0.0
        for j in range(J):
            a[j] = max(a[j]/tot, mu_floor)
            tot2 += a[j]
        for j in range(J):
            ou[u, j] = a[j]/tot2
    return om, oc, ou, True, healthy


# ------------------------------------------------------------------------------------------------ coarse gate
@njit(error_model='numpy')
def coarse_ok(mean, cov, t, time, xy, C, has_xy, kind, pose, z, vs, rr_var, dof, half, ql, qd, gate2, gate1,
              rr_margin2, min_depth):
    """(n, D) association.coarse_candidates' test of every (track, detection) pair: mean (n, 4), cov (n, 4, 4) at
    times t (n,); the DetTable arrays of the D real detections (rr_var: R[:, 2, 2]; half: a camera's half field of
    view plus the margin); ql, qd the squared accelerations; gate2, gate1 the 2D and range-rate gates."""
    n, Dn = len(mean), len(time)
    ok = np.zeros((n, Dn), np.bool_)
    for d in range(Dn):
        if not dof[d] > 0:
            continue
        kd = kind[d]
        u0, u1 = 0.0, 0.0
        if kd == RADAR or kd == CAMERA:
            phi = pose[d, 2]+z[d, 1] if kd == RADAR else pose[d, 2]
            u0, u1 = cos(phi), sin(phi)
        for i in range(n):
            dt = time[d]-t[i]
            ex = mean[i, 0]+mean[i, 2]*dt-xy[d, 0]
            ey = mean[i, 1]+mean[i, 3]*dt-xy[d, 1]
            c = cov[i]
            dt2 = dt*dt
            adt = abs(dt)
            a3 = adt**3
            if has_xy[d]:
                a = c[0, 0]+2*dt*c[0, 2]+dt2*c[2, 2]+C[d, 0, 0]+ql*a3/3
                b = c[0, 1]+dt*(c[0, 3]+c[2, 1])+dt2*c[2, 3]+C[d, 0, 1]
                dd = c[1, 1]+2*dt*c[1, 3]+dt2*c[3, 3]+C[d, 1, 1]+qd*a3/3
                m2 = (dd*ex*ex-2*b*ex*ey+a*ey*ey)/(a*dd-b*b)
                if not m2 <= gate2:
                    continue
            if kd == RADAR:
                rr = (mean[i, 2]*u0+mean[i, 3]*u1)-(vs[d, 0]*u0+vs[d, 1]*u1)
                var = (c[2, 2]*u0**2+2*c[2, 3]*u0*u1+c[3, 3]*u1**2+rr_var[d]+rr_margin2
                       + adt*(ql*u0**2+qd*u1**2))
                if not (rr-z[d, 2])**2 <= gate1*var:
                    continue
            elif kd == CAMERA:
                px, py = ex+xy[d, 0]-pose[d, 0], ey+xy[d, 1]-pose[d, 1]
                depth = px*u0+py*u1
                lat = -px*u1+py*u0
                if not (depth > min_depth and abs(atan2(lat, depth)) <= half[d]):
                    continue
            ok[i, d] = True
    return ok


# ------------------------------------------------------------------------------------------------ visibility
@njit(error_model='numpy')
def _gap(a, b):
    """existence._angle_gap of two floats."""
    d = abs(a-b)
    return _nmin(d, TWO_PI-d)


@njit(error_model='numpy')
def visibility(o, B, targets, active, want, samples):
    """existence.visibility (C, T) of target boxes targets (T,) among boxes B (nb, 4, 2) from origins o (C, 2);
    active (C, nb) the boxes each origin does not skip, want (C, T) the pairs to evaluate."""
    C, T, nb = len(o), len(targets), len(B)
    out = np.zeros((C, T))
    if C == 0 or T == 0 or nb == 0:
        return out
    cen = np.empty((nb, 2))
    hl, hw = np.empty(nb), np.empty(nb)
    ux, uy = np.empty((nb, 2)), np.empty((nb, 2))
    for b in range(nb):
        for q in range(2):
            cen[b, q] = (((B[b, 0, q]+B[b, 1, q])+B[b, 2, q])+B[b, 3, q])/4
        ex0, ex1 = B[b, 1, 0]-B[b, 0, 0], B[b, 1, 1]-B[b, 0, 1]
        ey0, ey1 = B[b, 3, 0]-B[b, 0, 0], B[b, 3, 1]-B[b, 0, 1]
        hl[b], hw[b] = hypot(ex0, ex1)/2, hypot(ey0, ey1)/2
        ux[b, 0], ux[b, 1] = ex0/(2*hl[b]), ex1/(2*hl[b])
        uy[b, 0], uy[b, 1] = ey0/(2*hw[b]), ey1/(2*hw[b])
    dc, bear, half = np.empty((C, nb)), np.empty((C, nb)), np.empty((C, nb))
    reach, ox, oy = np.empty((C, nb)), np.empty((C, nb)), np.empty((C, nb))
    inside = np.empty((C, nb), np.bool_)
    any_in = np.zeros(C, np.bool_)
    for c in range(C):
        for b in range(nb):
            r0, r1 = cen[b, 0]-o[c, 0], cen[b, 1]-o[c, 1]
            dc[c, b] = hypot(r0, r1)
            bear[c, b] = atan2(r1, r0)
            h = -inf
            for q in range(4):
                h = _nmax(h, _gap(atan2(B[b, q, 1]-o[c, 1], B[b, q, 0]-o[c, 0]), bear[c, b]))
            half[c, b] = h
            reach[c, b] = dc[c, b]-hypot(hl[b], hw[b])
            ox[c, b] = -(r0*ux[b, 0]+r1*ux[b, 1])
            oy[c, b] = -(r0*uy[b, 0]+r1*uy[b, 1])
            inside[c, b] = abs(ox[c, b]) <= hl[b] and abs(oy[c, b]) <= hw[b]
            if inside[c, b] and active[c, b]:
                any_in[c] = True
    blk = np.empty(nb, np.int64)
    for c in range(C):
        for t in range(T):
            if not want[c, t]:
                continue
            tb = targets[t]
            nblk = 0
            for b in range(nb):
                if b == tb or not active[c, b]:
                    continue
                if any_in[c] or (_gap(bear[c, tb], bear[c, b]) <= half[c, tb]+half[c, b]+1e-9
                                 and reach[c, b] < dc[c, tb]):
                    blk[nblk] = b
                    nblk += 1
            if nblk == 0 and not inside[c, tb]:
                out[c, t] = 1.0
                continue
            count, seen = 0, 0
            for e in range(4):
                e1 = (e+1) % 4
                for s in range(samples):
                    st = s/samples
                    v0 = B[tb, e, 0]+st*(B[tb, e1, 0]-B[tb, e, 0])-o[c, 0]
                    v1 = B[tb, e, 1]+st*(B[tb, e1, 1]-B[tb, e, 1])-o[c, 1]
                    dr = hypot(v0, v1)
                    if not dr <= dc[c, tb]:
                        continue
                    count += 1
                    hit = False
                    for q in range(nblk):
                        b = blk[q]
                        dx = v0*ux[b, 0]+v1*ux[b, 1]
                        dy = v0*uy[b, 0]+v1*uy[b, 1]
                        tx1, tx2 = (-hl[b]-ox[c, b])/dx, (hl[b]-ox[c, b])/dx
                        ty1, ty2 = (-hw[b]-oy[c, b])/dy, (hw[b]-oy[c, b])/dy
                        t_in = _nmax(_nmin(tx1, tx2), _nmin(ty1, ty2))
                        t_out = _nmin(_nmax(tx1, tx2), _nmax(ty1, ty2))
                        t_hit = t_in if t_in > 0 else t_out
                        if t_in <= t_out and t_hit > 0 and t_hit*dr < dr-1e-6:
                            hit = True
                            break
                    if not hit:
                        seen += 1
            out[c, t] = seen/max(count, 1) if count > 0 else 0.0
    return out


# ------------------------------------------------------------------------------------------------ coverage
SPEC_FIELDS = ('reach_angle', 'reach_range', 'fov', 'range_max', 'beam_spacing', 'layers', 'dropout', 'far_dropout',
               'fov_far', 'near_range', 'pd_ref', 'ref_range', 'sigma_range_rate', 'sigma_azimuth', 'hfov', 'focal',
               'min_height_px', 'min_visible', 'pd')
(S_REACH_A, S_REACH_R, S_FOV, S_RMAX, S_BEAM, S_LAYERS, S_DROP, S_FAR, S_FOV_FAR, S_NEAR_R, S_PD_REF, S_REF_R, S_SRR,
 S_SAZ, S_HFOV, S_FOCAL, S_MIN_H, S_MIN_VIS, S_PD) = range(19)
COVER_FIELDS = ('slack_angle', 'slack_range', 'origin_merge', 'samples', 'stationary_sigmas', 'lidar_min_points',
                'rear', 'front', 'length', 'width', 'car_height')
C_SLACK_A, C_SLACK_R, C_MERGE, C_SAMPLES, C_SIGMAS, C_MIN_PTS, C_REAR, C_FRONT, C_LEN, C_WID, C_HEIGHT = range(11)
SQRT1_2 = 0.7071067811865476


@njit(error_model='numpy')
def _ndtr(a):
    """scipy.special.ndtr, the standard normal CDF, from erf and erfc as cephes computes it."""
    x = a*SQRT1_2
    z = abs(x)
    if z < SQRT1_2:
        return 0.5+0.5*erf(x)
    y = 0.5*erfc(z)
    return 1.0-y if x > 0 else y


@njit(error_model='numpy')
def _footprint(x, y, h, cp, out, row):
    """out[row] (4, 2) <- geometry.car_boxes of one CG state (x, y, heading h); cp the COVER row."""
    c, s = cos(h), sin(h)
    rear, front, hw = cp[C_REAR], cp[C_FRONT], cp[C_WID]/2
    for q in range(4):
        lx = rear if q == 0 or q == 3 else front
        ly = -hw if q < 2 else hw
        out[row, q, 0] = lx*c+ly*(-s)+x
        out[row, q, 1] = lx*s+ly*c+y


@njit(error_model='numpy')
def _expected_pd(kd, sp, rng, az, los, vis, cp):
    """existence.expected_pd (through=False) of one track for one scan; sp the sensor's SPEC row."""
    L, W = cp[C_LEN], cp[C_WID]
    cl, sl = abs(cos(los)), abs(sin(los))
    near = max(rng-(cl*L/2+sl*W/2), 0.1)
    a = abs(az)
    if kd == LIDAR:
        if not (a <= sp[S_FOV]/2 and near <= sp[S_RMAX]):
            return 0.0
        extent = (cl*W+sl*L)/max(rng, 1.0)
        nb = floor(max(extent*vis, 0.0)/sp[S_BEAM])*max(floor(sp[S_LAYERS]), 1.0)
        p = _clip(1-sp[S_DROP]-sp[S_FAR]*_clip(near/sp[S_RMAX], 0.0, 1.0), 0.0, 1.0)
        q = 1-p
        kmin = int(cp[C_MIN_PTS])
        tail = 0.0
        for k in range(kmin):
            coef = 1.0
            for i in range(k):
                coef = coef*(nb-i)/(i+1)
            if nb >= k:
                tail = tail+coef*p**k*q**max(nb-k, 0.0)
        return _clip(1-tail, 0.0, 1.0) if nb >= kmin else 0.0
    if kd == RADAR:
        if not (near <= sp[S_RMAX] and a <= (sp[S_FOV] if near <= sp[S_NEAR_R] else sp[S_FOV_FAR])/2):
            return 0.0
        p, ref, rmax = _clip(sp[S_PD_REF], 0.0, 1.0), sp[S_REF_R], sp[S_RMAX]
        curve = p
        if p > 0.5 and p < 1 and ref > 0 and ref < rmax:
            n = log(p/(1-p))/log(rmax/ref)
            curve = _nmin(p, 1/(1+(max(near, 0.0)/rmax)**n))
        return (curve if near <= rmax else 0.0)*vis
    depth = near*cos(min(a, pi/2))
    if not (a <= sp[S_HFOV]/2 and depth > 1.0):
        return 0.0
    height = sp[S_FOCAL]*cp[C_HEIGHT]/max(depth, 1e-3)
    return sp[S_PD] if height >= sp[S_MIN_H] and vis >= sp[S_MIN_VIS] else 0.0


@njit(error_model='numpy')
def coverage(poses, tp, cars, kinds, spec, mate, mc, t, likely, team, has_team, T, cp, gp):
    """TeamTracker._coverage of the scans with a pose (Q,): pd (Q, n), each scan's expected detection probability of
    each track (existence.expected_pd with through=False, times existence.doppler_pass for a radar), with the
    visibility of TeamTracker._visibility. poses (Q, 4) and pose times tp, carrying cars, kind codes, SPEC rows
    (Q, F); mate (Q, 4) the carrying car's state [x, y, yaw, v] at tp (zeros when unknown); tracks mc (n, 4) at
    times t, likely (n,) the tracks that occlude; team (k, 4) at the cycle time T (has_team False: none given); cp the
    COVER row, gp the GEOM row (the radar model)."""
    Q, n, k = len(poses), len(mc), len(team)
    pd = np.zeros((Q, n))
    if Q == 0 or n == 0:
        return pd
    rng, az, los = np.empty((Q, n)), np.empty((Q, n)), np.empty((Q, n))
    pos = np.empty((Q, n, 2))
    need = np.zeros((Q, n), np.bool_)
    for q in range(Q):
        for j in range(n):
            px = mc[j, 0]+mc[j, 2]*(tp[q]-t[j])
            py = mc[j, 1]+mc[j, 3]*(tp[q]-t[j])
            pos[q, j, 0], pos[q, j, 1] = px, py
            r0, r1 = px-poses[q, 0], py-poses[q, 1]
            ang = atan2(r1, r0)
            rng[q, j] = hypot(r0, r1)
            az[q, j] = _wrap(ang-poses[q, 2])
            los[q, j] = ang
            need[q, j] = (abs(az[q, j]) <= spec[q, S_REACH_A]+cp[C_SLACK_A]
                          and rng[q, j] <= spec[q, S_REACH_R]+cp[C_SLACK_R])
    vis = np.ones((Q, n))
    nok = 0
    for q in range(Q):
        if cars[q] < k:
            nok += 1
    if has_team and nok > 0:
        ok = np.empty(nok, np.int64)
        origin = np.empty((nok, 2))
        m = 0
        for q in range(Q):
            if cars[q] < k:
                ok[m] = q
                origin[m, 0] = team[cars[q], 0]+(poses[q, 0]-mate[q, 0])
                origin[m, 1] = team[cars[q], 1]+(poses[q, 1]-mate[q, 1])
                m += 1
        group = np.zeros(nok, np.int64)
        reps = np.empty(nok, np.int64)
        nr = 0
        for a in range(nok):
            g = -1
            for b in range(nr):
                r = reps[b]
                if (hypot(origin[r, 0]-origin[a, 0], origin[r, 1]-origin[a, 1]) <= cp[C_MERGE]
                        and cars[ok[r]] == cars[ok[a]]):
                    g = b
                    break
            if g < 0:
                g = nr
                reps[nr] = a
                nr += 1
            group[a] = g
        boxes = np.empty((n+k, 4, 2))
        for j in range(n):
            h = atan2(mc[j, 3], max(mc[j, 2], gp[G_VMIN]))
            _footprint(mc[j, 0]+mc[j, 2]*(T-t[j]), mc[j, 1]+mc[j, 3]*(T-t[j]), h, cp, boxes, j)
        for c in range(k):
            _footprint(team[c, 0], team[c, 1], team[c, 2], cp, boxes, n+c)
        active = np.ones((nr, n+k), np.bool_)
        want = np.zeros((nr, n), np.bool_)
        orep = np.empty((nr, 2))
        for b in range(nr):
            orep[b, 0], orep[b, 1] = origin[reps[b], 0], origin[reps[b], 1]
            for j in range(n):
                active[b, j] = likely[j]
            active[b, n+cars[ok[reps[b]]]] = False
        for a in range(nok):
            for j in range(n):
                if need[ok[a], j]:
                    want[group[a], j] = True
        v = visibility(orep, boxes, np.arange(n), active, want, int(cp[C_SAMPLES]))
        for a in range(nok):
            for j in range(n):
                if need[ok[a], j]:
                    vis[ok[a], j] = v[group[a], j]
    x, out, zero = np.empty(4), np.empty(4), np.zeros(2)
    for q in range(Q):
        kd = kinds[q]
        for j in range(n):
            p = _expected_pd(kd, spec[q], rng[q, j], az[q, j], los[q, j], vis[q, j], cp)
            if kd == RADAR and p > 0:        # existence.doppler_pass of the box point nearest the radar
                x[0], x[1], x[2], x[3] = pos[q, j, 0], pos[q, j, 1], mc[j, 2], mc[j, 3]
                _h_radar(x, poses[q], zero, gp, out)
                a = poses[q, 2]+out[1]
                slope = mate[q, 3]*(sin(a)*cos(mate[q, 2])-cos(a)*sin(mate[q, 2]))
                sigma = sqrt(spec[q, S_SRR]**2+(slope*spec[q, S_SAZ])**2)
                zz = out[2]/max(sigma, 1e-9)
                p = p*(1-(_ndtr(cp[C_SIGMAS]-zz)-_ndtr(-cp[C_SIGMAS]-zz)))
            pd[q, j] = p
    return pd


# ------------------------------------------------------------------------------------------------ the belief
@njit(error_model='numpy')
def combine(means, covs, mu):
    """imm.imm_combine of tracks (P, J, n), (P, J, n, n), (P, J): the moment-matched (mean (P, n), cov (P, n, n))."""
    P, J, n = means.shape
    m = np.zeros((P, n))
    C = np.empty((P, n, n))
    Pm = np.empty((n, n))
    for p in range(P):
        for c in range(n):
            acc = 0.0
            for j in range(J):
                acc += mu[p, j]*means[p, j, c]
            m[p, c] = acc
        for r in range(n):
            for c in range(n):
                acc, spread = 0.0, 0.0
                for j in range(J):
                    acc += mu[p, j]*covs[p, j, r, c]
                    spread += mu[p, j]*(means[p, j, r]-m[p, r])*(means[p, j, c]-m[p, c])
                Pm[r, c] = acc+spread
        for r in range(n):
            for c in range(n):
                C[p, r, c] = 0.5*(Pm[r, c]+Pm[c, r])
    return m, C


@njit(error_model='numpy')
def lane_rows(d, mu, steps, lanes, w, half):
    """imm.lane_probability of tracks with model lateral positions d (P, J) and probabilities mu (P, J): (P, lanes),
    each model's models.target_lane (steps (J,) the models' lateral steps; half = lanes w / 2)."""
    P, J = d.shape
    out = np.zeros((P, lanes))
    for p in range(P):
        for j in range(J):
            lane = int(_clip(floor((d[p, j]+half)/w), 0.0, lanes-1.0))
            step = steps[j]
            centre = (lane+0.5)*w-half
            tgt = lane+step if step*(d[p, j]-centre) >= -1e-9 else lane
            tgt = min(max(tgt, 0), lanes-1)
            out[p, tgt] += mu[p, j]
    return out


# ------------------------------------------------------------------------------------------------ existence
LOGE2 = 0.6931471805599453


@njit(error_model='numpy')
def _logaddexp(x, y):
    """np.logaddexp of two floats (numpy's npy_logaddexp)."""
    if x == y:
        return x+LOGE2
    tmp = x-y
    if tmp > 0:
        return x+log1p(exp(-tmp))
    if tmp <= 0:
        return y+log1p(exp(tmp))
    return tmp


@njit(error_model='numpy')
def existence(p, dtp, pd, hit, hit_tr, log_lr, hit_pd, ep):
    """TeamTracker._existence: the tracks' existence (n,) after survival over dtp (n,) and the cycle's misses (pd
    (Q, n) where not hit (Q, n)) and hits (hit_tr, with log likelihood ratio log_lr and pd hit_pd; in that order, as
    np.add.at). ep the EXIST row: gate, the floor of a cycle's summed misses, exist_survive, EXIST_MIN, p_max,
    PD_DETECTED_MIN."""
    gate, floor_, surv, pmin, pmax, pdmin = ep[0], ep[1], ep[2], ep[3], ep[4], ep[5]
    Q, n = pd.shape
    L = np.zeros(n)
    for q in range(Q):
        for j in range(n):
            v = 0.0
            if pd[q, j] > 0 and not hit[q, j]:
                v = log1p(-_clip(pd[q, j]*gate, 0.0, 1-1e-12))
            L[j] = L[j]+v
    for j in range(n):
        L[j] = _nmax(L[j], floor_)
    for h in range(len(hit_tr)):
        d = _clip(_nmax(hit_pd[h], pdmin), 1e-12, 1.0)
        L[hit_tr[h]] += _logaddexp(log1p(-_clip(d*gate, 0.0, 1-1e-12)), log(d)+log_lr[h])
    out = np.empty(n)
    for j in range(n):
        x = _clip(p[j]*surv**max(dtp[j], 0.0), pmin, 1-1e-15)
        lo = log(x)-log1p(-x)+L[j]
        out[j] = _clip(0.5*(1+tanh(0.5*lo)), pmin, pmax)
    return out


@njit(error_model='numpy')
def cv_rows(mean, cov, dt, ql, qd, noise):
    """tracker._cv: constant-velocity prediction of combined states (n, 4), (n, 4, 4) by dt (n,), F cov F^T plus
    (noise True) the white-acceleration noise of q_long^2 = ql and q_lat^2 = qd over |dt|."""
    n = len(dt)
    m, P = np.empty((n, 4)), np.empty((n, 4, 4))
    F, A2 = np.empty((4, 4)), np.empty((4, 4))
    for i in range(n):
        t = dt[i]
        for r in range(4):
            for c in range(4):
                F[r, c] = 1.0 if r == c else 0.0
        F[0, 2] = F[1, 3] = t
        for r in range(4):
            acc = 0.0
            for c in range(4):
                acc += F[r, c]*mean[i, c]
            m[i, r] = acc
            for c in range(4):
                acc = 0.0
                for k in range(4):
                    acc += F[r, k]*cov[i, k, c]
                A2[r, c] = acc
        for r in range(4):
            for c in range(4):
                acc = 0.0
                for k in range(4):
                    acc += A2[r, k]*F[c, k]
                P[i, r, c] = acc
        if noise:
            a = abs(t)
            q3, q2 = a**3/3, a**2/2
            P[i, 0, 0] += ql*q3
            P[i, 0, 2] += ql*q2
            P[i, 2, 0] += ql*q2
            P[i, 2, 2] += ql*a
            P[i, 1, 1] += qd*q3
            P[i, 1, 3] += qd*q2
            P[i, 3, 1] += qd*q2
            P[i, 3, 3] += qd*a
    return m, P


# ------------------------------------------------------------------------------------------------ births
BIRTH_FIELDS = ('shift', 'length', 'width', 'init_vd_var', 'radar_speed_min_cos', 'radar_birth_var')


@njit(error_model='numpy')
def _image_moments(Y, wm, wc, m, P):
    """m (n,), P (n, n) <- tracker._moments of one set of sigma-point images Y (K, n): the weighted mean and the
    symmetrized covariance."""
    K, n = Y.shape
    for c in range(n):
        acc = 0.0
        for k in range(K):
            acc += wm[k]*Y[k, c]
        m[c] = acc
    for r in range(n):
        for c in range(n):
            acc = 0.0
            for k in range(K):
                acc += (Y[k, r]-m[r])*wc[k]*(Y[k, c]-m[c])
            P[r, c] = acc
    for r in range(n):
        for c in range(r):
            v = 0.5*(P[r, c]+P[c, r])
            P[r, c] = v
            P[c, r] = v


@njit(error_model='numpy')
def birth_rows(kind, z, R, pose, vs, bp, wm3, wc3, sc3, wm4, wc4, sc4):
    """The lidar and radar rows of TeamTracker._birth_states for detections of kind (u,) with the DetTable rows z,
    R, pose, vs: (mean (u, 4), cov (u, 4, 4), have_v (u,), ok). A lidar box's CG position from the unscented
    transform of [x, y, heading]; a radar detection's [s, d, v_s, v_d] from [range, azimuth, range rate] and a lateral
    speed prior (v_s from the range rate where the line of sight is along the road, have_v). bp the BIRTH row;
    wm3, wc3, sc3 and wm4, wc4, sc4 the sigma-point weights and scales of 3 and 4 dimensions. Camera rows stay zero.
    ok False: a covariance without a plain Cholesky factor (the caller redoes the call in numpy)."""
    u = len(kind)
    mean, cov = np.zeros((u, 4)), np.zeros((u, 4, 4))
    have_v = np.zeros(u, np.bool_)
    m3, P3, L3, S3, X3 = np.empty(3), np.empty((3, 3)), np.empty((3, 3)), np.empty((3, 3)), np.empty((7, 3))
    m4, P4, L4, S4, X4 = np.empty(4), np.empty((4, 4)), np.empty((4, 4)), np.empty((4, 4)), np.empty((9, 4))
    Y2, Y4 = np.empty((7, 2)), np.empty((9, 4))
    mo2, Po2 = np.empty(2), np.empty((2, 2))
    shift, Lc, Wc = bp[0], bp[1], bp[2]
    for i in range(u):
        if kind[i] == LIDAR:
            for a in range(3):
                m3[a] = z[i, a]
                for b in range(3):
                    P3[a, b] = R[i, a, b]
            if not _points(m3, P3, sc3, X3, L3, S3):
                return mean, cov, have_v, False
            for k in range(7):
                Y2[k, 0] = X3[k, 0]-shift*cos(X3[k, 2])
                Y2[k, 1] = X3[k, 1]-shift*sin(X3[k, 2])
            _image_moments(Y2, wm3, wc3, mo2, Po2)
            for a in range(2):
                mean[i, a] = mo2[a]
                for b in range(2):
                    cov[i, a, b] = Po2[a, b]
        elif kind[i] == RADAR:
            for a in range(4):
                m4[a] = z[i, a] if a < 3 else 0.0
                for b in range(4):
                    P4[a, b] = R[i, a, b] if a < 3 and b < 3 else 0.0
            P4[3, 3] = bp[3]
            if not _points(m4, P4, sc4, X4, L4, S4):
                return mean, cov, have_v, False
            speed = abs(cos(pose[i, 2]+z[i, 1])) >= bp[4]
            for k in range(9):
                phi = pose[i, 2]+X4[k, 1]
                c, s = cos(phi), sin(phi)
                reach = X4[k, 0]+abs(c)*Lc/2+abs(s)*Wc/2
                cs = 1e-3 if abs(c) < 1e-3 else c
                Y4[k, 0] = pose[i, 0]+reach*c-shift
                Y4[k, 1] = pose[i, 1]+reach*s
                Y4[k, 2] = vs[i, 0]+(X4[k, 2]-(X4[k, 3]-vs[i, 1])*s)/cs if speed else 0.0
                Y4[k, 3] = X4[k, 3]
            _image_moments(Y4, wm4, wc4, m4, P4)
            P4[0, 0] += bp[5]
            P4[1, 1] += bp[5]
            if not speed:
                for a in range(4):
                    P4[2, a] = 0.0
                    P4[a, 2] = 0.0
            for a in range(4):
                mean[i, a] = m4[a]
                for b in range(4):
                    cov[i, a, b] = P4[a, b]
            have_v[i] = speed
    return mean, cov, have_v, True


# ------------------------------------------------------------------------------------------------ assignment
@njit(error_model='numpy')
def assign_parts(ci, cj, cost, idx, bounds):
    """The cost matrices of TeamTracker._assign's per-scan problems: part p is pairs bounds[p]:bounds[p+1] of
    (track ci, detection cj, cost, pair index idx); its matrix has the sorted unique detections as rows and the
    sorted unique tracks as columns, inf (index -1) where no pair. Returns (C, I) flat, matrix after matrix, and the
    shapes (parts, 2)."""
    P = len(bounds)-1
    shapes = np.empty((P, 2), np.int64)
    total = 0
    for p in range(P):
        a, b = bounds[p], bounds[p+1]
        shapes[p, 0] = len(np.unique(cj[a:b]))
        shapes[p, 1] = len(np.unique(ci[a:b]))
        total += shapes[p, 0]*shapes[p, 1]
    C = np.full(total, np.inf)
    I = np.full(total, -1, np.int64)
    off = 0
    for p in range(P):
        a, b = bounds[p], bounds[p+1]
        ui, uj = np.unique(ci[a:b]), np.unique(cj[a:b])
        nc = len(ui)
        ri, rj = np.searchsorted(ui, ci[a:b]), np.searchsorted(uj, cj[a:b])
        for q in range(b-a):
            C[off+rj[q]*nc+ri[q]] = cost[a+q]
            I[off+rj[q]*nc+ri[q]] = idx[a+q]
        off += shapes[p, 0]*nc
    return C, I, shapes


@njit(error_model='numpy')
def occupied(time, xy, has_xy, means, mu, t, length, width):
    """TeamTracker._occupied of detections (time, rough CG xy, has_xy) against the likely tracks (means (n, J, 4),
    mu (n, J) at times t): (D,) bool, the detection's CG inside a track's box."""
    Dn, n, J = len(time), len(t), mu.shape[1]
    m = np.zeros((n, 4))
    for i in range(n):
        for c in range(4):
            acc = mu[i, 0]*means[i, 0, c]
            for j in range(1, J):
                acc = acc+mu[i, j]*means[i, j, c]
            m[i, c] = acc
    out = np.zeros(Dn, np.bool_)
    for d in range(Dn):
        if not has_xy[d]:
            continue
        for i in range(n):
            dt = time[d]-t[i]
            if abs(xy[d, 0]-(m[i, 0]+m[i, 2]*dt)) < length and abs(xy[d, 1]-(m[i, 1]+m[i, 3]*dt)) < width:
                out[d] = True
                break
    return out
