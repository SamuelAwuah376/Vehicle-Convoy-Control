"""Compiled box fit of one lidar cluster: lidar.fit_box's algorithm as one numba kernel (nohesi.perception._jit).

The same steps, in the same order, as lidar.py's numpy path (lshape_heading, _assign, _rows, _linearize, _normal,
_anchor, _ransac, _consensus, the ICP loop, the covariance), written as loops over the cluster's points. Results match
the numpy path to rounding (sums are taken in a different order); the discrete choices (thinning indices, face
assignment, RANSAC hypotheses) use the same formulas. fit_box calls fit_box_kernel when _jit.ENABLED; with
NOHESI_JIT=0 the numpy path runs instead and this module's functions run uncompiled.

Rows of the residual model: at most one per point plus two per axis (free ends, or the extent prior), so every
array is preallocated with n + 4 rows and a count.
"""
import numpy as np
from .._jit import njit

LSHAPE_POINTS = 32
GATE = 0.3
POSE_SIGMA = 0.05
CORNER_MIN = 0.15
STRAY_MAX = 0.5
TAN_VIS = np.tan(np.radians(3.0))
HEADING_PRIOR = np.radians(5.0)
STEP_MAX_C, STEP_MAX_H = 0.5, np.radians(3.0)
ICP_ITERS, ICP_ROUNDS = 5, 2
COS_RANSAC = np.cos(np.radians(10.0))
HEADINGS = np.radians(np.arange(-20.0, 20.5, 1.0))
FACE_AX = np.array([0, 0, 1, 1])
FACE_SG = np.array([-1.0, 1.0, -1.0, 1.0])


@njit
def _wrap(a):
    return (a+np.pi) % (2*np.pi)-np.pi


@njit
def lshape_heading_kernel(P, o, headings):
    """lidar.lshape_heading: P (n, 2) world points, o (2,) the sensor."""
    n = P.shape[0]
    if n > LSHAPE_POINTS:                     # np.linspace(0, n-1, 32).astype(int), numpy's own formula
        step = (n-1)/(LSHAPE_POINTS-1)
        Q = np.empty((LSHAPE_POINTS, 2))
        for k in range(LSHAPE_POINTS-1):
            i = int(k*step)
            Q[k, 0], Q[k, 1] = P[i, 0], P[i, 1]
        Q[LSHAPE_POINTS-1, 0], Q[LSHAPE_POINTS-1, 1] = P[n-1, 0], P[n-1, 1]
    else:
        Q = P.copy()
    m = Q.shape[0]
    mx, my = 0.0, 0.0
    for i in range(m):
        mx += Q[i, 0]
        my += Q[i, 1]
    mx /= m
    my /= m
    ox, oy = o[0]-mx, o[1]-my
    H = headings.shape[0]
    cost = np.empty(H)
    C1 = np.empty(m)
    C2 = np.empty(m)
    for h in range(H):
        c, s = np.cos(headings[h]), np.sin(headings[h])
        lo1, hi1, lo2, hi2 = np.inf, -np.inf, np.inf, -np.inf
        for i in range(m):
            px, py = Q[i, 0]-mx, Q[i, 1]-my
            a, b = c*px+s*py, c*py-s*px
            C1[i], C2[i] = a, b
            lo1, hi1 = min(lo1, a), max(hi1, a)
            lo2, hi2 = min(lo2, b), max(hi2, b)
        o1, o2 = c*ox+s*oy, c*oy-s*ox
        r1 = lo1 if o1 < (lo1+hi1)/2 else hi1
        r2 = lo2 if o2 < (lo2+hi2)/2 else hi2
        k1, k2, s1, s2, q1, q2 = 0, 0, 0.0, 0.0, 0.0, 0.0
        for i in range(m):
            d1, d2 = abs(C1[i]-r1), abs(C2[i]-r2)
            if d1 < d2:
                k1 += 1
                s1 += d1
                q1 += d1*d1
            else:
                k2 += 1
                s2 += d2
                q2 += d2*d2
        k1, k2 = max(k1, 1), max(k2, 1)
        mean1, mean2 = s1/k1, s2/k2
        cost[h] = 0.0+(q1/k1-mean1*mean1)+(q2/k2-mean2*mean2)
    k = int(np.argmin(cost))
    if 0 < k < H-1:
        y0, y1, y2 = cost[k-1], cost[k], cost[k+1]
        curv = y0-2*y1+y2
        if curv > 0:
            return headings[k]+min(max(0.5*(y0-y2)/curv, -0.5), 0.5)*(headings[k+1]-headings[k])
    return headings[k]


@njit
def _assign(P, o, half, rx, ry, sr, sm, use, pose):
    """lidar._assign: (c, s, B, j, var, on, cand)."""
    n = P.shape[0]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    hx, hy = half[0], half[1]
    B = np.empty((n, 2))
    for i in range(n):
        dx, dy = P[i, 0]-pose[0], P[i, 1]-pose[1]
        B[i, 0] = dx*c+dy*s
        B[i, 1] = dx*(-s)+dy*c
    ox, oy = o[0]-pose[0], o[1]-pose[1]
    obx, oby = c*ox+s*oy, c*oy-s*ox
    tx, ty = TAN_VIS*abs(oby), TAN_VIS*abs(obx)
    cand = np.array([-obx > hx-tx, obx > hx-tx, -oby > hy-ty, oby > hy-ty])
    j = np.full(n, -1)
    var = np.ones(n)
    on = np.zeros(n, np.bool_)
    if not cand.any():
        return c, s, B, j, var, on, cand
    V = np.empty((n, 2))
    best = np.full(4, -1.0)
    ps2 = POSE_SIGMA**2
    for i in range(n):
        a0, a1 = abs(B[i, 0]), abs(B[i, 1])
        ov0, ov1 = max(a0-hx, 0.0)**2, max(a1-hy, 0.0)**2
        v0 = (sr*(rx[i]*c+ry[i]*s))**2+sm**2
        v1 = (sr*(rx[i]*(-s)+ry[i]*c))**2+sm**2
        V[i, 0], V[i, 1] = v0, v1
        bj, bv = 0, np.inf
        for f in range(4):
            if not cand[f]:
                continue
            if FACE_AX[f] == 0:
                D = (B[i, 0]-FACE_SG[f]*hx)**2+ov1
                q = D/(v0+ps2)
            else:
                D = (B[i, 1]-FACE_SG[f]*hy)**2+ov0
                q = D/(v1+ps2)
            if q < bv:
                bj, bv = f, q
        j[i] = bj
        var[i] = V[i, FACE_AX[bj]]
        if use[i]:
            clear = hy-a1 if FACE_AX[bj] == 0 else hx-a0
            best[bj] = max(best[bj], clear)
    keep = best.copy()
    for f in range(4):
        if best[f ^ 1] > best[f]:
            keep[f] = -1.0
    for i in range(n):
        on[i] = keep[j[i]] >= CORNER_MIN
    zone = 3*np.sqrt(sr**2+sm**2+ps2)
    for f0 in range(2):
        for f1 in range(2, 4):
            if cand[f0] and cand[f1]:
                for i in range(n):
                    if abs(B[i, 0]-FACE_SG[f0]*hx) < zone and abs(B[i, 1]-FACE_SG[f1]*hy) < zone:
                        var[i] = var[i]+zone*zone
    return c, s, B, j, var, on, cand


@njit
def _rows(P, o, half, size, rx, ry, dist, tau, sr, sm, ends_i, ends_nb, ends_pd, use, pose):
    """lidar._rows: (pts, ax, sg, h, var, tau, src) with m rows, the per-point face axis, the seen faces mask (4,),
    their count, and E (2, 2)."""
    c, s, B, j, var, on, cand = _assign(P, o, half, rx, ry, sr, sm, use, pose)
    n = P.shape[0]
    cap = n+4
    pts = np.empty((cap, 2))
    ax = np.empty(cap, np.int64)
    sg = np.empty(cap)
    h = np.empty(cap)
    rv = np.empty(cap)
    rt = np.empty(cap)
    src = np.empty(cap, np.int64)
    seen = np.zeros(4, np.bool_)
    E = np.empty((2, 2))
    E[0, 0], E[0, 1], E[1, 0], E[1, 1] = c, s, -s, c
    m = 0
    for i in range(n):
        if not (use[i] and on[i]):
            continue
        f = j[i]
        a = FACE_AX[f]
        g = FACE_SG[f]
        if abs(g*B[i, a]-half[a]) <= GATE:
            pts[m, 0], pts[m, 1] = P[i, 0], P[i, 1]
            ax[m], sg[m], h[m], rv[m], rt[m], src[m] = a, g, half[a], var[i], tau[i], i
            seen[f] = True
            m += 1
    for k in range(2):
        if seen[2*k] or seen[2*k+1]:
            continue
        lo, hi, mid = np.inf, -np.inf, 0.0
        imin, imax = 0, 0
        for i in range(n):
            v = B[i, k]
            if v < lo:
                lo, imin = v, i
            if v > hi:
                hi, imax = v, i
            mid += v
        mid /= n
        nc = 0
        cd = np.empty(2)
        ci = np.empty(2, np.int64)
        cs = np.empty(2)
        cg = np.empty(2)
        cp = np.empty(2)
        for e in range(ends_i.shape[0]):
            i, nb, pd = ends_i[e], ends_nb[e], ends_pd[e]
            gap = min(abs(B[i, k]-B[nb, k]), size[k])
            sign = 1.0 if B[i, k] > mid else -1.0
            if abs(B[i, k]-(hi if sign > 0 else lo)) <= gap+0.1:
                cd[nc], ci[nc], cs[nc], cg[nc], cp[nc] = dist[i], i, sign, gap, pd
                nc += 1
        first = 0
        if nc == 2 and (cs[0] == cs[1] or hi-lo < size[k]-0.3):
            # min() of the (dist, i, sign, gap, pd) tuples
            if cd[1] < cd[0] or (cd[1] == cd[0] and ci[1] < ci[0]):
                first = 1
            nc = first+1
        for q in range(first, nc):
            i, sign, gap, pd = ci[q], cs[q], cg[q], cp[q]
            ck = abs(E[k, 0]*rx[i]+E[k, 1]*ry[i])
            lost = pd/(1-pd)
            sh = (0.5+lost)*gap*sign
            pts[m, 0], pts[m, 1] = P[i, 0]+sh*E[k, 0], P[i, 1]+sh*E[k, 1]
            ax[m], sg[m], h[m] = k, sign, half[k]
            rv[m] = gap*gap*(1/12+lost/(1-pd))+(sr*ck)**2+sm**2
            rt[m], src[m] = tau[i], i
            m += 1
        if nc == 0:
            pts[m, 0] = 0.5*(P[imin, 0]+P[imax, 0])
            pts[m, 1] = 0.5*(P[imin, 1]+P[imax, 1])
            ax[m], sg[m], h[m] = k, 1.0, 0.0
            rv[m] = max(size[k]-(hi-lo), 0.1)**2/12+sm**2
            rt[m], src[m] = 0.0, -1
            m += 1
    pax = np.empty(n, np.int64)
    for i in range(n):
        pax[i] = FACE_AX[j[i]] if j[i] >= 0 else -1
    nseen = 0
    for f in range(4):
        nseen += seen[f]
    return pts[:m], ax[:m], sg[:m], h[:m], rv[:m], rt[:m], src[:m], pax, seen, nseen, E


@njit
def _linearize(pts, ax, sg, h, var, tau, pose):
    """lidar._linearize: r, J (m, 3), w, Bv (m, 2)."""
    m = pts.shape[0]
    c, s = np.cos(pose[2]), np.sin(pose[2])
    r = np.empty(m)
    J = np.empty((m, 3))
    w = np.empty(m)
    Bv = np.empty((m, 2))
    for i in range(m):
        dx, dy = pts[i, 0]-pose[0], pts[i, 1]-pose[1]
        b0, b1 = dx*c+dy*s, dx*(-s)+dy*c
        g = sg[i]
        if ax[i] == 0:
            r[i] = g*b0-h[i]
            e0, e1 = c*g, s*g
            J[i, 2] = g*b1
        else:
            r[i] = g*b1-h[i]
            e0, e1 = -s*g, c*g
            J[i, 2] = g*(-b0)
        J[i, 0], J[i, 1] = -e0, -e1
        w[i] = 1.0/var[i]
        Bv[i, 0], Bv[i, 1] = e0*tau[i], e1*tau[i]
    return r, J, w, Bv


@njit
def _normal(r, J, w, pose):
    """lidar._normal: A (3, 3), g (3,) with the heading prior."""
    A = np.zeros((3, 3))
    g = np.zeros(3)
    for i in range(r.shape[0]):
        for a in range(3):
            wa = w[i]*J[i, a]
            g[a] += J[i, a]*(w[i]*r[i])
            for b in range(3):
                A[a, b] += J[i, b]*wa
    A[2, 2] += 1/HEADING_PRIOR**2
    g[2] += _wrap(pose[2])/HEADING_PRIOR**2
    return A, g


@njit
def _consensus(Q, d0, d1, tol):
    """lidar._consensus: the mask of the points to keep from the best line through Q along (d0, d1)."""
    n = Q.shape[0]
    L = min(n, 12)
    idx = np.empty(L, np.int64)          # np.unique(np.round(np.linspace(0, n-1, L)).astype(int))
    step = (n-1)/(L-1) if L > 1 else 0.0
    nu = 0
    for k in range(L):
        v = int(np.rint(k*step)) if k < L-1 else n-1
        if nu == 0 or v != idx[nu-1]:    # linspace is increasing, so equal values are adjacent
            idx[nu] = v
            nu += 1
    best_score, best_h = -np.inf, -1
    cosr = COS_RANSAC
    for a in range(nu):
        for b in range(a+1, nu):
            i, jj = idx[a], idx[b]
            ux, uy = Q[jj, 0]-Q[i, 0], Q[jj, 1]-Q[i, 1]
            norm = np.hypot(ux, uy)
            dn = max(norm, 1e-9)
            ux, uy = ux/dn, uy/dn
            if not (norm > 0.05 and abs(ux*d0+uy*d1) >= cosr):
                continue
            cnt, sd = 0, 0.0
            for p in range(n):
                dist = abs((Q[p, 0]-Q[i, 0])*uy-(Q[p, 1]-Q[i, 1])*ux)
                if dist < tol:
                    cnt += 1
                    sd += dist
            score = cnt-sd/(tol*n)
            if score > best_score:
                best_score, best_h = score, a*nu+b
    keep = np.ones(n, np.bool_)
    if best_h < 0:
        return keep
    i, jj = idx[best_h//nu], idx[best_h % nu]
    ux, uy = Q[jj, 0]-Q[i, 0], Q[jj, 1]-Q[i, 1]
    dn = max(np.hypot(ux, uy), 1e-9)
    ux, uy = ux/dn, uy/dn
    cnt = 0
    for p in range(n):
        dist = abs((Q[p, 0]-Q[i, 0])*uy-(Q[p, 1]-Q[i, 1])*ux)
        inl = dist < tol
        cnt += inl
        keep[p] = inl or dist > STRAY_MAX
    if cnt < 0.5*n:
        keep[:] = True
    return keep


@njit
def _ransac(P, r, pax, seen, E, tol):
    """lidar._ransac: the points kept for ICP."""
    n = P.shape[0]
    keep = np.ones(n, np.bool_)
    if np.all(np.abs(r) <= tol):
        return keep
    for k in range(2):
        if not (seen[2*k] or seen[2*k+1]):
            continue
        cnt = 0
        for i in range(n):
            cnt += pax[i] == k
        if cnt < 4:
            continue
        sel = np.empty(cnt, np.int64)
        q = 0
        for i in range(n):
            if pax[i] == k:
                sel[q] = i
                q += 1
        Q = np.empty((cnt, 2))
        for q in range(cnt):
            Q[q, 0], Q[q, 1] = P[sel[q], 0], P[sel[q], 1]
        kp = _consensus(Q, E[1-k, 0], E[1-k, 1], tol)
        for q in range(cnt):
            keep[sel[q]] = kp[q]
    return keep


@njit
def _anchor(P, o, half, size, rx, ry, dist, tau, sr, sm, ends_i, ends_nb, ends_pd, use, th, passes):
    """lidar._anchor: the pose with the heading held at th, and the rows there."""
    c, s = np.cos(th), np.sin(th)
    n = P.shape[0]
    lo0, hi0, lo1, hi1 = np.inf, -np.inf, np.inf, -np.inf
    for i in range(n):
        b0, b1 = c*P[i, 0]+s*P[i, 1], c*P[i, 1]-s*P[i, 0]
        lo0, hi0 = min(lo0, b0), max(hi0, b0)
        lo1, hi1 = min(lo1, b1), max(hi1, b1)
    ob0, ob1 = c*o[0]+s*o[1], c*o[1]-s*o[0]
    cb = np.empty(2)
    cb[0] = lo0+half[0] if ob0 < lo0 else (hi0-half[0] if ob0 > hi0 else (lo0+hi0)/2)
    cb[1] = lo1+half[1] if ob1 < lo1 else (hi1-half[1] if ob1 > hi1 else (lo1+hi1)/2)
    pose = np.empty(3)
    pose[0], pose[1], pose[2] = c*cb[0]-s*cb[1], s*cb[0]+c*cb[1], th
    out = _rows(P, o, half, size, rx, ry, dist, tau, sr, sm, ends_i, ends_nb, ends_pd, use, pose)
    for p in range(passes):
        if p:
            pose[0], pose[1], pose[2] = c*cb[0]-s*cb[1], s*cb[0]+c*cb[1], th
            out = _rows(P, o, half, size, rx, ry, dist, tau, sr, sm, ends_i, ends_nb, ends_pd, use, pose)
        pts, ax, sg, h, var = out[0], out[1], out[2], out[3], out[4]
        num = np.zeros(2)
        den = np.zeros(2)
        for i in range(pts.shape[0]):
            if ax[i] == 0:
                t = c*pts[i, 0]+s*pts[i, 1]
            else:
                t = c*pts[i, 1]-s*pts[i, 0]
            t = t-sg[i]*h[i]
            num[ax[i]] += t/var[i]
            den[ax[i]] += 1/var[i]
        cb[0], cb[1] = num[0]/den[0], num[1]/den[1]
    pose[0], pose[1], pose[2] = c*cb[0]-s*cb[1], s*cb[0]+c*cb[1], th
    return pose, out


@njit
def fit_box_kernel(points, times, sensor, size, sr, sm, free0, free1, dropout, far_dropout, range_max, velocity,
                   velocity_sigma, refine, headings):
    """lidar.fit_box without its Python wrapping: (pose (3,), cov (3, 3), sides seen, t_ref, dz_dv (3, 2),
    R_fit (3, 3))."""
    n = points.shape[0]
    t_ref = 0.0
    for i in range(n):
        t_ref += times[i]
    t_ref /= n
    tau = t_ref-times
    P = np.empty((n, 2))
    for i in range(n):
        P[i, 0] = points[i, 0]+tau[i]*velocity[0]
        P[i, 1] = points[i, 1]+tau[i]*velocity[1]
    o = sensor
    half = 0.5*size
    rx, ry, dist = np.empty(n), np.empty(n), np.empty(n)
    for i in range(n):
        ux, uy = P[i, 0]-o[0], P[i, 1]-o[1]
        d = max(np.hypot(ux, uy), 1e-9)
        dist[i], rx[i], ry[i] = d, ux/d, uy/d
    ne = 0
    ends_i, ends_nb, ends_pd = np.empty(2, np.int64), np.empty(2, np.int64), np.empty(2)
    if n > 1:
        for e in range(2):
            if (free0 if e == 0 else free1):
                i = 0 if e == 0 else n-1
                ends_i[ne] = i
                ends_nb[ne] = min(1, n-1) if e == 0 else max(n-2, 0)
                r = np.hypot(P[i, 0]-o[0], P[i, 1]-o[1])
                ends_pd[ne] = min(max(dropout+far_dropout*min(r/range_max, 1.0), 1e-3), 0.95)
                ne += 1
    ends_i, ends_nb, ends_pd = ends_i[:ne], ends_nb[:ne], ends_pd[:ne]
    use = np.ones(n, np.bool_)
    th = lshape_heading_kernel(P, o, headings) if n >= 8 else 0.0
    pose, out = _anchor(P, o, half, size, rx, ry, dist, tau, sr, sm, ends_i, ends_nb, ends_pd, use, th,
                        1 if refine else 2)
    pts, ax, sg, h, var, rt, src, pax, seen, nseen, E = out
    if refine:
        r, J, w, Bv = _linearize(pts, ax, sg, h, var, rt, pose)
        use = _ransac(P, r, pax, seen, E, 3*np.hypot(sr, sm)+0.02)
        if not use.all():
            pts, ax, sg, h, var, rt, src, pax, seen, nseen, E = _rows(P, o, half, size, rx, ry, dist, tau, sr, sm,
                                                                       ends_i, ends_nb, ends_pd, use, pose)
            r, J, w, Bv = _linearize(pts, ax, sg, h, var, rt, pose)
        for _ in range(ICP_ROUNDS):                # Gauss-Newton on fixed correspondences, then re-assign
            for _ in range(ICP_ITERS):
                A, g = _normal(r, J, w, pose)
                step = -np.linalg.solve(A, g)
                f = min(1.0, STEP_MAX_C/max(np.hypot(step[0], step[1]), 1e-12), STEP_MAX_H/max(abs(step[2]), 1e-12))
                step = step*f
                pose = pose+step
                r, J, w, Bv = _linearize(pts, ax, sg, h, var, rt, pose)
                if abs(step[0])+abs(step[1]) < 5e-4 and abs(step[2]) < 2e-5:
                    break
            src0, ax0, sg0 = src, ax, sg
            pts, ax, sg, h, var, rt, src, pax, seen, nseen, E = _rows(P, o, half, size, rx, ry, dist, tau, sr, sm,
                                                                       ends_i, ends_nb, ends_pd, use, pose)
            same = src0.shape[0] == src.shape[0]
            if same:
                same = np.array_equal(src0, src) and np.array_equal(ax0, ax) and np.array_equal(sg0, sg)
            if same:
                break
            r, J, w, Bv = _linearize(pts, ax, sg, h, var, rt, pose)
    r, J, w, Bv = _linearize(pts, ax, sg, h, var, rt, pose)
    A, _g = _normal(r, J, w, pose)
    core = np.linalg.inv(A)
    JB = np.zeros((3, 2))
    for i in range(r.shape[0]):
        for a in range(3):
            JB[a, 0] += J[i, a]*(w[i]*Bv[i, 0])
            JB[a, 1] += J[i, a]*(w[i]*Bv[i, 1])
    S = core @ JB
    if not refine:
        G = np.empty(3)
        G[:2] = -np.linalg.solve(A[:2, :2].copy(), A[:2, 2].copy())
        G[2] = 1.0
        dh = headings[1]-headings[0]
        core = core+np.outer(G, G)*dh**2/12
    vs = np.empty(2)
    vs[0], vs[1] = velocity_sigma[0]**2, velocity_sigma[1]**2
    cov = core+(S*vs) @ S.T
    pose[2] = _wrap(pose[2])
    return pose, 0.5*(cov+cov.T), nseen, t_ref, -S, 0.5*(core+core.T)
