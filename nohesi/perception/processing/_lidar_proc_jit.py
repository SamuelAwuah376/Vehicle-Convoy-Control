"""Compiled lidar processing around the box fit: lidar.process_lidar's per-sweep steps as two numba kernels
(nohesi.perception._jit).

sweep_kernel: de-skew, the world frame, the teammate mask and off-road labels, segmentation, merging, splitting and
the free ends, i.e. everything process_lidar does before the per-cluster callbacks; it returns the clusters (their
points, beam times, free ends and mean beam times). fit_clusters: _lidar_jit.fit_box_kernel on every cluster of a sweep
in one call.

The same steps, formulas and order of floating-point operations as lidar.py's numpy path, written as loops: numpy's
elementwise arithmetic is reproduced operation by operation, and its reductions in numpy's own order (np.sum and
np.mean of a contiguous row: pairwise summation, _pwsum; a column mean: a running sum), so that the L-shape headings of
the merge and split tests are those of lidar.lshape_heading, not of _lidar_jit's kernel (which sums in its own order).
The clusters and detections are bit-identical to the numpy path on recorded v9 sweeps (numba's sin, cos, hypot, log,
rint and float % give numpy's results on the platforms tested); tests/test_perception_jit_lidar_proc.py compares the
two. process_lidar uses these kernels when _jit.ENABLED; with NOHESI_JIT=0 the numpy path runs and this module's
functions run uncompiled.

Constants are lidar.py's, duplicated (the test checks they match).
"""
import numpy as np
from .._jit import njit
from ._lidar_jit import fit_box_kernel

BREAK_ANGLE = np.radians(10.0)
HEADINGS = np.radians(np.arange(-20.0, 20.5, 1.0))
COS_H, SIN_H = np.cos(HEADINGS), np.sin(HEADINGS)     # np.cos(headings) of lidar._lshape_cost
SIN_MAX = np.sin(HEADINGS[-1])                       # lidar._box_fits
LSHAPE_POINTS = 32
MATE_MARGIN = 0.3
EDGE_MARGIN = 0.5
FIT_SLACK = (0.5, 0.25)
OCCLUDER_MARGIN = 0.5
GAP_ARC = 0.5
GAP_P = 1e-4
LOG_GAP_P = np.log(GAP_P)
END_LOOK = 3
NONE, KEEP, MATE, OFF = 0, 1, 2, 3

# the kernels' scalar parameters, packed into one float64 array (fewer arguments to dispatch): the scan's times, the
# steady flag and the carrying car's twist, the mount offset and its yaw's cos and sin, the sensor at t_end and its
# yaw's cos and sin, whether teammates are given, the CG to body-centre shift, the teammate box half size plus margin,
# the off-road limit, the velocity prior, the mid time, the spec's beam spacing, 3 range sigma, dropout, far dropout and
# range, the car's size, range sigma, sigma_model and the velocity prior's sigma
(P_T_END, P_T0, P_STEADY, P_VX, P_VY, P_R, P_MX, P_MY, P_CM, P_SM, P_OX, P_OY, P_OC, P_OS, P_MATES, P_SHIFT, P_HL,
 P_HW, P_EDGE, P_VWX, P_VWY, P_T_MID, P_BS, P_SIG3, P_DROP, P_FAR, P_RMAX, P_LEN, P_WID, P_SR, P_SMOD, P_VS0,
 P_VS1) = range(33)
N_PARAMS = 33


@njit
def _pw_block(a, lo, n):
    """numpy's pairwise summation of a[lo:lo+n] for n <= 128: a running sum under 8 terms, else 8 accumulators."""
    if n < 8:
        r = 0.0
        for i in range(lo, lo+n):
            r += a[i]
        return r
    r0, r1, r2, r3 = a[lo], a[lo+1], a[lo+2], a[lo+3]
    r4, r5, r6, r7 = a[lo+4], a[lo+5], a[lo+6], a[lo+7]
    i = 8
    m = n-n % 8
    while i < m:
        b = lo+i
        r0 += a[b]
        r1 += a[b+1]
        r2 += a[b+2]
        r3 += a[b+3]
        r4 += a[b+4]
        r5 += a[b+5]
        r6 += a[b+6]
        r7 += a[b+7]
        i += 8
    res = ((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))
    while i < n:
        res += a[lo+i]
        i += 1
    return res


@njit
def _pw(a, lo, n):
    """numpy's pairwise summation of a[lo:lo+n]: blocks of up to 128 (_pw_block), halved at a multiple of 8 above
    that, left half first. Iterative (a recursive njit function breaks numba's on-disk cache)."""
    if n <= 128:
        return _pw_block(a, lo, n)
    fl = np.empty(64, np.int64)        # frames: (lo, n, stage); stage 1 adds the two halves' sums
    fn = np.empty(64, np.int64)
    fs = np.empty(64, np.int64)
    vals = np.empty(64)
    nf, nv = 1, 0
    fl[0], fn[0], fs[0] = lo, n, 0
    while nf:
        nf -= 1
        l, m, stage = fl[nf], fn[nf], fs[nf]
        if stage:
            nv -= 1
            vals[nv-1] = vals[nv-1]+vals[nv]
        elif m <= 128:
            vals[nv] = _pw_block(a, l, m)
            nv += 1
        else:
            n2 = m//2
            n2 -= n2 % 8
            fl[nf], fn[nf], fs[nf] = l, m, 1
            fl[nf+1], fn[nf+1], fs[nf+1] = l+n2, m-n2, 0
            fl[nf+2], fn[nf+2], fs[nf+2] = l, n2, 0
            nf += 3
    return vals[0]


@njit
def _pwsum(a, lo, n):
    """np.sum of the contiguous float64 row a[lo:lo+n] (np.mean is this over n): the reduction's 0.0 plus numpy's
    pairwise sum."""
    return 0.0+_pw(a, lo, n)


@njit
def _sinc(x):
    """np.sinc: sin(pi x)/(pi x), with x = 0 taken as 1e-20."""
    y = np.pi*(1e-20 if x == 0 else x)
    return np.sin(y)/y


@njit
def lshape_heading_np(Q, o):
    """lidar.lshape_heading(Q, o) with numpy's own order of operations (the merge and split tests' heading)."""
    n = Q.shape[0]
    if n > LSHAPE_POINTS:                     # np.linspace(0, n-1, 32).astype(int)
        step = (n-1)/(LSHAPE_POINTS-1)
        m = LSHAPE_POINTS
        P = np.empty((m, 2))
        for k in range(m-1):
            i = int(k*step)
            P[k, 0], P[k, 1] = Q[i, 0], Q[i, 1]
        P[m-1, 0], P[m-1, 1] = Q[n-1, 0], Q[n-1, 1]
    else:
        m = n
        P = Q
    mx, my = 0.0, 0.0                         # P.mean(0): a running sum down each column
    for i in range(m):
        mx += P[i, 0]
        my += P[i, 1]
    mx /= m
    my /= m
    px, py = np.empty(m), np.empty(m)
    for i in range(m):
        px[i], py[i] = P[i, 0]-mx, P[i, 1]-my
    ox, oy = o[0]-mx, o[1]-my
    H = COS_H.shape[0]
    cost = np.empty(H)
    C1, C2, D1, D2 = np.empty(m), np.empty(m), np.empty(m), np.empty(m)
    ds, dd = np.empty(m), np.empty(m)
    for h in range(H):
        c, s = COS_H[h], SIN_H[h]
        lo1, hi1, lo2, hi2 = np.inf, -np.inf, np.inf, -np.inf
        for i in range(m):
            a = c*px[i]+s*py[i]
            b = c*py[i]-s*px[i]
            C1[i], C2[i] = a, b
            lo1, hi1 = min(lo1, a), max(hi1, a)
            lo2, hi2 = min(lo2, b), max(hi2, b)
        o1, o2 = c*ox+s*oy, c*oy-s*ox
        r1 = lo1 if o1 < (lo1+hi1)/2 else hi1
        r2 = lo2 if o2 < (lo2+hi2)/2 else hi2
        for i in range(m):
            D1[i], D2[i] = abs(C1[i]-r1), abs(C2[i]-r2)
        cst = 0.0
        for side in range(2):
            kk = 0
            for i in range(m):
                near = D1[i] < D2[i]
                sel = near if side == 0 else not near
                d = D1[i] if side == 0 else D2[i]
                v = d if sel else 0.0
                ds[i], dd[i] = v, v*d
                kk += sel
            kk = max(kk, 1)
            mean = _pwsum(ds, 0, m)/kk
            cst = cst+_pwsum(dd, 0, m)/kk-mean*mean
        cost[h] = cst
    k = 0
    for h in range(1, H):
        if cost[h] < cost[k]:
            k = h
    if 0 < k < H-1:
        y0, y1, y2 = cost[k-1], cost[k], cost[k+1]
        curv = y0-2*y1+y2
        if curv > 0:
            return HEADINGS[k]+min(max(0.5*(y0-y2)/curv, -0.5), 0.5)*(HEADINGS[k+1]-HEADINGS[k])
    return HEADINGS[k]


@njit
def _box_fits(e0, e1, L, W):
    """lidar._box_fits with L, W the size plus slack: 1 (fits), 0 (cannot), -1 (undecided)."""
    if e0 <= L and e1 <= W:
        return 1
    if e0 > L+W*SIN_MAX or e1 > L*SIN_MAX+W:
        return 0
    return -1


@njit
def _fits(Q, o, L, W, main):
    """lidar._fits(Q, o, size, main) (main: Q itself for none)."""
    lo0, hi0, lo1, hi1 = np.inf, -np.inf, np.inf, -np.inf
    for i in range(Q.shape[0]):
        lo0, hi0 = min(lo0, Q[i, 0]), max(hi0, Q[i, 0])
        lo1, hi1 = min(lo1, Q[i, 1]), max(hi1, Q[i, 1])
    ok = _box_fits(hi0-lo0, hi1-lo1, L, W)
    if ok >= 0:
        return ok == 1
    th = lshape_heading_np(main if main.shape[0] >= 5 else Q, o)
    c, s = np.cos(th), np.sin(th)
    lo0, hi0, lo1, hi1 = np.inf, -np.inf, np.inf, -np.inf
    for i in range(Q.shape[0]):
        a = c*Q[i, 0]+s*Q[i, 1]
        b = c*Q[i, 1]-s*Q[i, 0]
        lo0, hi0 = min(lo0, a), max(hi0, a)
        lo1, hi1 = min(lo1, b), max(hi1, b)
    return hi0-lo0 <= L and hi1-lo1 <= W


@njit
def _free_gap(miss, r, drop, far, rmax, bs):
    """lidar._free_gap for one gap."""
    pd = min(max(drop+far*min(r/rmax, 1.0), 1e-3), 0.95)
    return miss >= max(2.0, np.ceil(LOG_GAP_P/np.log(pd))) and miss*r*bs >= GAP_ARC


@njit
def _free_end(row, step, rd, label):
    """lidar._free_end."""
    n = rd.shape[0]
    k = row+step
    if not 0 <= k < n:
        return False
    for _ in range(END_LOOK):
        if not 0 <= k < n or label[k] == OFF:
            return True
        if label[k] != NONE:
            return rd[k] > rd[row]+OCCLUDER_MARGIN
        k += step
    return True


@njit
def _gather(Wc, rows):
    Q = np.empty((rows.shape[0], 2))
    for q in range(rows.shape[0]):
        Q[q, 0], Q[q, 1] = Wc[rows[q], 0], Wc[rows[q], 1]
    return Q


@njit
def sweep_kernel(rng, az_in, tm_in, M0, M1, prm, X3, ts3):
    """process_lidar up to the box fits. rng, az_in, tm_in: the scan's beams; M0, M1 (k, 3) the teammates at t_start
    and t_end; prm the scalars (P_*); X3 (3, 3), ts3 (3,) the poses and times of lidar._lagrange when the motion is not
    steady. Returns (C, offs (C+1,), points (N, 2), times (N,), free (C, 2), t_ref (C,)): C clusters (-1 for under 3
    returns kept), the rows of cluster c at offs[c]:offs[c+1], in order of their first beam."""
    n = rng.shape[0]
    t_end, t0 = prm[P_T_END], prm[P_T0]
    mx, my, cm, sm = prm[P_MX], prm[P_MY], prm[P_CM], prm[P_SM]
    bs = prm[P_BS]
    srt = True
    for q in range(n-1):
        if not az_in[q] <= az_in[q+1]:
            srt = False
            break
    order = np.arange(n) if srt else np.argsort(az_in, kind='mergesort')     # stable: identity when sorted
    az, t = np.empty(n), np.empty(n)
    P = np.empty((n, 2))
    rd = np.empty(n)
    Wp = np.empty((n, 2))
    Wc = np.empty((n, 2))
    # ---- de-skew (lidar.deskew), then the world frame at t_end (geometry.to_world)
    steady = prm[P_STEADY] != 0.0
    vx, vy, rr = prm[P_VX], prm[P_VY], prm[P_R]
    xe = np.zeros(3)
    if not steady:
        for k in range(3):
            i, j = (1, 2) if k == 0 else ((0, 2) if k == 1 else (0, 1))
            w = (t_end-ts3[i])*(t_end-ts3[j])/((ts3[k]-ts3[i])*(ts3[k]-ts3[j]))
            for q in range(3):
                xe[q] += w*X3[k, q]
    xi = np.zeros(3)
    for q in range(n):
        u = order[q]
        tt = tm_in[u]
        az[q], t[q] = az_in[u], tt
        if not np.isfinite(rng[u]):          # no return: every coordinate is non-finite (rd is all that is read)
            P[q, 0] = P[q, 1] = rd[q] = Wp[q, 0] = Wp[q, 1] = np.nan
            continue
        sx, sy = rng[u]*np.cos(az_in[u]), rng[u]*np.sin(az_in[u])
        px, py = cm*sx-sm*sy+mx, sm*sx+cm*sy+my
        if steady:
            tau = t_end-tt
            th = rr*tau
            sa_ = _sinc(th/np.pi)
            sb_ = _sinc(th/(2*np.pi))
            sb_ = 0.5*th*(sb_*sb_)
            dpsi, dx, dy = th, tau*(vx*sa_-vy*sb_), tau*(vx*sb_+vy*sa_)
        else:
            xi[0], xi[1], xi[2] = 0.0, 0.0, 0.0
            for k in range(3):
                i, j = (1, 2) if k == 0 else ((0, 2) if k == 1 else (0, 1))
                w = (tt-ts3[i])*(tt-ts3[j])/((ts3[k]-ts3[i])*(ts3[k]-ts3[j]))
                for q3 in range(3):
                    xi[q3] += w*X3[k, q3]
            dpsi = xe[2]-xi[2]
            ex, ey = xe[0]-xi[0], xe[1]-xi[1]
            ci, si = np.cos(xi[2]), np.sin(xi[2])
            dx, dy = ci*ex+si*ey, -si*ex+ci*ey
        cp, sp = np.cos(dpsi), np.sin(dpsi)
        qx, qy = px-dx, py-dy
        ex, ey = cp*qx+sp*qy-mx, -sp*qx+cp*qy-my
        p0, p1 = cm*ex+sm*ey, -sm*ex+cm*ey
        P[q, 0], P[q, 1] = p0, p1
        rd[q] = np.hypot(p0, p1)
        oc, os_ = prm[P_OC], prm[P_OS]
        Wp[q, 0] = oc*p0-os_*p1+prm[P_OX]
        Wp[q, 1] = os_*p0+oc*p1+prm[P_OY]
    # ---- labels: kept, teammate (lidar._mate_mask), off the road
    label = np.zeros(n, np.int8)
    for q in range(n):
        if np.isfinite(rd[q]):
            label[q] = KEEP
    if prm[P_MATES] != 0.0:
        t1 = t_end
        shift, hl, hw = prm[P_SHIFT], prm[P_HL], prm[P_HW]
        km = M0.shape[0]
        dyaw = np.empty(km)
        for m in range(km):
            dyaw[m] = (M1[m, 2]-M0[m, 2]+np.pi) % (2*np.pi)-np.pi
        reach = np.hypot(hl, hw)+abs(shift)+1.0     # no point farther than this from a CG is inside its box
        reach2 = reach*reach
        for q in range(n):
            if label[q] != KEEP:
                continue
            w = (t[q]-t0)/(t1-t0) if t1 > t0 else 0.0
            for m in range(km):
                x = M0[m, 0]+w*(M1[m, 0]-M0[m, 0])
                y = M0[m, 1]+w*(M1[m, 1]-M0[m, 1])
                if (Wp[q, 0]-x)**2+(Wp[q, 1]-y)**2 > reach2:
                    continue
                yaw = M0[m, 2]+w*dyaw[m]
                cy, sy_ = np.cos(yaw), np.sin(yaw)
                dx, dy = Wp[q, 0]-(x+shift*cy), Wp[q, 1]-(y+shift*sy_)
                if abs(cy*dx+sy_*dy) <= hl and abs(cy*dy-sy_*dx) <= hw:
                    label[q] = MATE
                    break
    edge = prm[P_EDGE]
    nk = 0
    for q in range(n):
        if label[q] == KEEP and abs(Wp[q, 1]) > edge:
            label[q] = OFF
        nk += label[q] == KEEP
    if nk < 3:
        return -1, np.zeros(1, np.int64), np.zeros((0, 2)), np.zeros(0), np.zeros((0, 2), np.bool_), np.zeros(0)
    vwx, vwy, tmid = prm[P_VWX], prm[P_VWY], prm[P_T_MID]
    for q in range(n):
        dt = tmid-t[q]
        Wc[q, 0], Wc[q, 1] = Wp[q, 0]+dt*vwx, Wp[q, 1]+dt*vwy
    drop, far, rmax, sig3 = prm[P_DROP], prm[P_FAR], prm[P_RMAX], prm[P_SIG3]
    # ---- segments (lidar._segments): starts and ends index K
    K = np.empty(nk, np.int64)
    cum_other = np.empty(n, np.int64)
    cum_off = np.empty(n, np.int64)
    nka = nko = 0
    for q in range(n):
        if label[q] == KEEP:
            K[nka] = q
            nka += 1
        nko += label[q] == MATE or label[q] == OFF
        cum_other[q] = nko
        cum_off[q] = (cum_off[q-1] if q else 0)+(label[q] == OFF)
    starts = np.empty(nk, np.int64)
    ends = np.empty(nk, np.int64)
    S = 1
    starts[0] = 0
    for q in range(nk-1):
        ia, ib = K[q], K[q+1]
        dphi = az[ib]-az[ia]
        de = max(dphi, bs)
        brk = de >= BREAK_ANGLE
        if not brk:
            dist = np.hypot(P[ib, 0]-P[ia, 0], P[ib, 1]-P[ia, 1])
            thr = rd[ia]*np.sin(de)/np.sin(BREAK_ANGLE-de)+sig3
            brk = dist > thr or cum_other[ib] > cum_other[ia] or \
                _free_gap(max(np.rint(dphi/bs)-1, 0.0), rd[ia], drop, far, rmax, bs)
        if brk:
            ends[S-1] = q
            starts[S] = q+1
            S += 1
    ends[S-1] = nk-1
    # ---- merge (lidar._merge)
    first, last = np.empty(S, np.int64), np.empty(S, np.int64)
    lo = np.empty((S, 2))
    hi = np.empty((S, 2))
    rmx = np.empty(S)
    for g in range(S):
        first[g], last[g] = K[starts[g]], K[ends[g]]
        l0, l1, h0, h1, rm = np.inf, np.inf, -np.inf, -np.inf, -np.inf
        for q in range(starts[g], ends[g]+1):
            row = K[q]
            l0, h0 = min(l0, Wc[row, 0]), max(h0, Wc[row, 0])
            l1, h1 = min(l1, Wc[row, 1]), max(h1, Wc[row, 1])
            rm = max(rm, rd[row])
        lo[g, 0], lo[g, 1], hi[g, 0], hi[g, 1], rmx[g] = l0, l1, h0, h1, rm
    size_l, size_w = prm[P_LEN], prm[P_WID]
    L, W = size_l+FIT_SLACK[0], size_w+FIT_SLACK[1]
    o = np.empty(2)
    o[0], o[1] = prm[P_OX], prm[P_OY]
    head = np.arange(S)            # each cluster's member segments as a linked list in lidar._merge's list order
    tail = np.arange(S)
    nxt = np.full(S, -1)
    count = np.empty(S, np.int64)  # rows per cluster
    for g in range(S):
        count[g] = ends[g]-starts[g]+1
    root = np.arange(S)
    alive = np.ones(S, np.bool_)
    if S >= 2:
        gmate = np.full(S-1, -np.inf)
        g = 0
        for q in range(n):
            if label[q] != MATE:
                continue
            while g < S and first[g] < q:      # g = np.searchsorted(first, q) - 1, q ascending
                g += 1
            if 0 <= g-1 < S-1:
                gmate[g-1] = max(gmate[g-1], rd[q])
        nbad = np.zeros(S, np.int64)
        for g in range(S-1):
            miss = max(np.rint((az[first[g+1]]-az[last[g]])/bs)-1, 0.0)
            bad = (_free_gap(miss, rd[last[g]], drop, far, rmax, bs) and not np.isfinite(gmate[g])) or \
                cum_off[first[g+1]] > cum_off[last[g]]
            nbad[g+1] = nbad[g]+bad
        for d in range(1, 4):
            for i in range(S-d):
                near = min(rd[last[i]], rd[first[i+d]])-OCCLUDER_MARGIN
                ok = nbad[i+d] == nbad[i]
                for m in range(d):
                    ok = ok and gmate[i+m] < near
                    if m:
                        ok = ok and rmx[i+m] < near
                if not ok:
                    continue
                ci, cj = root[i], root[i+d]
                if ci == cj:
                    continue
                b0, b1 = min(lo[ci, 0], lo[cj, 0]), min(lo[ci, 1], lo[cj, 1])
                c0, c1 = max(hi[ci, 0], hi[cj, 0]), max(hi[ci, 1], hi[cj, 1])
                fit = _box_fits(c0-b0, c1-b1, L, W)
                if fit < 0:
                    ni, nj = count[ci], count[cj]
                    rows = np.empty(ni+nj, np.int64)
                    p = 0
                    for cc in (ci, cj):
                        mseg = head[cc]
                        while mseg >= 0:
                            for q in range(starts[mseg], ends[mseg]+1):
                                rows[p] = K[q]
                                p += 1
                            mseg = nxt[mseg]
                    Q = _gather(Wc, rows)
                    main = Q[:ni].copy() if ni >= nj else Q[ni:].copy()
                    fit = 1 if _fits(Q, o, L, W, main) else 0
                if fit == 1:
                    mseg = head[cj]
                    while mseg >= 0:
                        root[mseg] = ci
                        mseg = nxt[mseg]
                    nxt[tail[ci]] = head[cj]
                    tail[ci] = tail[cj]
                    count[ci] += count[cj]
                    alive[cj] = False
                    lo[ci, 0], lo[ci, 1], hi[ci, 0], hi[ci, 1] = b0, b1, c0, c1
    # ---- each merged cluster's rows in beam order, then split (lidar._split)
    pieces_a = np.empty(nk, np.int64)       # pieces as ranges of `ordered`
    pieces_b = np.empty(nk, np.int64)
    npc = 0
    ordered = np.empty(nk, np.int64)
    p = 0
    stack_a = np.empty(64, np.int64)
    stack_b = np.empty(64, np.int64)
    stack_d = np.empty(64, np.int64)
    seg_of = np.zeros(S, np.bool_)
    for c in range(S):
        if not alive[c]:
            continue
        seg_of[:] = False
        mseg = head[c]
        while mseg >= 0:
            seg_of[mseg] = True
            mseg = nxt[mseg]
        a0 = p
        for g in range(S):                  # np.sort of the members' rows: segments ascending
            if seg_of[g]:
                for q in range(starts[g], ends[g]+1):
                    ordered[p] = K[q]
                    p += 1
        ns = 1
        stack_a[0], stack_b[0], stack_d[0] = a0, p, 0
        while ns:
            ns -= 1
            sa, sb, depth = stack_a[ns], stack_b[ns], stack_d[ns]
            m = sb-sa
            if m >= 3 and depth < 4:
                Q = _gather(Wc, ordered[sa:sb])
                if not _fits(Q, o, L, W, Q):
                    best, kbest = -1.0, 0
                    for q in range(m-1):
                        hq = np.hypot(Q[q+1, 0]-Q[q, 0], Q[q+1, 1]-Q[q, 1])
                        if hq > best:
                            best, kbest = hq, q
                    k = kbest+1
                    stack_a[ns], stack_b[ns], stack_d[ns] = sa+k, sb, depth+1
                    stack_a[ns+1], stack_b[ns+1], stack_d[ns+1] = sa, sa+k, depth+1
                    ns += 2
                    continue
            if m >= 3:
                pieces_a[npc], pieces_b[npc] = sa, sb
                npc += 1
    # ---- clusters in order of their first row, with free ends and mean beam time
    firsts = np.empty(npc, np.int64)
    for c in range(npc):
        firsts[c] = ordered[pieces_a[c]]
    srt = np.argsort(firsts)
    total = 0
    for c in range(npc):
        total += pieces_b[c]-pieces_a[c]
    offs = np.empty(npc+1, np.int64)
    pts = np.empty((total, 2))
    tms = np.empty(total)
    free = np.empty((npc, 2), np.bool_)
    tref = np.empty(npc)
    p = 0
    offs[0] = 0
    for c in range(npc):
        sa, sb = pieces_a[srt[c]], pieces_b[srt[c]]
        for q in range(sa, sb):
            row = ordered[q]
            pts[p, 0], pts[p, 1] = Wp[row, 0], Wp[row, 1]
            tms[p] = t[row]
            p += 1
        offs[c+1] = p
        free[c, 0] = _free_end(ordered[sa], -1, rd, label)
        free[c, 1] = _free_end(ordered[sb-1], 1, rd, label)
        tref[c] = _pwsum(tms, offs[c], sb-sa)/(sb-sa)
    return npc, offs, pts, tms, free, tref


@njit
def fit_clusters(pts, tms, offs, sensors, free, refine, refine_min, velocity, prm):
    """_lidar_jit.fit_box_kernel on each cluster c: rows offs[c]:offs[c+1] of pts, tms, seen from sensors[c], refined
    when refine and it has refine_min points or more (prm: sweep_kernel's parameters). Stacked (z (C, 3), R (C, 3, 3),
    sides (C,), dz_dv (C, 3, 2), R_fit (C, 3, 3))."""
    C = offs.shape[0]-1
    Z = np.empty((C, 3))
    R = np.empty((C, 3, 3))
    sides = np.empty(C, np.int64)
    DZ = np.empty((C, 3, 2))
    RF = np.empty((C, 3, 3))
    size = np.empty(2)
    size[0], size[1] = prm[P_LEN], prm[P_WID]
    vs = np.empty(2)
    vs[0], vs[1] = prm[P_VS0], prm[P_VS1]
    headings = HEADINGS.copy()
    for c in range(C):
        a, b = offs[c], offs[c+1]
        z, cov, ns, _, dz, rf = fit_box_kernel(pts[a:b].copy(), tms[a:b].copy(), sensors[c].copy(), size, prm[P_SR],
                                               prm[P_SMOD], free[c, 0], free[c, 1], prm[P_DROP], prm[P_FAR],
                                               prm[P_RMAX], velocity, vs, refine and b-a >= refine_min, headings)
        Z[c] = z
        R[c] = cov
        sides[c] = ns
        DZ[c] = dz
        RF[c] = rf
    return Z, R, sides, DZ, RF
