"""Compiled kernels of the Tier A radar (sim/radar.py): one sim step's scans (every radar on every team car) in three
calls, instead of numpy's many small array operations per step.

radar._scans_compiled runs them between its random draws, which stay in Python with the numpy path's sizes and order
(the random stream is unchanged):

    view_pd        sensor poses and velocities, each car's nearest outline point, the field of view, the visible
                   fraction (radar._visibility) and the detection probability of every candidate;
                   then the detection draw, the point counts and offsets, and the clutter draws (radar.py);
    points_merge   the reflection points, which of them are kept, their range rates, the clutter points and the
                   resolution merge (radar._merge);
                   then the noise draw (radar.py);
    noise_sort     the noise, and each scan's points by measured range.

Each kernel is radar.py's numpy code written out per element, in the same order of operations, so the two paths agree
to rounding (the math library's last bit; on the test scenes they agree to the bit). Shortcuts skip work without
changing a decision: a box more than range_max + 2 diagonals + REACH_MARGIN away can neither be in view nor block a
candidate (its nearest point is not computed); the nearest of a box's four edge points is chosen by squared distance
(hypot only for near ties, NEAR_TIE); the field of view and a sample's side of a bearing span are decided by a cosine
when it is more than COS_MARGIN from the edge's (atan2 as numpy otherwise); two boxes whose enclosing cones are more
than CONE_MARGIN apart cannot overlap in bearing (no span computed); a candidate that nothing can block is seen in
full once one sample is near; spans, bearings and ray directions are computed once, when first needed.

radar.py's numpy code stays the reference and the path with NOHESI_JIT=0 (_jit.py); there these kernels would run as
plain Python, correct but slow, and radar.py does not call them. The spec of each scan is a row of the TAB layout
below; the module constants are radar.py's, duplicated (numba's on-disk cache keys on this file only) and checked
equal by tests/test_perception_jit_radar_sim.py.
"""
from math import atan2, cos, hypot, inf, log, sin, sqrt
import numpy as np
from .._jit import njit

__all__ = ['TAB_FIELDS', 'view_pd', 'points_merge', 'noise_sort']

CLUTTER_RANGE_MIN = 5.0              # radar.CLUTTER_RANGE_MIN
CLUTTER_STATIONARY = 0.7             # radar.CLUTTER_STATIONARY
NEXT = np.array([1, 2, 3, 0])        # radar.NEXT
REACH_MARGIN = 1.0                   # m: a box nearer than range_max + 2 diagonals + this gets its nearest point
CONE_MARGIN = 1e-6                   # in cosine: two boxes' enclosing cones this far apart cannot overlap in bearing
COS_MARGIN = 1e-7                    # in cosine: a bearing this far from an edge is decided without atan2
NEAR_TIE = 1e-12                     # relative, in squared distance: the edge points whose hypot decides the nearest

TAB_FIELDS = ('range_max', 'near_range', 'fov', 'fov_far', 'pd_ref', 'ref_range', 'occluded_pd', 'res_range',
              'res_azimuth', 'sigma_range', 'sigma_azimuth', 'sigma_range_rate', 'mount_x', 'mount_y', 'mount_yaw')
RMAX, NEAR, FOV, FOV_FAR, PD_REF, REF, OCC, RES_R, RES_AZ, SIG_R, SIG_AZ, SIG_RR, OX, OY, MYAW = range(15)


@njit(error_model='numpy')
def _wrap(a):
    """geometry.wrap: numpy's remainder takes the sign of the divisor, as Python's float % does."""
    return (a+np.pi) % (2*np.pi)-np.pi


@njit(error_model='numpy')
def _gap(a):
    """radar._angle_gap."""
    m = abs(a)
    return min(m, 2*np.pi-m)


@njit(error_model='numpy')
def _half_fov(tab, s, r):
    return (tab[s, FOV] if r <= tab[s, NEAR] else tab[s, FOV_FAR])/2


@njit(error_model='numpy')
def _pd_range(pd_ref, ref_range, range_max, r):
    """radar._pd_range of one range."""
    p = min(max(pd_ref, 0.0), 1.0)
    if p > 0.5 and p < 1 and ref_range > 0 and ref_range < range_max:
        n = log(p/(1-p))/log(range_max/ref_range)
        return min(p, 1/(1+(max(r, 0.0)/range_max)**n))
    return p


@njit(error_model='numpy')
def _span(boxes, E, ox, oy, b, out):
    """out <- (mid, half, far, cos mid, sin mid, cos half) of box b seen from (ox, oy): radar._visibility's bearing
    span and farthest corner."""
    cx = 0.0
    cy = 0.0
    for k in range(4):
        cx += boxes[b, k, 0]-ox
        cy += boxes[b, k, 1]-oy
    lo, hi, far = inf, -inf, -inf
    inside = True
    for k in range(4):
        rx, ry = boxes[b, k, 0]-ox, boxes[b, k, 1]-oy
        side = atan2(cx*ry-cy*rx, cx*rx+cy*ry)
        lo, hi = min(lo, side), max(hi, side)
        inside = inside and E[b, k, 0]*ry-E[b, k, 1]*rx < 0
        far = max(far, hypot(rx, ry))
    out[0] = _wrap(atan2(cy, cx)+(lo+hi)/2)
    out[1] = np.pi if inside else (hi-lo)/2
    out[2] = far
    out[3], out[4], out[5] = cos(out[0]), sin(out[0]), cos(out[1])


@njit(error_model='numpy')
def _cone(centre, rad, ox, oy, b, out):
    """out <- (ux, uy, sin a, cos a): the unit direction from (ox, oy) to box b's centre and the half angle a of the
    circle of radius rad[b] about it, which holds the box (sin a = 1 when the sensor is inside the circle)."""
    dx, dy = centre[b, 0]-ox, centre[b, 1]-oy
    d = sqrt(dx*dx+dy*dy)
    if d <= rad[b]:
        out[2] = 1.0
        return
    out[0], out[1] = dx/d, dy/d
    out[2] = rad[b]/d
    out[3] = sqrt(1-out[2]*out[2])


@njit(error_model='numpy')
def _angle(ang, has, px, py, ox, oy, i):
    """The bearing of sample i (radar._visibility's ang), computed once per candidate."""
    if not has[i]:
        ang[i] = atan2(py[i]-oy, px[i]-ox)
        has[i] = True
    return ang[i]


@njit(error_model='numpy')
def _visible(boxes, E, ox, oy, s, t, own, reach, f, centre, rad, span, have, cone, have_cone, cand, px, py, dist,
             sx, sy, near, ang, has_ang, dcs, dsn, has_dir, inb, blocked):
    """radar._visibility of box t from sensor s at (ox, oy): reach (M,) the sensor's nearest distances (inf where not
    computed: such a box cannot block), own the box it skips; centre, rad (M,) the boxes' centres and enclosing radii;
    span (S, M, 6), cone (S, M, 4) and have, have_cone (S, M) the spans and cones computed so far; cand (M,) and
    px .. blocked (4 len(f),) scratch.

    numpy's bearings (atan2) are computed only where a decision needs them: a box whose cone (_cone) is more than
    CONE_MARGIN (in cosine) from the target's cannot overlap its bearing span; a sample whose bearing is more than
    COS_MARGIN in cosine from a span's edge is inside or outside it by its dot product with the span's mid direction
    (the cosine is 1-Lipschitz in the angle, so such a sample is also more than 1e-9 from the edge: not grazing). The
    rest, and the ray test, use numpy's formulas on numpy's bearings."""
    nf = f.shape[0]
    ns = 4*nf
    mx = (((boxes[t, 0, 0]+boxes[t, 1, 0])+boxes[t, 2, 0])+boxes[t, 3, 0])/4
    my = (((boxes[t, 0, 1]+boxes[t, 1, 1])+boxes[t, 2, 1])+boxes[t, 3, 1])/4
    dx, dy = mx-ox, my-oy
    dc = sqrt(dx*dx+dy*dy)
    nb = 0                                   # the boxes that can block (numpy's pair) before the samples: reach below
    for b in range(boxes.shape[0]):          # dc (>= the farthest near sample) and the bearing spans overlap
        if b == t or b == own or not reach[b] < dc:
            continue
        if not have_cone[s, t]:
            _cone(centre, rad, ox, oy, t, cone[s, t])
            have_cone[s, t] = True
        if not have_cone[s, b]:
            _cone(centre, rad, ox, oy, b, cone[s, b])
            have_cone[s, b] = True
        sb, st = cone[s, b, 2], cone[s, t, 2]
        if sb < 1 and st < 1:                    # the separation against the sum of the half angles (< pi), in cosine
            cb, ct = cone[s, b, 3], cone[s, t, 3]
            if cone[s, b, 0]*cone[s, t, 0]+cone[s, b, 1]*cone[s, t, 1] < cb*ct-sb*st-CONE_MARGIN:
                continue
        if not have[s, t]:
            _span(boxes, E, ox, oy, t, span[s, t])
            have[s, t] = True
        if not have[s, b]:
            _span(boxes, E, ox, oy, b, span[s, b])
            have[s, b] = True
        if _gap(span[s, b, 0]-span[s, t, 0]) <= span[s, b, 1]+span[s, t, 1]+1e-9:
            cand[nb] = b
            nb += 1
    count = 0
    nmax, nmin = 0.0, inf
    for k in range(4):
        for j in range(nf):
            i = k*nf+j
            px[i] = boxes[t, k, 0]+f[j]*E[t, k, 0]
            py[i] = boxes[t, k, 1]+f[j]*E[t, k, 1]
            dx, dy = px[i]-ox, py[i]-oy
            dist[i] = sqrt(dx*dx+dy*dy)
            if dist[i] <= dc:
                if nb == 0:                      # nothing blocks: every near sample is seen
                    return 1.0
                count += 1
                nmax, nmin = max(nmax, dist[i]), min(nmin, dist[i])
                near[i] = True
            else:
                near[i] = False
            if dist[i] > 0:
                sx[i], sy[i] = dx/dist[i], dy/dist[i]
            blocked[i] = False
            has_ang[i] = False
            has_dir[i] = False
    if count == 0:
        return 0.0
    for q in range(nb):
        b = cand[q]
        if not reach[b] < nmax:
            continue
        mid, half, far = span[s, b, 0], span[s, b, 1], span[s, b, 2]
        rays = far >= nmin-1e-6
        if not rays:
            cm, sm, ch = span[s, b, 3], span[s, b, 4], span[s, b, 5]
            for i in range(ns):
                cg = sx[i]*cm+sy[i]*sm if dist[i] > 0 else ch
                if cg-ch > COS_MARGIN:
                    inb[i] = True
                elif ch-cg > COS_MARGIN:
                    inb[i] = False
                else:
                    g = _gap(_angle(ang, has_ang, px, py, ox, oy, i)-mid)-half
                    inb[i] = g <= 0
                    if abs(g) <= 1e-9:               # a grazing sample: the ray test for every sample
                        rays = True
                        break
        if not rays:
            for i in range(ns):
                if inb[i]:
                    blocked[i] = True
            continue
        for i in range(ns):
            if not near[i] or blocked[i]:
                continue
            if not has_dir[i]:
                a = _angle(ang, has_ang, px, py, ox, oy, i)
                dcs[i], dsn[i] = cos(a), sin(a)
                has_dir[i] = True
            ux, uy = dcs[i], dsn[i]
            for k in range(4):
                ex, ey = E[b, k, 0], E[b, k, 1]
                wx, wy = boxes[b, k, 0]-ox, boxes[b, k, 1]-oy
                den = ux*ey-uy*ex
                if abs(den) > 1e-12:
                    tr = (wx*ey-wy*ex)/den
                    ue = (wx*uy-wy*ux)/den
                    if tr > 0 and ue >= 0 and ue <= 1 and tr < dist[i]-1e-6:
                        blocked[i] = True
                        break
    seen = 0
    for i in range(ns):
        if near[i] and not blocked[i]:
            seen += 1
    return seen/max(count, 1)


@njit(error_model='numpy')
def _in_fov(qx, qy, d, cb, sb, yaw, hf, ch):
    """abs(wrap(atan2(qy, qx) - yaw)) <= hf for a point at distance d = hypot(qx, qy), the boresight's cos and sin
    (cb, sb) and ch = cos(hf): by the cosine of the angle off the boresight when it is more than COS_MARGIN from ch."""
    if hf >= np.pi:
        return True
    if hf >= 0 and d > 0:
        c = (qx*cb+qy*sb)/d
        if c-ch > COS_MARGIN:
            return True
        if ch-c > COS_MARGIN:
            return False
    return abs(_wrap(atan2(qy, qx)-yaw)) <= hf


@njit(error_model='numpy')
def view_pd(team, team_vy, team_r, vx, cars, n_traffic, tab, boxes, f):
    """The candidates of every scan and their detection probabilities, as radar._scans_numpy up to the detection draw.

    team (3, 4) [x, y, yaw, v], team_vy, team_r, vx (3,) the team's lateral speed, yaw rate and forward speed; cars (S,)
    each scan's team car; n_traffic the number of traffic boxes (team car j's box is n_traffic + j); tab (S, 15) the
    specs (TAB_FIELDS); boxes (M, 4, 2); f (samples,) the sample fractions along an edge. Returns the sensor positions
    O (S, 2), boresights yaw (S,) and velocities VS (S, 2), and the candidates' sensors si, boxes ti and pd (C,).

    The nearest point of a box is the first of its four edge points with the least hypot, as numpy's argmin: hypot is
    taken only for the points whose squared distance is within a relative NEAR_TIE of the least."""
    S, M = cars.shape[0], boxes.shape[0]
    O, yaw, VS = np.empty((S, 2)), np.empty(S), np.empty((S, 2))
    for s in range(S):
        car = cars[s]
        c, sn = cos(team[car, 2]), sin(team[car, 2])
        ox, oy = tab[s, OX], tab[s, OY]
        O[s, 0] = team[car, 0]+c*ox-sn*oy
        O[s, 1] = team[car, 1]+sn*ox+c*oy
        yaw[s] = _wrap(team[car, 2]+tab[s, MYAW])
        yr = team_r[car]
        bx, by = vx[car]-yr*oy, team_vy[car]+yr*ox
        VS[s, 0] = c*bx-sn*by
        VS[s, 1] = sn*bx+c*by
    E, L2 = np.empty((M, 4, 2)), np.empty((M, 4))
    centre, rad = np.empty((M, 2)), np.empty(M)
    diag = 0.0
    for b in range(M):
        mx, my = 0.0, 0.0
        for k in range(4):
            E[b, k, 0] = boxes[b, NEXT[k], 0]-boxes[b, k, 0]
            E[b, k, 1] = boxes[b, NEXT[k], 1]-boxes[b, k, 1]
            L2[b, k] = E[b, k, 0]*E[b, k, 0]+E[b, k, 1]*E[b, k, 1]
            mx += boxes[b, k, 0]
            my += boxes[b, k, 1]
        centre[b, 0], centre[b, 1] = mx/4, my/4
        r = 0.0
        for k in range(4):
            r = max(r, hypot(boxes[b, k, 0]-centre[b, 0], boxes[b, k, 1]-centre[b, 1]))
        rad[b] = r*(1+1e-9)+1e-9
        diag = max(diag, hypot(boxes[b, 2, 0]-boxes[b, 0, 0], boxes[b, 2, 1]-boxes[b, 0, 1]))
    reach = np.full((S, M), inf)
    view = np.zeros((S, M), np.bool_)
    qx4, qy4, d24 = np.empty(4), np.empty(4), np.empty(4)
    C = 0
    for s in range(S):
        ox, oy = O[s, 0], O[s, 1]
        lim = tab[s, RMAX]+2*diag+REACH_MARGIN    # beyond it a box is out of view and cannot block one in view
        lim2 = lim*lim
        own = n_traffic+cars[s]
        cb, sb = cos(yaw[s]), sin(yaw[s])
        hn, hf = tab[s, FOV]/2, tab[s, FOV_FAR]/2
        chn, chf = cos(hn), cos(hf)
        for b in range(M):
            dx, dy = boxes[b, 0, 0]-ox, boxes[b, 0, 1]-oy
            if dx*dx+dy*dy > lim2:
                continue
            least = inf
            for k in range(4):
                rx, ry = boxes[b, k, 0]-ox, boxes[b, k, 1]-oy
                ex, ey = E[b, k, 0], E[b, k, 1]
                u = min(max(-(rx*ex+ry*ey)/L2[b, k], 0.0), 1.0)
                qx4[k], qy4[k] = rx+u*ex, ry+u*ey
                d24[k] = qx4[k]*qx4[k]+qy4[k]*qy4[k]
                least = min(least, d24[k])
            best, qx, qy = inf, 0.0, 0.0
            for k in range(4):
                if d24[k] <= least*(1+NEAR_TIE):
                    d = hypot(qx4[k], qy4[k])
                    if d < best:
                        best, qx, qy = d, qx4[k], qy4[k]
            reach[s, b] = best
            if b != own and best <= tab[s, RMAX]:
                if best <= tab[s, NEAR]:
                    seen = _in_fov(qx, qy, best, cb, sb, yaw[s], hn, chn)
                else:
                    seen = _in_fov(qx, qy, best, cb, sb, yaw[s], hf, chf)
                if seen:
                    view[s, b] = True
                    C += 1
    si, ti, pd = np.empty(C, np.int64), np.empty(C, np.int64), np.empty(C)
    ns = 4*f.shape[0]
    px, py, dist, ang, dcs, dsn = (np.empty(ns), np.empty(ns), np.empty(ns), np.empty(ns), np.empty(ns),
                                   np.empty(ns))
    sx, sy, cand = np.empty(ns), np.empty(ns), np.empty(M, np.int64)
    near, blocked, inb = np.empty(ns, np.bool_), np.empty(ns, np.bool_), np.empty(ns, np.bool_)
    has_ang, has_dir = np.empty(ns, np.bool_), np.empty(ns, np.bool_)
    span, have = np.empty((S, M, 6)), np.zeros((S, M), np.bool_)
    cone, have_cone = np.empty((S, M, 4)), np.zeros((S, M), np.bool_)
    c = 0
    for s in range(S):
        occ = tab[s, OCC]
        for b in range(M):
            if not view[s, b]:
                continue
            si[c], ti[c] = s, b
            p = _pd_range(tab[s, PD_REF], tab[s, REF], tab[s, RMAX], reach[s, b])
            if occ < 1:
                v = _visible(boxes, E, O[s, 0], O[s, 1], s, b, n_traffic+cars[s], reach[s], f, centre, rad, span,
                             have, cone, have_cone, cand, px, py, dist, sx, sy, near, ang, has_ang, dcs, dsn, has_dir,
                             inb, blocked)
                p *= occ+(1-occ)*v
            pd[c] = p
            c += 1
    return O, yaw, VS, si, ti, pd


@njit(error_model='numpy')
def points_merge(O, yaw, VS, tab, ds, dt, centre, heading, vel, ids, hx, hy, n_pts, offsets, cs, u, uni):
    """Every scan's points before noise, as radar._scans_numpy from the detection draw to the merge: the reflection
    points of the detected cars (sensors ds, boxes dt (D,), point counts n_pts (D,), arc offsets (D, m - 1)) that are
    kept, then the clutter of sensors cs (n,) from the uniform draws u (4, n) and the moving range rates uni (n,),
    merged (radar._merge). centre, heading, vel, ids: every box's body centre, heading, world CG velocity and truth id;
    hx, hy the half size. Returns (r, az, rr, tid, sensor) of the merged points."""
    D, m, nc = ds.shape[0], offsets.shape[1]+1, cs.shape[0]
    cap = D*m+nc
    r, az, rr = np.empty(cap), np.empty(cap), np.empty(cap)
    tid, sen = np.empty(cap, np.int64), np.empty(cap, np.int64)
    n = 0
    for d in range(D):
        s, t = ds[d], dt[d]
        c, sn = cos(heading[t]), sin(heading[t])
        ox, oy = O[s, 0], O[s, 1]
        dx, dy = ox-centre[t, 0], oy-centre[t, 1]
        lx, ly = c*dx+sn*dy, c*dy-sn*dx
        end, side = abs(lx) > hx, abs(ly) > hy
        lo = -2*hy if end else 0.0
        hi = 2*hx if side else 0.0
        q = (0.0 if side else abs(ly)-hy) if end else hx-abs(lx)
        sx = -1.0 if lx < 0 else 1.0
        sy = -1.0 if ly < 0 else 1.0
        rvx, rvy = vel[t, 0]-VS[s, 0], vel[t, 1]-VS[s, 1]
        for j in range(m):
            arc = q if j == 0 else q+offsets[d, j-1]
            arc = min(max(hi-abs(hi-lo-abs(arc-lo)), lo), hi)
            pxl = (hx if arc < 0 else hx-arc)*sx
            pyl = (hy+arc if arc < 0 else hy)*sy
            x = centre[t, 0]+c*pxl-sn*pyl-ox
            y = centre[t, 1]+sn*pxl+c*pyl-oy
            pr = hypot(x, y)
            paz = _wrap(atan2(y, x)-yaw[s])
            if j == 0 or (j < n_pts[d] and pr <= tab[s, RMAX] and abs(paz) <= _half_fov(tab, s, pr)):
                r[n], az[n], rr[n] = pr, paz, (x*rvx+y*rvy)/max(pr, 1e-9)
                tid[n], sen[n] = ids[t], s
                n += 1
    r0 = CLUTTER_RANGE_MIN
    for i in range(nc):
        s = cs[i]
        rmax, near_r, fov, fov_far = tab[s, RMAX], tab[s, NEAR], tab[s, FOV], tab[s, FOV_FAR]
        near_c = min(near_r, rmax)
        a1 = max(near_c-r0, 0.0)*fov
        a2 = max(rmax-max(near_r, r0), 0.0)*fov_far
        outer = u[0, i]*(a1+a2) < a2
        cr = max(near_r, r0) if outer else r0
        cr = cr+((rmax if outer else near_c)-cr)*u[1, i]
        caz = (fov_far if outer else fov)*(u[2, i]-0.5)
        if u[3, i] < CLUTTER_STATIONARY:
            rr[n] = -(cos(yaw[s]+caz)*VS[s, 0]+sin(yaw[s]+caz)*VS[s, 1])
        else:
            rr[n] = uni[i]
        r[n], az[n], tid[n], sen[n] = cr, caz, -1, s
        n += 1
    return _merge(r[:n], az[:n], rr[:n], tid[:n], sen[:n], tab)


@njit(error_model='numpy')
def _root(parent, i):
    while parent[i] != i:
        i = parent[i]
    return i


@njit(error_model='numpy')
def _merge(r, az, rr, tid, sen, tab):
    """radar._merge: connected groups (labelled by their smallest index, in that order), the representative the
    nearest car point (ties by index), means summed in index order."""
    n = r.shape[0]
    if n < 2:
        return r, az, rr, tid, sen
    parent = np.arange(n)
    joined = False
    for i in range(n):
        s = sen[i]
        res_r, res_a = tab[s, RES_R], tab[s, RES_AZ]
        for j in range(i+1, n):
            if sen[j] == s and abs(r[i]-r[j]) < res_r and _gap(az[i]-az[j]) < res_a:
                joined = True
                a, b = _root(parent, i), _root(parent, j)
                if a < b:
                    parent[b] = a
                elif b < a:
                    parent[a] = b
    if not joined:
        return r, az, rr, tid, sen
    lab = np.empty(n, np.int64)
    gid = np.full(n, -1, np.int64)
    G = 0
    for i in range(n):
        root = _root(parent, i)
        if gid[root] < 0:
            gid[root] = G
            G += 1
        lab[i] = gid[root]
    count = np.zeros(G, np.int64)
    rep = np.full(G, -1, np.int64)
    for i in range(n):
        g = lab[i]
        count[g] += 1
        k = rep[g]
        if k < 0 or (tid[i] != -1 and tid[k] == -1) or ((tid[i] == -1) == (tid[k] == -1) and r[i] < r[k]):
            rep[g] = i
    mr, maz, mrr = np.zeros(G), np.zeros(G), np.zeros(G)
    for i in range(n):
        g = lab[i]
        mr[g] += r[i]
        mrr[g] += rr[i]
        maz[g] += _wrap(az[i]-az[rep[g]])
    mtid, msen = np.empty(G, np.int64), np.empty(G, np.int64)
    for g in range(G):
        mr[g] /= count[g]
        mrr[g] /= count[g]
        maz[g] = _wrap(az[rep[g]]+maz[g]/count[g])
        mtid[g], msen[g] = tid[rep[g]], sen[rep[g]]
    return mr, maz, mrr, mtid, msen


@njit(error_model='numpy')
def noise_sort(r, az, rr, tid, sen, z, tab, S):
    """The measurement noise (z (3, n) standard normals) and the scans' points by sensor, then measured range (stable,
    as np.lexsort): (r, az, rr, tid, edge (S + 1,)), scan s holding edge[s]:edge[s + 1]."""
    n = r.shape[0]
    edge = np.zeros(S+1, np.int64)
    for i in range(n):
        edge[sen[i]+1] += 1
    for s in range(S):
        edge[s+1] += edge[s]
    fill = edge[:S].copy()
    order = np.empty(n, np.int64)
    nr = np.empty(n)
    for i in range(n):
        s = sen[i]
        nr[i] = abs(r[i]+tab[s, SIG_R]*z[0, i])
        order[fill[s]] = i
        fill[s] += 1
    for s in range(S):                       # insertion sort of each scan's points by range (stable; scans are short)
        for a in range(edge[s]+1, edge[s+1]):
            k = order[a]
            b = a
            while b > edge[s] and nr[order[b-1]] > nr[k]:
                order[b] = order[b-1]
                b -= 1
            order[b] = k
    out_r, out_az, out_rr, out_tid = np.empty(n), np.empty(n), np.empty(n), np.empty(n, np.int64)
    for a in range(n):
        i = order[a]
        s = sen[i]
        out_r[a] = nr[i]
        out_az[a] = _wrap(az[i]+tab[s, SIG_AZ]*z[1, i])
        out_rr[a] = rr[i]+tab[s, SIG_RR]*z[2, i]
        out_tid[a] = tid[i]
    return out_r, out_az, out_rr, out_tid, edge
