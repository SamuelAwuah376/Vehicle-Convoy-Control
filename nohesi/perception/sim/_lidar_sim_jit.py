"""Compiled ray casting of one lidar sweep: lidar.cast_sweep's (beam, box) pairs and ray-segment test as one numba
kernel (nohesi.perception._jit).

The same steps and formulas as cast_sweep's numpy path, written as loops: each box's bearing span over the knots, the
beams inside it (three turns for a span across the +-pi cut, every beam for a box holding the sensor or spanning
2 pi), and for every pair the box and the sensor interpolated to the beam's time and the nearest crossing of its four
edges. A beam keeps the nearest hit within range_max (on equal ranges the earlier pair, as the numpy path's stable
sort). Results match the numpy path to rounding; with NOHESI_JIT=0 the numpy path runs and this module is unused.
"""
import numpy as np
from .._jit import njit


@njit
def _wrap(a):
    return (a+np.pi) % (2*np.pi)-np.pi


@njit
def _left(a, v):
    """np.searchsorted(a, v) (side 'left') for ascending a."""
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < v:
            lo = mid+1
        else:
            hi = mid
    return lo


@njit
def _right(a, v):
    """np.searchsorted(a, v, side='right') for ascending a."""
    lo, hi = 0, a.shape[0]
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] <= v:
            lo = mid+1
        else:
            hi = mid
    return lo


@njit
def cast_kernel(az, time, knots, sensor, corners, beam_spacing, range_max, half_diag):
    """cast_sweep after the truth lookup. az, time (n,) the beams; knots (K,); sensor (K, 3) [x, y, yaw] at the knots;
    corners (K, B, 4, 2) the boxes at the knots; half_diag = hypot(length, width)/2. Returns (range (n,), np.inf for
    no hit within range_max; box (n,) the index of the box hit, -1 for none)."""
    n, K, B = az.shape[0], knots.shape[0], corners.shape[1]
    ranges = np.full(n, np.inf)
    hit_box = np.full(n, -1, np.int64)
    if B == 0 or n == 0:
        return ranges, hit_box
    two_pi = 2*np.pi
    yaw0 = sensor[0, 2]

    # Each beam's knot segment and weight, the sensor's position and yaw at its time (lidar._lerp), its direction.
    seg = np.zeros(n, np.int64)
    w = np.zeros(n)
    ox, oy, beam_rel = np.empty(n), np.empty(n), np.empty(n)
    dx, dy = np.empty(n), np.empty(n)
    for i in range(n):
        t = time[i]
        if K == 1:
            ox[i], oy[i], yaw = sensor[0, 0], sensor[0, 1], sensor[0, 2]
        else:
            k = min(max(_right(knots, t)-1, 0), K-2)
            wi = (t-knots[k])/(knots[k+1]-knots[k])
            seg[i], w[i] = k, wi
            ox[i] = sensor[k, 0]+wi*(sensor[k+1, 0]-sensor[k, 0])
            oy[i] = sensor[k, 1]+wi*(sensor[k+1, 1]-sensor[k, 1])
            yaw = sensor[k, 2]+wi*(sensor[k+1, 2]-sensor[k, 2])
        beam_rel[i] = az[i]+yaw-yaw0
        phi = yaw+az[i]
        dx[i], dy[i] = np.cos(phi), np.sin(phi)

    turns = (-two_pi, 0.0, two_pi)
    for b in range(B):
        # The bearing span over the knots (mid unwrapped along them), the sensor inside the box, the nearest reach.
        lo, hi = np.inf, -np.inf
        inside = False
        reach = np.inf
        mid0 = 0.0
        acc = 0.0
        prev = 0.0
        for k in range(K):
            sx, sy = sensor[k, 0], sensor[k, 1]
            rx0, ry0 = corners[k, b, 0, 0]-sx, corners[k, b, 0, 1]-sy
            rx1, ry1 = corners[k, b, 1, 0]-sx, corners[k, b, 1, 1]-sy
            rx2, ry2 = corners[k, b, 2, 0]-sx, corners[k, b, 2, 1]-sy
            rx3, ry3 = corners[k, b, 3, 0]-sx, corners[k, b, 3, 1]-sy
            cx = (((rx0+rx1)+rx2)+rx3)/4
            cy = (((ry0+ry1)+ry2)+ry3)/4
            bearing = np.arctan2(cy, cx)
            smin, smax = np.inf, -np.inf
            for rx, ry in ((rx0, ry0), (rx1, ry1), (rx2, ry2), (rx3, ry3)):
                s = _wrap(np.arctan2(ry, rx)-bearing)
                smin = min(smin, s)
                smax = max(smax, s)
            m = yaw0+_wrap(bearing-yaw0)
            if k == 0:
                mid0 = m
                mk = mid0+0.0
            else:
                acc = acc+_wrap(m-prev)
                mk = mid0+acc
            prev = m
            lo = min(lo, mk+smin)
            hi = max(hi, mk+smax)
            ins = True
            for c in range(4):
                c1 = (c+1) % 4
                ex = corners[k, b, c1, 0]-corners[k, b, c, 0]
                ey = corners[k, b, c1, 1]-corners[k, b, c, 1]
                rx = corners[k, b, c, 0]-sx
                ry = corners[k, b, c, 1]-sy
                if not (ey*rx-ex*ry >= 0):
                    ins = False
            inside = inside or ins
            reach = min(reach, np.hypot(cx, cy)-half_diag)
        if not (reach <= range_max):
            continue
        lo = lo-yaw0-beam_spacing
        hi = hi-yaw0+beam_spacing
        whole = inside or (hi-lo >= two_pi)

        for turn in turns:
            if whole:
                if turn != 0.0:
                    continue
                i0, i1 = 0, n
            else:
                i0 = _left(beam_rel, lo+turn)
                i1 = _right(beam_rel, hi+turn)
            for i in range(i0, i1):
                # The box at the beam's time and geopmetry.raycast's ray-segment test on its four edges.
                k, wi = seg[i], w[i]
                best = np.inf
                for c in range(4):
                    c1 = (c+1) % 4
                    ax = corners[k, b, c, 0]
                    ay = corners[k, b, c, 1]
                    bx = corners[k, b, c1, 0]
                    by = corners[k, b, c1, 1]
                    if K > 1:
                        ax = ax+wi*(corners[k+1, b, c, 0]-ax)
                        ay = ay+wi*(corners[k+1, b, c, 1]-ay)
                        bx = bx+wi*(corners[k+1, b, c1, 0]-bx)
                        by = by+wi*(corners[k+1, b, c1, 1]-by)
                    ex, ey = bx-ax, by-ay
                    vx, vy = ax-ox[i], ay-oy[i]
                    denom = dx[i]*ey-dy[i]*ex
                    if abs(denom) > 1e-12:
                        tr = (vx*ey-vy*ex)/denom
                        u = (vx*dy[i]-vy*dx[i])/denom
                        if tr > 0 and u >= 0 and u <= 1 and tr < best:
                            best = tr
                if best < ranges[i] and best <= range_max:
                    ranges[i] = best
                    hit_box[i] = b
    return ranges, hit_box
