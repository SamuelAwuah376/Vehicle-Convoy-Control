"""Compiled kernel of radar processing (processing/radar.py): one sim step's scans, every point of every scan, in one
call (clusters), instead of numpy's many small array operations per step.

clusters is radar.py's numpy code (_process_numpy) written out per point and per pair, in the same order of operations:
the sensor poses (sensor_pose) and the noise floor, the validity test, the Doppler test (stationary), the
sensor-to-world transform, the teammate filter (box_distance against MATE_MARGIN / MATE_SIGMAS), cluster_distance,
dbscan and the cluster statistics (bincount means, the add.at spread, R). Pairs of different scans are never adjacent in
numpy (their distance is set to inf), so DBSCAN runs per scan here; its label propagation is numpy's synchronous minimum
over linked core points (d2 is not exactly symmetric: wrap(a - b) and wrap(b - a) can differ in the last bit), the
border rule numpy's first nearest core, the labels numpy's rank of the propagated values. Sums run in numpy's order
(bincount and add.at add point by point), and cos, sin, atan2, np.hypot and % are the same C-library calls as numpy's,
so the two paths agree to the bit on the recorded v9 scans and the synthetic ones, compiled and as plain Python
(tests/test_perception_jit_radar_proc.py).

What numpy computes with BLAS (the sensor velocity's 2 x 2 rotation, car_boxes' matmul) stays in radar.py: OpenBLAS
fuses multiply-adds, which plain arithmetic here would not reproduce to the bit.

radar.py's numpy code stays the reference and the path with NOHESI_JIT=0 (_jit.py); there this kernel would run as
plain Python, correct but slow, and radar.py does not call it. The scan table is the TAB layout below; the module
constants are radar.py's, duplicated (numba's on-disk cache keys on this file only) and checked equal by the test;
SIGMA_FLOOR comes in as an argument.
"""
from math import atan2, cos, inf, isfinite, sin, sqrt
import numpy as np
from .._jit import njit

__all__ = ['TAB_FIELDS', 'clusters']

STATIONARY_SIGMAS = 3.0              # radar.STATIONARY_SIGMAS
MATE_MARGIN = 1.0                    # radar.MATE_MARGIN
MATE_SIGMAS = 3.0                    # radar.MATE_SIGMAS
TAB_FIELDS = ('x', 'y', 'yaw', 'vx', 'vy', 'r', 'mount_x', 'mount_y', 'mount_yaw', 'mount_z', 'sigma_range',
              'sigma_azimuth', 'sigma_range_rate')   # a scan's EgoState, mount_offset, mount yaw and z, spec sigmas
EX, EY, EYAW, EVX, EVY, ER, OX, OY, MYAW, MZ, SR, SA, SRR = range(13)


@njit(error_model='numpy')
def _wrap(a):
    return (a+np.pi) % (2*np.pi)-np.pi


@njit(error_model='numpy')
def _maximum(a, b):
    """np.maximum of two floats: NaN if either is."""
    if a != a or b != b:
        return np.nan
    return a if a > b else b


@njit(error_model='numpy')
def _box_distance(px, py, boxes, b):
    """radar.box_distance of the point (px, py) to box b of boxes (B, 4, 2): the least distance to the four edges
    (NaN if any is NaN, as numpy's min), 0 inside."""
    d = inf
    nan_seen = False
    inside = True
    for e in range(4):
        f = (e+1) % 4
        ax, ay = boxes[b, e, 0], boxes[b, e, 1]
        abx, aby = boxes[b, f, 0]-ax, boxes[b, f, 1]-ay
        apx, apy = px-ax, py-ay
        t = (apx*abx+apy*aby)/(abx*abx+aby*aby)
        if t < 0.0:
            t = 0.0
        elif t > 1.0:
            t = 1.0
        qx, qy = apx-t*abx, apy-t*aby
        de = sqrt(qx*qx+qy*qy)
        if de != de:
            nan_seen = True
        elif de < d:
            d = de
        if not abx*apy-aby*apx >= 0:
            inside = False
    if inside:
        return 0.0
    return np.nan if nan_seen else d


@njit(error_model='numpy')
def clusters(r, az, rr, off, tab, vel, group, boxes, box_off, scale, eps, min_samples, rel_speed, floor):
    """radar._process_numpy's arithmetic for scans q = 0..S-1 whose points are r, az, rr[off[q]:off[q+1]] (float64):
    tab (S, TAB_FIELDS) the scans' cars and sensors, vel (S, 2) the sensors' world velocities (sensor_velocity),
    teammate boxes boxes[box_off[g]:box_off[g+1]] (B, 4, 2) of group[q] (-1: none), DBSCAN scale (3,), eps,
    min_samples and rel_speed, floor radar.SIGMA_FLOOR. Returns (z (K, 3), R (K, 3, 3), owner (K,), count (K,),
    pose (S, 3)): the clusters in numpy's label order (by scan), each one's [range, azimuth, range_rate], R, scan and
    point count, and each scan's sensor pose (x, y, yaw) as sensor_pose."""
    n = r.shape[0]
    S = off.shape[0]-1
    pose = np.empty((S, 3))
    noise = np.empty((S, 3))
    for q in range(S):                                 # sensor_pose, the floored noise
        c, s = cos(tab[q, EYAW]), sin(tab[q, EYAW])
        ox, oy = tab[q, OX], tab[q, OY]
        pose[q, 0], pose[q, 1] = tab[q, EX]+c*ox-s*oy, tab[q, EY]+s*ox+c*oy
        pose[q, 2] = _wrap(tab[q, EYAW]+tab[q, MYAW])
        for a in range(3):
            noise[q, a] = _maximum(tab[q, SR+a], floor[a])
    kr, kaz, krr = np.empty(n), np.empty(n), np.empty(n)
    lx, ly, wx, wy = np.empty(n), np.empty(n), np.empty(n), np.empty(n)   # sensor-frame and world points
    start = np.zeros(S+1, np.int64)
    m = 0
    for q in range(S):                                 # validity, the Doppler test, the world point, teammates
        start[q] = m
        px, py, pyaw, vx, vy = pose[q, 0], pose[q, 1], pose[q, 2], vel[q, 0], vel[q, 1]
        sr, sa, rsa, rsrr = noise[q, 0], noise[q, 1], tab[q, SA], tab[q, SRR]
        cq, sq = cos(pyaw), sin(pyaw)
        g = group[q]
        for i in range(off[q], off[q+1]):
            ri, ai, rri = r[i], az[i], rr[i]
            if not (isfinite(ri) and isfinite(ai) and isfinite(rri) and ri > 0):
                continue
            a = pyaw+ai
            ca, sa_ = cos(a), sin(a)
            slope = sa_*vx-ca*vy
            sw = slope*rsa
            sigma = sqrt(rsrr*rsrr+sw*sw)
            if abs(rri+(ca*vx+sa_*vy)) < STATIONARY_SIGMAS*sigma:
                continue
            x0, y0 = ri*cos(ai), ri*sin(ai)
            x, y = cq*x0-sq*y0+px, sq*x0+cq*y0+py
            if g >= 0:
                ra = ri*sa
                margin = MATE_SIGMAS*sqrt(sr*sr+ra*ra)
                margin = _maximum(MATE_MARGIN, margin)
                hit = False
                for b in range(box_off[g], box_off[g+1]):
                    if _box_distance(x, y, boxes, b) <= margin:
                        hit = True
                        break
                if hit:
                    continue
            kr[m], kaz[m], krr[m], lx[m], ly[m], wx[m], wy[m] = ri, ai, rri, x0, y0, x, y
            m += 1
    start[S] = m
    s2 = scale[2]
    e2 = eps*eps
    for i in range(m):                                 # cluster_distance's scaled positions
        wx[i] /= scale[0]
        wy[i] /= scale[1]
    big = 0
    for q in range(S):
        big = max(big, start[q+1]-start[q])
    d2_all = np.empty(big*big)
    core = np.empty(big, np.bool_)
    cur_all, new_all, fin = np.empty(big, np.int64), np.empty(big, np.int64), np.empty(big, np.int64)
    lab = np.empty(m, np.int64)
    for q in range(S):                                 # cluster_distance and dbscan, scan by scan
        lo, hi = start[q], start[q+1]
        nb = hi-lo
        if nb == 0:
            continue
        d2 = d2_all[:nb*nb].reshape((nb, nb))
        for i in range(nb):
            pxi, pyi = wx[lo+i], wy[lo+i]
            for j in range(nb):
                dx, dy = pxi-wx[lo+j], pyi-wy[lo+j]
                v = abs(krr[lo+i]-krr[lo+j])
                if rel_speed > 0:
                    v = v-rel_speed*abs(_wrap(kaz[lo+i]-kaz[lo+j]))
                    v = _maximum(v, 0.0)
                v = v/s2
                d2[i, j] = (dx*dx+dy*dy)+v*v
        for i in range(nb):
            c = 0
            for j in range(nb):
                if d2[i, j] <= e2:
                    c += 1
            core[i] = c >= min_samples
        cur, new = cur_all, new_all
        for i in range(nb):
            cur[i] = i if core[i] else nb
        while True:                                    # each core point: the least label over its linked cores
            same = True
            for i in range(nb):
                v_ = nb
                if core[i]:
                    for j in range(nb):
                        if core[j] and d2[i, j] <= e2 and cur[j] < v_:
                            v_ = cur[j]
                new[i] = v_
                if v_ != cur[i]:
                    same = False
            if same:
                break
            cur, new = new, cur
        for i in range(nb):                            # a border point: its first nearest core's label
            fin[i] = cur[i]
            if core[i]:
                continue
            best, jb = inf, -1
            for j in range(nb):
                if core[j] and d2[i, j] <= e2:
                    if jb < 0 or d2[i, j] < best:
                        best, jb = d2[i, j], j
            if jb >= 0:
                fin[i] = cur[jb]
        for i in range(nb):
            lab[lo+i] = lo+fin[i] if fin[i] < nb else -1
    rank = np.full(m+1, -1, np.int64)                  # np.unique's inverse: labels by rank of the propagated values
    for i in range(m):
        if lab[i] >= 0:
            rank[lab[i]] = 0
    K = 0
    for v in range(m):
        if rank[v] == 0:
            rank[v] = K
            K += 1
    owner = np.zeros(K, np.int64)
    count = np.zeros(K, np.int64)
    sx, sy, srr = np.zeros(K), np.zeros(K), np.zeros(K)
    q = 0
    for i in range(m):
        if lab[i] < 0:
            continue
        while start[q+1] <= i:
            q += 1
        k = rank[lab[i]]
        lab[i] = k
        owner[k] = q
        count[k] += 1
        sx[k] += lx[i]
        sy[k] += ly[i]
        srr[k] += krr[i]
    z = np.empty((K, 3))
    for k in range(K):                                 # np.hypot, not math.hypot: Python's is its own algorithm, not
        cx, cy = sx[k]/count[k], sy[k]/count[k]        # the C library's (the plain-Python run would differ in bits)
        z[k, 0], z[k, 1], z[k, 2] = np.hypot(cx, cy), atan2(cy, cx), srr[k]/count[k]
    spread = np.zeros((K, 3, 3))
    dev = np.empty(3)
    for i in range(m):
        k = lab[i]
        if k < 0:
            continue
        dev[0], dev[1], dev[2] = kr[i]-z[k, 0], _wrap(kaz[i]-z[k, 1]), krr[i]-z[k, 2]
        for a in range(3):
            for b in range(3):
                spread[k, a, b] += dev[a]*dev[b]
    R = np.empty((K, 3, 3))
    for k in range(K):
        q = owner[k]
        for a in range(3):
            for b in range(3):
                e = 1.0 if a == b else 0.0
                R[k, a, b] = (noise[q, a]*e*noise[q, b])/count[k]+spread[k, a, b]/count[k]
    return z, R, owner, count, pose
