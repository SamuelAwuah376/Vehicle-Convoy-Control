"""Compiled kernels of the Tier A camera simulator (sim/camera.py), for one simulate_cameras call: the footprints of
the cars in view (scene), each camera's projection, truncation and size cut-off (candidates, one call per camera),
the pd draw and the occlusion pass of all the cameras (select), a frame's false boxes (fakes, when it has any), and
the frames' noisy boxes, scores and shuffle (finish), in place of numpy's many small array operations per frame.

Each kernel is camera.py's numpy code written out per car, corner and sample ray, in the same order of operations, so
the compiled path gives the numpy path's frames to rounding. On the development machine that is bit for bit: the math
library's cos, sin and atan2 are the ones numpy calls, and scene fuses the multiply-add of geometry.footprints the way
numpy's matmul (OpenBLAS) does. Nothing here draws random numbers: camera.py keeps every rng call, in the same order
and with the same sizes, and passes the draws in. camera.py's numpy functions stay the reference and the path with
NOHESI_JIT=0 (_jit.py); there these kernels would run as plain Python, correct but slow, and camera.py does not call
them.

The tuning constants come in as arguments (the TUNE layout below, camera.py's values read at call time), not as
module globals: numba's on-disk cache keys on this file only, so a global read from camera.py would keep its old
value after an edit there. A camera is a row of camera.py's _cams (the CAM layout).

Buffers (from scene) hold up to M rows per camera, camera k's at rows k*M ...: candidates fills them, select compacts
them in place, counts (ncam,) says how many rows each camera has.
"""
from math import atan2, cos, sin, sqrt
import numpy as np
from .. import _jit
from .._jit import njit

__all__ = ['CAM_FIELDS', 'TUNE_FIELDS', 'scene', 'candidates', 'select', 'fakes', 'finish', 'visible']

CAM_FIELDS = ('ox', 'oy', 'mount_yaw', 'mount_z', 'f', 'width_px', 'height_px', 'min_height_px', 'pd', 'min_visible',
              'sigma_px')
OX, OY, MYAW, MZ, F, W, H, MINH, PD, MINVIS, SIGMA = range(11)
TUNE_FIELDS = ('MIN_DEPTH', 'KEEP_AREA', 'SCORE_HALF_PX', '_TOL', '_GAP', '_PARALLEL')
T_DEPTH, T_KEEP, T_HALF, T_TOL, T_GAP, T_PAR = range(6)
PI = np.pi
TWO_PI = 2*np.pi

if _jit.ENABLED:
    from llvmlite import ir
    from numba.core import types
    from numba.extending import intrinsic

    @intrinsic
    def _fma(typingctx, a, b, c):
        """a*b + c rounded once (llvm.fma), as OpenBLAS computes the 2 x 2 products of geometry.footprints."""
        def codegen(context, builder, signature, args):
            d = ir.DoubleType()
            fn = builder.module.declare_intrinsic('llvm.fma', [d], ir.FunctionType(d, (d, d, d)))
            return builder.call(fn, args)
        return types.float64(types.float64, types.float64, types.float64), codegen
else:
    try:
        from math import fma as _fma            # Python >= 3.13
    except ImportError:                         # pragma: no cover - plain Python reference only
        def _fma(a, b, c):
            return a*b+c


@njit(error_model='numpy')
def _wrap(a):
    return (a+PI) % TWO_PI-PI


@njit(error_model='numpy')
def _clip(x, hi):
    """np.clip(x, 0.0, hi) of a finite x, signed zeros included."""
    y = x if x > 0.0 else 0.0
    return y if y < hi else hi


@njit(error_model='numpy')
def scene(traffic, traffic_id, team, car, rear, front, half_width, ncam):
    """The footprints fp (M, 4, 2) (geometry.car_boxes) and ids (M,) of the cars camera.simulate_cameras projects:
    the traffic (N, >= 3) [x, y, heading] with ids traffic_id (N,), then the teammates (team rows other than car)
    with id -2 - their index; rear, front from geometry.cg_offsets, corners rear-right, front-right, front-left,
    rear-left. Also the buffers of ncam cameras: indices (ncam M,), boxes (ncam M, 4), pd draws (ncam M,), edge noise
    (ncam M, 4), sensor poses (ncam, 4) and counts (ncam,)."""
    n = traffic.shape[0]
    M = n+team.shape[0]-1
    fp = np.empty((M, 4, 2))
    ids = np.empty(M, np.int64)
    lx = (rear, front, front, rear)
    ly = (-half_width, -half_width, half_width, half_width)
    m = 0
    for r in range(n+team.shape[0]):
        if r < n:
            x, y, h = traffic[r, 0], traffic[r, 1], traffic[r, 2]
            ids[m] = traffic_id[r]
        else:
            k = r-n
            if k == car:
                continue
            x, y, h = team[k, 0], team[k, 1], team[k, 2]
            ids[m] = -2-k
        c, s = cos(h), sin(h)
        for j in range(4):                  # local @ [[c, s], [-s, c]] + (x, y)
            fp[m, j, 0] = _fma(ly[j], -s, lx[j]*c)+x
            fp[m, j, 1] = _fma(ly[j], c, lx[j]*s)+y
        m += 1
    R = ncam*M
    return (fp, ids, np.empty(R, np.int64), np.empty((R, 4)), np.empty(R), np.empty((R, 4)), np.empty((ncam, 4)),
            np.zeros(ncam, np.int64))


@njit(error_model='numpy')
def candidates(fp, height, team, car, cams, k, pitch, tune, poses, idx, box, counts):
    """camera._candidates after the pitch draw and before the pd draw, for camera k (cams[k], a CAM row) on team car
    `car` (team rows [x, y, yaw, ...]) with body pitch `pitch`: poses[k] <- its pose (geometry.sensor_pose); rows
    k M ... of idx and box <- the indices into fp of the cars wholly deeper than MIN_DEPTH whose box survives
    truncation (KEEP_AREA) and the size cut-off, and their clipped boxes (camera.project_boxes, clip_boxes,
    geometry.project); counts[k] <- their number, which it returns."""
    cam = cams[k]
    x, y, yaw = team[car, 0], team[car, 1], team[car, 2]
    c, s = cos(yaw), sin(yaw)
    px = x+c*cam[OX]-s*cam[OY]
    py = y+s*cam[OX]+c*cam[OY]
    pyaw = _wrap(yaw+cam[MYAW])
    pz = cam[MZ]
    poses[k, 0], poses[k, 1], poses[k, 2], poses[k, 3] = px, py, pyaw, pz
    c0, s0 = cos(pyaw), sin(pyaw)
    sa, ca = sin(cam[MYAW]), cos(cam[MYAW])            # geometry._tilt
    sp, kp = sin(pitch), 1-cos(pitch)
    r00, r01, r02 = 1-kp*ca*ca, kp*sa*ca, sp*ca
    r10, r11, r12 = kp*sa*ca, 1-kp*sa*sa, -sp*sa
    r20, r21, r22 = -sp*ca, sp*sa, 1-kp
    f, Wpx, Hpx = cam[F], cam[W], cam[H]
    cu, cv = Wpx/2, Hpx/2
    zs = (0.0-pz, height-pz)
    near, keep_area, min_h = tune[T_DEPTH], tune[T_KEEP], cam[MINH]
    M = fp.shape[0]
    base = k*M
    n = 0
    for m in range(M):
        u0, u1, v0, v1 = np.inf, -np.inf, np.inf, -np.inf
        deep = True
        for j in range(4):
            X, Y = fp[m, j, 0]-px, fp[m, j, 1]-py
            xs, ys = c0*X+s0*Y, -s0*X+c0*Y               # geometry.to_sensor
            for z in zs:
                xc = r00*xs+r10*ys+r20*z
                if not xc > near:
                    deep = False
                    break
                yc = r01*xs+r11*ys+r21*z
                zc = r02*xs+r12*ys+r22*z
                u, v = cu-f*yc/xc, cv-f*zc/xc
                u0, u1, v0, v1 = min(u0, u), max(u1, u), min(v0, v), max(v1, v)
            if not deep:
                break
        if not deep:
            continue
        b0, b1 = _clip(u0, Wpx), _clip(u1, Wpx)                    # clip_boxes
        b2, b3 = _clip(v0, Hpx), _clip(v1, Hpx)
        full = (u1-u0)*(v1-v0)
        share = (b1-b0)*(b3-b2)/full if full > 0 else 0.0
        if share >= keep_area and b3-b2 >= min_h:
            idx[base+n] = m
            box[base+n, 0], box[base+n, 1], box[base+n, 2], box[base+n, 3] = b0, b1, b2, b3
            n += 1
    counts[k] = n
    return n


@njit(error_model='numpy')
def visible(origin, B, targets, steps, tune):
    """camera._visibility (with _ray_fractions) of one camera: the visible fraction (K,) of each box targets[k] of B
    (M, 4, 2) seen from origin (2,), by the same bearing-interval and distance shortcuts and sample rays. Per box:
    q its corners from the origin, the bearing interval [lo, hi] from the reference bearing of the targets' centres
    (wide: none usable), and its nearest and farthest distance bounds; a box nearer than no target's farthest corner
    hides nothing and gets no interval."""
    K = targets.shape[0]
    frac = np.zeros(K)
    if K == 0:
        return frac
    M = B.shape[0]
    tol, gap, par = tune[T_TOL], tune[T_GAP], tune[T_PAR]
    ox, oy = origin[0], origin[1]
    q = np.empty((M, 4, 2))
    nearest, farthest = np.empty(M), np.empty(M)
    for m in range(M):
        far = -np.inf
        for j in range(4):
            qx, qy = B[m, j, 0]-ox, B[m, j, 1]-oy
            q[m, j, 0], q[m, j, 1] = qx, qy
            far = max(far, sqrt(qx*qx+qy*qy))
        cx, cy = (q[m, 0, 0]+q[m, 2, 0])*0.5, (q[m, 0, 1]+q[m, 2, 1])*0.5
        dx, dy = q[m, 0, 0]-q[m, 2, 0], q[m, 0, 1]-q[m, 2, 1]
        nearest[m] = sqrt(cx*cx+cy*cy)-0.5*sqrt(dx*dx+dy*dy)
        farthest[m] = far
    sx, sy, reach = 0.0, 0.0, -np.inf               # np.bincount: the targets' centres summed in order
    for k in range(K):
        t = targets[k]
        sx += (q[t, 0, 0]+q[t, 2, 0])*0.5
        sy += (q[t, 0, 1]+q[t, 2, 1])*0.5
        reach = max(reach, farthest[t])
    ref = atan2(sy, sx)
    lo, hi = np.empty(M), np.empty(M)
    wide = np.zeros(M, np.bool_)
    useful = np.zeros(M, np.bool_)
    for m in range(M):
        if not nearest[m] < reach:
            continue
        useful[m] = True
        a, b = np.inf, -np.inf
        for j in range(4):
            g = _wrap(atan2(q[m, j, 1], q[m, j, 0])-ref)
            a, b = min(a, g), max(b, g)
        lo[m], hi[m] = a, b
        wide[m] = b-a >= PI-tol
    occ = np.empty(M, np.int64)
    S = steps.shape[0]
    for k in range(K):
        t = targets[k]
        tl, th, tw, tf, tn = lo[t], hi[t], wide[t], farthest[t], nearest[t]
        n_occ, covered = 0, False
        for m in range(M):
            if useful[m] and m != t and nearest[m] < tf and (wide[m] or tw or (lo[m]-tol <= th and hi[m]+tol >= tl)):
                occ[n_occ] = m
                n_occ += 1
                if not (wide[m] or tw) and lo[m]+tol < tl and th < hi[m]-tol and farthest[m] < tn-gap:
                    covered = True
        if covered:
            frac[k] = 0.0
            continue
        if n_occ == 0:
            frac[k] = 1.0
            continue
        mx = (((B[t, 0, 0]+B[t, 1, 0])+B[t, 2, 0])+B[t, 3, 0])/4-ox      # _ray_fractions
        my = (((B[t, 0, 1]+B[t, 1, 1])+B[t, 2, 1])+B[t, 3, 1])/4-oy
        centre = sqrt(mx*mx+my*my)
        rays, seen = 0, 0
        for j in range(4):
            jn = (j+1) % 4
            ex, ey = B[t, jn, 0]-B[t, j, 0], B[t, jn, 1]-B[t, j, 1]
            for i in range(S):
                px = (B[t, j, 0]+steps[i]*ex)-ox
                py = (B[t, j, 1]+steps[i]*ey)-oy
                d = sqrt(px*px+py*py)
                if not d <= centre:
                    continue
                rays += 1
                ang = atan2(py, px)
                A = _wrap(ang-ref)
                hidden = False
                for n in range(n_occ):                 # the shortcut: strictly inside a box wholly nearer
                    m = occ[n]
                    if not wide[m] and nearest[m] < d and lo[m]+tol < A and A < hi[m]-tol and farthest[m] < d-gap:
                        hidden = True
                        break
                if not hidden:
                    rx, ry = cos(ang), sin(ang)
                    for n in range(n_occ):             # exactly (geometry.raycast)
                        m = occ[n]
                        if not (nearest[m] < d and (wide[m] or (lo[m]-tol <= A and A <= hi[m]+tol))):
                            continue
                        for e in range(4):
                            en = (e+1) % 4
                            bx, by = B[m, en, 0]-B[m, e, 0], B[m, en, 1]-B[m, e, 1]
                            qx, qy = q[m, e, 0], q[m, e, 1]
                            denom = rx*by-ry*bx
                            tt = (qx*by-qy*bx)/denom
                            ss = (qx*ry-qy*rx)/denom
                            if abs(denom) > par and tt > 0 and ss >= 0 and ss <= 1 and tt < d-gap:
                                hidden = True
                                break
                        if hidden:
                            break
                if not hidden:
                    seen += 1
        frac[k] = seen/max(rays, 1)
    return frac


@njit(error_model='numpy')
def select(fp, cams, poses, counts, idx, box, draws, steps, tune):
    """The pd draw and the occlusion cut of camera.simulate_cameras for every camera k: of its counts[k] candidates
    (rows k M ... of idx and box), those with draws < pd, then, when min_visible > 0, those of them seen at least
    min_visible past the other cars (visible, from the camera's pose); compacted in place, counts updated."""
    M = fp.shape[0]
    origin = np.empty(2)
    for k in range(cams.shape[0]):
        base, n = k*M, 0
        for i in range(counts[k]):
            if draws[base+i] < cams[k, PD]:
                idx[base+n] = idx[base+i]
                for a in range(4):
                    box[base+n, a] = box[base+i, a]
                n += 1
        if cams[k, MINVIS] > 0 and n > 0:
            origin[0], origin[1] = poses[k, 0], poses[k, 1]
            frac = visible(origin, fp, idx[base:base+n], steps, tune)
            kept = 0
            for i in range(n):
                if frac[i] >= cams[k, MINVIS]:
                    idx[base+kept] = idx[base+i]
                    for a in range(4):
                        box[base+kept, a] = box[base+i, a]
                    kept += 1
            n = kept
        counts[k] = n


@njit(error_model='numpy')
def fakes(cams, k, pitch, car_height, ranges, widths, columns):
    """camera._fake_boxes after its draws, for camera k with body pitch `pitch`: ranges, widths and columns (n,)
    the uniform draws (FALSE_NEAR..1, width..length, 0..width_px); the boxes (n', 4) clipped to the image and with
    some area left."""
    cam = cams[k]
    f, Wpx, Hpx = cam[F], cam[W], cam[H]
    horizon = Hpx/2-f*np.tan(pitch*cos(cam[MYAW]))
    min_h = max(cam[MINH], 1.0)
    out = np.empty((ranges.shape[0], 4))
    n = 0
    for i in range(ranges.shape[0]):
        r = ranges[i]*f*car_height/min_h
        w = f*widths[i]/r
        u = columns[i]
        bottom = horizon+f*cam[MZ]/r
        b0, b1 = _clip(u-w/2, Wpx), _clip(u+w/2, Wpx)
        b2, b3 = _clip(bottom-f*car_height/r, Hpx), _clip(bottom, Hpx)
        if (b1-b0)*(b3-b2) > 0:
            out[n, 0], out[n, 1], out[n, 2], out[n, 3] = b0, b1, b2, b3
            n += 1
    return out[:n]


@njit(error_model='numpy')
def finish(cams, counts, idx, box, noise, ids, fakes, fake_counts, fake_score, orders, tune):
    """camera._frame's arrays of every camera k, one after the other: its detections (rows k M ... of box, ids[idx])
    plus edge noise (standard normal draws times sigma_px), re-ordered within each axis and clipped to the image,
    then its fake_counts[k] false boxes (the next rows of fakes); scores 0.6..1.0 by box height and the false boxes'
    fake_score; truth ids and -1; each frame permuted by its part of orders (counts + fake_counts entries)."""
    M = ids.shape[0]
    total = 0
    for k in range(cams.shape[0]):
        total += counts[k]+fake_counts[k]
    boxes = np.empty((total, 4))
    score = np.empty(total)
    tid = np.empty(total, np.int64)
    det = np.empty((M+fakes.shape[0], 4))
    sc = np.empty(M+fakes.shape[0])
    tr = np.empty(M+fakes.shape[0], np.int64)
    out, fake = 0, 0
    for k in range(cams.shape[0]):
        n, nf = counts[k], fake_counts[k]
        sig = cams[k, SIGMA]
        lim = (cams[k, W], cams[k, W], cams[k, H], cams[k, H])
        for i in range(n):
            r = k*M+i
            for a in range(0, 4, 2):
                p = box[r, a]+(0.0+sig*noise[r, a])
                s = box[r, a+1]+(0.0+sig*noise[r, a+1])
                p, s = (p, s) if p <= s else (s, p)
                det[i, a], det[i, a+1] = _clip(p, lim[a]), _clip(s, lim[a+1])
            h = box[r, 3]-box[r, 2]
            sc[i] = 0.6+0.4*h/(h+tune[T_HALF])
            tr[i] = ids[idx[r]]
        for i in range(nf):
            for a in range(4):
                det[n+i, a] = fakes[fake+i, a]
            sc[n+i] = fake_score[fake+i]
            tr[n+i] = -1
        for i in range(n+nf):
            o = orders[out+i]
            for a in range(4):
                boxes[out+i, a] = det[o, a]
            score[out+i], tid[out+i] = sc[o], tr[o]
        out += n+nf
        fake += nf
    return boxes, score, tid
