"""The compiled lidar processing (processing/_lidar_proc_jit.py, lidar._process_lidar_jit) against process_lidar's
numpy path (lidar._process_lidar_py): the same detections, in the same order, on ray-cast sweeps with noise, dropout,
teammates, off-road cars, occluders, lane-changing cars (the L-shape tests of the merge and split), unsorted beams,
unsteady motion (the pose interpolation) and either lidar. Also checks the kernels' constants are lidar.py's and that
their numpy-order reductions (_pwsum, lshape_heading_np) and lidar.py's float helpers give numpy's results exactly.

With numba on and its sin, cos, log and hypot giving numpy's results (the platforms tested), every field is
bit-identical; otherwise the floats are compared to 1e-9. With NOHESI_JIT=0 the kernels run uncompiled and the box fits
differ by rounding (_lidar_jit.fit_box_kernel against lidar._fit_box_py): the clusters are still identical."""
import math
import time
from dataclasses import replace
import numpy as np
import pytest
from nohesi.perception import _jit
from nohesi.perception.processing import _lidar_proc_jit as K
from nohesi.perception.processing import lidar as L
from nohesi.perception.types import EgoState, LidarScan
from tests.test_perception_lidar_proc import EGO, FRONT, LANE, REAR, ROAD, SCENE, SPEEDS, V_EGO, VEH, cars, sweep, \
    twist_motion

FLOAT_META = ('dz_dv', 'R_fit')


def _libm_matches_numpy():
    """Whether the compiled kernels' sin, cos, log and hypot give numpy's results on this platform."""
    if not _jit.ENABLED:
        return False

    @_jit.njit
    def probe(x):
        out = np.empty((4, x.shape[0]))
        for i in range(x.shape[0]):
            out[0, i], out[1, i] = np.sin(x[i]), np.cos(x[i])
            out[2, i], out[3, i] = np.log(abs(x[i])+1e-3), np.hypot(x[i], 0.7)
        return out
    x = np.random.default_rng(0).uniform(-4.0, 4.0, 20000)
    return np.array_equal(probe(x), np.stack((np.sin(x), np.cos(x), np.log(np.abs(x)+1e-3), np.hypot(x, 0.7))))


EXACT = _libm_matches_numpy()


def test_kernel_constants_are_lidar_constants():
    for name in ('BREAK_ANGLE', 'HEADINGS', 'LSHAPE_POINTS', 'MATE_MARGIN', 'EDGE_MARGIN', 'FIT_SLACK',
                 'OCCLUDER_MARGIN', 'GAP_ARC', 'GAP_P', 'END_LOOK', 'NONE', 'KEEP', 'MATE', 'OFF'):
        assert np.array_equal(getattr(K, name), getattr(L, name)), name
    assert np.array_equal(K.COS_H, np.cos(L.HEADINGS)) and np.array_equal(K.SIN_H, np.sin(L.HEADINGS))
    assert K.SIN_MAX == np.sin(L.HEADINGS[-1]) and K.LOG_GAP_P == np.log(L.GAP_P)
    assert L.REFINE_MIN == L.LSHAPE_MIN


def test_pairwise_sum_is_numpy_sum():
    """_pwsum is np.sum bit for bit (numpy's pairwise summation), and over n np.mean, for lengths across the 8 and
    128 blocks and the halving above them."""
    rng = np.random.default_rng(5)
    for n in list(range(1, 40))+[127, 128, 129, 255, 256, 257, 300, 777, 1024, 1201, 2500]:
        for scale in (1e-3, 1.0, 1e4):
            a = rng.normal(0.0, scale, n)+rng.uniform(-scale, scale)
            assert K._pwsum(a, 0, n) == a.sum(), n
            assert K._pwsum(a, 0, n)/n == a.mean(), n
            lo = n//3
            assert K._pwsum(a, lo, n-lo) == a[lo:].sum(), n


def car_points(rng, n_cases):
    """Noisy L and I views of car outlines (world points in scan order) and the sensor, some with strays."""
    out = []
    while len(out) < n_cases:
        x, y, th = rng.uniform(5.0, 90.0), rng.choice([-LANE, 0.0, LANE])+rng.normal(0.0, 0.4), rng.normal(0.0, 0.08)
        c, s = np.cos(th), np.sin(th)
        hx, hy = VEH.length/2, VEH.width/2
        m = int(rng.integers(3, 120))
        u = rng.uniform(-1.0, 1.0, m)
        side = rng.random(m) < 0.5
        bx = np.where(side, -hx, u*hx)
        by = np.where(side, u*hy, np.sign(y+1e-9)*-hy if rng.random() < 0.7 else hy)
        P = np.column_stack((x+c*bx-s*by, y+s*bx+c*by))+rng.normal(0.0, 0.03, (m, 2))
        if rng.random() < 0.3:
            P[rng.integers(m)] += rng.normal(0.0, 0.6, 2)
        o = np.array([0.0, rng.normal(0.0, 1.0)])
        P = P[np.argsort(np.arctan2(P[:, 1]-o[1], P[:, 0]-o[0]))]
        out.append((np.ascontiguousarray(P), o))
    return out


def test_lshape_heading_np_is_lidar_lshape_heading():
    """The kernels' L-shape heading (the merge and split tests) is lidar.lshape_heading bit for bit, thinning
    included."""
    rng = np.random.default_rng(9)
    for P, o in car_points(rng, 200 if _jit.ENABLED else 40):
        assert K.lshape_heading_np(P, o) == L.lshape_heading(P, o), len(P)


def test_float_helpers_are_numpy():
    """lidar.py's float helpers of the compiled path: _wrap1 is geometry.wrap, _unwrap3 is np.unwrap, and math's cos
    and sin are numpy's (the inlined sensor poses)."""
    rng = np.random.default_rng(2)
    x = np.concatenate((rng.uniform(-9.0, 9.0, 20000), np.arange(-12, 13)*np.pi/2, [np.pi, -np.pi, 0.0]))
    assert [L._wrap1(v) for v in x.tolist()] == L.wrap(x).tolist()
    assert [math.cos(v) for v in x.tolist()] == np.cos(x).tolist()
    assert [math.sin(v) for v in x.tolist()] == np.sin(x).tolist()
    for k in range(3000):
        p = rng.uniform(-np.pi, np.pi, 3) if k % 3 else rng.choice([-np.pi, np.pi, 0.0, 3.1], 3)+rng.normal(0, 1e-3, 3)
        assert L._unwrap3(*p.tolist()) == np.unwrap(p).tolist(), p


# ---------------------------------------------------------------------------------------------------- end to end
def unsteady_ego(x0=500.0, v0=33.0, a=-6.0, j=3.0):
    """Hard braking with the yaw rate ramping (|ax| > 3 m/s^2: the de-skew interpolates the poses)."""
    def at(t):
        r = j*t
        yaw = 0.5*j*t*t
        v = v0+a*t
        return EgoState(float(t), float(x0+v0*t+0.5*a*t*t), float(0.3+v0*t*yaw/3), float(yaw), float(v), 0.0, r, a)
    return at


def scenes():
    """(scan, spec, ego_at, mates, kwargs) cases for process_lidar."""
    rng = np.random.default_rng(17)
    out = []
    cars_at = cars(SCENE, SPEEDS)
    for spec in (FRONT, REAR):                                 # the rear lidar 20-80 m ahead of the cars
        ego = EGO if spec is FRONT else twist_motion(600.0, 0.0, 0.0, V_EGO)
        out.append((sweep(spec, ego, cars_at)[0], spec, ego, None, {}))
        out.append((sweep(spec, ego, cars_at, rng=rng)[0], spec, ego, None, {}))
    scan = sweep(FRONT, EGO, cars_at, rng=rng)[0]
    out.append((scan, FRONT, EGO, None, {'refine': False}))
    out.append((scan, FRONT, EGO, None, {'target_v': (31.0, 0.2), 'target_sigma': (2.0, 0.5), 'sigma_model': 0.01}))
    perm = rng.permutation(len(scan.range))                    # beams out of azimuth order
    out.append((LidarScan(scan.sensor, scan.car, scan.t_start, scan.t_end, scan.azimuth[perm], scan.range[perm],
                          scan.time[perm]), FRONT, EGO, None, {}))
    # lane changers (headings 4-8 deg: the split's L-shape test) and a car bumper to bumper
    lc = np.array([[515.0, 0.0, 0.08], [521.5, 0.3, 0.07], [540.0, LANE-0.8, -0.1], [560.0, -LANE+1.0, 0.12],
                   [566.0, -LANE+1.4, 0.1]])
    out.append((sweep(FRONT, EGO, cars(lc), rng=rng)[0], FRONT, EGO, None, {}))
    # a teammate ahead, a car parked off the road, an occluding post
    mate = cars([[518.0, 0.1, 0.01]], [V_EGO+1.0])
    traffic = cars([[535.0, LANE, 0.0], [529.0, -LANE, 0.02]])
    parked = np.array([[540.0, ROAD.width/2+2.2, 0.0]])
    square = np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])
    post = lambda t: (np.array([512.0+V_EGO*t, 1.2])+square)[None]
    scan = sweep(FRONT, EGO, lambda t: np.vstack((traffic(t), mate(t), parked)), rng=rng, boxes_at=post)[0]
    out.append((scan, FRONT, EGO, mate, {}))
    out.append((scan, FRONT, EGO, lambda t: np.zeros((0, 3)), {}))
    # unsteady carrying car, coarse beams (free-space gaps)
    ego = unsteady_ego()
    out.append((sweep(FRONT, ego, cars_at, rng=rng)[0], FRONT, ego, None, {}))
    coarse = replace(FRONT, beam_spacing=np.radians(0.3))
    out.append((sweep(coarse, EGO, cars_at, rng=rng)[0], coarse, EGO, None, {}))
    # random dense traffic with teammates and a turning carrying car (test_random_traffic_robust's scenes)
    for trial in range(24 if _jit.ENABLED else 4):
        spec = FRONT if trial % 2 == 0 else REAR
        ego = twist_motion(500.0, rng.uniform(-0.5, 0.5), rng.uniform(-0.03, 0.03), V_EGO+rng.uniform(-3, 3), 0.0,
                           rng.uniform(-0.1, 0.1))
        n = rng.integers(3, 25)
        st = np.column_stack((500+(1 if spec is FRONT else -1)*rng.uniform(6, 120, n),
                              rng.choice([-LANE, 0.0, LANE], n)+rng.uniform(-0.6, 0.6, n), rng.uniform(-0.12, 0.12, n)))
        keep = []
        for k in range(n):
            if all(abs(st[k, 0]-st[m, 0]) > 5.2 or abs(st[k, 1]-st[m, 1]) > 2.0 for m in keep) and \
                    (abs(st[k, 0]-500) > 5.5 or abs(st[k, 1]-ego(0.0).y) > 2.1):
                keep.append(k)
        cars_at_k = cars(st[keep], V_EGO+rng.uniform(-6, 6, len(keep)))
        ids = rng.choice(len(keep), min(2, len(keep)), replace=False) if trial % 3 == 0 else []
        mates = (lambda t, ids=ids, c=cars_at_k: c(t)[ids]) if len(ids) else None
        out.append((sweep(spec, ego, cars_at_k, rng=rng)[0], spec, ego, mates, {}))
    return out


def same(a, b, exact, what):
    if exact:
        np.testing.assert_array_equal(a, b, err_msg=what)
    else:
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-9*max(np.abs(b).max(), 1.0), err_msg=what)


def test_process_lidar_matches_numpy_path(monkeypatch):
    """Every case gives the numpy path's detections: the same clusters (points, times, free ends), stamps, sensor
    poses and ego states exactly, the boxes and covariances exactly (or to 1e-9, see the module docstring). The
    merge's and split's L-shape tests run in some of the cases."""
    n_lshape = [0]                   # the numpy path's L-shape tests (with numba on, its box fits are compiled)
    lshape = L.lshape_heading

    def counted(*a, **kw):
        n_lshape[0] += 1
        return lshape(*a, **kw)
    monkeypatch.setattr(L, 'lshape_heading', counted)
    fits_exact = EXACT
    n_dets = n_two = 0
    for k, (scan, spec, ego_at, mates, kw) in enumerate(scenes()):
        refine = kw.pop('refine', True)
        kw = {'target_v': None, 'target_sigma': (5.0, 1.0), 'sigma_model': 0.003, **kw}
        args = (scan, spec, ego_at, mates, VEH, ROAD, refine, kw['target_v'], kw['target_sigma'], kw['sigma_model'])
        want = L._process_lidar_py(*args)
        got = L._process_lidar_jit(*args)
        assert len(got) == len(want), k
        for g, w in zip(got, want):
            assert (g.kind, g.sensor, g.car, g.time, g.ego) == (w.kind, w.sensor, w.car, w.time, w.ego), k
            assert g.sensor_pose == w.sensor_pose and [type(v) for v in g.sensor_pose] == \
                [type(v) for v in w.sensor_pose], k
            assert list(g.meta) == list(w.meta), k
            for key in ('n_points', 'sides', 'scan', 'free'):
                assert g.meta[key] == w.meta[key] and type(g.meta[key]) is type(w.meta[key]), (k, key)
            for key in ('points', 'times', 'v_prior'):
                same(g.meta[key], w.meta[key], EXACT or not _jit.ENABLED, f'case {k} {key}')
            same(g.z, w.z, fits_exact, f'case {k} z')
            same(g.R, w.R, fits_exact, f'case {k} R')
            for key in FLOAT_META:
                same(g.meta[key], w.meta[key], fits_exact, f'case {k} {key}')
            n_two += g.meta['sides'] == 2
        n_dets += len(got)
    assert n_dets > (100 if _jit.ENABLED else 40) and n_two > 10
    assert n_lshape[0] > 0 or not _jit.ENABLED      # the cases reach the merge's or split's L-shape test


def test_no_returns_and_small_sweeps():
    """Sweeps with under 3 beams, or under 3 returns kept (none at all, or 1-2 on a far car with 1 deg beams), give no
    detections on either path, with or without a mates callback (returning no teammate)."""
    coarse = replace(FRONT, beam_spacing=np.radians(1.0))
    scan, _ = sweep(coarse, EGO, cars([[600.0, 0.0, 0.0]]))
    assert 1 <= np.isfinite(scan.range).sum() <= 2
    empty = np.full(len(scan.range), np.nan)
    for s in (LidarScan(scan.sensor, 1, 0.0, 0.1, scan.azimuth[:2], scan.range[:2], scan.time[:2]),
              LidarScan(scan.sensor, 1, 0.0, 0.1, scan.azimuth, empty, scan.time), scan):
        assert L.process_lidar(s, coarse, EGO, None, VEH, ROAD) == []
        if len(s.range) >= 3:
            for run in (L._process_lidar_py, L._process_lidar_jit):
                for mates in (None, lambda t: np.zeros((0, 3))):
                    assert run(s, coarse, EGO, mates, VEH, ROAD, True, None, (5.0, 1.0), 0.003) == []


@pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')
def test_compiled_path_is_faster():
    """The 20-car sweep of test_timing_twenty_cars: the compiled path at least twice as fast as the numpy path
    (medians of 7; about 5 times on an idle machine)."""
    rng = np.random.default_rng(11)
    xs = np.sort(rng.uniform(508.0, 650.0, 20))
    lanes = np.array([-LANE, 0.0, LANE])[np.arange(20) % 3]+rng.uniform(-0.3, 0.3, 20)
    cars_at = cars(np.column_stack((xs, lanes, rng.uniform(-0.02, 0.02, 20))), V_EGO+rng.uniform(-5, 5, 20))
    scan, _ = sweep(FRONT, EGO, cars_at, rng=rng, window=False)
    args = (scan, FRONT, EGO, lambda t: np.zeros((0, 3)), VEH, ROAD, True, None, (5.0, 1.0), 0.003)
    times = {}
    for run in (L._process_lidar_jit, L._process_lidar_py):
        run(*args)
        ts = []
        for _ in range(7):
            tic = time.perf_counter()
            run(*args)
            ts.append(time.perf_counter()-tic)
        times[run.__name__] = float(np.median(ts))
    jit, py = times['_process_lidar_jit'], times['_process_lidar_py']
    print(f'\n20 cars: compiled {1e3*jit:.2f} ms, numpy {1e3*py:.2f} ms')
    assert jit < 0.5*py
