"""Radar processing's compiled path (processing/_radar_jit.py, radar._process_compiled) against its numpy reference
(radar._process_numpy, the path with NOHESI_JIT=0): the same Detections to the bit (kind, sensor, car, time, z, R,
sensor pose with its element types, ego, meta), in the same order. On the simulator's scans of short v9 episodes
(every radar on every team car, the teammates' V2V states) and on randomized synthetic scans: clumps of points for
multi-point clusters with border and noise points (min_samples 1 to 3), duplicates, stationary clutter, points on and
inside teammates' boxes, NaN / inf / non-positive entries, empty scans, noise-free and noisy specs, turning cars with
the ego moved to the scan time, other scales, eps and rel_speed 0. And at the knife edges, where a rounding difference
would flip a decision: two points exactly eps apart (adjacent) and one ulp further (not), a border point equally near
two cores (the first one wins), and the ranges and range rates within a few ulps of the teammate margin and of the
Doppler test's threshold, where both paths must flip at the same float. Also checks that the kernel's constants are
radar.py's and that it has one specialization.

The kernel follows the numpy code operation by operation with the same C-library calls, so no tolerance is
needed, compiled or (NOHESI_JIT=0, fewer synthetic batches) as plain Python; the knife-edge cases confirm it where it
matters."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import _jit
from nohesi.perception.config import RadarSpec, default_radars
from nohesi.perception.geometry import car_boxes, sensor_pose
from nohesi.perception.processing import _radar_jit as K
from nohesi.perception.processing import radar as RP
from nohesi.perception.sim.radar import simulate_radars
from nohesi.perception.types import EgoState, RadarScan, TruthHistory
from nohesi.world import World

VEH = scenario_v9().vehicle
RADARS = default_radars()
FRONT = RADARS[0]
compiled = pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')
invalid_ok = pytest.mark.filterwarnings('ignore:invalid value:RuntimeWarning')   # numpy on the inf entries


def test_kernel_constants_are_radar_constants():
    assert (K.STATIONARY_SIGMAS, K.MATE_MARGIN, K.MATE_SIGMAS) == (RP.STATIONARY_SIGMAS, RP.MATE_MARGIN,
                                                                   RP.MATE_SIGMAS)
    assert np.array_equal(RP._FLOOR, RP.SIGMA_FLOOR)
    names = {f.name for f in RadarSpec.__dataclass_fields__.values()}
    assert all(f in names for f in K.TAB_FIELDS[10:]) and len(K.TAB_FIELDS) == K.SRR+1
    assert K.TAB_FIELDS[:6] == ('x', 'y', 'yaw', 'vx', 'vy', 'r')
    assert all(f in EgoState.__dataclass_fields__ for f in K.TAB_FIELDS[:6])


def run(path, scans, specs, egos, mates, params=VEH, scale=RP.SCALE, eps=1.0, min_samples=1,
        rel_speed=RP.MAX_REL_SPEED):
    egos = [RP.ego_at(e, sc.time) for e, sc in zip(egos, scans)]
    return path(scans, specs, egos, mates, params, scale, eps, min_samples, rel_speed)


def assert_same(got, want):
    """Compiled Detections `got` (one list per scan) equal numpy's `want` to the bit; returns their number."""
    assert len(got) == len(want)
    n = 0
    for g, w in zip(got, want):
        assert len(g) == len(w)
        for a, b in zip(g, w):
            assert (a.kind, a.sensor, a.car, a.time) == (b.kind, b.sensor, b.car, b.time)
            assert type(a.time) is type(b.time)
            for x, y in ((a.z, b.z), (a.R, b.R)):
                assert x.shape == y.shape and x.dtype == y.dtype
                assert x.tobytes() == y.tobytes(), (x, y)
            assert len(a.sensor_pose) == len(b.sensor_pose)
            for u, v in zip(a.sensor_pose, b.sensor_pose):
                assert type(u) is type(v) and np.float64(u).tobytes() == np.float64(v).tobytes()
            assert a.ego == b.ego
            assert a.meta == b.meta and [type(v) for v in a.meta.values()] == [type(v) for v in b.meta.values()]
        n += len(w)
    return n


def both(*args, **kw):
    """(compiled, numpy) of the same inputs, checked equal; the numpy result."""
    got, want = run(RP._process_compiled, *args, **kw), run(RP._process_numpy, *args, **kw)
    assert_same(got, want)
    return want


# ------------------------------------------------------------------------------------------------- v9 episodes
def v9_steps(seed, steps=60):
    """Per step of v9 seed `seed` driven straight: (scans, specs, egos, mates) as the pipeline passes them (every
    radar on every team car, car-major; one EgoState and one mates array per car, shared by its scans)."""
    world = World(scenario_v9(), seed)
    truth = TruthHistory()
    truth.push(world.snapshot, np.zeros(3))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(steps):
        snap, applied = world.step(np.zeros((3, 2)))
        truth.push(snap, applied[:, 0].copy())
        t = float(snap.time)
        scans = simulate_radars(RADARS, [0, 1, 2], truth, t, world.plant.params, rng)
        team = np.asarray(snap.team, float)
        egos = [EgoState(t, *team[c, :3].tolist(), float(team[c, 3]), float(snap.team_vy[c]), float(snap.team_r[c]))
                for c in range(3)]
        mates = [np.delete(team[:, :3], c, axis=0) for c in range(3)]
        jobs = [(c, s) for c in range(3) for s in RADARS]
        out.append((scans, [s for _, s in jobs], [egos[c] for c, _ in jobs], [mates[c] for c, _ in jobs],
                    world.plant.params))
    return out


@compiled
def test_v9_episodes():
    n_scans = n_points = n_dets = 0
    for seed in (1000, 1001):
        for scans, specs, egos, mates, params in v9_steps(seed):
            n_dets += sum(len(d) for d in both(scans, specs, egos, mates, params, scale=(3.0, 1.5, 1.0)))
            n_scans += len(scans)
            n_points += sum(len(s.range) for s in scans)
            got = RP.process_radars(scans, specs, egos, mates, params, scale=(3.0, 1.5, 1.0))
            assert_same(got, run(RP._process_numpy, scans, specs, egos, mates, params, scale=(3.0, 1.5, 1.0)))
    print(f'v9: {n_scans} scans, {n_points} points, {n_dets} detections')
    assert n_points > 5000 and n_dets > 2000
    assert len(K.clusters.signatures) == 1


# ------------------------------------------------------------------------------------------------- synthetic scans
def synthetic(rng):
    """A random batch of scans: (scans, specs, egos, mates, kwargs)."""
    n_cars = int(rng.integers(1, 4))
    egos, mates_of = [], []
    for c in range(n_cars):
        t0 = 1.0-rng.choice([0.0, 0.0, 0.01, 0.05])         # some egos are moved to the scan time
        egos.append(EgoState(t0, rng.uniform(-50, 50), rng.uniform(-4, 8), rng.normal(0.0, 0.3), rng.uniform(0, 35),
                             rng.normal(0.0, 0.3), rng.normal(0.0, 0.2)))
        kind = rng.random()
        if kind < 0.15:
            mates_of.append(None)
        elif kind < 0.25:
            mates_of.append(np.zeros((0, 3)))
        else:
            k = int(rng.integers(1, 4))
            m = np.column_stack((egos[-1].x+rng.uniform(-30, 30, k), egos[-1].y+rng.uniform(-4, 4, k),
                                 rng.normal(0.0, 0.1, k), rng.uniform(0, 30, k)))   # 4 columns: [:, :3] is used
            mates_of.append(m if rng.random() < 0.8 else m.tolist())
    specs = []
    for spec in RADARS:
        change = {}
        u = rng.random()
        if u < 0.15:
            change.update(sigma_range=0.0, sigma_azimuth=0.0, sigma_range_rate=0.0)
        elif u < 0.4:
            change.update(sigma_range=rng.uniform(0.0, 1.0), sigma_azimuth=rng.uniform(0.0, 0.05),
                          sigma_range_rate=rng.uniform(0.0, 0.5))
        specs.append(replace(spec, **change))
    scans, sp_, eg_, ma_ = [], [], [], []
    for c in range(n_cars):
        for spec in specs:
            if rng.random() < 0.15:
                n = 0
                r, az, rr = np.zeros(0), np.zeros(0), np.zeros(0)
            else:
                parts = []
                for _ in range(int(rng.integers(0, 8))):        # clumps: one car's points
                    k = int(rng.integers(1, 7))
                    r0, a0, v0 = rng.uniform(1, 150), rng.uniform(-1.2, 1.2), rng.uniform(-15, 15)
                    parts.append(np.column_stack((r0+rng.normal(0, rng.choice([0.2, 1.0, 3.0]), k),
                                                  a0+rng.normal(0, 0.01, k), v0+rng.normal(0, 0.5, k))))
                k = int(rng.integers(0, 6))                     # scattered points
                parts.append(np.column_stack((rng.uniform(0.5, 150, k), rng.uniform(-np.pi, np.pi, k),
                                              rng.uniform(-30, 30, k))))
                pts = np.concatenate(parts) if parts else np.zeros((0, 3))
                ego = RP.ego_at(egos[c], 1.0)
                pose = sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), VEH)
                v = RP.sensor_velocity(spec.mount, ego, VEH)
                k = int(rng.integers(0, 4))                     # stationary clutter
                a = rng.uniform(-1, 1, k)
                pts = np.vstack((pts, np.column_stack((rng.uniform(5, 100, k), a, RP.stationary_range_rate(a, pose, v)
                                                       + rng.normal(0, spec.sigma_range_rate, k)))))
                m = mates_of[c]
                if m is not None and len(m) and rng.random() < 0.7:   # points on and inside a teammate
                    box = car_boxes(np.asarray(m, float)[:1, :3], VEH)[0]
                    w = box[0]+rng.random((3, 1))*(box[1]-box[0])+rng.random((3, 1))*(box[3]-box[0])
                    w = np.vstack((w, box[1]+rng.normal(0, 0.5, (2, 2))))
                    d = w-np.array(pose[:2])
                    pts = np.vstack((pts, np.column_stack((np.hypot(d[:, 0], d[:, 1]),
                                                           np.arctan2(d[:, 1], d[:, 0])-pose[2],
                                                           rng.uniform(-10, 10, len(w))))))
                if len(pts) and rng.random() < 0.3:             # duplicates
                    pts = np.vstack((pts, pts[rng.integers(0, len(pts), 2)]))
                if len(pts) and rng.random() < 0.2:             # invalid entries
                    bad = pts[:3].copy()
                    bad[:, int(rng.integers(0, 3))] = rng.choice([np.nan, np.inf, -np.inf])
                    pts = np.vstack((pts, bad, [[0.0, 0.1, 5.0], [-1.0, 0.1, 5.0]]))
                pts = pts[rng.permutation(len(pts))]
                r, az, rr = pts.T.copy()
            scans.append(RadarScan(spec.name, c, 1.0, r, az, rr))
            sp_.append(spec)
            eg_.append(egos[c])
            ma_.append(mates_of[c])
    kw = {}
    if rng.random() < 0.3:
        kw['scale'] = tuple(rng.uniform(0.5, 4.0, 3))
    if rng.random() < 0.3:
        kw['eps'] = float(rng.uniform(0.3, 2.0))
    if rng.random() < 0.4:
        kw['min_samples'] = int(rng.integers(2, 4))
    if rng.random() < 0.2:
        kw['rel_speed'] = 0.0
    return scans, sp_, eg_, ma_, kw


@invalid_ok
def test_synthetic_scans():
    rng = np.random.default_rng(23)
    n_batches = 300 if _jit.ENABLED else 30                 # the kernel runs as plain Python with NOHESI_JIT=0
    n_dets = n_multi = n_noise = 0
    for _ in range(n_batches):
        scans, specs, egos, mates, kw = synthetic(rng)
        want = both(scans, specs, egos, mates, **kw)
        dets = [d for ds in want for d in ds]
        n_dets += len(dets)
        n_multi += sum(d.meta['n_points'] > 1 for d in dets)
        if kw.get('min_samples', 1) > 1:
            n_noise += 1
    print(f'synthetic: {n_batches} batches, {n_dets} detections ({n_multi} of several points)')
    assert n_dets > 10*n_batches and n_multi > 2*n_batches and n_noise > 0.2*n_batches


# ------------------------------------------------------------------------------------------------- knife edges
STILL = EgoState(1.0, 0.0, 0.0, 0.0, 30.0)                      # front radar along +x, pose yaw 0


def one_scan(r, az, rr, spec=FRONT, ego=STILL, mates=None, **kw):
    sc = RadarScan(spec.name, 0, 1.0, np.asarray(r, float), np.asarray(az, float), np.asarray(rr, float))
    return both([sc], [spec], [ego], [mates], **kw)[0]


def test_points_exactly_eps_apart():
    """Two points at the same range and azimuth whose range rates differ by exactly eps join; by eps plus one ulp they
    do not (scale 1 on the range rate: d2 is exactly 1, then just above)."""
    for rel in (0.0, RP.MAX_REL_SPEED):
        joined = one_scan([40.0, 40.0], [0.1, 0.1], [5.0, 6.0], scale=(1.5, 1.5, 1.0), rel_speed=rel)
        apart = one_scan([40.0, 40.0], [0.1, 0.1], [5.0, np.nextafter(6.0, 7.0)], scale=(1.5, 1.5, 1.0),
                         rel_speed=rel)
        assert [d.meta['n_points'] for d in joined] == [2]
        assert [d.meta['n_points'] for d in apart] == [1, 1]
        for ms, want in ((2, [2]), (3, [])):                    # each the other's neighbour: core at 2, noise at 3
            dets = one_scan([40.0, 40.0], [0.1, 0.1], [5.0, 6.0], scale=(1.5, 1.5, 1.0), rel_speed=rel,
                            min_samples=ms)
            assert [d.meta['n_points'] for d in dets] == want


@pytest.mark.parametrize('first', ['A', 'B'])
def test_border_point_equally_near_two_cores(first):
    """min_samples 4: a border point exactly eps from a core of each of two clusters joins the first one's."""
    A, B, b = [-0.9, -0.6, -0.3, 0.0], [2.0, 2.3, 2.6, 2.9], [1.0]
    rr = (A+b+B) if first == 'A' else (B+b+A)
    dets = one_scan(np.full(9, 40.0), np.full(9, 0.1), rr, scale=(1.5, 1.5, 1.0), rel_speed=0.0, min_samples=4)
    counts = [d.meta['n_points'] for d in dets]
    assert counts == [5, 4]                                     # the border point is in the first cluster


def boundary(decide, lo, hi):
    """The float x between lo and hi (either order) at which decide(x), False at lo and True at hi, first holds going
    from lo: bisection to adjacent floats."""
    assert not decide(lo) and decide(hi)
    while True:
        mid = lo+(hi-lo)/2
        if mid == lo or mid == hi:
            return hi
        lo, hi = (lo, mid) if decide(mid) else (mid, hi)


def sweep(make, x0, n=6):
    """make(x) -> Detections, for the n floats either side of x0 through both paths; the outcomes (detection count)
    seen."""
    xs = [x0]
    for direction in (-np.inf, np.inf):
        x = x0
        for _ in range(n):
            x = np.nextafter(x, direction)
            xs.append(x)
    return {len(make(float(x))) for x in xs}


def test_teammate_margin_knife_edge():
    """A point along the front radar's boresight at ranges within ulps of the teammate margin (a teammate ahead):
    both paths drop it from the same float on."""
    for spec in (FRONT, replace(FRONT, sigma_range=0.4, sigma_azimuth=0.02)):   # margin 1 m, and the noise's
        mates = np.array([[60.0, 0.0, 0.0]])
        make = lambda r: one_scan([r], [0.0], [-5.0], spec=spec, mates=mates)
        numpy_dropped = lambda r: len(run(RP._process_numpy, [RadarScan(spec.name, 0, 1.0, np.array([r]), np.zeros(1),
                                                                        np.array([-5.0]))], [spec], [STILL],
                                          [mates])[0]) == 0
        r0 = boundary(numpy_dropped, 30.0, 58.0)
        assert sweep(make, r0) == {0, 1}


def test_point_exactly_at_the_teammate_margin():
    """A point exactly MATE_MARGIN (1 m, the noise's margin being smaller) behind a teammate's rear face is the
    teammate's: dropped by both paths."""
    spec = replace(FRONT, sigma_range=0.1, sigma_azimuth=0.001)
    mates = np.array([[60.0, 0.0, 0.0]])
    box = car_boxes(mates, VEH)
    px, py, _, _ = sensor_pose(spec.mount, (STILL.x, STILL.y, STILL.yaw), VEH)
    r = float(box[0, :, 0].min()-1.0-px)
    for _ in range(50):                                         # the range whose world point is 1 m off, as numpy
        d = RP.box_distance(np.array([[1.0*r+px, py]]), box)[0, 0]   # computes it (pose yaw 0: x = r + px)
        if d == 1.0:
            break
        r = float(np.nextafter(r, np.inf if d > 1.0 else -np.inf))
    assert d == 1.0
    assert one_scan([r], [0.0], [-5.0], spec=spec, mates=mates) == []
    assert len(one_scan([r-1e-6], [0.0], [-5.0], spec=spec, mates=mates)) == 1


def test_doppler_threshold_knife_edge():
    """Range rates within ulps of the Doppler test's threshold (3 sigma from a fixed point's), for a side radar on a
    turning car: both paths reject from the same float on."""
    ego = EgoState(1.0, 0.0, 0.0, 0.2, 25.0, 0.3, 0.15)
    for spec in (FRONT, RADARS[2]):
        e = RP.ego_at(ego, 1.0)
        pose = sensor_pose(spec.mount, (e.x, e.y, e.yaw), VEH)
        still = float(RP.stationary_range_rate(np.array([0.3]), pose, RP.sensor_velocity(spec.mount, e, VEH))[0])
        make = lambda rr: one_scan([30.0], [0.3], [rr], spec=spec, ego=ego)
        rejected = lambda rr: len(run(RP._process_numpy, [RadarScan(spec.name, 0, 1.0, np.array([30.0]),
                                                                    np.array([0.3]), np.array([rr]))], [spec], [ego],
                                      [None])[0]) == 0
        r0 = boundary(rejected, still+5.0, still)               # kept 5 m/s off, rejected at the fixed point's
        assert sweep(make, r0) == {0, 1}


@invalid_ok
def test_dispatch_and_fallbacks(monkeypatch):
    """process_radars runs the compiled path when _JIT is on (and gives what the numpy path does either way); eps inf
    and mismatched arrays go to the numpy path."""
    rng = np.random.default_rng(5)
    scans, specs, egos, mates, kw = synthetic(rng)
    want = run(RP._process_numpy, scans, specs, egos, mates, **kw)
    calls = []
    real = RP._process_compiled
    monkeypatch.setattr(RP, '_process_compiled', lambda *a: (calls.append(1), real(*a))[1])
    for flag in (True, False):
        monkeypatch.setattr(RP, '_JIT', flag)
        assert_same(RP.process_radars(scans, specs, egos, mates, VEH, **kw), want)
    assert len(calls) == 1
    monkeypatch.setattr(RP, '_JIT', True)
    assert RP.process_radars([], [], [], [], VEH) == []
    sc = RadarScan(FRONT.name, 0, 1.0, np.array([40.0, 41.0]), np.array([0.1, 0.1]), np.array([5.0, 5.2]))
    assert_same(RP.process_radars([sc, sc], [FRONT, FRONT], [STILL, STILL], [None, None], VEH, eps=np.inf),
                run(RP._process_numpy, [sc, sc], [FRONT, FRONT], [STILL, STILL], [None, None], eps=np.inf))
    assert len(calls) == 1                                      # eps inf: the numpy path (scans would link)
    bad = RadarScan(FRONT.name, 0, 1.0, np.array([40.0, 41.0]), np.array([0.1]), np.array([5.0, 5.2]))
    assert run(RP._process_compiled, [bad], [FRONT], [STILL], [None]) is None
    lists = RadarScan(FRONT.name, 0, 1.0, [40.0, 41.0], [0.1, 0.1], [5.0, 5.2])   # lists, not arrays
    assert_same(run(RP._process_compiled, [lists], [FRONT], [STILL], [None]),
                run(RP._process_numpy, [lists], [FRONT], [STILL], [None]))
