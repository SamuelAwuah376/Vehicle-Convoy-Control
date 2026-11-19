"""Tier A lidar simulator (nohesi/perception/sim/lidar.py): analytic hits, occlusion, own body, scan skew, dropout,
layers, determinism, the truth interpolation and speed."""
import time
from dataclasses import replace
from math import radians
import numpy as np
from nohesi.config import scenario_v9
from nohesi.world import Snapshot
from nohesi.perception.config import default_lidars
from nohesi.perception.geometry import sensor_pose, wrap
from nohesi.perception.types import LidarScan, TruthHistory
from nohesi.perception.sim.lidar import CHUNK, _knot_states, beam_azimuths, cast_sweep, simulate_lidar

P = scenario_v9().vehicle
FRONT, REAR = default_lidars()
EXACT = replace(FRONT, range_sigma=0.0, dropout=0.0, far_dropout=0.0)
EXACT_REAR = replace(REAR, range_sigma=0.0, dropout=0.0, far_dropout=0.0)
STEPS = np.round(np.arange(0.0, 0.551, 0.05), 3)
AWAY = [[-1000.0, 20.0, 0.0, 0.0], [1000.0, 20.0, 0.0, 0.0]]   # teammates parked out of every sensor's range


def history(team, traffic, times=STEPS):
    """TruthHistory from team(t) -> (3, 4) [x, y, yaw, v] and traffic(t) -> rows [id, s, d, heading, v]."""
    h = TruthHistory()
    for t in times:
        tm = np.asarray(team(t), float)
        tr = np.asarray(traffic(t), float).reshape(-1, 5)
        v = np.where(tr[:, 4] > 0, tr[:, 4], 1.0)
        h.push(Snapshot(float(t), tm, np.zeros(3), tm[:, 0], tm[:, 1], tr[:, 1], tr[:, 2], v, tr[:, 0].astype(int),
                        (), False, traffic_vd=v*np.tan(tr[:, 3])))
    return h


def static(ego, *cars, mates=AWAY):
    """History of a frozen scene: ego [x, y, yaw], cars [s, d, heading] (ids 0, 1, ...)."""
    team = [[*ego, 0.0], *mates]
    return history(lambda t: team, lambda t: [[i, *c, 0.0] for i, c in enumerate(cars)])


def body(state):
    """Body centre and heading of a CG state [x, y, heading], from the vehicle parameters directly."""
    shift = (P.wheelbase-2*P.cg_to_rear_axle)/2
    return np.asarray(state[:2], float)+shift*np.array([np.cos(state[2]), np.sin(state[2])]), state[2]


def origin(spec, ego):
    """Sensor position and boresight of a bumper mount, from the vehicle parameters directly."""
    l_r, overhang = P.cg_to_rear_axle, (P.length-P.wheelbase)/2
    edge = P.wheelbase-l_r+overhang if spec.mount.edge == 'front' else -(l_r+overhang)
    ox, oy = edge+spec.mount.x, spec.mount.y
    c, s = np.cos(ego[2]), np.sin(ego[2])
    return np.array([ego[0]+c*ox-s*oy, ego[1]+s*ox+c*oy]), ego[2]+spec.mount.yaw


def slab(o, angle, state):
    """Entry distance of a ray into a car's body box (np.inf if missed): the slab test in the box frame."""
    centre, h = body(state)
    half = np.array([P.length/2, P.width/2])
    c, s = np.cos(h), np.sin(h)
    rel = o-centre
    p = np.array([c*rel[0]+s*rel[1], -s*rel[0]+c*rel[1]])
    d = np.array([np.cos(angle-h), np.sin(angle-h)])
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (-half-p)/d, (half-p)/d
    near, far = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
    return near if 0 < near <= far else np.inf


def expected(spec, car, truth, azimuth, times, others=None):
    """Per-beam nearest hit with every car at the beam's own fire time (no chunking), and the hit's label."""
    out, lab = np.full(len(azimuth), np.inf), np.full(len(azimuth), -1)
    for i, (a, t) in enumerate(zip(azimuth, times)):
        s = truth.at(t)
        o, yaw = origin(spec, s.team[car])
        cars = [(int(k), st) for k, st in zip(s.traffic_id, s.traffic)]
        cars += [(-2-j, st) for j, st in enumerate(s.team) if j != car]
        for k, st in cars:
            r = slab(o, yaw+a, st)
            if r < out[i] and r <= spec.range_max:
                out[i], lab[i] = r, k
    return out, lab


def test_beam_pattern_and_scan_fields():
    truth = static([0.0, 0.0, 0.0], [40.0, 0.0, 0.0])
    for spec, n in ((FRONT, 1201), (REAR, 901)):
        az = beam_azimuths(spec)
        assert len(az) == n
        assert np.isclose(az[0], -spec.fov/2) and np.isclose(az[-1], spec.fov/2)
        assert np.allclose(np.diff(az), spec.beam_spacing)
        scan = simulate_lidar(spec, 0, truth, 0.4, P, np.random.default_rng(0))
        assert isinstance(scan, LidarScan) and scan.sensor == spec.name and scan.car == 0 and scan.points is None
        assert np.isclose(scan.t_start, 0.3) and scan.t_end == 0.4
        assert np.array_equal(scan.azimuth, az)
        assert np.isclose(scan.time[0], scan.t_start) and scan.time[-1] == scan.t_end
        assert np.all(np.diff(scan.time) > 0) and np.allclose(np.diff(scan.time), 0.1/(n-1))
        assert scan.range.shape == (n,)


def test_box_straight_ahead_matches_closed_form():
    truth = static([0.0, 0.0, 0.0], [40.0, 0.0, 0.0])
    az, times, r, label = cast_sweep(EXACT, 0, truth, 0.4, P)
    sensor_x = 2.3235-0.05
    face = 40.0-0.0715-P.length/2-sensor_x            # target's rear face ahead of the sensor
    across = face*np.tan(az)
    inside = np.abs(across) < P.width/2-1e-6
    assert np.isclose(r[np.argmin(np.abs(az))], face)
    assert np.allclose(r[inside], face/np.cos(az[inside]), atol=1e-9)
    assert np.all(label[inside] == 0)
    assert np.all(np.isinf(r[np.abs(across) > P.width/2+1e-6]))
    assert inside.sum() == 2*int(np.arctan(0.95/face)/EXACT.beam_spacing)+1


def test_angled_boxes_match_slab_test():
    ego = [10.0, 1.0, 0.05]
    cars = ([35.0, 5.5, 0.25], [18.0, -4.0, -0.1], [-20.0, 2.0, 0.15], [-9.0, -3.0, 0.0])
    truth = static(ego, *cars)
    for spec in (EXACT, EXACT_REAR):
        az, times, r, label = cast_sweep(spec, 0, truth, 0.4, P)
        want, want_label = expected(spec, 0, truth, az, times)
        assert np.array_equal(np.isfinite(r), np.isfinite(want))
        assert np.allclose(r[np.isfinite(r)], want[np.isfinite(want)], atol=1e-9)
        assert np.array_equal(label, want_label)
        assert len(set(label[label >= 0])) == 2          # two cars ahead for the front, two behind for the rear
        # the cast origin is geometry.sensor_pose's
        o = np.array(sensor_pose(spec.mount, np.r_[ego, 0.0], P)[:2])
        assert np.allclose(o, origin(spec, ego)[0])
    scan = simulate_lidar(FRONT, 0, truth, 0.4, P, np.random.default_rng(3))
    hit = np.isfinite(scan.range)
    az, times, r, _ = cast_sweep(FRONT, 0, truth, 0.4, P)
    assert np.all(np.isfinite(r[hit]))
    assert abs(np.std(scan.range[hit]-r[hit])-FRONT.range_sigma) < 0.01


def test_hidden_car_gets_no_beams():
    hidden = static([0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [60.0, 0.2, 0.0], [60.0, 2.6, 0.0])
    az, _, r, label = cast_sweep(EXACT, 0, hidden, 0.4, P)
    assert not np.any(label == 1)                         # fully behind car 0
    assert np.any(label == 2)                             # partly behind it: only its uncovered part is hit
    shadow = np.abs(az) <= np.arctan(0.95/(30-0.0715-P.length/2-2.2735))
    assert np.all(label[shadow] == 0)
    assert np.all(r[label == 2] > 50) and np.all(r[label == 0] < 30)
    alone = static([0.0, 0.0, 0.0], [60.0, 0.2, 0.0])
    assert np.sum(cast_sweep(EXACT, 0, alone, 0.4, P)[3] == 0) >= 10


def test_own_body_is_never_hit():
    for car in range(3):
        mates = [[x, d, 0.0, 0.0] for x, d in ((-1000.0, 20.0), (1000.0, 20.0))]
        team = mates[:car]+[[5.0, 1.0, 0.4, 0.0]]+mates[car:]
        truth = history(lambda t: team, lambda t: np.zeros((0, 5)))
        for spec in (FRONT, REAR):
            scan = simulate_lidar(spec, car, truth, 0.4, P, np.random.default_rng(car))
            assert np.all(np.isnan(scan.range))
    # teammates close around car 1 are seen; car 1 itself never is
    team = [[0.0, 0.0, 0.0, 30.0], [-8.0, 0.0, 0.0, 30.0], [-16.0, 0.3, 0.0, 30.0]]
    truth = history(lambda t: team, lambda t: [[7, 15.0, 7.4, 0.0, 30.0]])
    labels = np.concatenate([cast_sweep(s, 1, truth, 0.4, P)[3] for s in (EXACT, EXACT_REAR)])
    assert set(labels) == {-1, 7, -2, -4}
    for spec in (EXACT, EXACT_REAR):
        r = cast_sweep(spec, 1, truth, 0.4, P)[2]
        assert np.nanmin(np.where(np.isfinite(r), r, np.nan)) > 2.0


def moving(rate=0.0):
    """Ego at 30 m/s (yaw rate `rate`), car 0 ahead at 70 m/s (40 m/s faster) and car 1 alongside-ahead at 30 m/s."""
    team = lambda t: [[30*t, 0.0, rate*t, 30.0], [30*t-300, 0.0, 0.0, 30.0], [30*t-320, 0.0, 0.0, 30.0]]
    return history(team, lambda t: [[0, 12+70*t, 0.0, 0.0, 70.0], [1, 20+30*t, 3.7, 0.0, 30.0]])


def test_scan_skew_follows_beam_times():
    truth, t_end = moving(), 0.4
    az, times, r, label = cast_sweep(EXACT, 0, truth, t_end, P)
    want, _ = expected(EXACT, 0, truth, az, times)
    both = np.isfinite(r) & np.isfinite(want)
    fast, slow = both & (label == 0), both & (label == 1)
    assert fast.sum() > 20 and slow.sum() > 20
    # within a chunk every beam sees the state at the chunk's mid time: at most CHUNK/2 of relative motion
    assert np.all(np.abs(r[fast]-want[fast]) <= 40*CHUNK/2/np.cos(az[fast])+1e-9)
    assert np.allclose(r[slow], want[slow], atol=1e-9)
    # against a scan frozen at t_end, the fast car is where it was at each beam's time: about 2 m nearer
    end = truth.at(t_end)
    o, boresight = origin(EXACT, end.team[0])
    face = slab(o, boresight, end.traffic[0])          # the fast car's rear face straight ahead at t_end
    skew = r[fast]-face/np.cos(az[fast])
    assert np.allclose(skew, 40*(times[fast]-t_end)/np.cos(az[fast]), atol=40*CHUNK/2/np.cos(az[fast]).min()+1e-9)
    assert np.all(skew < -1.5)
    assert abs(np.isfinite(r).sum()-np.isfinite(want).sum()) <= 2


def test_fine_chunks_converge_to_per_beam_truth():
    truth, t_end = moving(rate=0.2), 0.45
    az, times, r, label = cast_sweep(EXACT, 0, truth, t_end, P, chunk=1e-4)
    want, _ = expected(EXACT, 0, truth, az, times)
    both = np.isfinite(r) & np.isfinite(want)
    assert both.sum() > 50
    assert np.max(np.abs(r[both]-want[both])) < 5e-3


def test_dropout_fraction_near_spec():
    narrow = replace(FRONT, fov=radians(10.0), beam_spacing=radians(0.01), range_sigma=0.0, dropout=0.05,
                     far_dropout=0.5)
    rng = np.random.default_rng(11)
    for gap, scans in ((12.0, 10), (154.74, 40)):
        truth = static([0.0, 0.0, 0.0], [gap, 0.0, 0.0])
        true_range = cast_sweep(narrow, 0, truth, 0.4, P)[2]
        hit = np.isfinite(true_range)
        p = narrow.dropout+narrow.far_dropout*np.mean(true_range[hit])/narrow.range_max
        lost = np.mean([np.isnan(simulate_lidar(narrow, 0, truth, 0.4, P, rng).range[hit]) for _ in range(scans)])
        n = hit.sum()*scans
        assert n > 2500
        assert abs(lost-p) < 4*np.sqrt(p*(1-p)/n)+1e-3
    beyond = static([0.0, 0.0, 0.0], [FRONT.range_max+10, 0.0, 0.0])
    assert np.all(np.isnan(simulate_lidar(FRONT, 0, beyond, 0.4, P, rng).range))


def test_layers_repeat_each_beam_with_independent_noise():
    truth = static([0.0, 0.0, 0.0], [12.0, 0.0, 0.0])
    layered = replace(FRONT, layers=4, dropout=0.0, far_dropout=0.0)
    scan = simulate_lidar(layered, 0, truth, 0.4, P, np.random.default_rng(5))
    az, times, r, _ = cast_sweep(layered, 0, truth, 0.4, P)
    assert np.array_equal(scan.azimuth, np.repeat(az, 4)) and np.array_equal(scan.time, np.repeat(times, 4))
    m = scan.range.reshape(-1, 4)
    hit = np.isfinite(r)
    assert np.all(np.isfinite(m[hit])) and np.all(np.isnan(m[~hit]))
    assert abs(np.std(np.diff(m[hit], axis=1))-np.sqrt(2)*FRONT.range_sigma) < 0.005


def test_seeded_determinism():
    truth = moving()
    a = simulate_lidar(FRONT, 0, truth, 0.4, P, np.random.default_rng(42))
    b = simulate_lidar(FRONT, 0, truth, 0.4, P, np.random.default_rng(42))
    c = simulate_lidar(FRONT, 0, truth, 0.4, P, np.random.default_rng(43))
    assert np.array_equal(a.range, b.range, equal_nan=True) and np.array_equal(a.time, b.time)
    assert not np.array_equal(a.range, c.range, equal_nan=True)
    # the draws do not depend on what is hit, so the generator stays in step across scenes
    g1, g2 = np.random.default_rng(7), np.random.default_rng(7)
    simulate_lidar(FRONT, 0, truth, 0.4, P, g1)
    simulate_lidar(FRONT, 0, static([0.0, 0.0, 0.0]), 0.4, P, g2)
    assert g1.random() == g2.random()


def test_truth_interpolation_matches_truth_history():
    def traffic(t):
        ids = [0, 1, 2, 3 if t < 0.32 else 99]           # car 3 recycled as id 99 between 0.30 and 0.35
        speed = 30+20*(t >= 0.35)                         # piecewise-linear motion with a kink at 0.35
        return [[i, 10*k+speed*t, 3.7*(k % 3), 0.02*k, speed] for k, i in enumerate(ids)]
    team = lambda t: [[30*t, 0.1*t, 0.5*t, 30.0], [30*t-8, 3.7, wrap(2.8+1.0*t), 30.0], [30*t-16, 7.4, 0.0, 30.0]]
    truth = history(team, traffic, times=np.round(np.arange(0.0, 0.401, 0.05), 3))
    times = np.r_[0.2925, np.linspace(0.3, 0.44, 23), 0.45]     # snapshot times inside, clamping beyond 0.4
    team, traffic, ids = _knot_states(truth, times)
    last = truth.at(times[-1])
    assert np.array_equal(ids, last.traffic_id) and team.shape == (len(times), 3, 3)
    assert np.all(np.abs(np.diff(team[:, :, 2], axis=0)) < 0.2)       # yaw unwrapped along the knots
    for t, tm, tr in zip(times, team, traffic):
        s = truth.at(t)
        assert np.allclose(tm[:, :2], s.team[:, :2], atol=1e-9)
        assert np.allclose(wrap(tm[:, 2]-s.team[:, 2]), 0.0, atol=1e-9)
        here = np.isin(ids, s.traffic_id)                              # car 99 exists only from 0.35 on
        rows = [int(np.flatnonzero(s.traffic_id == i)[0]) for i in ids[here]]
        assert np.allclose(tr[here], s.traffic[rows, :3], atol=1e-9)
        assert np.allclose(tr[~here], last.traffic[~here, :3], atol=1e-9)


def test_cast_equals_truth_at_per_chunk():
    class AtOnly:                                         # no snapshot list: one truth.at call per chunk
        def __init__(self, h):
            self.at = h.at
    truth = moving(rate=0.2)
    for spec in (EXACT, EXACT_REAR):
        fast, slow = cast_sweep(spec, 0, truth, 0.43, P), cast_sweep(spec, 0, AtOnly(truth), 0.43, P)
        assert np.array_equal(np.isfinite(fast[2]), np.isfinite(slow[2])) and np.array_equal(fast[3], slow[3])
        assert np.allclose(fast[2][np.isfinite(fast[2])], slow[2][np.isfinite(slow[2])], atol=1e-9)


def test_unlimited_range_and_single_snapshot():
    truth = static([0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [1000.0, 40.0, 1.2], mates=[[-30.0, 0.0, 0.0, 0.0]]*2)
    ideal = replace(EXACT, range_max=np.inf)
    _, _, r, label = cast_sweep(ideal, 0, truth, 0.4, P)
    assert set(label) == {-1, 0, 1} and np.all(np.isinf(r[label == -1])) and np.all(np.isfinite(r[label >= 0]))
    assert np.all(np.isnan(simulate_lidar(ideal, 0, truth, 0.4, P, np.random.default_rng(0)).range[label == -1]))
    one = history(lambda t: [[0.0, 0.0, 0.0, 0.0], *AWAY], lambda t: [[0, 40.0, 0.0, 0.0, 0.0]], times=[0.2])
    assert np.array_equal(cast_sweep(EXACT, 0, one, 0.4, P)[3] == 0, label == 0)


def test_timing_front_lidar_45_cars():
    rng = np.random.default_rng(0)
    s0, lane, v = rng.uniform(-150, 400, 45), rng.integers(0, 3, 45), rng.uniform(24.6, 35.8, 45)
    team = lambda t: [[30*t, 3.7, 0.0, 30.0], [30*t-12, 3.7, 0.0, 30.0], [30*t-24, 0.0, 0.0, 30.0]]
    truth = history(team, lambda t: np.column_stack((np.arange(45), s0+v*t, 3.7*lane, np.zeros(45), v)))
    for car in range(3):
        assert np.isfinite(simulate_lidar(FRONT, car, truth, 0.4, P, rng).range).sum() > 50
    reps = 60
    start = time.perf_counter()
    for k in range(reps):
        simulate_lidar(FRONT, k % 3, truth, 0.35+0.002*(k % 25), P, rng)
    ms = (time.perf_counter()-start)*1e3/reps
    print(f'front lidar, 45 cars: {ms:.2f} ms per scan')
    assert ms < 15.0
