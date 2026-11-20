"""The radar simulator's compiled path (sim/_radar_sim_jit.py, radar._scans_compiled) against its numpy reference
(radar._scans_numpy, the path with NOHESI_JIT=0): from identically seeded generators, the same scans (sensor, car,
time), the same truth ids and point counts per scan, range, azimuth and range rate to rounding, and the generator left
in the same state (the same random calls). On the truth of short v9 episodes (at snapshot times and between them) and
on randomized synthetic scenes: dense traffic changing lanes with the team bunched (occlusion, grazing rays from the
mirror radars), specs with more points, coarse resolutions (merging), heavy clutter, a fully hidden pd of 0, no
occlusion, a flat pd curve, a 360 degree field of view and near_range beyond range_max. Also checks that the kernels'
constants are radar.py's and that each kernel has one specialization.

A scene where a discrete outcome (the field of view, a reflection point kept, two points merged, a detection draw)
sits within rounding of its boundary could legitimately differ; such scenes are found by perturbing the numpy path's
own input by 1e-15 and are left out (none of the real scenes is; KNIFE_MAX of the synthetic ones may be, none is at
present). The kernels' shortcuts (bounding cones, decisions by the cosine away from an edge) only skip work, so the two
paths agree to the bit on these scenes."""
from dataclasses import fields, replace
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import _jit
from nohesi.perception.config import RadarSpec, default_radars
from nohesi.perception.geometry import wrap
from nohesi.perception.sim import _radar_sim_jit as K
from nohesi.perception.sim import radar as R
from nohesi.perception.types import TruthHistory
from nohesi.world import Snapshot, World

pytestmark = pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')

VEH = scenario_v9().vehicle
RADARS = default_radars()
LANE = 3.6
ATOL = 1e-9
KNIFE_MAX = 3            # synthetic scenes left out as knife-edge cases (of N_SYNTH)
N_SYNTH = 300


def test_kernel_constants_are_radar_constants():
    assert (K.CLUTTER_RANGE_MIN, K.CLUTTER_STATIONARY) == (R.CLUTTER_RANGE_MIN, R.CLUTTER_STATIONARY)
    assert np.array_equal(K.NEXT, R.NEXT)
    names = {f.name for f in fields(RadarSpec)}
    assert all(name in names for name in K.TAB_FIELDS[:12])
    assert K.TAB_FIELDS[12:] == ('mount_x', 'mount_y', 'mount_yaw') and len(K.TAB_FIELDS) == K.MYAW+1
    assert np.array_equal(R._SAMPLES, np.linspace(0, 1, R.VIS_SAMPLES, endpoint=False))


def run(path, jobs, scene, t, seed):
    """The scans of one path from default_rng(seed), and the generator's next draw (the stream's position)."""
    rng = np.random.default_rng(seed)
    out = path(jobs, scene, t, VEH, rng)
    return out, rng.random()


def signature(scans):
    return [(s.sensor, s.car, s.time, len(s.range), tuple(s.truth_id)) for s in scans]


def assert_same(got, want):
    """Compiled scans `got` equal numpy's `want`; returns the number of points compared and the largest difference."""
    (g, gnext), (w, wnext) = got, want
    assert signature(g) == signature(w)
    assert gnext == wnext
    worst, n = 0.0, 0
    for a, b in zip(g, w):
        assert a.truth_id.dtype == b.truth_id.dtype and a.range.dtype == b.range.dtype
        d = (np.abs(a.range-b.range), np.abs(wrap(a.azimuth-b.azimuth)), np.abs(a.range_rate-b.range_rate))
        for x in d:
            if len(x):
                worst = max(worst, float(x.max()))
        n += len(a.range)
    assert worst <= ATOL
    return n, worst


def perturbed(scene, k):
    """The scene with its boxes, centres and velocities scaled by 1 + k 1e-15."""
    state, boxes, centre, heading, vel, ids, vx = scene
    e = 1+k*1e-15
    return state, boxes*e, centre*e, heading, vel*e, ids, vx


def knife_edge(jobs, scene, t, seed):
    """Whether numpy's own scans change discretely when its input moves by rounding."""
    ref = signature(run(R._scans_numpy, jobs, scene, t, seed)[0])
    return any(signature(run(R._scans_numpy, jobs, perturbed(scene, k), t, seed)[0]) != ref for k in (1, -1))


@pytest.fixture(scope='module')
def episodes():
    """(truth, times) of v9 seeds 1000 and 1001 driven straight for 3 s: every step's snapshot time and the midpoint
    before it."""
    out = []
    for seed in (1000, 1001):
        world = World(scenario_v9(), seed)
        truth = TruthHistory()
        truth.push(world.snapshot, np.zeros(3))
        times = []
        for _ in range(60):
            snap, applied = world.step(np.zeros((3, 2)))
            truth.push(snap, applied[:, 0].copy())
            times.append(float(snap.time))
        out.append((truth, times, world.plant.params))
    return out


def test_v9_episodes(episodes):
    jobs = [(spec, car) for car in range(3) for spec in RADARS]
    n_scans = n_points = n_edge = 0
    worst = 0.0
    for k, (truth, times, params) in enumerate(episodes):
        for j, t in enumerate(times[1:]):
            for tt in (t, t-0.025):
                scene = R._scene(truth, tt, params)
                seed = 1000*k+j
                if knife_edge(jobs, scene, tt, seed):
                    n_edge += 1
                    continue
                n, w = assert_same(run(R._scans_compiled, jobs, scene, tt, seed),
                                   run(R._scans_numpy, jobs, scene, tt, seed))
                n_scans, n_points, worst = n_scans+len(jobs), n_points+n, max(worst, w)
    print(f'v9: {n_scans} scans, {n_points} points, max diff {worst:.2e}, {n_edge} knife-edge steps')
    assert n_edge == 0 and n_points > 5000
    for kernel in (K.view_pd, K.points_merge, K.noise_sort):
        assert len(kernel.signatures) == 1, kernel


def snapshot(traffic, team):
    """A true Snapshot: traffic rows (s, d, v_s, v_d), team rows (x, y, yaw, v)."""
    traffic, team = np.asarray(traffic, float).reshape(-1, 4), np.asarray(team, float)
    return Snapshot(time=0.0, team=team, team_steer=np.zeros(3), team_s=team[:, 0].copy(), team_d=team[:, 1].copy(),
                    traffic_s=traffic[:, 0].copy(), traffic_d=traffic[:, 1].copy(), traffic_v=traffic[:, 2].copy(),
                    traffic_id=np.arange(100, 100+len(traffic)), collisions=(), done=False,
                    traffic_vd=traffic[:, 3].copy(), team_vy=np.array([0.3, -0.2, 0.0]),
                    team_r=np.array([0.05, -0.02, 0.0]))


def synthetic(rng):
    """A dense random scene (up to 60 cars on and between three lanes, some changing lanes, bunched around a bunched
    team) and a random variant of the default radars."""
    n = int(rng.integers(0, 61))
    s = rng.uniform(-120.0, 260.0, n)
    d = np.where(rng.random(n) < 0.8, LANE*rng.integers(0, 3, n), rng.uniform(-1.0, 2*LANE+1.0, n))
    team = np.array([[0.0, LANE, rng.normal(0.0, 0.03), 30.0],
                     [rng.uniform(-25.0, -7.0), LANE*rng.integers(0, 3), rng.normal(0.0, 0.03), 29.0],
                     [rng.uniform(7.0, 25.0), LANE*rng.integers(0, 3), rng.normal(0.0, 0.03), 31.0]])
    keep = np.ones(n, bool)
    for k in range(n):                         # no car overlapping a team car or an earlier car
        others = np.vstack((team[:, :2], np.column_stack((s[:k], d[:k]))[keep[:k]]))
        keep[k] = not np.any((np.abs(others[:, 0]-s[k]) < 6.0) & (np.abs(others[:, 1]-d[k]) < 2.4))
    traffic = np.column_stack((s, d, rng.uniform(20.0, 38.0, n), rng.normal(0.0, 0.6, n)))[keep]
    history = TruthHistory()
    history.push(snapshot(traffic, team))
    specs = []
    for spec in RADARS:
        change = {}
        if rng.random() < 0.4:
            change.update(points_max=int(rng.integers(1, 7)))
        if rng.random() < 0.4:
            change.update(res_range=rng.uniform(0.0, 3.0), res_azimuth=np.radians(rng.uniform(0.0, 10.0)))
        if rng.random() < 0.3:
            change.update(false_alarms=rng.uniform(0.0, 20.0))
        if rng.random() < 0.3:
            change.update(occluded_pd=float(rng.choice([0.0, 1.0, 0.7])))
        if rng.random() < 0.15:
            change.update(pd_ref=float(rng.choice([1.0, 0.4])))
        if rng.random() < 0.1:
            change.update(fov=2*np.pi, fov_far=2*np.pi)
        if rng.random() < 0.1:
            change.update(near_range=spec.range_max+20.0)
        if rng.random() < 0.1:
            change.update(sigma_range=0.0, sigma_azimuth=0.0, sigma_range_rate=0.0)
        specs.append(replace(spec, **change))
    cars = [int(c) for c in rng.permutation(3)[:int(rng.integers(1, 4))]]
    return history, [(spec, car) for car in cars for spec in specs]


def test_synthetic_scenes():
    rng = np.random.default_rng(17)
    n_scenes = n_points = n_edge = 0
    worst = 0.0
    for k in range(N_SYNTH):
        history, jobs = synthetic(rng)
        scene = R._scene(history, 0.0, VEH)
        if knife_edge(jobs, scene, 0.0, k):
            n_edge += 1
            continue
        got, want = run(R._scans_compiled, jobs, scene, 0.0, k), run(R._scans_numpy, jobs, scene, 0.0, k)
        n, w = assert_same(got, want)
        n_points, n_scenes, worst = n_points+n, n_scenes+1, max(worst, w)
    print(f'synthetic: {n_scenes} scenes, {n_points} points, max diff {worst:.2e}, '
          f'{n_edge} knife-edge scenes')
    assert n_edge <= KNIFE_MAX and n_points > 20000


def test_merging_and_occlusion_are_exercised(monkeypatch):
    """The synthetic scenes do merge points and hide cars partly and fully: what the comparison covers."""
    merges, seen = [], []
    merge, visibility = R._merge, R._visibility
    monkeypatch.setattr(R, '_merge', lambda r, *a: (merges.append(len(r)), merge(r, *a))[1])
    monkeypatch.setattr(R, '_visibility', lambda *a: (seen.append(visibility(*a)), seen[-1])[1])
    rng = np.random.default_rng(17)
    merged = 0
    for k in range(40):
        history, jobs = synthetic(rng)
        merges.clear()
        out = R._scans_numpy(jobs, R._scene(history, 0.0, VEH), 0.0, VEH, np.random.default_rng(k))
        merged += merges[0] > sum(len(s.range) for s in out)
    v = np.concatenate(seen)
    assert merged >= 10
    assert (v == 0).sum() > 50 and ((v > 0) & (v < 1)).sum() > 50 and (v == 1).sum() > 50


def test_edge_cases():
    """No traffic (clutter only), no clutter and no candidates, one scan (simulate_radar), no occlusion model."""
    empty = TruthHistory()
    empty.push(snapshot(np.zeros((0, 4)), [[0.0, LANE, 0.0, 30.0], [-3000.0, 0.0, 0.0, 30.0],
                                           [3000.0, 0.0, 0.0, 30.0]]))
    quiet = [replace(s, false_alarms=0.0) for s in RADARS]
    for specs in (RADARS, quiet, [replace(s, occluded_pd=1.0) for s in RADARS]):
        for cars in ([0], [0, 1, 2]):
            jobs = [(spec, car) for car in cars for spec in specs]
            scene = R._scene(empty, 0.0, VEH)
            for seed in range(5):
                assert_same(run(R._scans_compiled, jobs, scene, 0.0, seed), run(R._scans_numpy, jobs, scene, 0.0, seed))
    one = [(RADARS[0], 0)]
    scene = R._scene(empty, 0.0, VEH)
    assert all(len(s.range) == 0 for s in run(R._scans_compiled, [(q, 0) for q in quiet], scene, 0.0, 0)[0])
    assert_same(run(R._scans_compiled, one, scene, 0.0, 3), run(R._scans_numpy, one, scene, 0.0, 3))


def test_simulate_radars_takes_the_compiled_path(episodes):
    truth, times, params = episodes[0]
    t = times[10]
    a = R.simulate_radars(RADARS, range(3), truth, t, params, np.random.default_rng(4))
    jobs = [(spec, car) for car in range(3) for spec in RADARS]
    b = R._scans_compiled(jobs, R._scene(truth, t, params), t, params, np.random.default_rng(4))
    assert signature(a) == signature(b)
    for x, y in zip(a, b):
        for k in ('range', 'azimuth', 'range_rate'):
            assert np.array_equal(getattr(x, k), getattr(y, k))


def test_field_of_view_test_at_its_boundary():
    """The kernels' field-of-view decision (by the cosine, atan2 within COS_MARGIN of the edge) against numpy's
    abs(wrap(atan2 - yaw)) <= half, for points from 1e-16 to 1e-3 rad either side of the edge."""
    rng = np.random.default_rng(5)
    eps = np.concatenate([s*np.logspace(-16, -3, 40) for s in (1, -1)]+[[0.0]])
    n_close = 0
    for _ in range(300):
        yaw = rng.uniform(-np.pi, np.pi)
        hf = float(rng.choice([rng.uniform(0.0, np.pi), 0.0, np.pi/2, np.pi-1e-9, np.pi, 4.0, -0.1]))
        d = rng.uniform(0.5, 300.0)
        for side in (1, -1):
            a = yaw+side*(hf+eps)
            qx, qy = d*np.cos(a), d*np.sin(a)
            want = np.abs(wrap(np.arctan2(qy, qx)-yaw)) <= hf
            got = [K._in_fov(x, y, np.hypot(x, y), np.cos(yaw), np.sin(yaw), yaw, hf, np.cos(hf))
                   for x, y in zip(qx, qy)]
            assert np.array_equal(got, want)
            n_close += int(np.sum(np.abs(np.cos(np.arctan2(qy, qx)-yaw)-np.cos(hf)) <= K.COS_MARGIN))
    assert n_close > 1000                      # the atan2 fallback was taken
