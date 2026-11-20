"""The camera simulator's compiled path (sim/_camera_sim_jit.py) against its numpy reference (sim/camera.py,
NOHESI_JIT=0): from identically seeded generators, the same frames (truth ids in the same order, the same pitch, boxes
and scores to rounding) and the generator left in the same state, on the frames of a real v9 episode (the pipeline's
calls) and on randomized dense scenes (at and between snapshots) with occlusion, truncation, pitching, teammates
alongside, false boxes and the spec's options; plus the kernels one by one and their constants."""
from dataclasses import replace
from math import radians
from types import SimpleNamespace
import numpy as np
import pytest
from nohesi.config import Planner, scenario_v9
from nohesi.geometry import cg_offsets
from nohesi.perception import _jit
from nohesi.perception import pipeline as P
from nohesi.perception.config import Perception, default_cameras
from nohesi.perception.geometry import car_boxes, sensor_pose
from nohesi.perception.sim import _camera_sim_jit as K
from nohesi.perception.sim import camera as C
from nohesi.perception.types import TruthHistory
from nohesi.world import Snapshot

pytestmark = pytest.mark.skipif(not _jit.ENABLED, reason='numba off (NOHESI_JIT=0 or not installed)')

SC = scenario_v9()
VEH = SC.vehicle
H_CAR = 1.45
CAMERAS = default_cameras()
LANE = (1.8, 5.4, 9.0)
TOL_PX = 1e-9


def run(path, specs, car, hist, t, params, state, height=H_CAR):
    """simulate_cameras on the compiled (True) or numpy (False) path from generator state `state`: (frames, the
    generator's state after)."""
    rng = np.random.default_rng()
    rng.bit_generator.state = state
    fn = C._simulate_cameras_jit if path else C._simulate_cameras_numpy
    return fn(specs, car, hist, t, params, rng, height), rng.bit_generator.state


def assert_same(specs, car, hist, t, params, state, height=H_CAR):
    """Both paths give the same frames and leave the generator alike; returns (frames, detections, false boxes)."""
    got, g_state = run(True, specs, car, hist, t, params, state, height)
    want, w_state = run(False, specs, car, hist, t, params, state, height)
    assert g_state == w_state
    assert len(got) == len(want) == len(specs)
    dets = fakes = 0
    for g, w in zip(got, want):
        assert (g.sensor, g.car, g.time, g.pitch) == (w.sensor, w.car, w.time, w.pitch)
        assert type(g.time) is float and type(g.pitch) is float
        assert g.truth_id.dtype == w.truth_id.dtype and np.array_equal(g.truth_id, w.truth_id)
        assert g.boxes.shape == w.boxes.shape == (len(w.truth_id), 4) and g.score.shape == w.score.shape
        assert g.boxes.dtype == w.boxes.dtype == float and g.score.dtype == float
        assert np.all(np.abs(g.boxes-w.boxes) <= TOL_PX) and np.all(np.abs(g.score-w.score) <= 1e-12)
        dets += int(np.count_nonzero(w.truth_id != -1))
        fakes += int(np.count_nonzero(w.truth_id == -1))
    return want, dets, fakes


def snapshot(t, team, traffic, ids=None):
    """A true Snapshot: team rows [s, d, yaw, v], traffic rows [s, d, lateral speed, v]."""
    team, traffic = np.asarray(team, float), np.asarray(traffic, float).reshape(-1, 4)
    return Snapshot(time=float(t), team=team, team_steer=np.zeros(3), team_s=team[:, 0].copy(),
                    team_d=team[:, 1].copy(), traffic_s=traffic[:, 0].copy(), traffic_d=traffic[:, 1].copy(),
                    traffic_v=traffic[:, 3].copy(), traffic_id=np.arange(len(traffic)) if ids is None else ids,
                    collisions=(), done=False, traffic_vd=traffic[:, 2].copy())


def random_history(rng, n=45):
    """Two snapshots 0.1 s apart of v9-like traffic around a bunched team (teammates beside and in front of each
    other, so they fill and block each other's cameras), some cars changing lanes, braking or accelerating; one car
    leaves and one is new in the second (TruthHistory matches by id)."""
    lanes = np.array(LANE)[rng.integers(0, 3, n)]+rng.normal(0.0, 0.4, n)
    traffic = np.column_stack((rng.uniform(-120, 200, n), lanes, rng.normal(0.0, 0.8, n), rng.uniform(25, 35, n)))
    team = np.array([[0.0, LANE[1], 0.0, 30.0],
                     [rng.uniform(-15, -6), LANE[1]+rng.choice((-3.6, 0.0, 3.6)), rng.normal(0.0, 0.03), 30.0],
                     [rng.uniform(-3, 15), LANE[1]+rng.choice((-3.6, 3.6)), rng.normal(0.0, 0.03), 30.0]])
    hist = TruthHistory()
    hist.push(snapshot(0.0, team, traffic), rng.uniform(-6.0, 3.0, 3))
    later = traffic+0.1*np.column_stack((traffic[:, 3], traffic[:, 2], np.zeros(n), rng.normal(0.0, 1.0, n)))
    ids = np.arange(n)
    ids[rng.integers(n)] = 1000
    team2 = team+0.1*np.column_stack((team[:, 3], np.zeros(3), rng.normal(0.0, 0.05, 3), np.zeros(3)))
    hist.push(snapshot(0.1, team2, later, ids), rng.uniform(-6.0, 3.0, 3))
    return hist


def spec_variants(rng):
    """The default cameras, and variants: noisy pitch (large), no occlusion cut, a high cut, misses, many false
    boxes, no noise, a wide lens (truncation and side views), a small cut-off."""
    pitchy = tuple(replace(s, pitch_sigma=radians(2.0)) for s in CAMERAS)
    mixed = (replace(CAMERAS[0], min_visible=0.0, false_rate=3.0), replace(CAMERAS[1], pd=0.5, min_visible=0.9),
             replace(CAMERAS[2], sigma_px=0.0, pitch_sigma=0.0, pd=1.0, false_rate=0.0),
             replace(CAMERAS[3], hfov=radians(150.0), min_height_px=3.0, false_rate=1.0))
    wide = tuple(replace(s, hfov=radians(rng.uniform(40.0, 170.0)), min_visible=rng.uniform(0.0, 1.0),
                         false_rate=rng.uniform(0.0, 2.0)) for s in CAMERAS)
    return [CAMERAS, pitchy, mixed, wide, CAMERAS[1:3], (CAMERAS[0],)]


def test_kernel_constants_match_camera_py():
    assert set(K.TUNE_FIELDS) <= set(vars(C)) and np.array_equal(C._tune(), [getattr(C, n) for n in K.TUNE_FIELDS])
    rows = C._cams(CAMERAS, VEH, tuple(map(float, cg_offsets(VEH))))
    assert rows.shape == (4, len(K.CAM_FIELDS)) and len(K.CAM_FIELDS) == K.SIGMA+1
    for row, spec in zip(rows, CAMERAS):
        pose = sensor_pose(spec.mount, np.array([0.0, 0.0, 0.0]), VEH)
        assert (row[K.OX], row[K.OY]) == pose[:2] and row[K.MYAW] == spec.mount.yaw and row[K.MZ] == pose[3]
        assert row[K.F] == C.focal(spec) and (row[K.W], row[K.H]) == (spec.width_px, spec.height_px)
        assert (row[K.MINH], row[K.PD], row[K.MINVIS], row[K.SIGMA]) == (spec.min_height_px, spec.pd,
                                                                       spec.min_visible, spec.sigma_px)
    assert K.PI == np.pi and K.TWO_PI == 2*np.pi


def test_kernels_match_numpy_pieces():
    """scene against geometry.car_boxes, candidates against project_boxes and clip_boxes, visible against
    visible_fractions (camera.py's numpy functions), bit for bit."""
    rng = np.random.default_rng(5)
    tune = C._tune()
    partial = 0
    for _ in range(30):
        state = random_history(rng).at(0.0)
        car = int(rng.integers(3))
        fp, ids, idx, box, _, _, poses, counts = K.scene(np.array(state.traffic), np.array(state.traffic_id),
                                                        np.array(state.team), car, *map(float, cg_offsets(VEH)),
                                                        VEH.width/2, 4)
        others = [k for k in range(3) if k != car]
        states = np.concatenate((state.traffic[:, :3], state.team[others, :3]))
        assert np.array_equal(fp, car_boxes(states, VEH))
        assert np.array_equal(ids, np.concatenate((state.traffic_id, -2-np.array(others))))
        cams = C._cams(CAMERAS, VEH, tuple(map(float, cg_offsets(VEH))))
        for k, spec in enumerate(CAMERAS):
            pitch = rng.normal(0.0, 0.02)
            n = K.candidates(fp, H_CAR, np.array(state.team), car, cams, k, pitch, tune, poses, idx, box, counts)
            pose = sensor_pose(spec.mount, state.team[car], VEH)
            assert tuple(poses[k]) == pose
            boxes, ok = C.project_boxes(spec, pose, states, VEH, H_CAR, pitch)
            clipped, share = C.clip_boxes(boxes[ok], spec)
            keep = (share >= C.KEEP_AREA) & (clipped[:, 3]-clipped[:, 2] >= spec.min_height_px)
            assert counts[k] == n and np.array_equal(idx[k*len(fp):k*len(fp)+n], np.flatnonzero(ok)[keep])
            assert np.array_equal(box[k*len(fp):k*len(fp)+n], clipped[keep])
            targets = idx[k*len(fp):k*len(fp)+n].copy()
            got = K.visible(poses[k, :2].copy(), fp, targets, C._steps(9), tune)
            want = C.visible_fractions(pose[:2], fp, targets)
            assert np.array_equal(got, want)
            partial += np.count_nonzero((want > 0) & (want < 1))
    assert partial > 30
    cams = C._cams(CAMERAS, VEH, tuple(map(float, cg_offsets(VEH))))
    cut = 0
    for _ in range(200):                     # false boxes: many per frame, some cut at the border or left out
        k, pitch = int(rng.integers(4)), rng.normal(0.0, 0.03)
        spec = replace(CAMERAS[k], false_rate=6.0)
        a, b = np.random.default_rng(), np.random.default_rng()
        b.bit_generator.state = a.bit_generator.state
        want = C._false_boxes(spec, (0.0, 0.0, 0.0, CAMERAS[k].mount.z), VEH, a, H_CAR, pitch)
        nf = b.poisson(spec.false_rate)
        got = K.fakes(cams, k, pitch, H_CAR, b.uniform(C.FALSE_NEAR, 1.0, nf), b.uniform(VEH.width, VEH.length, nf),
                      b.uniform(0.0, spec.width_px, nf))
        assert np.array_equal(got, want) and a.bit_generator.state == b.bit_generator.state
        cut += nf-len(want)+np.count_nonzero((want == 0) | (want == [spec.width_px]*2+[spec.height_px]*2))
    assert cut > 20


@pytest.mark.parametrize('seed', range(6))
def test_randomized_scenes(seed):
    """Dense random scenes, every team car, the spec variants, at a snapshot (read-only truth arrays), between the
    two snapshots (interpolated) and past the end, against simple and SimpleNamespace vehicle params."""
    rng = np.random.default_rng(100+seed)
    plain = SimpleNamespace(**{k: getattr(VEH, k) for k in ('wheelbase', 'cg_to_rear_axle', 'length', 'width')})
    dets = fakes = frames = 0
    for _ in range(3):
        hist = random_history(rng)
        for specs in spec_variants(rng):
            for car in range(3):
                for t in (0.0, float(rng.uniform(0.0, 0.1)), 0.1, 0.25):
                    params = plain if rng.random() < 0.3 else VEH
                    state = np.random.default_rng(int(rng.integers(1 << 30))).bit_generator.state
                    out, d, f = assert_same(specs, car, hist, t, params, state)
                    dets, fakes, frames = dets+d, fakes+f, frames+len(out)
    assert dets > 1000 and fakes > 50 and frames == 3*19*3*4


def test_truncation_occlusion_and_pitch_edge_cases():
    """Hand-made scenes: cars at the image border (truncation on both sides of KEEP_AREA), a car hidden behind
    another and half seen past it, cars alongside (partly behind the front and rear cameras), hard braking pitch,
    an empty road and a team alone."""
    dets = 0
    team = np.array([[0.0, LANE[1], 0.0, 30.0], [-30.0, LANE[1], 0.0, 30.0], [25.0, LANE[2], 0.02, 30.0]])
    exact = tuple(replace(s, sigma_px=0.0, pitch_sigma=0.0, pd=1.0, false_rate=0.0) for s in CAMERAS)
    for dy in np.linspace(-30.0, 30.0, 61):
        traffic = [[40.0, LANE[1]+dy, 0.0, 30.0], [80.0, LANE[1], 0.0, 30.0], [80.0, LANE[1]+1.2, 0.0, 30.0],
                   [1.0, LANE[0], 0.0, 30.0], [-60.0, LANE[1]+dy/4, 0.0, 30.0]]
        for ax in ((0.0, 0.0, 0.0), (-9.0, 4.0, -2.0)):
            hist = TruthHistory()
            hist.push(snapshot(0.0, team, traffic), ax)
            for specs in (CAMERAS, exact):
                for car in range(3):
                    state = np.random.default_rng(int(dy*10)+500).bit_generator.state
                    dets += assert_same(specs, car, hist, 0.0, VEH, state)[1]
    for traffic in (np.zeros((0, 4)), [[5000.0, LANE[0], 0.0, 30.0]]):
        hist = TruthHistory()
        hist.push(snapshot(0.0, team, traffic), (0.0, 0.0, 0.0))
        for car in range(3):
            assert_same(CAMERAS, car, hist, 0.0, VEH, np.random.default_rng(1).bit_generator.state)
    assert dets > 500


def test_real_v9_episode():
    """Every simulate_cameras call of a short shadow v9 episode (seed 1000, the pre-roll included): the pipeline's
    own truth history, times, accelerations and generator states; the compiled path's frames go on to the pipeline."""
    seen = {'calls': 0, 'dets': 0}
    real = P.simulate_cameras

    def checked(specs, car, truth, t, params, rng, car_height):
        state = rng.bit_generator.state
        frames, d, _ = assert_same(specs, car, truth, t, params, state, car_height)
        seen['calls'] += 1
        seen['dets'] += d
        return real(specs, car, truth, t, params, rng, car_height)

    from nohesi.evaluation.runner import EpisodeRun
    sc = replace(SC, timing=replace(SC.timing, duration=2.0), perception=Perception(enabled=True, mode='shadow'))
    mp = pytest.MonkeyPatch()
    mp.setattr(P, 'simulate_cameras', checked)
    try:
        r = EpisodeRun(sc, 1000, 'overtaker', Planner())
        while not r.done:
            r.step()
    finally:
        mp.undo()
    assert seen['calls'] >= 150 and seen['dets'] > 1500, seen
    for kernel in (K.scene, K.candidates, K.select, K.fakes, K.finish):
        assert len(kernel.signatures) == 1, (kernel, kernel.signatures)   # one specialization each
