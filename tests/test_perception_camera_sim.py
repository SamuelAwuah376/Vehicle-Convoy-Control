"""Tier A camera simulator (nohesi/perception/sim/camera.py): pinhole geometry, occlusion, the size cut-off,
truncation, side cameras, braking pitch, noise, false boxes, determinism and speed."""
import time
from dataclasses import replace
from math import radians
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.geometry import cg_offsets
from nohesi.perception.config import default_cameras
from nohesi.perception.geometry import box_corners_3d, car_boxes, focal, project, sensor_pose, visible_fraction
from nohesi.perception.sim.camera import clip_boxes, project_boxes, simulate_camera, simulate_cameras, visible_fractions
from nohesi.perception.types import TruthHistory
from nohesi.world import Snapshot

VEH = scenario_v9().vehicle
H_CAR = 1.45
LANE = (1.8, 5.4, 9.0)
FRONT, REAR, LEFT, RIGHT = default_cameras()
FAR_TEAM = ((-3000.0, LANE[1]), (-4000.0, LANE[1]))   # teammates parked far away unless a test places them


def exact(spec, **kw):
    """The spec without noise, misses or false boxes."""
    return replace(spec, **{**dict(sigma_px=0.0, pitch_sigma=0.0, pd=1.0, false_rate=0.0), **kw})


def truth(traffic=(), team=((0.0, LANE[1]),)+FAR_TEAM, team_ax=(0.0, 0.0, 0.0), t=0.0):
    """TruthHistory with one snapshot: traffic rows (s, d[, v]) and team rows (s, d)."""
    tr = np.array([(*row, 30.0)[:3] for row in traffic], float).reshape(-1, 3)
    tm = np.zeros((3, 4))
    tm[:, :2] = np.array(team, float)
    tm[:, 3] = 30.0
    snap = Snapshot(time=t, team=tm, team_steer=np.zeros(3), team_s=tm[:, 0].copy(), team_d=tm[:, 1].copy(),
                    traffic_s=tr[:, 0].copy(), traffic_d=tr[:, 1].copy(), traffic_v=tr[:, 2].copy(),
                    traffic_id=np.arange(len(tr))+100, collisions=(), done=False)
    hist = TruthHistory()
    hist.push(snap, team_ax)
    return hist


def frame(spec, hist, car=0, seed=0, t=0.0):
    return simulate_camera(spec, car, hist, t, VEH, np.random.default_rng(seed), H_CAR)


def near_face(spec, ego, s_car):
    """Distance along the boresight from a forward camera on ego (s, d) to the rear bumper of a car at s_car."""
    pose = sensor_pose(spec.mount, np.array([*ego, 0.0]), VEH)
    return s_car+cg_offsets(VEH)[0]-pose[0], pose


def test_car_ahead_projects_by_pinhole_formulas():
    spec = exact(FRONT)
    f, cx, cy = focal(spec), spec.width_px/2, spec.height_px/2
    for gap in (15.0, 40.0, 90.0):
        fr = frame(spec, truth([(gap, LANE[1])]))
        assert fr.truth_id.tolist() == [100] and fr.time == 0.0 and fr.pitch == 0.0
        x, pose = near_face(spec, (0.0, LANE[1]), gap)
        h = pose[3]
        ul, ur, vt, vb = fr.boxes[0]
        assert vb == pytest.approx(cy+f*h/x, abs=1e-6)
        assert vt == pytest.approx(cy-f*(H_CAR-h)/x, abs=1e-6)
        assert vb-vt == pytest.approx(f*H_CAR/x, abs=1e-6)
        assert (ul, ur) == pytest.approx((cx-f*VEH.width/2/x, cx+f*VEH.width/2/x), abs=1e-6)
        assert 0.6 <= fr.score[0] <= 1.0


def test_score_grows_with_box_height():
    spec = exact(FRONT)
    scores = [frame(spec, truth([(gap, LANE[1])])).score[0] for gap in (20.0, 50.0, 100.0, 150.0)]
    assert all(a > b for a, b in zip(scores, scores[1:]))
    assert 0.6 < scores[-1] < scores[0] < 1.0


def test_min_height_cutoff_at_long_range():
    spec = exact(FRONT)
    f = focal(spec)
    for height, seen in ((16.0, True), (14.0, False)):
        x = f*H_CAR/height
        s_car = x-near_face(spec, (0.0, LANE[1]), 0.0)[0]
        fr = frame(spec, truth([(s_car, LANE[1])]))
        assert (len(fr.boxes) == 1) == seen
        if seen:
            assert fr.boxes[0, 3]-fr.boxes[0, 2] == pytest.approx(height, abs=1e-6)
    assert 150.0 < f*H_CAR/spec.min_height_px < 170.0      # the front camera's ~150 m detection limit


def test_occlusion_by_nearer_car():
    spec = exact(FRONT)
    fr = frame(spec, truth([(30.0, LANE[1]), (60.0, LANE[1])]))
    assert fr.truth_id.tolist() == [100]
    fr = frame(spec, truth([(30.0, LANE[1]), (60.0, LANE[2])]))
    assert sorted(fr.truth_id.tolist()) == [100, 101]
    fr = frame(replace(spec, min_visible=0.0), truth([(30.0, LANE[1]), (60.0, LANE[1])]))
    assert sorted(fr.truth_id.tolist()) == [100, 101]   # the cut is the occlusion test, not the projection
    # partly hidden: a car half a width aside behind the leader is seen only past it
    hist = truth([(30.0, LANE[1]), (60.0, LANE[1]+1.2)])
    boxes = car_boxes(np.array([[30.0, LANE[1], 0.0], [60.0, LANE[1]+1.2, 0.0]]), VEH)
    pose = sensor_pose(spec.mount, np.array([0.0, LANE[1], 0.0]), VEH)
    vis = visible_fraction(pose[:2], boxes, 1)
    assert 0.05 < vis < 0.95
    assert (101 in frame(replace(spec, min_visible=vis-0.01), hist).truth_id) is True
    assert (101 in frame(replace(spec, min_visible=vis+0.01), hist).truth_id) is False


def test_batched_visibility_matches_geometry():
    rng = np.random.default_rng(3)
    fracs = []
    for _ in range(6):
        s = np.concatenate([np.cumsum(rng.uniform(7.0, 30.0, 12))-150.0 for _ in LANE])
        d = np.repeat(LANE, 12)+rng.uniform(-0.5, 0.5, 36)
        boxes = car_boxes(np.column_stack((s, d, rng.uniform(-0.05, 0.05, 36))), VEH)
        for own in (5, 17, 30):
            for dx, dy in ((1.9, 0.0), (-2.2, 0.0), (0.7, 0.95), (0.7, -0.95)):   # front, rear and mirror mounts
                origin = np.array([s[own]+dx, d[own]+dy])
                keep = np.setdiff1d(np.arange(36), [own])         # the own car's body is left out
                targets = rng.permutation(35)[:rng.integers(1, 36)]
                got = visible_fractions(origin, boxes[keep], targets)
                want = [visible_fraction(origin, boxes[keep], k) for k in targets]
                assert got == pytest.approx(want, abs=1e-12)
                fracs += list(got)
    fracs = np.array(fracs)
    assert np.mean(fracs == 0) > 0.1 and np.mean(fracs == 1) > 0.1 and np.mean((fracs > 0) & (fracs < 1)) > 0.05


def test_truncation_rule_at_the_image_border():
    spec = replace(exact(FRONT), min_visible=0.0)
    pose = sensor_pose(spec.mount, np.array([0.0, LANE[1], 0.0]), VEH)
    kept = dropped = partial = 0
    for dy in np.linspace(-35.0, 35.0, 281):
        state = np.array([[40.0, LANE[1]+dy, 0.0]])
        u, v, depth = project(spec, pose, box_corners_3d(state, VEH, H_CAR))
        box = np.array([[u.min(), u.max(), v.min(), v.max()]])
        share = clip_boxes(box, spec)[1][0]
        if abs(share-0.5) < 0.01:
            continue
        seen = len(frame(spec, truth([(40.0, LANE[1]+dy)])).boxes) == 1
        assert seen == (share >= 0.5)
        kept += seen and share < 1
        dropped += (not seen) and share > 0
        partial += 0 < share < 1
    assert kept >= 4 and dropped >= 4 and partial >= 8


def test_side_cameras_see_a_car_alongside():
    hist = truth([(0.0, LANE[2]), (1.0, LANE[0])])
    left, right = frame(exact(LEFT), hist), frame(exact(RIGHT), hist)
    assert left.truth_id.tolist() == [100]
    assert right.truth_id.tolist() == [101]
    for fr, spec in ((left, LEFT), (right, RIGHT)):
        ul, ur, vt, vb = fr.boxes[0]
        assert 0.0 <= ul < ur <= spec.width_px and vt < spec.height_px/2 < vb
    for spec in (FRONT, REAR):
        assert len(frame(exact(spec), hist).boxes) == 0     # partly behind those cameras: skipped


def test_rear_camera_and_teammates():
    team = ((0.0, LANE[1]), (30.0, LANE[1]), (-25.0, LANE[1]))
    hist = truth([(-60.0, LANE[0])], team=team)
    assert frame(exact(FRONT), hist, car=0).truth_id.tolist() == [-3]
    assert sorted(frame(exact(REAR), hist, car=0).truth_id.tolist()) == [-4, 100]
    assert sorted(frame(exact(REAR), hist, car=1).truth_id.tolist()) == [-2]   # car 0 hides car 2 behind it
    assert -4 not in frame(exact(FRONT), hist, car=2).truth_id                   # never the carrying car itself


def test_braking_pitches_the_nose_down_and_boxes_up():
    spec = exact(FRONT)
    f = focal(spec)
    still = frame(spec, truth([(40.0, LANE[1])]))
    brake = frame(spec, truth([(40.0, LANE[1])], team_ax=(-4.9, 0.0, 0.0)))
    accel = frame(spec, truth([(40.0, LANE[1])], team_ax=(2.45, 0.0, 0.0)))
    assert brake.pitch == pytest.approx(radians(1.0)) and accel.pitch == pytest.approx(-radians(0.5))
    shift = brake.boxes[0, 2:]-still.boxes[0, 2:]
    assert shift == pytest.approx([-f*np.tan(radians(1.0))]*2, abs=0.2)
    assert np.all(accel.boxes[0, 2:] > still.boxes[0, 2:])
    assert brake.boxes[0, :2] == pytest.approx(still.boxes[0, :2], abs=0.1)
    # the other cars' pitch does not matter, and the residual noise has the specified spread
    assert frame(spec, truth([(40.0, LANE[1])], team_ax=(0.0, -6.0, 3.0))).pitch == 0.0
    noisy = replace(spec, pitch_sigma=radians(0.1))
    pitches = [frame(noisy, truth([(40.0, LANE[1])]), seed=k).pitch for k in range(400)]
    assert np.std(pitches) == pytest.approx(radians(0.1), rel=0.15)


def test_edge_noise_and_detection_rate():
    spec = replace(FRONT, pitch_sigma=0.0, false_rate=0.0)
    hist = truth([(50.0, LANE[1])])
    ref = frame(exact(FRONT), hist).boxes[0]
    rng = np.random.default_rng(11)
    frames = [simulate_camera(spec, 0, hist, 0.0, VEH, rng, H_CAR) for _ in range(2000)]
    hits = np.array([fr.boxes[0] for fr in frames if len(fr.boxes)])
    rate = len(hits)/len(frames)
    assert abs(rate-spec.pd) < 3*np.sqrt(spec.pd*(1-spec.pd)/len(frames))
    err = hits-ref
    assert np.abs(err.mean(0)).max() < 0.2
    assert err.std(0) == pytest.approx([spec.sigma_px]*4, rel=0.08)
    assert abs(np.corrcoef(err.T)[0, 1]) < 0.1                 # the edges are independent


def test_false_boxes():
    spec = replace(FRONT, pitch_sigma=0.0, false_rate=2.0)
    rng = np.random.default_rng(5)
    frames = [simulate_camera(spec, 0, truth(), 0.0, VEH, rng, H_CAR) for _ in range(1500)]
    counts = np.array([len(fr.boxes) for fr in frames])
    assert counts.mean() == pytest.approx(2.0, abs=0.12) and counts.var() == pytest.approx(2.0, rel=0.15)
    boxes = np.concatenate([fr.boxes for fr in frames])
    score = np.concatenate([fr.score for fr in frames])
    ids = np.concatenate([fr.truth_id for fr in frames])
    assert np.all(ids == -1) and np.all((score >= 0.3) & (score <= 0.7))
    assert np.all(boxes[:, 3] > spec.height_px/2)                       # standing on the road below the horizon
    assert np.all((boxes[:, [0, 2]] >= 0) & (boxes[:, [1, 3]] <= [spec.width_px, spec.height_px]))
    whole = (boxes[:, 0] > 0) & (boxes[:, 1] < spec.width_px)
    h, w = boxes[whole, 3]-boxes[whole, 2], boxes[whole, 1]-boxes[whole, 0]
    assert np.all(h >= spec.min_height_px-1e-9)
    assert np.all((w/h >= VEH.width/H_CAR-1e-9) & (w/h <= VEH.length/H_CAR+1e-9))
    assert len(frame(exact(FRONT), truth()).boxes) == 0


def dense_scene(seed=0):
    """A v9-like scene: 45 traffic cars on three lanes around the team, no overlaps."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in LANE:
        s = -150.0+np.cumsum(rng.uniform(8.0, 30.0, 15))
        rows += [(x, d+rng.uniform(-0.3, 0.3)) for x in s if not (d == LANE[1] and -40.0 < x < 50.0)]
    team = ((0.0, LANE[1]), (18.0, LANE[1]), (-18.0, LANE[1]))
    return truth(rows, team=team, team_ax=(-2.0, 1.0, 0.0))


def test_deterministic_given_the_seed():
    hist = dense_scene()
    for spec in default_cameras():
        a, b, c = frame(spec, hist, seed=7), frame(spec, hist, seed=7), frame(spec, hist, seed=8)
        for k in ('boxes', 'score', 'truth_id'):
            assert np.array_equal(getattr(a, k), getattr(b, k))
        assert a.pitch == b.pitch != c.pitch
    a = frame(FRONT, hist, seed=7)
    assert len(a.boxes) >= 3 and a.boxes.shape == (len(a.score), 4) and a.truth_id.dtype.kind == 'i'


def test_batched_cameras_equal_single_calls():
    """Without randomness that matters, one simulate_cameras call (one occlusion pass for all four cameras) gives
    the frames of four simulate_camera calls; with it, the same seed gives the same frames."""
    specs = tuple(exact(s) for s in default_cameras())
    occluded = 0
    for seed in (2, 3):
        hist = dense_scene(seed)
        for car in range(3):
            batch = simulate_cameras(specs, car, hist, 0.0, VEH, np.random.default_rng(0), H_CAR)
            free = simulate_cameras(tuple(replace(s, min_visible=0.0) for s in specs), car, hist, 0.0, VEH,
                                    np.random.default_rng(0), H_CAR)
            for a, b, spec in zip(batch, free, specs):
                one = frame(spec, hist, car=car)
                assert a.sensor == one.sensor == spec.name and a.car == car and a.pitch == one.pitch
                order = np.argsort(a.truth_id)
                assert np.array_equal(a.truth_id[order], np.sort(one.truth_id))
                assert np.array_equal(a.boxes[order], one.boxes[np.argsort(one.truth_id)])
                occluded += len(b.truth_id)-len(a.truth_id)
    assert occluded >= 10
    noisy = tuple(replace(s, false_rate=1.0) for s in default_cameras())
    a = simulate_cameras(noisy, 1, hist, 0.0, VEH, np.random.default_rng(4), H_CAR)
    b = simulate_cameras(noisy, 1, hist, 0.0, VEH, np.random.default_rng(4), H_CAR)
    for x, y in zip(a, b):
        assert x.pitch == y.pitch and all(np.array_equal(getattr(x, k), getattr(y, k))
                                          for k in ('boxes', 'score', 'truth_id'))


def test_project_boxes_marks_cars_behind_the_camera():
    pose = sensor_pose(FRONT.mount, np.array([0.0, LANE[1], 0.0]), VEH)
    states = np.array([[30.0, LANE[1], 0.0], [-30.0, LANE[1], 0.0], [2.5, LANE[2], 0.0]])
    boxes, ok = project_boxes(FRONT, pose, states, VEH, H_CAR)
    assert ok.tolist() == [True, False, False]
    assert project_boxes(FRONT, pose, np.zeros((0, 3)), VEH, H_CAR)[0].shape == (0, 4)


def test_speed_dense_scene():
    hist = dense_scene(1)
    rng = np.random.default_rng(0)
    specs = default_cameras()
    simulate_cameras(specs, 0, hist, 0.0, VEH, rng, H_CAR)
    best = np.inf
    for _ in range(3):           # best of three: robust to other load on the machine
        t0 = time.perf_counter()
        for _ in range(10):
            for car in range(3):
                simulate_cameras(specs, car, hist, 0.0, VEH, rng, H_CAR)
        best = min(best, (time.perf_counter()-t0)/10)
    assert best < 0.006, f'{best*1e3:.2f} ms per sim step for the team\'s 12 frames'
