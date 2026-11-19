"""Camera processing (nohesi/perception/processing/camera.py): the box model against the contract's projection, the
single-frame position (meta['init']) consistent with its covariance (NEES) ahead, behind and beside, the pitch
effect, teammate removal, truncation flags, the detection fields and speed. Frames are synthesized here from the
contract geometry (not from the simulator): boxes are the extent of the projected 3D corners, clipped to the image,
with pixel noise, clipped again, as a detector reports them."""
import time
from dataclasses import replace
from math import radians
import numpy as np
import pytest
from scipy.stats import chi2
from nohesi.config import scenario_v9
from nohesi.geometry import cg_offsets
from nohesi.perception.config import default_cameras
from nohesi.perception.geometry import box_corners_3d, focal, project, sensor_pose
from nohesi.perception.processing.camera import (PITCH_EXTRA, box_model, cue_init, iou, mate_boxes, pitch_jacobian,
                                                 process_camera)
from nohesi.perception.types import CameraFrame, EgoState

VEH = scenario_v9().vehicle
H_CAR = 1.45
FRONT, REAR, LEFT, RIGHT = default_cameras()
LANE = (1.8, 5.4, 9.0)
EGO = EgoState(time=2.0, x=0.0, y=LANE[1], yaw=0.0, vx=30.0)


def pose_of(spec, ego=EGO):
    return sensor_pose(spec.mount, (ego.x, ego.y, ego.yaw), VEH)


def true_boxes(spec, cars, pitch=0.0, ego=EGO):
    """Unclipped boxes (n, 4) of cars (n, 3) [x, y, heading] by geometry.project of geometry.box_corners_3d; pitch a
    scalar or one per car."""
    u, v, depth = project(spec, pose_of(spec, ego), box_corners_3d(np.asarray(cars, float), VEH, H_CAR),
                          np.reshape(pitch, (-1, 1)))
    assert (depth > 0.5).all()
    return np.stack((u.min(1), u.max(1), v.min(1), v.max(1)), -1)


def detector(spec, boxes, rng=None):
    lim = (spec.width_px, spec.width_px, spec.height_px, spec.height_px)
    b = np.clip(boxes, 0.0, lim)
    return b if rng is None else np.clip(b+rng.normal(0.0, spec.sigma_px, b.shape), 0.0, lim)


def frame(spec, boxes, ego=EGO, score=None):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return CameraFrame(spec.name, 0, ego.time, boxes, np.full(len(boxes), 0.9) if score is None else score)


def lane_change_headings(rng, n, share=0.2):
    """Headings to the road of v9 traffic: most cars keep their lane (0); `share` are changing lanes, at atan(v_d/v)
    of the quintic profile (one 3.6 m lane in 3 s) at a uniform phase, 25 m/s: up to 5 deg either way."""
    tau = rng.random(n)
    h = np.arctan(30*tau**2*(1-tau)**2*3.6/3.0/25.0)*rng.choice((-1.0, 1.0), n)
    return np.where(rng.random(n) < share, h, 0.0)


def init_errors(spec, car, n=300, seed=0, ego=EGO, refine=True, pitch_error=True):
    """n noisy frames of one car (x, y): v9 headings and a true pitch around ego.pitch with the processing's own
    pitch sigma; returns the init errors (n, 2), covariances (n, 2, 2), NEES (n,) and the detections."""
    rng = np.random.default_rng(seed)
    cars = np.column_stack((np.tile(car, (n, 1)), lane_change_headings(rng, n)))
    pitch = ego.pitch+(rng.normal(0.0, spec.pitch_sigma+PITCH_EXTRA, n) if pitch_error else np.zeros(n))
    dets = process_camera(frame(spec, detector(spec, true_boxes(spec, cars, pitch, ego), rng), ego), spec, ego, None,
                          VEH, H_CAR, refine=refine)
    assert len(dets) == n
    err = np.array([d.meta['init'][0] for d in dets])-np.asarray(car, float)
    P = np.array([d.meta['init'][1] for d in dets])
    return err, P, np.einsum('ni,nij,nj->n', err, np.linalg.inv(P), err), dets


def consistent(nees, inside=0.88):
    """The init covariance is honest: mean NEES near the 2 of a 2-dof chi-square (allowing the non-Gaussian headings
    and first-order errors some room), and most errors inside the 95 % ellipse."""
    return 0.5 < nees.mean() < 2.6 and np.mean(nees < chi2.ppf(0.95, 2)) > inside


def test_box_model_matches_the_contract_projection():
    rng = np.random.default_rng(0)
    for spec in default_cameras():
        ego = replace(EGO, yaw=radians(3.0))
        pose = pose_of(spec, ego)
        a = pose[2]+rng.uniform(-0.3, 0.3, 40)*spec.hfov
        r = rng.uniform(8.0, 60.0, 40)
        cars = np.column_stack((pose[0]+r*np.cos(a), pose[1]+r*np.sin(a), rng.uniform(-0.1, 0.1, 40)))
        pitch = rng.uniform(-0.02, 0.02, 40)
        want = true_boxes(spec, cars, pitch, ego)
        assert box_model(spec, pose, cars[:, :2], cars[:, 2], pitch, VEH, H_CAR) == pytest.approx(want, abs=1e-9)


def test_box_model_cuts_the_part_behind_the_camera():
    """A car in the next lane alongside the front camera reaches behind its image plane: the box of the part in front
    runs past the image's left edge (a detector reports it truncated); a car wholly behind has no box."""
    pose = pose_of(FRONT)
    car = (pose[0]+0.5, LANE[2], 0.0)
    b = box_model(FRONT, pose, car[:2], car[2], 0.0, VEH, H_CAR)
    assert np.isfinite(b).all() and b[0] < -1e3 and b[1] < 0
    assert np.isnan(box_model(FRONT, pose, (pose[0]-10.0, LANE[1]), 0.0, 0.0, VEH, H_CAR)).all()


@pytest.mark.parametrize('gap', [20.0, 50.0, 100.0])
def test_init_is_consistent_for_a_car_ahead(gap):
    err, P, nees, dets = init_errors(FRONT, (gap, LANE[1]), seed=int(gap))
    assert consistent(nees), (nees.mean(), np.mean(nees < 5.99))
    sig = np.sqrt(P[:, [0, 1], [0, 1]]).mean(0)
    assert sig[0] > (3 if gap < 30 else 20)*sig[1]        # long along the ray (here the road), narrow across
    assert np.abs(err.mean(0)) == pytest.approx(0, abs=0.25*sig[0]+0.02)
    w, v = np.linalg.eigh(P[0])
    assert abs(v[0, 1]) > 0.99 and w[1] > 9*w[0]


def test_init_is_consistent_for_cars_off_the_ray_and_behind():
    for spec, car in ((FRONT, (20.0, LANE[2])), (FRONT, (50.0, LANE[0])), (REAR, (-30.0, LANE[1])),
                      (REAR, (-25.0, LANE[2]))):
        err, P, nees, _ = init_errors(spec, car, seed=7)
        assert consistent(nees), (spec.name, car, nees.mean())


def test_init_is_consistent_beside_a_side_camera():
    """Left camera: a car two lanes over abeam it (5.3 m off), and one in the next lane ahead of it whose front runs
    out of the image (right edge truncated); right camera: a car two lanes over, behind."""
    ego0 = replace(EGO, y=LANE[0])
    ego2 = replace(EGO, y=LANE[2])
    for spec, ego, car, cut in ((LEFT, ego0, (0.8, LANE[2]), (False,)*4), (LEFT, EGO, (3.0, LANE[2]), (False, True, False, False)),
                                (RIGHT, ego2, (-2.0, LANE[0]), (False,)*4)):
        err, P, nees, dets = init_errors(spec, car, seed=3, ego=ego)
        assert consistent(nees), (spec.name, car, nees.mean())
        assert np.mean([tuple(d.meta['truncated']) == cut for d in dets]) > 0.7
        assert np.sqrt(np.linalg.eigvalsh(P).max(1)).mean() < 0.15      # within a few metres the init is sharp


def test_cue_method_alone_is_consistent_on_the_ray():
    """The two-cue estimate (refine=False), design step 3, for cars on the camera's axis."""
    for spec, car, ego in ((FRONT, (20.0, LANE[1]), EGO), (FRONT, (50.0, LANE[1]), EGO), (FRONT, (100.0, LANE[1]), EGO),
                           (REAR, (-30.0, LANE[1]), EGO), (LEFT, (0.8, LANE[2]), replace(EGO, y=LANE[0]))):
        err, P, nees, _ = init_errors(spec, car, seed=11, ego=ego, refine=False)
        assert consistent(nees), (spec.name, car, nees.mean())


def test_refinement_removes_the_cue_bias_off_the_ray():
    """A car 20 m ahead in the next lane: its visible flank widens the box to one side, so the box centre's bearing
    misses the car's centre line by ~0.2 m (ten times the bearing's pixel noise across the ray, and the cue estimate
    is inconsistent); the box-model fit is not biased."""
    car = (20.0, LANE[2])
    cue, Pc, nees_cue, _ = init_errors(FRONT, car, seed=5, refine=False)
    fit, Pf, nees, _ = init_errors(FRONT, car, seed=5)
    assert abs(cue[:, 1].mean()) > 0.15 > 0.03 > abs(fit[:, 1].mean())
    assert nees_cue.mean() > 3.0 and consistent(nees)


def test_pitch_effect():
    """1 deg of braking pitch at 50 m ahead: with the pitch known (ego.pitch) the init is unbiased; with it ignored the
    ground-plane cue alone would put the car ~60 m too far (dr ~ r^2 dtheta / h to first order), but the fit leans on
    the pitch-free size cue. Known exactly, the pitch makes the ground cue count and the along-ray sigma shrinks; at
    1 deg of pitch sigma (the ground cue alone: r^2 sigma / h ~ 30 m) the size cue holds it where it was."""
    car, pitch = (50.0, LANE[1]), radians(1.0)
    box = detector(FRONT, true_boxes(FRONT, [(*car, 0.0)], pitch))
    known = process_camera(frame(FRONT, box), FRONT, replace(EGO, pitch=pitch), None, VEH, H_CAR)[0]
    xy, P = known.meta['init']
    assert abs(xy[0]-car[0]) < 0.25*np.sqrt(P[0, 0]) and abs(xy[1]-car[1]) < 0.02

    pose = pose_of(FRONT)
    f, cy, h = focal(FRONT), FRONT.height_px/2, pose[3]
    ground_only = pose[0]+h*f/(box[0, 3]-cy)-cg_offsets(VEH)[0]       # flat ground, pitch 0: rear face, then the CG
    off = abs(ground_only-car[0])
    assert off > 40.0
    ignored = process_camera(frame(FRONT, box), FRONT, EGO, None, VEH, H_CAR)[0]
    assert abs(ignored.meta['init'][0][0]-car[0]) < 0.1*off
    cue = process_camera(frame(FRONT, box), FRONT, EGO, None, VEH, H_CAR, refine=False)[0]
    assert abs(cue.meta['init'][0][0]-car[0]) < 0.1*off

    def cov(spec, **kw):
        return process_camera(frame(spec, box), spec, replace(EGO, pitch=pitch), None, VEH, H_CAR, **kw)[0].meta['init'][1]
    for refine in (True, False):
        default = cov(FRONT, refine=refine)
        exact = cov(replace(FRONT, pitch_sigma=0.0), pitch_extra=0.0, refine=refine)
        shaky = cov(replace(FRONT, pitch_sigma=radians(1.0)), refine=refine)
        assert np.sqrt(exact[0, 0]) < 0.9*np.sqrt(default[0, 0]) < 0.9*np.sqrt(shaky[0, 0]) < np.sqrt(default[0, 0])
        assert exact[1, 1] == pytest.approx(default[1, 1], rel=0.05) and shaky[1, 1] == pytest.approx(default[1, 1], rel=0.05)


def test_teammate_boxes_are_dropped():
    rng = np.random.default_rng(4)
    mates = np.array([[25.0, LANE[1], 0.0], [-300.0, LANE[1], 0.0]])
    car = (40.0, LANE[2], 0.0)
    boxes = detector(FRONT, true_boxes(FRONT, [mates[0], car]), rng)
    dets = process_camera(frame(FRONT, boxes), FRONT, EGO, mates, VEH, H_CAR)
    assert len(dets) == 1 and dets[0].z == pytest.approx(boxes[1])
    assert len(process_camera(frame(FRONT, boxes), FRONT, EGO, None, VEH, H_CAR)) == 2
    assert len(process_camera(frame(FRONT, boxes), FRONT, EGO, mates[:, :3].tolist(), VEH, H_CAR)) == 1

    beside = np.array([[EGO.x+0.8, LANE[2], 0.0]])                    # fills the left camera, cut on both sides
    side = detector(LEFT, true_boxes(LEFT, beside), rng)
    assert side[0, 0] == 0.0 and side[0, 1] == LEFT.width_px
    assert process_camera(frame(LEFT, side), LEFT, EGO, beside, VEH, H_CAR) == []
    assert iou(side, mate_boxes(LEFT, pose_of(LEFT), beside, VEH, H_CAR))[0, 0] > 0.9


def test_truncation_flags():
    pose = pose_of(FRONT)
    rear = -cg_offsets(VEH)[0]
    cars = [(30.0, LANE[1], 0.0), (pose[0]+7.0, LANE[2], 0.0), (pose[0]+7.0, LANE[0], 0.0), (pose[0]+rear+3.5, LANE[1], 0.0)]
    want = [(False, False, False, False), (True, False, False, False), (False, True, False, False),
            (False, False, False, True)]
    dets = process_camera(frame(FRONT, detector(FRONT, true_boxes(FRONT, cars))), FRONT, EGO, None, VEH, H_CAR)
    assert [tuple(d.meta['truncated']) for d in dets] == want
    for d in dets:
        xy, P = d.meta['init']
        assert np.isfinite(xy).all() and np.all(np.linalg.eigvalsh(P) > 0)
    W, H = FRONT.width_px, FRONT.height_px
    near = [[1.5, 900.0, 400.0, 600.0], [2.5, W-1.5, 1.5, H-2.5]]
    flags = [tuple(d.meta['truncated']) for d in process_camera(frame(FRONT, near), FRONT, EGO, None, VEH, H_CAR)]
    assert flags == [(True, False, False, False), (False, True, True, False)]


def test_truncated_car_is_still_located():
    """A car 9 m ahead of the front camera in the next lane, its left side out of the image: the known size places it.
    At 6 m its bottom is cut too, leaving the right edge and a top edge 5 cm above the camera's height for range: the
    cut edges bound the fit (the car reaches past them) and the mean NEES stays honest, but that top-edge cue is
    strongly nonlinear in range (dr ~ r^2 dpitch / 5 cm), so a 2.5 sigma pitch error pulls the estimate in to where the
    linearized covariance is too sharp: a heavier tail than a Gaussian's."""
    for gap, cut, worst, inside in ((9.0, (True, False, False, False), 0.6, 0.88), (6.0, (True, False, False, True), 4.0, 0.8)):
        err, P, nees, dets = init_errors(FRONT, (pose_of(FRONT)[0]+gap, LANE[2]), n=200, seed=9)
        assert np.mean([tuple(d.meta['truncated']) == cut for d in dets]) > 0.6   # a cut edge's noise: within 2 px or not
        assert consistent(nees, inside) and np.abs(err).max() < worst, (gap, nees.mean(), np.abs(err).max())


def test_detection_fields_and_ego_moved_to_frame_time():
    early = replace(EGO, time=1.9, x=-3.0)
    box = detector(FRONT, true_boxes(FRONT, [(40.0, LANE[1], 0.0)]))
    (d,) = process_camera(frame(FRONT, box, score=np.array([0.77])), FRONT, early, None, VEH, H_CAR)
    assert (d.kind, d.sensor, d.car, d.time) == ('camera_box', FRONT.name, 0, 2.0)
    assert d.ego.time == 2.0 and d.ego.x == pytest.approx(0.0)
    assert d.sensor_pose == pytest.approx(pose_of(FRONT))
    j = pitch_jacobian(box, FRONT)[0]       # the frame's pitch noise moves the edges together (a front camera: no roll)
    assert d.z == pytest.approx(box[0])
    assert d.R == pytest.approx(FRONT.sigma_px**2*np.eye(4)+FRONT.pitch_sigma**2*np.outer(j, j))
    assert set(d.meta) == {'init', 'truncated', 'score', 'scan'}
    assert d.meta['score'] == 0.77 and d.meta['scan'] == 2.0
    xy, P = d.meta['init']
    assert xy.shape == (2,) and P.shape == (2, 2) and np.allclose(P, P.T) and np.all(np.linalg.eigvalsh(P) > 0)
    assert d.meta['truncated'].dtype == bool and d.meta['truncated'].shape == (4,)


def test_empty_and_degenerate_boxes():
    assert process_camera(frame(FRONT, np.zeros((0, 4))), FRONT, EGO, None, VEH, H_CAR) == []
    bad = [[100.0, 90.0, 400.0, 600.0], [np.nan, 200.0, 400.0, 600.0], [100.0, 200.0, 600.0, 600.0]]
    assert process_camera(frame(FRONT, bad), FRONT, EGO, None, VEH, H_CAR) == []


def test_noise_free_camera():
    spec = replace(FRONT, sigma_px=0.0, pitch_sigma=0.0)
    box = detector(spec, true_boxes(spec, [(30.0, LANE[2], 0.0)]))
    (d,) = process_camera(frame(spec, box), spec, EGO, None, VEH, H_CAR)
    xy, P = d.meta['init']
    assert np.all(np.linalg.eigvalsh(d.R) > 0) and np.all(np.linalg.eigvalsh(P) > 0)
    assert xy == pytest.approx((30.0, LANE[2]), abs=0.3)


def test_cue_init_without_depth_cues_is_wide():
    """A box whose bottom is cut has neither depth cue: the fallback depth from the width gets a wide covariance."""
    pose = pose_of(FRONT)
    box = np.array([[800.0, 1100.0, 300.0, FRONT.height_px]])
    xy, P = cue_init(box, np.array([[True, True, True, False]]), FRONT, pose, 0.0, VEH, H_CAR, radians(0.35))
    depth = np.linalg.norm(xy[0]-pose[:2])
    assert np.sqrt(np.linalg.eigvalsh(P[0]).max()) > 0.9*depth


def test_speed():
    rng = np.random.default_rng(6)
    cars = [(15.0+12*i, LANE[i % 3], 0.0) for i in range(8)]
    fr = frame(FRONT, detector(FRONT, true_boxes(FRONT, cars), rng))
    mates = np.array([[-20.0, LANE[1], 0.0], [-40.0, LANE[1], 0.0]])
    process_camera(fr, FRONT, EGO, mates, VEH, H_CAR)
    n = 100
    t0 = time.perf_counter()
    for _ in range(n):
        dets = process_camera(fr, FRONT, EGO, mates, VEH, H_CAR)
    per = (time.perf_counter()-t0)/n
    assert len(dets) == 8 and per < 5e-3, per
