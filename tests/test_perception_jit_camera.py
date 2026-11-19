"""Camera processing's compiled path (processing/_camera_jit.py) against its numpy reference (processing/camera.py,
NOHESI_JIT=0): the same detections, counts, truncation flags and scores, and z, R and the init (xy, P) to rounding, on
simulated frames of v9-like traffic around the team (the simulator's boxes, false boxes, teammates, truncation) and
on synthetic edge cases (cut boxes, a car beside, degenerate boxes, an ideal camera, the options)."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import _jit
from nohesi.perception.config import default_cameras
from nohesi.perception.processing import camera as C
from nohesi.perception.sim.camera import simulate_cameras
from nohesi.perception.types import CameraFrame, EgoState, TruthHistory
from nohesi.road import Road
from nohesi.world import Snapshot

pytestmark = pytest.mark.skipif(not _jit.ENABLED, reason='numba off (NOHESI_JIT=0 or not installed)')

SC = scenario_v9()
CAR = SC.vehicle
ROAD = Road(SC.road)
H = 1.45
CAMERAS = default_cameras()
FRONT, REAR, LEFT, RIGHT = CAMERAS
LANE = ROAD.lane_center(1)
RTOL = 1e-9            # relative to an array's largest entry. The refinement's 4 x 4 solve (LU in the kernel, LAPACK
                       # in numpy) differs by ~1e-12 on P, times the condition number of S: an ideal camera's 1e-3 px
                       # floor under its pixel-sized nuisance terms takes that to ~1e-9 (IDEAL_RTOL)
IDEAL_RTOL = 1e-7


def run(path, *args, **kw):
    """process_cameras on the compiled (True) or the numpy (False) path."""
    old = C._JIT
    C._JIT = path
    try:
        return C.process_cameras(*args, **kw)
    finally:
        C._JIT = old


def close(a, b, scale, rtol=RTOL):
    """a == b to rtol relative of `scale` (the array's largest entry), nan where the other is."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return a.shape == b.shape and np.array_equal(np.isfinite(a), np.isfinite(b)) and \
        np.all(np.abs(np.nan_to_num(a-b)) <= rtol*scale)


def assert_same(*args, rtol=RTOL, **kw):
    """The compiled and numpy paths give the same detections; returns their number."""
    got, want = run(True, *args, **kw), run(False, *args, **kw)
    assert len(got) == len(want)
    total = 0
    for g_frame, w_frame in zip(got, want):
        assert len(g_frame) == len(w_frame)
        for g, w in zip(g_frame, w_frame):
            assert (g.kind, g.sensor, g.car, g.time, g.sensor_pose, g.ego) == (w.kind, w.sensor, w.car, w.time,
                                                                              w.sensor_pose, w.ego)
            assert np.array_equal(g.z, w.z)
            assert close(g.R, w.R, np.abs(w.R).max(), rtol)
            (gx, gP), (wx, wP) = g.meta['init'], w.meta['init']
            assert close(gx, wx, np.abs(wx).max(), rtol)
            assert close(gP, wP, np.abs(wP).max(), rtol)
            assert g.meta['truncated'].dtype == bool and np.array_equal(g.meta['truncated'], w.meta['truncated'])
            assert (g.meta['score'], g.meta['scan'], set(g.meta)) == (w.meta['score'], w.meta['scan'], set(w.meta))
            total += 1
    return total


def snapshot(t, team, traffic):
    """A true Snapshot: team rows [s, d, yaw, v], traffic rows [s, d, lateral speed, v]."""
    team, traffic = np.asarray(team, float), np.asarray(traffic, float).reshape(-1, 4)
    return Snapshot(time=float(t), team=team, team_steer=np.zeros(3), team_s=team[:, 0].copy(),
                    team_d=team[:, 1].copy(), traffic_s=traffic[:, 0].copy(), traffic_d=traffic[:, 1].copy(),
                    traffic_v=traffic[:, 3].copy(), traffic_id=np.arange(len(traffic)), collisions=(), done=False,
                    traffic_vd=traffic[:, 2].copy())


def scene(seed, n=45):
    """One step's twelve frames (three team cars, four cameras each) of v9-like traffic: n cars over three lanes,
    some changing lanes, the team bunched (teammates fill each other's cameras) with braking pitch."""
    rng = np.random.default_rng(seed)
    lanes = ROAD.lane_center(rng.integers(0, 3, n))+rng.normal(0.0, 0.3, n)
    traffic = np.column_stack((rng.uniform(-80, 150, n), lanes, rng.normal(0.0, 0.8, n), rng.uniform(25, 35, n)))
    team = np.array([[0.0, LANE, 0.0, 30.0], [rng.uniform(-20, -8), LANE+rng.choice((-3.6, 0.0, 3.6)), 0.01, 30.0],
                     [rng.uniform(5, 15), LANE+3.6, rng.normal(0.0, 0.03), 30.0]])
    ax = rng.uniform(-3.0, 1.5, 3)
    hist = TruthHistory()
    hist.push(snapshot(0.0, team, traffic), tuple(ax))
    frames = [fr for c in range(3) for fr in simulate_cameras(CAMERAS, c, hist, 0.0, CAR, rng, H)]
    jobs = [(c, s) for c in range(3) for s in CAMERAS]
    egos = [EgoState(0.0, *team[c, :3], 30.0, ax=ax[c], pitch=s.pitch_per_accel*(-ax[c])) for c, s in jobs]
    mates = [np.delete(team[:, :3], c, 0) for c, _ in jobs]
    return frames, [s for _, s in jobs], egos, mates


@pytest.mark.parametrize('seed', range(8))
def test_simulated_frames(seed):
    frames, specs, egos, mates = scene(seed)
    assert assert_same(frames, specs, egos, mates, CAR, H) > 10


def test_simulated_frames_with_the_options():
    frames, specs, egos, mates = scene(11)
    assert assert_same(frames, specs, egos, mates, CAR, H, refine=False) > 10
    assert assert_same(frames, specs, egos, mates, CAR, H, iters=1, pitch_noise=0.004, pitch_extra=0.0) > 10
    assert assert_same(frames, specs, egos, [None]*len(frames), CAR, H, heading_sigma=0.0) > 10
    ideal = [replace(s, sigma_px=0.0, pitch_sigma=0.0) for s in specs]
    assert assert_same(frames, ideal, egos, mates, CAR, H, rtol=IDEAL_RTOL) > 10


def frame(spec, boxes, t=2.0, score=None):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return CameraFrame(spec.name, 0, t, boxes, np.linspace(0.5, 0.9, len(boxes)) if score is None else score)


def test_synthetic_edge_cases():
    """Boxes cut at the border (one side, two sides, the bottom), a car filling the side camera, a teammate's box,
    degenerate and empty frames, a frame without scores and an ego at another time."""
    ego = EgoState(time=2.0, x=0.0, y=LANE, yaw=0.01, vx=30.0, r=0.02, pitch=0.004)
    early = replace(ego, time=1.95)
    W, Hp = FRONT.width_px, FRONT.height_px
    front = [[1.5, 900.0, 400.0, 600.0], [2.5, W-1.5, 1.5, Hp-2.5], [800.0, 1100.0, 300.0, Hp],
             [900.0, 960.0, 520.0, 560.0], [0.0, 300.0, 380.0, Hp], [100.0, 90.0, 400.0, 600.0],
             [np.nan, 200.0, 400.0, 600.0], [100.0, 200.0, 600.0, 600.0], [W-200.0, W, 450.0, 620.0]]
    side = [[0.0, LEFT.width_px, 10.0, LEFT.height_px], [300.0, 700.0, 350.0, 600.0], [0.0, 400.0, 300.0, 700.0]]
    rear = [[600.0, 700.0, 500.0, 580.0], [0.0, 120.0, 450.0, 650.0]]
    frames = [frame(FRONT, front), frame(LEFT, side), frame(REAR, rear, score=None), frame(RIGHT, np.zeros((0, 4))),
              CameraFrame(FRONT.name, 1, 2.0, np.array(front[:4]), None)]
    specs = [FRONT, LEFT, REAR, RIGHT, FRONT]
    mates = [np.array([[25.0, LANE, 0.0]]), np.array([[0.8, LANE+3.6, 0.0], [-12.0, LANE, 0.0]]), None, [],
             [[-30.0, LANE, 0.0, 30.0]]]
    egos = [ego, early, ego, ego, early]
    assert assert_same(frames, specs, egos, mates, CAR, H) >= 10
    assert assert_same(frames, specs, egos, mates, CAR, H, refine=False) >= 10
    assert assert_same([frames[3]], [RIGHT], [ego], [None], CAR, H) == 0


def test_kernels_match_numpy_pieces():
    """The kernel's box model, cue estimate and refinement one by one against camera.box_model, cue_init and
    refine_init (the compiled kernels, called directly)."""
    from nohesi.perception.processing import _camera_jit as K
    from nohesi.perception.geometry import sensor_pose
    from nohesi.geometry import cg_offsets
    rng = np.random.default_rng(3)
    corners = C._local_corners(CAR, H)
    assert K.CAM_FIELDS == C._Cam._fields
    tune = np.array([getattr(C, name) for name in K.TUNE_FIELDS])
    car = np.array([CAR.length, CAR.width, *cg_offsets(CAR)])
    for spec in CAMERAS:
        pose = sensor_pose(spec.mount, (0.0, LANE, 0.02), CAR)
        cam = np.array(C._cam(spec, pose), float)
        a = pose[2]+rng.uniform(-0.6, 0.6, 30)*spec.hfov
        r = rng.uniform(1.0, 80.0, 30)
        xy = np.column_stack((pose[0]+r*np.cos(a), pose[1]+r*np.sin(a)))
        heading, pitch = rng.normal(0.0, 0.05, 30), rng.normal(0.0, 0.01, 30)
        want = C.box_model(spec, pose, xy, heading, pitch, CAR, H)
        out, s8 = np.empty(4), np.empty((3, 8))
        lxs, lys, R, kz = np.empty(8), np.empty(8), np.empty((3, 3)), np.empty((3, 8))
        for i in range(30):
            K._prepare(cam, heading[i], pitch[i], corners, lxs, lys, R, kz)
            K._evaluate(cam, np.cos(cam[K.PSI]), np.sin(cam[K.PSI]), lxs, lys, R, kz, xy[i, 0], xy[i, 1], C._EDGES,
                        C.NEAR, out, s8[0], s8[1], s8[2])
            assert close(out, want[i], np.nanmax(np.abs(want[i]), initial=1.0)), (spec.name, i, out, want[i])
        boxes = np.clip(want, 0.0, (spec.width_px, spec.width_px, spec.height_px, spec.height_px))
        ok = np.isfinite(boxes).all(1) & (boxes[:, 1] > boxes[:, 0]+1) & (boxes[:, 3] > boxes[:, 2]+1)
        boxes = boxes[ok]
        fit = ~np.column_stack((boxes[:, 0] <= 5, boxes[:, 1] >= spec.width_px-5, boxes[:, 2] <= 5,
                                boxes[:, 3] >= spec.height_px-5))
        sp = spec.pitch_sigma+C.PITCH_EXTRA
        xc, Pc = C.cue_init(boxes, fit, spec, pose, 0.003, CAR, H, sp)
        xr, Pr, okr = C.refine_init(boxes, fit, xc, spec, pose, 0.003, CAR, H, sp)
        x1, P1, x2, P2 = np.empty(2), np.empty((2, 2)), np.empty(2), np.empty((2, 2))
        for i in range(len(boxes)):
            K._cue(boxes[i], fit[i], cam, 0.003, sp, car, H, C.HEADING_SIGMA, C._STEPS, x1, P1, K._cue_work(6))
            assert close(x1, xc[i], np.abs(xc[i]).max()) and close(P1, Pc[i], np.abs(Pc[i]).max())
            assert K._refine(boxes[i], fit[i], xc[i], cam, 0.003, sp, corners, CAR.length, C.HEADING_SIGMA, C.ITERS,
                             C._EDGES, C._STEPS, tune, C._CUT_SIDE, x2, P2, K._refine_work()) == okr[i]
            if okr[i]:
                assert close(x2, xr[i], np.abs(xr[i]).max()) and close(P2, Pr[i], np.abs(Pr[i]).max())
