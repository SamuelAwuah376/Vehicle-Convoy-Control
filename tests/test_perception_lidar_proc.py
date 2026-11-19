"""Lidar processing (nohesi.perception.processing.lidar): de-skew, segmentation, merging, box fits and covariances on
synthetic sweeps ray-cast here beam by beam with geometry.raycast (each beam at its own time, the carrying car and
the other cars moving)."""
import time
from dataclasses import replace
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from nohesi.config import scenario_v9
from nohesi.geometry import body_centre, cg_offsets
from nohesi.road import Road
from nohesi.perception.config import default_lidars
from nohesi.perception.geometry import car_boxes, raycast, sensor_pose, to_sensor, wrap
from nohesi.perception.types import EgoState, LidarScan
from nohesi.perception.processing.lidar import deskew, fit_box, lshape_heading, process_lidar, refit

SCEN = scenario_v9()
VEH = SCEN.vehicle
ROAD = Road(SCEN.road)
LANE = SCEN.road.lane_width
FRONT, REAR = default_lidars()
V_EGO = 30.0


def twist_motion(x0, y0, yaw0, vx, vy=0.0, r=0.0):
    """Exact constant-twist motion of a CG (closed form: a circle, or a line for r = 0) -> ego_at(t)."""
    def at(t):
        yaw = yaw0+r*t
        if r == 0.0:
            c, s = np.cos(yaw0), np.sin(yaw0)
            x, y = x0+(vx*c-vy*s)*t, y0+(vx*s+vy*c)*t
        else:
            x = x0+(vx*(np.sin(yaw)-np.sin(yaw0))+vy*(np.cos(yaw)-np.cos(yaw0)))/r
            y = y0+(vx*(np.cos(yaw0)-np.cos(yaw))+vy*(np.sin(yaw)-np.sin(yaw0)))/r
        return EgoState(float(t), float(x), float(y), float(yaw), vx, vy, r)
    return at


def cars(states, speeds=None):
    """cars_at(t): (M, 3) CG states of cars driving straight along their headings from `states` at t = 0."""
    S = np.atleast_2d(np.asarray(states, float))
    v = np.full(len(S), V_EGO) if speeds is None else np.asarray(speeds, float)
    vel = np.column_stack((v*np.cos(S[:, 2]), v*np.sin(S[:, 2]), np.zeros(len(S))))
    return lambda t: S+vel*t


def sweep(spec, ego_at, cars_at, t0=0.0, rng=None, sigma=None, dropout=True, boxes_at=None, window=True):
    """A sweep cast beam by beam at each beam's time against the cars (cars_at(t) -> (M, 3) CG states) plus any extra
    polygons (boxes_at(t) -> (k, 4, 2)). With rng: range noise (spec.range_sigma unless sigma) and dropout
    (spec.dropout plus far_dropout reached linearly at range_max). Returns the scan and the true hit points (n, 2).
    window: cast only the beams near some box (the rest return nothing either way)."""
    n = int(round(spec.fov/spec.beam_spacing))+1
    az = np.linspace(-spec.fov/2, spec.fov/2, n)
    T = 1/spec.rate
    t = t0+np.linspace(0.0, T, n)
    boxes = lambda tt: np.concatenate((car_boxes(cars_at(tt), VEH), np.zeros((0, 4, 2)) if boxes_at is None
                                       else np.asarray(boxes_at(tt), float).reshape(-1, 4, 2)))
    cast = np.ones(n, bool)
    if window:
        cast[:] = False
        for tt in (t0, t0+T):
            e = ego_at(tt)
            p = sensor_pose(spec.mount, (e.x, e.y, e.yaw), VEH)
            for tb in (t0, t0+T):
                for box in boxes(tb):
                    a = wrap(np.arctan2(box[:, 1]-p[1], box[:, 0]-p[0])-p[2])
                    if np.ptp(a) > np.pi:
                        cast[:] = True
                    cast |= (az >= a.min()-np.radians(3)) & (az <= a.max()+np.radians(3))
    rng_, hits = np.full(n, np.nan), np.full((n, 2), np.nan)
    for i in np.flatnonzero(cast):
        e = ego_at(t[i])
        p = sensor_pose(spec.mount, (e.x, e.y, e.yaw), VEH)
        a = p[2]+az[i]
        r, _ = raycast(p[:2], [a], boxes(t[i]), spec.range_max)
        if np.isfinite(r[0]):
            rng_[i] = r[0]
            hits[i] = p[0]+r[0]*np.cos(a), p[1]+r[0]*np.sin(a)
    if rng is not None:
        rng_ = rng_+(spec.range_sigma if sigma is None else sigma)*rng.standard_normal(n)
        if dropout:
            pd = spec.dropout+spec.far_dropout*np.minimum(np.nan_to_num(rng_)/spec.range_max, 1.0)
            rng_[rng.random(n) < pd] = np.nan
    return LidarScan(spec.name, 1, t0, t0+T, az, rng_, t), hits


def truth(cars_at, t):
    S = cars_at(t)
    return np.column_stack((body_centre(S, VEH), S[:, 2]))


def nearest(dets, cars_at):
    """For each car, the detection whose centre is nearest its true body centre at the detection's time, and that
    distance."""
    if not dets:
        return [(None, np.inf)]*len(cars_at(0.0))
    d = np.array([np.hypot(*(truth(cars_at, det.time)[:, :2]-det.z[:2]).T) for det in dets])   # (dets, cars)
    return [(dets[int(np.argmin(d[:, k]))], float(d[:, k].min())) for k in range(d.shape[1])]


def run(spec, ego_at, cars_at, mates=None, refine=True, **kw):
    scan, _ = sweep(spec, ego_at, cars_at, **kw)
    return process_lidar(scan, spec, ego_at, mates, VEH, ROAD, refine=refine), scan


def check_box(det, cars_at, k, pos_tol, yaw_tol, noisy=False):
    """An L view's box within pos_tol and yaw_tol degrees. A one-sided view's position along its face rests on the
    face's free ends (beam quantization, dropout, occlusion) and its heading on one short face: 3 pos_tol and 3 yaw_tol
    without noise, its R alone with it. With noise every box is within 4 sigma of R per axis."""
    err = det.z-truth(cars_at, det.time)[k]
    err[2] = wrap(err[2])
    if noisy:
        assert (np.abs(err) < 4*np.sqrt(np.diag(det.R))).all(), (err, np.sqrt(np.diag(det.R)), det.meta['sides'])
    if det.meta['sides'] == 1:
        if noisy:
            return err
        pos_tol, yaw_tol = 3*pos_tol, 3*yaw_tol
    assert np.hypot(*err[:2]) < pos_tol, (err, det.meta['sides'])
    assert abs(err[2]) < np.radians(yaw_tol), (np.degrees(err[2]), det.meta['sides'])
    return err


# ------------------------------------------------------------------------------------------------------------ deskew
STATIC = np.array([[25.0, 0.0, 0.0], [40.0, LANE, 0.05], [18.0, -LANE, -0.03], [60.0, -LANE, 0.0], [12.0, LANE, 0.1]])


@pytest.mark.parametrize('vx,vy,r', [(35.0, 0.0, 0.0), (30.0, 0.4, 0.3), (25.0, -0.2, -0.5), (20.0, 0.0, 2e-5)])
def test_deskew_constant_twist_is_exact(vx, vy, r):
    """Static cars, the carrying car on an exact constant-twist path: the de-skewed points are the true hit points
    seen from the sensor at the scan end, with and without the start and end states given."""
    ego_at = twist_motion(0.0, 0.5, 0.02, vx, vy, r)
    scan, hits = sweep(FRONT, ego_at, lambda t: STATIC)
    e1 = ego_at(scan.t_end)
    want = to_sensor(hits, sensor_pose(FRONT.mount, (e1.x, e1.y, e1.yaw), VEH))
    ok = np.isfinite(scan.range)
    assert ok.sum() > 300
    for got in (deskew(scan, FRONT, ego_at(0.05), VEH), deskew(scan, FRONT, ego_at(0.05), VEH, ego_at(0.0), e1)):
        assert np.isnan(got[~ok]).all()
        np.testing.assert_allclose(got[ok], want[ok], atol=1e-9)
    raw = np.column_stack((scan.range*np.cos(scan.azimuth), scan.range*np.sin(scan.azimuth)))
    assert np.abs(raw[ok]-want[ok]).max() > 0.5       # the skew it removed: metres at these speeds


def test_deskew_rear_lidar_lever_arm():
    """A rearward lidar on a turning car: the mount's lever arm and yaw are handled."""
    ego_at = twist_motion(0.0, 0.0, 0.0, 30.0, 0.0, 0.4)
    cars_at = lambda t: np.array([[-20.0, 0.0, 0.0], [-35.0, LANE, 0.0], [-12.0, -LANE, 0.02]])
    scan, hits = sweep(REAR, ego_at, cars_at)
    e1 = ego_at(scan.t_end)
    want = to_sensor(hits, sensor_pose(REAR.mount, (e1.x, e1.y, e1.yaw), VEH))
    ok = np.isfinite(scan.range)
    np.testing.assert_allclose(deskew(scan, REAR, ego_at(0.05), VEH)[ok], want[ok], atol=1e-9)


def test_deskew_interpolates_unsteady_motion():
    """Hard braking (-6 m/s^2) with the yaw rate ramping 0 -> 0.3 rad/s over the sweep: constant twist is off by
    centimetres, the start/mid/end interpolation by about a millimetre."""
    a, j = -6.0, 3.0
    f = lambda t, q: [q[3]*np.cos(q[2]), q[3]*np.sin(q[2]), j*t, a]
    sol = solve_ivp(f, (0.0, 0.1), [0.0, 0.3, 0.0, 33.0], rtol=1e-12, atol=1e-12, dense_output=True)

    def ego_at(t):
        x, y, yaw, v = sol.sol(t)
        return EgoState(float(t), float(x), float(y), float(yaw), float(v), 0.0, j*t, a)
    scan, hits = sweep(FRONT, ego_at, lambda t: STATIC)
    e1 = ego_at(scan.t_end)
    want = to_sensor(hits, sensor_pose(FRONT.mount, (e1.x, e1.y, e1.yaw), VEH))
    ok = np.isfinite(scan.range)
    err = lambda got: np.hypot(*(got[ok]-want[ok]).T).max()
    twist = err(deskew(scan, FRONT, ego_at(0.05), VEH))
    interp = err(deskew(scan, FRONT, ego_at(0.05), VEH, ego_at(0.0), e1))
    assert twist > 0.02 and interp < 0.003, (twist, interp)


# ------------------------------------------------------------------------------------------------------------- boxes
EGO = twist_motion(500.0, 0.0, 0.0, V_EGO)
SCENE = np.array([[520.0, LANE, 0.03], [537.0, -LANE+0.3, -0.05], [552.0, 0.2, 0.0], [580.0, LANE, 0.02],
                  [566.0, -LANE, -0.02]])
SPEEDS = V_EGO+np.array([2.0, -3.0, 1.0, 0.0, -1.5])


def test_boxes_accurate_noise_free():
    """Cars at 20-80 m ahead, at angles and headings up to 3 deg, moving 0-3 m/s off the prior: within a few cm and
    well under 1 deg, every one found once."""
    cars_at = cars(SCENE, SPEEDS)
    dets, _ = run(FRONT, EGO, cars_at)
    assert len(dets) == len(SCENE)
    for k, (det, _) in enumerate(nearest(dets, cars_at)):
        check_box(det, cars_at, k, 0.03, 0.5)


def test_boxes_accurate_noisy():
    """The same scene with range noise and dropout, 10 sweeps: every box within 4 sigma, L views within 10 cm and
    2.5 deg (an L view's heading sigma is about 1 deg at 60-80 m with 3 cm range noise, and the centre sits 2.4 m
    from the seen corner)."""
    rng = np.random.default_rng(1)
    cars_at = cars(SCENE, SPEEDS)
    for _ in range(10):
        dets, _ = run(FRONT, EGO, cars_at, rng=rng)
        for k, (det, d) in enumerate(nearest(dets, cars_at)):
            check_box(det, cars_at, k, 0.10, 2.5, noisy=True)


def test_refine_off_and_rear_lidar():
    """Without ICP the L-shape fit and anchors alone are within 5 cm and 1 deg; the rear lidar's boxes behind, with
    noise, as the front's."""
    cars_at = cars(SCENE, SPEEDS)
    dets, _ = run(FRONT, EGO, cars_at, refine=False)
    for k, (det, _) in enumerate(nearest(dets, cars_at)):
        check_box(det, cars_at, k, 0.05, 1.0)
    behind = cars(SCENE*[-1.0, 1.0, 1.0]+[1000.0, 0.0, 0.0], SPEEDS)
    dets, _ = run(REAR, EGO, behind, rng=np.random.default_rng(2))
    for k, (det, _) in enumerate(nearest(dets, behind)):
        check_box(det, behind, k, 0.10, 2.5, noisy=True)


def test_one_sided_views():
    """A car straight ahead shows only its rear face: one side seen, sharp along-road position and lateral centre, a
    weaker heading than an L view of a car at the same range. A car alongside behind the rear lidar shows only its
    side, cut by the field of view at its front: anchored on its free rear end."""
    rng = np.random.default_rng(3)
    ahead, angled = cars([[531.0, 0.0, 0.0]]), cars([[531.0, LANE, 0.0]])
    sig_l, sig_a = [], []
    for _ in range(5):
        (det,), _ = run(FRONT, EGO, ahead, rng=rng)
        assert det.meta['sides'] == 1
        check_box(det, ahead, 0, 0.04, 1.0, noisy=True)
        rear_y = np.array([0.0, 1.0, -VEH.length/2])          # lateral centre of the rear face
        assert np.sqrt(det.R[0, 0]) < 0.02 and np.sqrt(rear_y @ det.R @ rear_y) < 0.025
        sig_a.append(np.sqrt(det.R[2, 2]))
        (det,), _ = run(FRONT, EGO, angled, rng=rng)
        assert det.meta['sides'] == 2
        sig_l.append(np.sqrt(det.R[2, 2]))
    assert np.mean(sig_a) > 1.5*np.mean(sig_l), (np.degrees(sig_a), np.degrees(sig_l))
    side = cars([[496.0, LANE, 0.0]])
    (det,), _ = run(REAR, EGO, side, rng=rng)
    assert det.meta['sides'] == 1 and det.meta['free'] == (False, True)     # swept from the field of view's edge
    check_box(det, side, 0, 0.06, 1.0, noisy=True)


def test_occlusion_split_is_merged():
    """A car whose rear face a nearer narrow object (a 0.4 m post on the road) splits in two: one box for the car,
    from both pieces, as accurate as unoccluded."""
    cars_at = cars([[530.0, LANE, 0.0]])
    far = truth(cars_at, 0.05)[0]
    ego = EGO(0.05)
    p = sensor_pose(FRONT.mount, (ego.x, ego.y, ego.yaw), VEH)
    aim = np.array([far[0]-VEH.length/2, far[1]+0.2])-p[:2]
    post = p[:2]+0.5*aim+V_EGO*0.05*np.array([1.0, 0.0])
    sq = np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])
    boxes_at = lambda t: (post+V_EGO*(t-0.05)*np.array([1.0, 0.0])+sq)[None]
    scan, _ = sweep(FRONT, EGO, cars_at, boxes_at=boxes_at)
    dets = process_lidar(scan, FRONT, EGO, None, VEH, ROAD)
    near = [d for d in dets if np.hypot(*(d.z[:2]-truth(cars_at, d.time)[0, :2])) < 3.0]
    assert len(near) == 1
    det = near[0]
    assert det.meta['sides'] == 2
    check_box(det, cars_at, 0, 0.03, 0.5)
    # the post really split it: the car's points lie on both sides of the post's beams
    hit_post = np.isfinite(scan.range) & (scan.range < np.hypot(*aim)*0.6)
    lo, hi = scan.azimuth[hit_post].min(), scan.azimuth[hit_post].max()
    az = np.arctan2(*(to_sensor(det.meta['points'], det.sensor_pose)[:, ::-1].T))
    assert (az < lo).any() and (az > hi).any()


def test_teammates_and_off_road_dropped():
    """A teammate ahead (V2V state, moving) and a car parked beyond the road edge give no box; the traffic car does."""
    mate = cars([[518.0, 0.1, 0.01]], [V_EGO+1.0])
    traffic = cars([[535.0, LANE, 0.0]])
    parked = np.array([[540.0, ROAD.width/2+2.2, 0.0]])
    allcars = lambda t: np.vstack((traffic(t), mate(t), parked))
    scan, _ = sweep(FRONT, EGO, allcars)
    dets = process_lidar(scan, FRONT, EGO, mate, VEH, ROAD)
    assert len(dets) == 1 and nearest(dets, allcars)[0][1] < 0.03
    dets = process_lidar(scan, FRONT, EGO, None, VEH, ROAD)
    assert len(dets) == 2 and nearest(dets, allcars)[1][1] < 0.1


def test_detection_contract():
    """Kind, stamps and poses: time is the cluster's mean beam time, sensor_pose and ego are at that time; R is
    symmetric positive definite; clusters of fewer than 3 points give nothing."""
    cars_at = cars(SCENE, SPEEDS)
    dets, scan = run(FRONT, EGO, cars_at)
    for det in dets:
        assert det.kind == 'lidar_box' and det.sensor == FRONT.name and det.car == scan.car
        assert scan.t_start < det.time < scan.t_end and det.time == pytest.approx(det.meta['times'].mean())
        e = EGO(det.time)
        assert det.ego == e
        assert det.sensor_pose == sensor_pose(FRONT.mount, (e.x, e.y, e.yaw), VEH)
        assert det.meta['scan'] == scan.t_end and det.meta['n_points'] >= 3 and det.meta['sides'] in (1, 2)
        assert det.z.shape == (3,) and det.R.shape == (3, 3)
        np.testing.assert_allclose(det.R, det.R.T)
        assert np.linalg.eigvalsh(det.R).min() > 0
    coarse = replace(FRONT, beam_spacing=np.radians(1.0))
    dets, scan = run(coarse, EGO, cars([[600.0, 0.0, 0.0]]))
    assert 1 <= np.isfinite(scan.range).sum() <= 2 and dets == []


def test_refit_with_track_velocity():
    """A car 12 m/s faster than the prior: the second pass with its true velocity tightens R and stays accurate."""
    cars_at = cars([[518.0, -LANE, 0.0]], [V_EGO+12.0])
    (det,), _ = run(FRONT, EGO, cars_at)
    better = refit(det, FRONT, VEH, (V_EGO+12.0, 0.0))
    check_box(better, cars_at, 0, 0.03, 0.5)
    assert better.time == det.time and np.trace(better.R) < np.trace(det.R)


def test_lshape_heading_grid_step():
    """The L-shape search resolves headings between its 1 deg grid points."""
    for yaw in np.radians([-12.3, -0.4, 0.0, 2.6, 7.75]):
        box = car_boxes([[0.0, 0.0, yaw]], VEH)[0]
        pts = np.concatenate([box[0]+np.linspace(0, 1, 40)[:, None]*(box[3]-box[0]),
                              box[0]+np.linspace(0, 1, 80)[:, None]*(box[1]-box[0])])
        assert abs(lshape_heading(pts, (-30.0, -20.0))-yaw) < np.radians(0.3)


# ------------------------------------------------------------------------------------------------------- covariance
def test_covariance_consistent():
    """NEES of the box over many noisy sweeps of cars at 15-70 m (random lanes, offsets, headings and speeds): the
    centre (2 dof) and heading (1 dof) averages near their dof, few beyond the 99 % bound."""
    rng = np.random.default_rng(7)
    sigma_v = (8/np.sqrt(12), V_EGO*0.06/np.sqrt(12))     # the speeds and headings drawn below, about the prior
    nees_xy, nees_h = [], []
    for _ in range(120):
        s = [500.0+rng.uniform(15.0, 70.0), rng.choice([-LANE, 0.0, LANE])+rng.uniform(-0.4, 0.4),
             rng.uniform(-0.03, 0.03)]
        cars_at = cars([s], [V_EGO+rng.uniform(-4.0, 4.0)])
        scan, _ = sweep(FRONT, EGO, cars_at, rng=rng)
        det, d = nearest(process_lidar(scan, FRONT, EGO, None, VEH, ROAD, target_sigma=sigma_v), cars_at)[0]
        assert d < 0.3
        e = det.z-truth(cars_at, det.time)[0]
        e[2] = wrap(e[2])
        nees_xy.append(e[:2] @ np.linalg.solve(det.R[:2, :2], e[:2]))
        nees_h.append(e[2]**2/det.R[2, 2])
    nees_xy, nees_h = np.array(nees_xy), np.array(nees_h)
    assert 1.0 < nees_xy.mean() < 3.0, nees_xy.mean()
    assert 0.5 < nees_h.mean() < 1.5, nees_h.mean()
    assert (nees_xy > 9.21).mean() < 0.06 and (nees_h > 6.63).mean() < 0.06


def test_fit_box_direct():
    """fit_box on a clean L of points (static, seen from the origin): the box is exact and R is small."""
    z = np.array([30.0, 4.0, 0.04])
    cg = z[:2]-np.array([np.cos(z[2]), np.sin(z[2])])*sum(cg_offsets(VEH))/2
    box = car_boxes([[*cg, z[2]]], VEH)[0]
    o = np.zeros(2)
    ang = np.arctan2(*(box.mean(0)-o)[::-1])+np.radians(np.arange(-8, 8, 0.1))
    r, _ = raycast(o, ang, box[None], 200.0)
    ok = np.isfinite(r)
    pts = np.column_stack((r*np.cos(ang), r*np.sin(ang)))[ok]
    got, R, sides, t_ref = fit_box(pts, np.zeros(len(pts)), o, VEH, FRONT)
    assert sides == 2 and t_ref == 0.0
    np.testing.assert_allclose(got, z, atol=5e-3)
    assert np.sqrt(np.diag(R)).max() < 0.02


def test_random_traffic_robust():
    """Random dense scenes for either lidar (3-24 cars, lateral offsets, headings to 3.4 deg, speeds 6 m/s off the
    prior, a turning carrying car, teammates in a third of them): finite boxes, R positive definite, no teammate boxes,
    and the boxes of clusters with 10 or more points within 4 sigma on every axis (all but 3 %)."""
    rng = np.random.default_rng(123)
    n10 = within = 0
    for trial in range(40):
        spec = FRONT if trial % 2 == 0 else REAR
        ego = twist_motion(500.0, rng.uniform(-0.5, 0.5), rng.uniform(-0.03, 0.03), V_EGO+rng.uniform(-3, 3), 0.0,
                           rng.uniform(-0.1, 0.1))
        n = rng.integers(3, 25)
        st = np.column_stack((500+(1 if spec is FRONT else -1)*rng.uniform(6, 120, n),
                              rng.choice([-LANE, 0.0, LANE], n)+rng.uniform(-0.6, 0.6, n), rng.uniform(-0.06, 0.06, n)))
        keep = []
        for k in range(n):
            if all(abs(st[k, 0]-st[m, 0]) > 5.5 or abs(st[k, 1]-st[m, 1]) > 2.1 for m in keep) and                     (abs(st[k, 0]-500) > 5.5 or abs(st[k, 1]-ego(0.0).y) > 2.1):
                keep.append(k)
        cars_at = cars(st[keep], V_EGO+rng.uniform(-6, 6, len(keep)))
        ids = rng.choice(len(keep), min(2, len(keep)), replace=False) if trial % 3 == 0 else []
        mates = (lambda t, ids=ids: cars_at(t)[ids]) if len(ids) else None
        scan, _ = sweep(spec, ego, cars_at, rng=rng)
        for det in process_lidar(scan, spec, ego, mates, VEH, ROAD):
            assert np.isfinite(det.z).all() and np.linalg.eigvalsh(det.R).min() > 0
            z = truth(cars_at, det.time)
            k = int(np.argmin(np.hypot(*(z[:, :2]-det.z[:2]).T)))
            assert k not in ids
            if det.meta['n_points'] >= 10:
                err = det.z-z[k]
                err[2] = wrap(err[2])
                n10 += 1
                within += bool((np.abs(err) < 4*np.sqrt(np.diag(det.R))).all())
    assert n10 > 50 and within >= 0.97*n10, (within, n10)


# ----------------------------------------------------------------------------------------------------------- timing
def test_timing_twenty_cars():
    """A full sweep with 20 cars in view (three lanes, 8-150 m ahead): processing stays within budget."""
    rng = np.random.default_rng(11)
    xs = np.sort(rng.uniform(508.0, 650.0, 20))
    lanes = np.array([-LANE, 0.0, LANE])[np.arange(20) % 3]+rng.uniform(-0.3, 0.3, 20)
    states = np.column_stack((xs, lanes, rng.uniform(-0.02, 0.02, 20)))
    cars_at = cars(states, V_EGO+rng.uniform(-5, 5, 20))
    scan, _ = sweep(FRONT, EGO, cars_at, rng=rng, window=False)
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        dets = process_lidar(scan, FRONT, EGO, lambda t: np.zeros((0, 3)), VEH, ROAD)
        best = min(best, time.perf_counter()-start)
    e = EGO(0.05)
    p = sensor_pose(FRONT.mount, (e.x, e.y, e.yaw), VEH)
    _, hit = raycast(p[:2], p[2]+scan.azimuth, car_boxes(cars_at(0.05), VEH), FRONT.range_max)
    visible = np.flatnonzero(np.bincount(hit[hit >= 0], minlength=20) >= 8)
    dist = np.array([d for _, d in nearest(dets, cars_at)])
    print(f'\n20 cars: {len(visible)} with 8+ beams, {len(dets)} boxes, {1e3*best:.1f} ms per sweep')
    assert len(visible) >= 5 and (dist[visible] < 0.5).all()
    assert best < 0.025
