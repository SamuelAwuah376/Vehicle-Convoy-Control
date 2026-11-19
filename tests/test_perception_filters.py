"""Perception filters: the UKF (ukf.py), the IMM motion and sensor measurement models (models.py) and the IMM
(imm.py). Monte Carlo checks run their seeded runs as one batched filter, so the whole file takes a few seconds."""
from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import chi2, multivariate_normal
from nohesi.config import scenario_v9
from nohesi.geometry import body_centre
from nohesi.road import Road
from nohesi.traffic import _profile
from nohesi.perception.config import TrackerConfig, default_radars, default_cameras
from nohesi.perception.types import Detection, EgoState
from nohesi.perception import geometry as G
from nohesi.perception.tracking.ukf import (sigma_points, ukf_predict, ukf_update, robust_cholesky,
                                            floor_eigenvalues, UpdateResult)
from nohesi.perception.tracking.models import (MODELS, motion, process_noise, h_lidar, residual_lidar, h_radar,
                                               residual_radar, h_camera, camera_R, measurement, lane_of,
                                               lane_centre, target_lane, stop_point, heading, sensor_velocity,
                                               TRUNCATED_VAR, NEAR, _EDGES)
from nohesi.perception.tracking.imm import (IMMState, init_state, imm_transition, imm_predict, imm_update,
                                            imm_combine, lane_probability)

SC = scenario_v9()
CAR = SC.vehicle
ROAD = Road(SC.road)
CFG = TrackerConfig()
LIDAR_R = np.diag([0.15**2, 0.1**2, np.radians(1.0)**2])
RADARS = {r.name: r for r in default_radars()}
CAMERAS = {c.name: c for c in default_cameras()}


def cv(dt):
    F = np.eye(4)
    F[0, 2] = F[1, 3] = dt
    return F


def psd(P):
    return np.allclose(P, np.swapaxes(P, -1, -2)) and np.all(np.linalg.eigvalsh(P) > 0)


def ego_det(kind, sensor, mount, ego, z=np.zeros(3), R=np.eye(3), meta=None):
    pose = G.sensor_pose(mount, [ego.x, ego.y, ego.yaw], CAR)
    return Detection(kind, sensor, 0, ego.time, np.asarray(z, float), np.asarray(R, float), pose, ego, meta or {})


# ---------------------------------------------------------------- UKF

def test_sigma_points_reproduce_moments_and_batch():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(5, 4, 4))
    P = A @ np.swapaxes(A, 1, 2)+0.1*np.eye(4)
    m = rng.normal(size=(5, 4))
    X, wm, wc = sigma_points(m, P)
    assert X.shape == (5, 9, 4) and np.isclose(wm.sum(), 1)
    mean = np.einsum('i,bij->bj', wm, X)
    dX = X-mean[:, None]
    assert np.allclose(mean, m) and np.allclose(np.einsum('i,bij,bik->bjk', wc, dX, dX), P)
    X0, _, _ = sigma_points(m[2], P[2])
    assert np.allclose(X0, X[2])
    for alpha, beta, kappa in ((1e-3, 2, 0), (0.5, 2, 1), (1, 0, 3-4)):
        X, wm, wc = sigma_points(m[0], P[0], alpha, beta, kappa)
        dX = X-m[0]
        assert np.allclose(wm @ X, m[0]) and np.allclose(np.einsum('i,ij,ik->jk', wc, dX, dX), P[0])


def test_ukf_equals_linear_kf():
    """Linear dynamics and measurements: the UKF is the Kalman filter (any sigma-point parameters)."""
    rng = np.random.default_rng(1)
    dt = 0.1
    F, Q = cv(dt), process_noise('keep', CFG, dt)
    H = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0], [0.3, -0.2, 1, 0.5]])
    R = np.diag([0.2, 0.1, 0.3])**2
    m, P = np.array([0.0, 0, 27, 0]), np.diag([4.0, 0.5, 9, 1])
    mk, Pk = m.copy(), P.copy()
    for _ in range(20):
        m, P = ukf_predict(m, P, lambda X: X @ F.T, Q)
        mk, Pk = F @ mk, F @ Pk @ F.T+Q
        assert np.allclose(m, mk) and np.allclose(P, Pk)
        zhat = H @ mk
        z = zhat+rng.normal(size=3)*0.3
        r = ukf_update(m, P, z, R, lambda X: X @ H.T)
        S = H @ Pk @ H.T+R
        K = Pk @ H.T @ np.linalg.inv(S)
        nu = z-zhat
        mk, Pk = mk+K @ nu, (np.eye(4)-K @ H) @ Pk
        assert isinstance(r, UpdateResult)
        assert np.allclose(r.mean, mk) and np.allclose(r.cov, Pk) and np.allclose(r.S, S)
        assert np.allclose(r.innovation, nu) and np.isclose(r.nis, nu @ np.linalg.solve(S, nu))
        assert np.isclose(r.loglik, multivariate_normal(zhat, S).logpdf(z))
        for alpha, kappa in ((1e-3, 0.0), (0.5, 1.0)):
            other = ukf_update(m, P, z, R, lambda X: X @ H.T, alpha=alpha, kappa=kappa)
            assert np.allclose(other.mean, mk) and np.allclose(other.cov, Pk, atol=1e-8)
        m, P = r.mean, r.cov
    # batched filters equal one-by-one filters, z and R per filter or shared
    ms, Ps = np.stack([m, m+1, m-2]), np.stack([P, 2*P, 0.5*P])
    zs = rng.normal(size=(3, 3))+ms @ H.T
    rb = ukf_update(ms, Ps, zs, R, lambda X: X @ H.T)
    for i in range(3):
        ri = ukf_update(ms[i], Ps[i], zs[i], R, lambda X: X @ H.T)
        assert all(np.allclose(a[i], b) for a, b in zip(rb, ri))


def _monte_carlo(model, x0, P0, steps, seed, runs=100, dt=0.1, cfg=CFG):
    """Batched Monte Carlo: truth from `model` plus its process noise, lidar boxes, one UKF per run with the same
    model. Returns (nees (steps, runs), nis (steps, runs))."""
    rng = np.random.default_rng(seed)
    f, Q = motion(model, cfg, ROAD), process_noise(model, cfg, dt)
    LQ, LR = np.linalg.cholesky(Q), np.linalg.cholesky(LIDAR_R)
    x = x0+rng.normal(size=(runs, 4)) @ np.linalg.cholesky(P0).T
    m, P = np.tile(x0, (runs, 1)), np.tile(P0, (runs, 1, 1))
    h = lambda X: h_lidar(X, CAR)
    nees, nis = [], []
    for _ in range(steps):
        x = f(x, dt)+rng.normal(size=(runs, 4)) @ LQ.T
        z = h(x)+rng.normal(size=(runs, 3)) @ LR.T
        m, P = ukf_predict(m, P, lambda X: f(X, dt), Q)
        r = ukf_update(m, P, z, LIDAR_R, h, residual_lidar)
        m, P = r.mean, r.cov
        e = x-m
        nees.append(np.einsum('bi,bi->b', e, np.linalg.solve(P, e[..., None])[..., 0]))
        nis.append(r.nis)
    return np.array(nees), np.array(nis)


def _consistent(values, dof, runs):
    """Run-averaged values per step inside the two-sided 95 % chi-square bounds: overall mean, and most steps."""
    lo, hi = chi2.ppf([0.025, 0.975], dof*runs)/runs
    per_step = values.mean(1)
    return lo < per_step.mean() < hi and np.mean((per_step > lo) & (per_step < hi)) >= 0.85, (lo, per_step.mean(), hi)


@pytest.mark.parametrize('model, x0', [('keep', [0.0, 0.0, 27.0, 0.0]), ('left', [0.0, -3.6, 27.0, 0.0]),
                                       ('right', [0.0, 3.4, 31.0, -0.3])])
def test_nees_nis_consistency(model, x0):
    """A car following the filter's own model with lidar boxes: NEES (4 dof) and NIS (3 dof) are chi-square."""
    P0 = np.diag([1.0, 0.1, 4.0, 0.2])
    nees, nis = _monte_carlo(model, np.array(x0), P0, steps=40, seed=7)
    ok, info = _consistent(nees, 4, nees.shape[1])
    assert ok, f'NEES {info}'
    ok, info = _consistent(nis, 3, nis.shape[1])
    assert ok, f'NIS {info}'


# ---------------------------------------------------------------- motion models

def test_lane_helpers_match_road():
    d = np.linspace(-7, 7, 57)
    assert np.array_equal(lane_of(d, ROAD), ROAD.lane_of(d)) and np.array_equal(lane_of(d, SC.road), ROAD.lane_of(d))
    assert np.allclose(lane_centre(np.arange(3), ROAD), [ROAD.lane_center(l) for l in range(3)])
    assert np.array_equal(target_lane('left', np.array([-3.6, 0.0, 3.6]), ROAD), [1, 2, 2])
    assert np.array_equal(target_lane('right', np.array([-3.6, 0.0, 3.6]), ROAD), [0, 0, 1])
    # completing a change: right of the new lane's centre a left change heads to that lane, not the next
    assert np.array_equal(target_lane('left', np.array([-1.7, -0.1, 0.1, 5.0]), ROAD), [1, 1, 2, 2])
    assert np.array_equal(target_lane('right', np.array([1.7, 0.1, -0.1, -5.0]), ROAD), [1, 1, 0, 0])
    assert np.array_equal(target_lane('keep', np.array([-1.7, 1.7]), ROAD), [1, 1])
    with pytest.raises(ValueError):
        motion('straight', CFG, ROAD)


def test_keep_model_settles_without_overshoot_and_is_exact():
    f = motion('keep', CFG, ROAD)
    x = np.array([[0.0, 1.5, 27.0, 0.0], [0.0, -1.0, 25.0, 0.8]])
    traj = [x]
    for _ in range(200):
        traj.append(f(traj[-1], 0.05))
    traj = np.array(traj)
    assert np.allclose(traj[:, :, 0], np.arange(201)[:, None]*0.05*x[:, 2]) and np.allclose(traj[:, :, 2], x[:, 2])
    assert np.all(np.diff(traj[:, 0, 1]) <= 0) and np.all(traj[:, 0, 1] >= 0)     # from rest: monotone, no overshoot
    assert np.allclose(traj[-1, :, 1], 0, atol=2e-3) and np.allclose(traj[-1, :, 3], 0, atol=2e-3)
    assert np.allclose(f(f(x, 0.05), 0.05), f(x, 0.1)) and np.allclose(f(x, 0.0), x)   # exact discretization
    e, v, k, t = 1.5, 0.0, CFG.lane_keep_rate, 0.7                                  # closed form, d'' = -2k d' - k^2 d
    assert np.isclose(f(np.array([[0, e, 27, v]]), t)[0, 1], np.exp(-k*t)*(e*(1+k*t)+v*t))


def test_change_models_stop_at_next_lane_centre():
    c, edge = lane_centre(np.arange(3), ROAD), ROAD.width/2
    left, right = motion('left', CFG, ROAD), motion('right', CFG, ROAD)
    x = np.array([[0.0, c[0], 27.0, 0.0]])
    assert np.allclose(left(x, 100.0)[0, [1, 3]], [c[1], 0.0])                    # one long step: the next centre
    assert np.allclose(left(np.array([[0.0, -1.0, 27.0, 0.5]]), 100.0)[0, [1, 3]], [c[2], 0.0])
    assert np.allclose(right(np.array([[0.0, c[2], 27.0, 0.0]]), 100.0)[0, [1, 3]], [c[1], 0.0])
    traj = [x]
    for _ in range(400):
        traj.append(left(traj[-1], 0.05))
    d, v = np.array(traj)[:, 0, 1], np.array(traj)[:, 0, 3]
    assert np.all(np.diff(d) >= 0) and np.isclose(d[-1], edge) and v[-1] == 0.0  # lane by lane, then to the edge
    assert v.max() <= CFG.lc_speed+1e-12 and np.all(v >= 0)
    half = left(left(x, 0.25), 0.25)                                              # exact before the stop
    assert np.allclose(half, left(x, 0.5))
    vt, k, t = CFG.lc_speed, CFG.lane_keep_rate, 0.5
    assert np.isclose(half[0, 1], c[0]+vt*t-vt*(1-np.exp(-k*t))/k) and np.isclose(half[0, 3], vt*(1-np.exp(-k*t)))
    # edge lanes: no lane beyond, so the change is a drift to the road's edge (not a copy of lane keep)
    for d0 in (c[2]-0.5, c[2]+0.5, edge+0.3):
        assert np.allclose(left(np.array([[0.0, d0, 27.0, 0.0]]), 30.0)[0, [1, 3]], [edge, 0.0])
    assert np.allclose(right(np.array([[0.0, c[0]+0.4, 27.0, 0.3]]), 30.0)[0, [1, 3]], [-edge, 0.0])
    assert np.allclose(stop_point('keep', np.array([5.0, -9.0]), ROAD), [c[2], c[0]])


def test_process_noise_is_white_acceleration():
    for dt in (0.05, 0.1, 0.37):
        Q = process_noise('left', CFG, dt)
        Qh, F = process_noise('left', CFG, dt/2), cv(dt/2)
        assert np.allclose(Q, F @ Qh @ F.T+Qh) and psd(Q)                       # consistent under splitting
        assert np.isclose(Q[2, 2], CFG.q_long**2*dt) and np.isclose(Q[3, 3], CFG.q_lat**2*dt)
        assert Q[0, 1] == Q[0, 3] == Q[2, 3] == 0
    assert np.allclose(process_noise('keep', CFG, 0.1), process_noise('right', CFG, 0.1))


# ---------------------------------------------------------------- measurement models

def test_h_lidar_body_centre():
    X = np.array([[100.0, 0.2, 27.0, 1.2], [50.0, -3.6, 0.0, 0.0]])
    z = h_lidar(X, CAR)
    yaw = np.arctan2(X[:, 3], np.maximum(X[:, 2], 1e-3))
    assert np.allclose(z[:, :2], body_centre(np.column_stack((X[:, :2], yaw)), CAR)) and np.allclose(z[:, 2], yaw)
    assert np.allclose(z[1], [X[1, 0]+sum(G.cg_offsets(CAR))/2, X[1, 1], 0.0])     # the centre 7 cm behind the CG
    assert h_lidar(np.tile(X, (3, 1, 1)), CAR).shape == (3, 2, 3)


def _ego(**kw):
    base = dict(time=0.0, x=100.0, y=0.3, yaw=0.05, vx=30.0, vy=0.4, r=0.15)
    base.update(kw)
    return EgoState(**base)


def _radar_truth(state, mount, ego):
    """Range, azimuth (geometry.nearest_box_point and to_sensor) and range rate by central differences, moving the
    car at its velocity and the carrying car along its twist."""
    def rng_az(eps):
        st = np.array([state[0]+state[2]*eps, state[1]+state[3]*eps, heading(state)])
        c, s = np.cos(ego.yaw), np.sin(ego.yaw)
        cg = [ego.x+(c*ego.vx-s*ego.vy)*eps, ego.y+(s*ego.vx+c*ego.vy)*eps, ego.yaw+ego.r*eps]
        pose = G.sensor_pose(mount, cg, CAR)
        p = G.to_sensor(G.nearest_box_point(pose[:2], G.car_boxes(st[None], CAR)[0]), pose)
        return np.hypot(*p), np.arctan2(p[1], p[0])
    eps = 1e-5
    r0, a0 = rng_az(0.0)
    return np.array([r0, a0, (rng_az(eps)[0]-rng_az(-eps)[0])/(2*eps)])


@pytest.mark.parametrize('sensor, state', [('radar_front', [160.0, 3.6, 25.0, 0.5]),
                                           ('radar_front', [112.0, -3.4, 36.0, -0.9]),
                                           ('radar_rear', [60.0, 0.1, 33.0, 0.0]),
                                           ('radar_left', [101.0, 3.9, 29.0, 0.2]),
                                           ('radar_right', [96.0, -3.5, 31.0, 1.1])])
def test_h_radar_matches_geometry_and_finite_difference(sensor, state):
    mount, ego = RADARS[sensor].mount, _ego()
    det = ego_det('radar', sensor, mount, ego)
    z = h_radar(np.array([state]), det, CAR)[0]
    assert np.allclose(z, _radar_truth(np.array(state), mount, ego), atol=1e-6)
    eps = 1e-6                                                                    # the sensor's velocity, lever arm
    pose = lambda e: np.array(G.sensor_pose(mount, [ego.x+(np.cos(ego.yaw)*ego.vx-np.sin(ego.yaw)*ego.vy)*e,
                                                    ego.y+(np.sin(ego.yaw)*ego.vx+np.cos(ego.yaw)*ego.vy)*e,
                                                    ego.yaw+ego.r*e], CAR)[:2])
    assert np.allclose(sensor_velocity(det), (pose(eps)-pose(-eps))/(2*eps), atol=1e-6)


def test_h_radar_nearest_point_not_centre():
    ego = _ego(yaw=0.0, vy=0.0, r=0.0, y=0.0)
    det = ego_det('radar', 'radar_front', RADARS['radar_front'].mount, ego)
    X = np.array([[150.0, 0.0, 30.0, 0.0]])
    rear, _ = G.mount_offset(RADARS['radar_front'].mount, CAR)
    z = h_radar(X, det, CAR)[0]
    assert np.isclose(z[0], 50.0+G.cg_offsets(CAR)[0]-rear) and abs(z[1]) < 1e-12 and np.isclose(z[2], 0.0)


def _cam(name='camera_front', **ego_kw):
    spec = CAMERAS[name]
    ego = _ego(**ego_kw)
    return spec, ego_det('camera_box', name, spec.mount, ego, np.zeros(4), np.eye(4)*4)


def test_h_camera_matches_project():
    spec, det = _cam(pitch=0.01)
    X = np.array([[135.0, 3.6, 27.0, 0.4], [180.0, -3.4, 33.0, 0.0]])
    z = h_camera(X, det, spec, CAR, 1.45)
    corners = G.box_corners_3d(np.column_stack((X[:, :2], heading(X))), CAR, 1.45)
    u, v, depth = G.project(spec, det.sensor_pose, corners, pitch=0.01)
    assert np.all(depth > 0)
    assert np.allclose(z, np.column_stack((u.min(1), u.max(1), v.min(1), v.max(1))))
    assert np.all((z[:, 0] > 0) & (z[:, 1] < spec.width_px) & (z[:, 2] > 0) & (z[:, 3] < spec.height_px))
    zp = h_camera(X, replace(det, ego=replace(det.ego, pitch=0.0)), spec, CAR, 1.45)
    assert np.all(zp[:, 2:] > z[:, 2:]) and np.allclose(zp[:, :2], z[:, :2], atol=2)   # nose down: scene moves up
    for name, state in (('camera_rear', [70.0, -3.5, 33.0, 0.1]), ('camera_left', [104.0, 4.0, 30.0, 0.0])):
        spec, det = _cam(name)
        corners = G.box_corners_3d(np.array([[*state[:2], heading(np.array(state))]]), CAR, 1.45)
        u, v, depth = G.project(spec, det.sensor_pose, corners)
        assert np.all(depth > 0)
        box = np.clip([u.min(), u.max(), v.min(), v.max()], 0, [spec.width_px]*2+[spec.height_px]*2)
        assert np.allclose(h_camera(np.array([state]), det, spec, CAR, 1.45)[0], box)


def test_h_camera_clips_to_image_and_near_plane():
    spec, det = _cam(yaw=0.0, y=0.0)
    X = np.array([[104.0, 3.6, 30.0, 0.0],      # alongside, reaching behind the camera plane
                  [108.0, 3.0, 30.0, 0.0],      # close, partly beyond the image's left side
                  [80.0, 0.0, 30.0, 0.0]])      # behind the camera: nothing in front
    z = h_camera(X, det, spec, CAR, 1.45)
    assert np.all(np.isfinite(z)) and np.allclose(z[2], [spec.width_px/2]*2+[spec.height_px/2]*2)
    corners = G.box_corners_3d(np.column_stack((X[:2, :2], heading(X[:2]))), CAR, 1.45)
    t = np.linspace(0, 1, 4001)
    pts = corners[:, _EDGES[:, 0], None]+t[:, None]*(corners[:, _EDGES[:, 1], None]-corners[:, _EDGES[:, 0], None])
    u, v, depth = G.project(spec, det.sensor_pose, pts.reshape(2, -1, 3))
    ok = depth > NEAR
    lim = np.array([spec.width_px]*2+[spec.height_px]*2)
    dense = np.clip(np.column_stack((np.where(ok, u, np.inf).min(1), np.where(ok, u, -np.inf).max(1),
                                     np.where(ok, v, np.inf).min(1), np.where(ok, v, -np.inf).max(1))), 0, lim)
    assert np.allclose(z[:2], dense, atol=1.0)
    assert z[0, 0] == 0 and z[1, 0] == 0 and 0 < z[1, 1] < spec.width_px/2       # running off the left side


def test_camera_R_inflates_truncated_edges():
    spec, det = _cam()
    R = np.full((4, 4), 0.5)+np.eye(4)*3.5
    flagged = replace(det, z=np.array([0.0, 300, 400, 700]), R=R, meta={'truncated': np.array([1, 0, 0, 1], bool)})
    Rt = camera_R(flagged, spec)
    assert Rt[0, 0] == Rt[3, 3] == TRUNCATED_VAR and np.all(Rt[0, 1:] == 0) and np.all(Rt[1:3, 3] == 0)
    assert np.allclose(Rt[1:3, 1:3], R[1:3, 1:3]) and np.allclose(det.R, np.eye(4)*4)
    border = replace(det, z=np.array([0.0, 300, 400, spec.height_px]), R=R, meta={})
    assert np.array_equal(np.diag(camera_R(border, spec)) == TRUNCATED_VAR, [True, False, False, True])
    assert np.allclose(camera_R(replace(border, meta={'truncated': False}), spec), R)
    h, Rm, res = measurement(flagged, CAR, tuple(CAMERAS.values()))
    assert np.allclose(Rm, Rt) and res is None and h(np.array([[135.0, 3.6, 27.0, 0.0]])).shape == (1, 4)


def test_measurement_dispatch():
    ego = _ego()
    lid = Detection('lidar_box', 'lidar_front', 0, 0.0, np.zeros(3), LIDAR_R, (0, 0, 0, 0.6), ego)
    h, R, res = measurement(lid, CAR)
    X = np.array([[1.0, 2, 27, 0]])
    assert res is residual_lidar and np.allclose(h(X), h_lidar(X, CAR)) and np.allclose(R, LIDAR_R)
    rad = ego_det('radar', 'radar_front', RADARS['radar_front'].mount, ego)
    assert measurement(rad, CAR)[2] is residual_radar
    with pytest.raises(KeyError):
        measurement(_cam()[1], CAR, ())


# ---------------------------------------------------------------- angle wrapping

def test_residuals_wrap_angles():
    a, b = np.array([0.0, 0.0, np.pi-0.01]), np.array([0.0, 0.0, -np.pi+0.01])
    assert np.isclose(residual_lidar(a, b)[2], -0.02) and np.isclose(residual_lidar(b, a)[2], 0.02)
    r = residual_radar(np.array([[10.0, -np.pi+0.02, 1.0]]), np.array([[9.0, np.pi-0.03, 0.0]]))
    assert np.allclose(r, [[1.0, 0.05, 1.0]])


def test_ukf_update_with_bearing_near_pi():
    """A bearing-only update behind the sensor (bearing near +-pi) is the mirror image of the same problem ahead
    (bearing near 0); without wrapping it fails."""
    def h(X):
        return np.arctan2(X[:, 1], X[:, 0])[:, None]

    def wrap_res(a, b):
        return G.wrap(np.subtract(a, b))
    m, P, R = np.array([-50.0, 0.3, 0.0, 0.0]), np.diag([4.0, 9.0, 1.0, 1.0]), np.array([[1e-4]])
    z = np.array([-np.pi+0.004])
    back = ukf_update(m, P, z, R, h, wrap_res)
    ahead = ukf_update(-m, P, G.wrap(z+np.pi), R, h, wrap_res)
    assert np.allclose(back.mean, -ahead.mean) and np.allclose(back.cov, ahead.cov)
    assert np.isclose(back.nis, ahead.nis) and abs(back.innovation[0]) < 0.02
    naive = ukf_update(m, P, z, R, h)
    assert abs(naive.mean[1]-back.mean[1]) > 1.0 and naive.S[0, 0] > 100*back.S[0, 0]


def test_h_radar_azimuth_wrapped():
    ego = _ego(yaw=np.pi-0.01, x=0.0, y=0.0, r=0.0, vy=0.0)                         # a car driving the other way
    det = ego_det('radar', 'radar_front', RADARS['radar_front'].mount, ego)
    X = np.array([[-60.0, 0.5, 20.0, 0.0], [-60.0, -0.5, 20.0, 0.0]])
    z = h_radar(X, det, CAR)
    assert np.all(np.abs(z[:, 1]) < 0.1) and np.all((z[:, 1] >= -np.pi) & (z[:, 1] < np.pi))
    ego = _ego(yaw=0.0, x=0.0, y=0.0, r=0.0, vy=0.0)
    det = ego_det('radar', 'radar_front', RADARS['radar_front'].mount, ego)
    zb = h_radar(np.array([[-60.0, 0.001, 20.0, 0.0], [-60.0, -0.001, 20.0, 0.0]]), det, CAR)
    assert np.allclose(np.abs(zb[:, 1]), np.pi, atol=1e-3) and abs(residual_radar(zb[0], zb[1])[1]) < 1e-3


# ---------------------------------------------------------------- numerical robustness

def test_tiny_and_huge_covariances():
    dt, F = 0.1, cv(0.1)
    H = np.array([[1.0, 0, 0, 0], [0, 1, 0, 0]])
    R = np.diag([0.04, 0.01])
    z = np.array([1.0, 0.2])
    lin = lambda X: X @ H.T
    tiny = ukf_update(np.zeros(4), 1e-14*np.eye(4), z, R, lin)
    assert np.allclose(tiny.mean, 0, atol=1e-10) and np.allclose(tiny.cov, 1e-14*np.eye(4), rtol=1e-6) and psd(tiny.cov)
    huge = ukf_update(np.zeros(4), 1e8*np.eye(4), z, R, lin)
    assert np.allclose(huge.mean[:2], z, atol=1e-6) and np.allclose(np.diag(huge.cov)[:2], np.diag(R), rtol=1e-3)
    assert np.allclose(np.diag(huge.cov)[2:], 1e8) and psd(huge.cov)
    m, P = ukf_predict(np.zeros(4), 1e-16*np.eye(4), lambda X: X @ F.T, np.zeros((4, 4)))
    assert psd(P) and np.allclose(P, 1e-16*F @ F.T)
    for scale in (1e-12, 1e10):                                                   # nonlinear models stay finite, PSD
        X0 = np.array([100.0, 0.0, 27.0, 0.1])
        r = ukf_update(X0, scale*np.eye(4), [102.5, 0.1, 0.0], LIDAR_R, lambda X: h_lidar(X, CAR), residual_lidar)
        assert np.all(np.isfinite(r.mean)) and psd(r.cov) and np.isfinite(r.loglik)
        spec, det = _cam()
        r = ukf_update(np.array([140.0, 3.6, 27.0, 0.0]), scale*np.eye(4), [700.0, 760, 500, 560], 4*np.eye(4),
                       lambda X: h_camera(X, det, spec, CAR))
        assert np.all(np.isfinite(r.mean)) and np.all(np.isfinite(r.cov)) and np.isfinite(r.nis)


def test_robust_cholesky_and_floor():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    singular = np.outer(v, v)
    L = robust_cholesky(singular)
    assert np.all(np.isfinite(L)) and np.allclose(L @ L.T, singular, atol=1e-6)
    indefinite = np.diag([1.0, 1.0, 1.0, -1e-14])
    assert np.all(np.linalg.eigvalsh(floor_eigenvalues(indefinite)) > 0)
    X, _, _ = sigma_points(np.zeros(4), np.stack([singular, indefinite, np.eye(4)]))
    assert np.all(np.isfinite(X))
    asym = np.eye(4)+np.triu(np.full((4, 4), 1e-9), 1)
    assert np.allclose(floor_eigenvalues(asym), floor_eigenvalues(asym).T)
    with pytest.raises(ValueError):
        robust_cholesky(np.full((4, 4), np.nan))


def test_imm_robust_to_outliers_and_extreme_probabilities():
    st = init_state([100.0, 0.0, 27.0, 0.0], np.diag([1.0, 0.1, 4.0, 0.1]), (1.0, 0.0, 0.0))
    h = lambda X: h_lidar(X, CAR)
    st = imm_predict(st, 0.1, CFG, ROAD)
    out, ll, nis = imm_update(st, [1e4, -50.0, 3.0], LIDAR_R, h, residual_lidar)
    assert np.all(np.isfinite(out.mu)) and np.isclose(out.mu.sum(), 1) and out.mu.min() > 0
    assert np.isfinite(ll) and ll < -1e6 and nis.shape == (3,) and np.all(nis > 1e6)
    st = init_state([100.0, 0.0, 27.0, 0.0], 1e9*np.eye(4))
    st = imm_predict(st, 0.1, CFG, ROAD)
    out, ll, _ = imm_update(st, [102.7, 0.05, 0.0], LIDAR_R, h, residual_lidar)
    assert np.all(np.isfinite(out.means)) and all(psd(P) for P in out.covs) and np.isfinite(ll)


# ---------------------------------------------------------------- IMM

def test_transition_matrix():
    P = imm_transition(CFG, 0.1)
    assert np.allclose(P.sum(1), 1) and np.all(P >= 0)
    assert np.allclose(np.diag(imm_transition(CFG, 0.0)), 1)
    assert np.isclose(P[0, 1], CFG.imm_switch*0.1, rtol=0.02) and np.isclose(P[1, 0], 10*CFG.imm_switch*0.1, rtol=0.02)
    assert np.isclose(P[0, 1], P[0, 2]) and P[1, 2] < 1e-4
    assert np.allclose(imm_transition(CFG, 0.3), np.linalg.matrix_power(P, 3))
    P[0, 0] = 5.0
    assert imm_transition(CFG, 0.1)[0, 0] < 1                                     # the cached matrix is not shared
    with pytest.raises(ValueError):
        imm_transition(CFG, -0.1)


def test_init_combine_and_lane_probability():
    st = init_state([10.0, 0.0, 27.0, 0.0], np.eye(4), (2.0, 1.0, 1.0))
    assert st.means.shape == (3, 4) and st.covs.shape == (3, 4, 4) and np.allclose(st.mu, [0.5, 0.25, 0.25])
    c = st.copy()
    c.means[0, 0] = -1
    assert st.means[0, 0] == 10.0
    means = np.array([[10.0, 0.0, 27, 0], [10.5, 1.0, 27, 0.8], [9.5, -1.4, 26, -0.8]])
    covs = np.stack([np.eye(4), 2*np.eye(4), np.diag([1.0, 2, 3, 4])])
    mu = np.array([0.6, 0.3, 0.1])
    st = IMMState(means, covs, mu)
    m, P = imm_combine(st)
    assert np.allclose(m, mu @ means)
    assert np.allclose(P, sum(mu[j]*(covs[j]+np.outer(means[j]-m, means[j]-m)) for j in range(3)))
    assert np.allclose(lane_probability(st, ROAD), [0.1, 0.6, 0.3])                # right from 1 -> 0, left -> 2
    edge = IMMState(means+np.array([0, 3.6, 0, 0]), covs, mu)                     # in the left lane: left is clipped
    assert np.allclose(lane_probability(edge, ROAD), [0.0, 0.1, 0.9])
    stacked = IMMState(np.stack([means, edge.means]), np.stack([covs, covs]), np.stack([mu, mu]))
    assert np.allclose(lane_probability(stacked, ROAD), [[0.1, 0.6, 0.3], [0.0, 0.1, 0.9]])
    assert np.allclose(imm_combine(stacked)[0][0], m)


def test_imm_batched_equals_single():
    rng = np.random.default_rng(3)
    tracks = [init_state([s, d, 27.0, 0.0], np.diag([1.0, 0.1, 4.0, 0.1]), mu)
              for s, d, mu in ((0.0, 0.0, None), (40.0, 3.4, (0.5, 0.2, 0.3)), (80.0, -3.7, (0.3, 0.6, 0.1)))]
    stacked = IMMState(*(np.stack([getattr(t, k) for t in tracks]) for k in ('means', 'covs', 'mu')))
    h = lambda X: h_lidar(X, CAR)
    z = np.array([[2.8, 0.1, 0.01], [42.6, 3.5, -0.02], [82.7, -3.6, 0.03]])+rng.normal(size=(3, 3))*0.05
    sb = imm_predict(stacked, 0.1, CFG, ROAD)
    ub, llb, nisb = imm_update(sb, z, LIDAR_R, h, residual_lidar)
    for i, t in enumerate(tracks):
        u, ll, nis = imm_update(imm_predict(t, 0.1, CFG, ROAD), z[i], LIDAR_R, h, residual_lidar)
        assert np.allclose(u.means, ub.means[i]) and np.allclose(u.covs, ub.covs[i]) and np.allclose(u.mu, ub.mu[i])
        assert np.isclose(ll, llb[i]) and np.allclose(nis, nisb[i])
    assert imm_predict(stacked, 0.0, CFG, ROAD).means is not stacked.means


def _lane_change_run(start, direction, seeds=20, dt=0.05, v=27.0, t_start=3.0, t_end=9.0):
    """IMM on a traffic-style lane change (quintic, 3 s, lane start -> start + direction, from t_start) with lidar
    boxes at 20 Hz (two team lidars), one filter per seed in one batch. Returns times, mode probabilities (steps,
    seeds, 3) and lane probabilities (steps, seeds, lanes)."""
    rng = np.random.default_rng(11)
    lc, w, c0 = SC.traffic.lane_change_time, SC.road.lane_width, lane_centre(start, ROAD)
    times = np.arange(1, int(round(t_end/dt))+1)*dt
    st = init_state(np.tile([0.0, c0, v, 0.0], (seeds, 1))+rng.normal(size=(seeds, 4))*[0.5, 0.1, 1.0, 0.1],
                    np.diag([1.0, 0.1, 4.0, 0.2]))
    h = lambda X: h_lidar(X, CAR)
    LR = np.linalg.cholesky(LIDAR_R)
    mus, lanes = [], []
    for t in times:
        p, dp = _profile((t-t_start)/lc)
        truth = np.array([v*t, c0+direction*w*p, v, direction*w*dp/lc])
        z = h(truth[None])+rng.normal(size=(seeds, 3)) @ LR.T
        st = imm_predict(st, dt, CFG, ROAD)
        st, _, _ = imm_update(st, z, LIDAR_R, h, residual_lidar)
        mus.append(st.mu)
        lanes.append(lane_probability(st, ROAD))
    return times, np.array(mus), np.array(lanes)


@pytest.mark.parametrize('start, direction', [(1, 1), (0, 1), (1, -1), (2, -1)])
def test_imm_detects_lane_change_within_one_second(start, direction):
    times, mu, lanes = _lane_change_run(start, direction)
    j, target = MODELS.index('left' if direction > 0 else 'right'), start+direction
    assert mu[(times > 1.0) & (times < 3.0), :, 1:].max() < 0.3                  # no false alarm while keeping lane
    hit = mu[:, :, j] > 0.7
    first = np.where(hit.any(0), times[np.argmax(hit, 0)], np.inf)-3.0
    assert np.all(first > 0) and np.all(first <= 1.0) and first.mean() < 0.9, first
    after = times > 4.0                                                           # from 1 s in: the target lane,
    assert np.mean(lanes[after].argmax(-1) == target) > 0.97                     # also while completing the change
    assert np.all(lanes[np.isclose(times, 4.0), :, target] > 0.7)
    end = times > 8.0
    assert np.all(mu[end, :, 0] > 0.7) and np.all(lanes[end, :, target] > 0.9)  # settled in the new lane
