"""IMM motion models and the sensor measurement models of the track state x = [s, d, v_s, v_d] (design section 6).

The state is the traffic car's CG in road coordinates (world frame = road frame on the straight road); its heading is
atan2(v_d, v_s) (v_s floored at V_MIN, as types.TruthHistory). Every car has the scenario's vehicle size `params`
(Scenario.vehicle) and Perception.car_height. All functions take sigma points X (k, 4) (or (..., 4)).

API (the tracker builds on it):
- MODELS = ('keep', 'left', 'right'): lane keep, change left (+d, lane + 1), change right (-d, lane - 1).
- motion(model, cfg, road) -> f(X, dt): one exact-discretized step for TrackerConfig cfg and nohesi.road.Road road
  (or its config). Along the road v_s is constant. Across it:
  - keep: d returns to the centre c of lane_of(d) and v_d to 0, critically damped at cfg.lane_keep_rate k
    (d'' = -2k d' - k^2 (d - c)): no overshoot, v_d stays a real state (a car caught mid-change carries on a little,
    then settles), and the step is the closed-form matrix exponential;
  - left / right: v_d relaxes toward cfg.lc_speed in the direction of stop_point at the same rate (first order,
    exact), and d stops there with v_d = 0. stop_point is the centre of lane lane_of(d) +- 1; where that lane does
    not exist (an edge lane) it is clipped to the road's edge (+-width/2), so the model stays a drift toward the
    edge, which the data rule out, instead of becoming a copy of lane keep that would hold its probability at the
    switch prior (a car settled in the left lane would read as changing left).
- process_noise(model, cfg, dt) -> Q (4, 4): continuous white acceleration of density q_long^2 on (s, v_s) and
  q_lat^2 on (d, v_d), Q = q^2 [[dt^3/3, dt^2/2], [dt^2/2, dt]] per axis (the same for every model). Full rank and
  consistent under splitting a step, which the tracker's irregular, asynchronous measurement times need.
- lane_of(d, road), lane_centre(lane, road): lane helpers (as nohesi.road.Road); stop_point(model, d, road).
- heading(X): atan2(v_d, max(v_s, V_MIN)), the heading every measurement model gives the car's box.
- target_lane(model, d, road): the lane a model's car is heading to, for imm.lane_probability: keep, lane_of(d);
  left, lane_of(d) + 1 once d is at or left of that lane's centre and lane_of(d) while right of it (the car is
  completing a change into it: the boundary crossing comes halfway through a change, and labelling the second half
  with the lane after would announce a second change for about a second each time); right mirrored; clipped to the
  road. The motion's stop point does not use this rule: holding the change models at the centre would make them a
  copy of lane keep near every centre, and the IMM would then be slower to leave them.
- h_lidar(X, params) -> (k, 3) [x, y, heading] of the body centre (nohesi.geometry.body_centre), as a 'lidar_box'
  Detection; residual_lidar(a, b) wraps the heading.
- h_radar(X, det, params) -> (k, 3) [range, azimuth, range_rate] of the point of the car's box nearest the sensor at
  det.sensor_pose (radar returns come from the near surface; the centre would bias range by ~2 m), range_rate the
  radial component of (car velocity - sensor velocity). The sensor velocity is the carrying car's det.ego body
  velocity rotated to the world plus yaw rate x lever arm (sensor position - ego CG). residual_radar wraps azimuth.
- h_camera(X, det, spec, params, car_height=1.45) -> (k, 4) [u_left, u_right, v_top, v_bottom]: the 3D box
  (geometry.box_corners_3d) projected with geometry.project at det.ego.pitch, clipped against the near plane
  (NEAR m in front of the camera: a box reaching behind the camera projects to the image edge, as a detector sees it)
  and then to the image [0, width_px] x [0, height_px] like a detector box. camera_R(det, spec) -> R with truncated
  edges' variance TRUNCATED_VAR (det.meta['truncated']: 4 bools [left, right, top, bottom], or one bool meaning
  "touches the border"; missing: edges within BORDER_PX of the border count as truncated).
- measurement(det, params, cameras=(), car_height=1.45) -> (h, R, residual) for any Detection kind, ready for
  ukf.ukf_update / imm.imm_update.
"""
import numpy as np
from ...geometry import body_centre
from ..geometry import wrap, car_boxes, box_corners_3d, project

__all__ = ['MODELS', 'V_MIN', 'NEAR', 'TRUNCATED_VAR', 'BORDER_PX', 'lane_of', 'lane_centre', 'target_lane',
           'stop_point', 'motion', 'process_noise', 'heading', 'h_lidar', 'residual_lidar', 'h_radar', 'residual_radar',
           'sensor_velocity', 'h_camera', 'camera_R', 'truncated_edges', 'measurement', 'RESIDUALS']

MODELS = ('keep', 'left', 'right')
V_MIN = 1e-3            # m/s: speed floor of the heading (as types.TruthHistory)
NEAR = 0.1              # m: camera near plane
TRUNCATED_VAR = 1e6     # px^2: variance of a truncated box edge (it carries no information)
BORDER_PX = 2.0         # an edge this close to the image border counts as truncated when meta has no flags
_STEP = {'keep': 0, 'left': 1, 'right': -1}


def _road(road):
    cfg = getattr(road, 'cfg', road)
    return int(cfg.lanes), float(cfg.lane_width)


def lane_of(d, road):
    """Lane index of lateral position(s) d, clipped to the road (lane 0 rightmost), as Road.lane_of."""
    lanes, w = _road(road)
    return np.clip(np.floor((np.asarray(d, float)+lanes*w/2)/w), 0, lanes-1).astype(int)


def lane_centre(lane, road):
    lanes, w = _road(road)
    return (np.asarray(lane, float)+0.5)*w-lanes*w/2


def target_lane(model, d, road):
    """The lane a model's car is heading to from d (module docstring): keep, lane_of(d); a change, the next lane in
    its direction once d has reached its own lane's centre, else its own lane; clipped to the road."""
    lanes, _ = _road(road)
    lane = lane_of(d, road)
    step = _STEP[model]
    ahead = step*(np.asarray(d, float)-lane_centre(lane, road)) >= -1e-9
    return np.clip(np.where(ahead, lane+step, lane), 0, lanes-1)


def stop_point(model, d, road):
    """Where a model's lateral motion ends from d: keep, the centre of lane_of(d); left / right, the centre of lane
    lane_of(d) +- 1, clipped to the road's edge (+-width/2) when that lane does not exist."""
    lanes, w = _road(road)
    return np.clip(lane_centre(lane_of(d, road)+_STEP[model], road), -lanes*w/2, lanes*w/2)


def _check(model):
    if model not in _STEP:
        raise ValueError(f'unknown motion model {model!r}; expected one of {MODELS}')


def motion(model, cfg, road):
    """f(X, dt) -> X after dt for one IMM model (module docstring)."""
    _check(model)
    k, speed, step = float(cfg.lane_keep_rate), float(cfg.lc_speed), _STEP[model]
    lanes, w = _road(road)
    half = lanes*w/2

    def f(X, dt):
        X = np.asarray(X, float)
        s, d, vs, vd = X[..., 0], X[..., 1], X[..., 2], X[..., 3]
        lane = np.clip(np.floor((d+half)/w), 0, lanes-1)
        c = np.clip((lane+step+0.5)*w-half, -half, half)                # stop_point, inlined (hot loop)
        E = np.exp(-k*dt)
        out = np.empty_like(X)
        out[..., 0] = s+vs*dt
        out[..., 2] = vs
        if step == 0:
            e = d-c
            out[..., 1] = c+E*(e*(1+k*dt)+vd*dt)
            out[..., 3] = E*(vd*(1-k*dt)-k*k*dt*e)
            return out
        sgn = np.sign(c-d)
        vt = sgn*speed
        g = -np.expm1(-k*dt)/k if k > 0 else dt
        dn = d+vt*dt+(vd-vt)*g
        stop = sgn*(dn-c) >= 0
        out[..., 1] = np.where(stop, c, dn)
        out[..., 3] = np.where(stop, 0.0, vt+(vd-vt)*E)
        return out
    return f


def process_noise(model, cfg, dt):
    """Q (4, 4) of continuous white acceleration q_long (s, v_s) and q_lat (d, v_d) over dt."""
    _check(model)
    a, b, c = dt**3/3, dt**2/2, dt
    ql, qd = cfg.q_long**2, cfg.q_lat**2
    return np.array([[ql*a, 0.0, ql*b, 0.0], [0.0, qd*a, 0.0, qd*b], [ql*b, 0.0, ql*c, 0.0], [0.0, qd*b, 0.0, qd*c]])


def heading(X):
    """Heading of track states relative to the road: atan2(v_d, max(v_s, V_MIN))."""
    X = np.asarray(X, float)
    return np.arctan2(X[..., 3], np.maximum(X[..., 2], V_MIN))


def _poses(X):
    X = np.asarray(X, float)
    flat = X.reshape(-1, 4)
    return X.shape[:-1], flat, np.column_stack((flat[:, 0], flat[:, 1], heading(flat)))


def h_lidar(X, params):
    """[x, y, heading] of the body centre (k, 3)."""
    shape, _, st = _poses(X)
    return np.column_stack((body_centre(st, params), st[:, 2])).reshape(shape+(3,))


def residual_lidar(a, b):
    r = np.subtract(a, b)
    r[..., 2] = wrap(r[..., 2])
    return r


def _nearest_points(origin, boxes):
    """Points of box polygons (k, 4, 2) nearest to origin (2,): (k, 2), as geometry.nearest_box_point."""
    a, ab = boxes, np.roll(boxes, -1, axis=1)-boxes
    t = np.clip(np.einsum('kij,kij->ki', origin-a, ab)/np.einsum('kij,kij->ki', ab, ab), 0, 1)
    p = a+t[..., None]*ab
    j = np.argmin(np.einsum('kij,kij->ki', p-origin, p-origin), axis=1)
    return p[np.arange(len(p)), j]


def sensor_velocity(det):
    """World velocity (2,) of the sensor of a Detection: ego body velocity rotated by its yaw plus yaw rate x lever
    arm (sensor position - ego CG)."""
    e = det.ego
    c, s = np.cos(e.yaw), np.sin(e.yaw)
    lx, ly = det.sensor_pose[0]-e.x, det.sensor_pose[1]-e.y
    return np.array([c*e.vx-s*e.vy-e.r*ly, s*e.vx+c*e.vy+e.r*lx])


def h_radar(X, det, params):
    """[range, azimuth, range_rate] (k, 3) of the box point nearest the sensor at det.sensor_pose."""
    shape, flat, st = _poses(X)
    o = np.asarray(det.sensor_pose[:2], float)
    rel = _nearest_points(o, car_boxes(st, params))-o
    rng = np.hypot(rel[:, 0], rel[:, 1])
    az = wrap(np.arctan2(rel[:, 1], rel[:, 0])-det.sensor_pose[2])
    vrel = flat[:, 2:4]-sensor_velocity(det)
    rr = np.einsum('ki,ki->k', vrel, rel)/np.maximum(rng, 1e-9)
    return np.column_stack((rng, az, rr)).reshape(shape+(3,))


def residual_radar(a, b):
    r = np.subtract(a, b)
    r[..., 1] = wrap(r[..., 1])
    return r


_EDGES = np.array([(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])


def h_camera(X, det, spec, params, car_height=1.45):
    """Predicted detector box [u_left, u_right, v_top, v_bottom] (k, 4) of the track's 3D box in camera `spec` at
    det.sensor_pose and det.ego.pitch. A box with nothing in front of the near plane (a sigma point behind the
    camera) predicts the degenerate box at the image centre."""
    shape, _, st = _poses(X)
    pose = det.sensor_pose
    pitch = float(det.ego.pitch) if det.ego is not None else 0.0
    P = box_corners_3d(st, params, car_height)                       # (k, 8, 3)
    u, v, depth = project(spec, pose, P, pitch=pitch)
    a, b = _EDGES.T
    da, db = depth[:, a], depth[:, b]
    cross = (da-NEAR)*(db-NEAR) < 0
    t = np.where(cross, (NEAR-da)/np.where(cross, db-da, 1.0), 0.0)
    ue, ve, _ = project(spec, pose, P[:, a]+t[..., None]*(P[:, b]-P[:, a]), pitch=pitch)
    U, V = np.concatenate((u, ue), 1), np.concatenate((v, ve), 1)
    ok = np.concatenate((depth > NEAR, cross), 1)
    W, H = float(spec.width_px), float(spec.height_px)
    box = np.column_stack((np.where(ok, U, np.inf).min(1), np.where(ok, U, -np.inf).max(1),
                           np.where(ok, V, np.inf).min(1), np.where(ok, V, -np.inf).max(1)))
    box = np.clip(box, 0.0, np.array([W, W, H, H]))
    box[~ok.any(1)] = (W/2, W/2, H/2, H/2)
    return box.reshape(shape+(4,))


def truncated_edges(z, spec, tol=BORDER_PX):
    """(4,) bool: box edges [left, right, top, bottom] within tol pixels of the image border."""
    z = np.asarray(z, float)
    return np.array([z[0] <= tol, z[1] >= spec.width_px-tol, z[2] <= tol, z[3] >= spec.height_px-tol])


def camera_R(det, spec):
    """The camera Detection's R (4, 4) with each truncated edge decoupled and given variance TRUNCATED_VAR."""
    R = np.array(det.R, float, copy=True)
    flags = det.meta.get('truncated') if det.meta else None
    if flags is None or np.ndim(flags) == 0:
        flags = truncated_edges(det.z, spec) if flags is None or bool(flags) else np.zeros(4, bool)
    idx = np.flatnonzero(np.asarray(flags, bool).reshape(4))
    R[idx, :] = 0.0
    R[:, idx] = 0.0
    R[idx, idx] = TRUNCATED_VAR
    return R


RESIDUALS = {'lidar_box': residual_lidar, 'radar': residual_radar, 'camera_box': None}


def measurement(det, params, cameras=(), car_height=1.45):
    """(h, R, residual) of a Detection for ukf_update / imm_update. cameras: the CameraSpecs (or a dict by name),
    needed for 'camera_box'."""
    R = np.asarray(det.R, float)
    if det.kind == 'lidar_box':
        return (lambda X: h_lidar(X, params)), R, residual_lidar
    if det.kind == 'radar':
        return (lambda X: h_radar(X, det, params)), R, residual_radar
    if det.kind == 'camera_box':
        specs = cameras if isinstance(cameras, dict) else {c.name: c for c in cameras}
        if det.sensor not in specs:
            raise KeyError(f'no CameraSpec named {det.sensor!r}')
        spec = specs[det.sensor]
        return (lambda X: h_camera(X, det, spec, params, car_height)), camera_R(det, spec), None
    raise ValueError(f'unknown detection kind {det.kind!r}')
