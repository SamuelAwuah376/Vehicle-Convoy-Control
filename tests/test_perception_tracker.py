"""Team tracker (tracking/tracker.py, association.py, existence.py): the batched pieces against the filter and geometry
modules, then synthetic multi-car scenes. The scenes generate noisy lidar, radar and camera Detections from known
trajectories with the tracker's own measurement models (models.h_lidar / h_radar / h_camera), occlusion-aware
detection and clutter; nohesi.perception.sim is not used."""
import time
from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import chi2
from nohesi.config import scenario_v9
from nohesi.road import Road
from nohesi.traffic import _profile
from nohesi.perception import geometry as G
from nohesi.perception.config import TrackerConfig, default_lidars, default_radars, default_cameras
from nohesi.perception.types import Detection, EgoState
from nohesi.perception.tracking import association as A
from nohesi.perception.tracking import existence as E
from nohesi.perception.tracking.imm import init_state, imm_predict, imm_transition, imm_update, IMMState
from nohesi.perception.tracking.models import (MODELS, motion, process_noise, measurement, h_lidar, h_radar,
                                               h_camera, camera_R, lane_centre, TRUNCATED_VAR)
from nohesi.perception.tracking.tracker import (TeamTracker, ScanInfo, NIS_DTYPE, predict_rows, joint_update,
                                                transition_rows, noise_rows)
from nohesi.perception.tracking.ukf import sigma_points

SC = scenario_v9()
CAR = SC.vehicle
ROAD = Road(SC.road)
CFG = TrackerConfig()
H = 1.45
LIDARS, RADARS, CAMERAS = default_lidars(), default_radars(), default_cameras()
SPECS = {s.name: s for s in (*LIDARS, *RADARS, *CAMERAS)}
MOTIONS = tuple(motion(m, CFG, ROAD) for m in MODELS)
DT = 0.05
LIDAR_R = np.diag([0.08**2, 0.08**2, np.radians(1.0)**2])


def lane(l):
    return float(lane_centre(l, ROAD))


# ------------------------------------------------------------------------------------------------ synthetic scenes
class Car:
    """A truth trajectory: constant speed v from s0 in lane `lane`, plus quintic lane changes (t_start, duration,
    lanes) as the traffic model drives them."""
    def __init__(self, s0, lane_, v, changes=()):
        self.s0, self.d0, self.v, self.changes = float(s0), lane(lane_), float(v), tuple(changes)

    def state(self, t):
        d, vd = self.d0, 0.0
        for ts, dur, dl in self.changes:
            p, dp = _profile((t-ts)/dur)
            d += dl*SC.road.lane_width*p
            vd += dl*SC.road.lane_width*dp/dur
        return np.array([self.s0+self.v*t, d, self.v, vd])


class Scene:
    """Detections of every sensor of the team cars (default suite), drawn with rng: radar and camera every step,
    lidar every other step (sweeps ending at the step, each car's box stamped at its beam time). Detection
    probabilities follow the tracker's nominal models with the true visible fraction (radar keeps occluded_pd of a
    hidden car), so the generator and the tracker agree; clutter as Poisson radar points (clutter_scale of the spec's
    false-alarm rate: the radar processing's stationary test removes most) and camera boxes."""
    def __init__(self, traffic, team, seed=0, clutter=True, specs=SPECS, split=0.0, clutter_scale=0.3):
        self.traffic, self.team, self.specs = list(traffic), list(team), dict(specs)
        self.rng = np.random.default_rng(seed)
        self.clutter, self.split, self.clutter_scale = clutter, split, clutter_scale

    def truth(self, t):
        return np.array([c.state(t) for c in self.traffic]).reshape(-1, 4)

    def team_states(self, t):
        X = np.array([c.state(t) for c in self.team])
        return np.column_stack((X[:, 0], X[:, 1], np.arctan2(X[:, 3], X[:, 2]), np.hypot(X[:, 2], X[:, 3])))

    def ego(self, c, t):
        x, y, yaw, v = self.team_states(t)[c]
        return EgoState(t, x, y, yaw, v)

    def _vis(self, pose, X, team, c):
        boxes = np.concatenate((G.car_boxes(np.column_stack((X[:, :2], np.arctan2(X[:, 3], X[:, 2]))), CAR),
                                G.car_boxes(team[:, :3], CAR)))
        return E.visible_fractions(pose[:2], boxes, np.arange(len(X)), skip=[len(X)+c]) if len(X) else np.zeros(0)

    def step(self, t, lidar):
        """(detections, scan infos) of every scan at time t."""
        out, infos = [], []
        team = self.team_states(t)
        X = self.truth(t)
        rng = self.rng
        for c in range(len(self.team)):
            ego = self.ego(c, t)
            for spec in self.specs.values():
                if isinstance(spec, type(LIDARS[0])):
                    if lidar:
                        out += self._lidar(spec, c, t)
                        infos.append(ScanInfo(spec.name, c, t, G.sensor_pose(spec.mount, team[c], CAR)))
                    continue
                pose = G.sensor_pose(spec.mount, team[c], CAR)
                infos.append(ScanInfo(spec.name, c, t, pose))
                tmpl = Detection('radar', spec.name, c, t, np.zeros(3), np.eye(3), pose, ego, {'scan': t})
                vis = self._vis(pose, X, team, c)
                if isinstance(spec, type(RADARS[0])):
                    zt = h_radar(X, tmpl, CAR) if len(X) else np.zeros((0, 3))
                    r, az = zt[:, 0], np.abs(zt[:, 1])
                    cov = (r <= spec.range_max) & (az <= np.where(r <= spec.near_range, spec.fov, spec.fov_far)/2)
                    pd = E.radar_pd_range(spec, r)*(spec.occluded_pd+(1-spec.occluded_pd)*vis)
                    R = np.diag([spec.sigma_range, spec.sigma_azimuth, spec.sigma_range_rate])**2
                    for k in np.flatnonzero(cov & (rng.random(len(X)) < pd)):
                        z = zt[k]+rng.normal(size=3)*np.sqrt(np.diag(R))
                        out.append(Detection('radar', spec.name, c, t, z, R, pose, ego, {'n_points': 2, 'scan': t}))
                    for _ in range(rng.poisson(spec.false_alarms*self.clutter_scale) if self.clutter else 0):
                        z = np.array([rng.uniform(5, spec.range_max), rng.uniform(-spec.fov/2, spec.fov/2),
                                      rng.uniform(-40, 10)])
                        out.append(Detection('radar', spec.name, c, t, z, R, pose, ego, {'n_points': 1, 'scan': t}))
                else:
                    out += self._camera(spec, c, t, pose, ego, X, vis)
        return out, infos

    def _camera(self, spec, c, t, pose, ego, X, vis):
        rng, out = self.rng, []
        R = spec.sigma_px**2*np.eye(4)
        tmpl = Detection('camera_box', spec.name, c, t, np.zeros(4), R, pose, ego, {'scan': t})
        if len(X):
            rel = G.to_sensor(X[:, :2], pose)
            depth, az = rel[:, 0], np.arctan2(rel[:, 1], rel[:, 0])
            zt = h_camera(X, tmpl, spec, CAR, H)
            ok = (depth > 3) & (np.abs(az) <= spec.hfov/2) & (zt[:, 3]-zt[:, 2] >= spec.min_height_px)
            ok &= (vis >= spec.min_visible) & (rng.random(len(X)) < spec.pd)
            for k in np.flatnonzero(ok):
                z = np.clip(zt[k]+rng.normal(0, spec.sigma_px, 4), 0, [spec.width_px]*2+[spec.height_px]*2)
                r = np.hypot(*rel[k])
                u = (X[k, :2]-pose[:2])/r
                Rot = np.array([[u[0], -u[1]], [u[1], u[0]]])
                P = Rot @ np.diag([(0.03*r+0.3)**2, 0.25**2]) @ Rot.T
                xy = X[k, :2]+np.linalg.cholesky(P) @ rng.normal(size=2)
                out.append(Detection('camera_box', spec.name, c, t, z, R, pose, ego,
                                     {'init': (xy, P), 'score': 0.9, 'scan': t}))
        if self.clutter and rng.random() < spec.false_rate:
            r, a = rng.uniform(10, 60), rng.uniform(-spec.hfov/3, spec.hfov/3)
            xy = np.array(pose[:2])+r*np.array([np.cos(pose[2]+a), np.sin(pose[2]+a)])
            fake = np.array([[xy[0], xy[1], 0.0, 0.0]])
            z = h_camera(fake, tmpl, spec, CAR, H)[0]
            P = np.diag([(0.03*r+0.3)**2, 0.25**2])
            out.append(Detection('camera_box', spec.name, c, t, z, R, pose, ego, {'init': (xy, P), 'score': 0.4,
                                                                                    'scan': t}))
        return out

    def _lidar(self, spec, c, t_end):
        rng, out = self.rng, []
        t0 = t_end-1/spec.rate
        team0 = self.team_states(t_end)
        pose_e = G.sensor_pose(spec.mount, team0[c], CAR)
        X = self.truth(t_end)
        if not len(X):
            return out
        rel = G.to_sensor(X[:, :2], pose_e)
        rng_, az = np.hypot(rel[:, 0], rel[:, 1]), np.arctan2(rel[:, 1], rel[:, 0])
        vis = self._vis(pose_e, X, team0, c)
        ok = (rng_ <= spec.range_max) & (np.abs(az) <= spec.fov/2) & (vis >= 0.3) & (rng.random(len(X)) < 0.98)
        for k in np.flatnonzero(ok):
            tau = t0+(az[k]+spec.fov/2)/spec.fov*(t_end-t0)
            ego = self.ego(c, tau)
            pose = G.sensor_pose(spec.mount, self.team_states(tau)[c], CAR)
            zt = h_lidar(self.traffic[k].state(tau)[None], CAR)[0]
            for off in ((-self.split, self.split) if self.split else (0.0,)):
                z = zt+np.array([off, 0, 0])+rng.normal(size=3)*np.sqrt(np.diag(LIDAR_R))
                out.append(Detection('lidar_box', spec.name, c, tau, z, LIDAR_R.copy(), pose, ego,
                                     {'n_points': 25, 'sides': 2, 'scan': t_end}))
        return out


def run(scene, tracker, t_end, t0=0.0, infos=True, latency=None, rng=None, observe=None):
    """Feed the scene's scans to the tracker step by step. latency None: delivered at their own time; else each scan
    is delivered latency(rng) s late (out of order across scans). observe(t, tracker) after every step."""
    queue = []
    steps = int(round((t_end-t0)/DT))
    for k in range(steps+1):
        t = round(t0+k*DT, 10)
        dets, inf = scene.step(t, lidar=k % 2 == 0)
        if latency is None:
            tracker.process(dets, t, scene.team_states(t), inf if infos else ())
        else:
            by = {}
            for d in dets:
                by.setdefault((d.sensor, d.car), []).append(d)
            for i in inf if infos else ():
                by.setdefault((i.sensor, i.car), [])
            for key, ds in by.items():
                due = t+latency(rng)
                queue.append((due, ds, [i for i in inf if (i.sensor, i.car) == key] if infos else []))
            ready = [q for q in queue if q[0] <= t+1e-9]
            queue = [q for q in queue if q[0] > t+1e-9]
            tracker.process([d for q in ready for d in q[1]], t, scene.team_states(t), [i for q in ready for i in q[2]])
        if observe is not None:
            observe(t, tracker)
    if queue:
        tracker.process([d for q in queue for d in q[1]], t, scene.team_states(t), [i for q in queue for i in q[2]])
    return tracker


def make_tracker(specs=SPECS, **kw):
    return TeamTracker(CFG, ROAD, CAR, specs, H, **kw)


def platoon(v=30.0):
    return [Car(0.0, 1, v), Car(-12.0, 0, v), Car(-24.0, 2, v)]


def match(belief, X, max_dist=3.0):
    """Confirmed track index per truth car (-1 when none within max_dist m)."""
    b = belief.select(belief.confirmed)
    out = np.full(len(X), -1)
    if len(b) == 0:
        return out, b
    d = np.hypot(b.mean[None, :, 0]-X[:, None, 0], b.mean[None, :, 1]-X[:, None, 1])
    r, c = A.gnn(np.where(d < max_dist, d, np.inf))
    out[r] = c
    return out, b


# ------------------------------------------------------------------------------------------------ batched pieces
def test_predict_rows_matches_imm_predict():
    rng = np.random.default_rng(0)
    m = np.column_stack((rng.normal(0, 50, 6), rng.uniform(-5, 5, 6), rng.uniform(20, 35, 6), rng.normal(0, 0.5, 6)))
    A_ = rng.normal(size=(6, 4, 4))
    st = init_state(m, A_ @ np.swapaxes(A_, 1, 2)*0.1+0.05*np.eye(4), (0.7, 0.2, 0.1))
    st.means += rng.normal(0, 0.2, st.means.shape)
    dts = np.array([0.05, 0.0, 0.1, 0.37, 0.05, 1.2])
    out = predict_rows(st.means, st.covs, st.mu, dts, CFG, MOTIONS)
    for i, dt in enumerate(dts):
        ref = imm_predict(IMMState(st.means[i], st.covs[i], st.mu[i]), dt, CFG, ROAD)
        assert np.allclose(out.means[i], ref.means, atol=1e-10) and np.allclose(out.covs[i], ref.covs, atol=1e-10)
        assert np.allclose(out.mu[i], ref.mu, atol=1e-12)
    assert np.array_equal(out.means[1], st.means[1]) and np.array_equal(out.covs[1], st.covs[1])
    for dt in (0.0, 0.02, 0.3, 4.0):
        assert np.allclose(transition_rows(CFG.imm_switch, [dt])[0], imm_transition(CFG, dt), atol=1e-13)
        assert np.allclose(noise_rows(CFG, [dt])[0], process_noise('keep', CFG, dt))
    # a small backward step moves the state back along its velocity; back and forth again returns to the start with
    # the covariance grown by the two steps' process noise
    back = predict_rows(st.means[:1], st.covs[:1], st.mu[:1], [-0.05], CFG, MOTIONS)
    assert np.allclose(back.means[0, :, 0], st.means[0, :, 0]-0.05*st.means[0, :, 2], atol=5e-3)   # plus mixing
    one = init_state([10.0, 0.2, 30.0, 0.1], np.diag([0.04, 0.01, 0.25, 0.04]))      # mid lane, models alike
    back = predict_rows(one.means[None], one.covs[None], one.mu[None], [-0.05], CFG, MOTIONS)
    trip = predict_rows(back.means, back.covs, back.mu, [0.05], CFG, MOTIONS)
    assert np.allclose(trip.means[0, 0], one.means[0], atol=1e-4)     # lane keep is linear; the rest is mixing
    assert np.all(np.linalg.eigvalsh(trip.covs[0, 0]-one.covs[0]) > -1e-12)


def test_visible_fractions_match_geometry():
    rng = np.random.default_rng(1)
    seen = []
    for trial in range(12):
        st = []
        while len(st) < 14:
            c = np.array([rng.uniform(-10, 80), rng.choice([-3.6, 0.0, 3.6])+rng.normal(0, 0.3), rng.normal(0, 0.05)])
            if all(abs(c[0]-o[0]) > 5.5 or abs(c[1]-o[1]) > 2.2 for o in st):
                st.append(c)
        boxes = G.car_boxes(np.array(st), CAR)
        origin = np.array([rng.uniform(-30, 30), rng.uniform(-4, 4)])
        skip = [0, 5]                   # the carrying car, and a box that is a target but blocks nothing
        ours = E.visible_fractions(origin, boxes, np.arange(1, 14), skip=skip)
        ref = [G.visible_fraction(origin, boxes, k, samples=E.VIS_SAMPLES, skip=skip) for k in range(1, 14)]
        assert np.allclose(ours, ref)
        seen.append(ref)
    seen = np.concatenate(seen)
    assert np.any(seen == 1.0) and np.any((seen > 0) & (seen < 1)) and np.any(seen == 0)
    assert np.array_equal(E.visible_fractions(origin, boxes, []), np.zeros(0))
    # several origins in one call, each with its own skipped boxes and pairs to evaluate; one origin inside a box
    origins = np.array([[rng.uniform(-30, 30), rng.uniform(-4, 4)] for _ in range(5)]+[st[3][:2]+[0.4, 0.2]])
    skip = np.zeros((6, len(boxes)), bool)
    skip[np.arange(5), rng.integers(0, len(boxes), 5)] = True
    want = rng.random((6, len(boxes))) < 0.8
    ours = E.visibility(origins, boxes, np.arange(len(boxes)), skip, want)
    for c in range(6):
        for k in range(len(boxes)):
            ref = G.visible_fraction(origins[c], boxes, k, samples=E.VIS_SAMPLES, skip=np.flatnonzero(skip[c]))
            assert np.isclose(ours[c, k], ref if want[c, k] else 0.0), (c, k)
    assert np.all(ours[5] == 0.0)                     # every ray from inside a box leaves it first


def test_small_matrix_helpers():
    rng = np.random.default_rng(14)
    M_ = rng.normal(size=(7, 3, 4, 4))
    S = M_ @ np.swapaxes(M_, -1, -2)+0.1*np.eye(4)
    S[0, 0] = np.diag([1.0, 1.0, 0.0, 1.0])                     # singular: the robust fallback
    L = A.cholesky(S)
    assert np.allclose(L[1:], np.linalg.cholesky(S[1:])) and np.allclose(L[0, 1:], np.linalg.cholesky(S[0, 1:]))
    assert np.all(np.isfinite(L[0, 0])) and np.allclose(L[0, 0] @ L[0, 0].T, S[0, 0], atol=1e-6)
    v, V = rng.normal(size=(7, 3, 4)), rng.normal(size=(7, 3, 4, 2))
    assert np.allclose(A.forward(L[1:], v[1:]), np.linalg.solve(L[1:], v[1:, ..., None])[..., 0])
    assert np.allclose(A.forward(L[1:], V[1:]), np.linalg.solve(L[1:], V[1:]))
    assert np.allclose(A.logdet(L[1:]), np.linalg.slogdet(S[1:])[1])
    a = rng.uniform(-12, 12, 1000)
    assert np.allclose(A.wrap_angle(a), G.wrap(a), atol=1e-12)


def test_truncated_camera_edges_carry_nothing():
    """A camera box cut by the image border: the cut edge leaves the update (variance TRUNCATED_VAR), its degree of
    freedom, the log-likelihood (ll_pad) and the clutter density alike, so a cut box of a tracked car supports its
    existence like a whole one instead of reading as clutter."""
    spec, ego = CAMERAS[0], EgoState(0.0, 0.0, 0.0, 0.0, 30.0)
    pose = G.sensor_pose(spec.mount, [0.0, 0.0, 0.0], CAR)
    x = np.array([[9.0, -2.4, 30.0, 0.0]])
    tmpl = Detection('camera_box', spec.name, 0, 0.0, np.zeros(4), np.eye(4), pose, ego, {})
    z = h_camera(x, tmpl, spec, CAR, H)[0]
    cut = z.copy()
    cut[0] = 0.0
    dets = [Detection('camera_box', spec.name, 0, 0.0, zz, 4.0*np.eye(4), pose, ego,
                      {'truncated': np.array([k == 1, False, False, False])}) for k, zz in enumerate((z, cut))]
    lc = np.log(E.clutter_density(spec))
    tab = A.DetTable.build([(dets, spec, lc)], CAR)
    assert list(tab.dof[:2]) == [4, 3] and tab.R[1, 0, 0] == TRUNCATED_VAR and tab.R[1, 0, 1] == 0
    assert np.isclose(tab.log_clutter[1]-tab.log_clutter[0], E.clutter_log_volumes(spec)[0])
    st = init_state(x[0], np.diag([0.3, 0.1, 1.0, 0.1]))
    X, wm, wc = sigma_points(st.means[None], st.covs[None])
    dj = np.arange(2)
    mo = A.moments(tab, dj, A.evaluate(tab, dj, np.repeat(X, 2, 0), CAR, H), wm, wc)
    ll, _, _ = A.imm_gate(mo, tab.R[dj], np.repeat(st.mu[None], 2, 0), tab, dj)
    lr = ll+tab.ll_pad[dj]-tab.log_clutter[dj]
    assert lr[1] > 0 and abs(lr[1]-lr[0]) < 0.5*lr[0]


def _pair_dets(ego, x):
    dets = []
    for spec, k in ((RADARS[0], 0), (RADARS[1], 1), (CAMERAS[0], 2), (LIDARS[0], 0)):
        pose = G.sensor_pose(spec.mount, [ego.x, ego.y, ego.yaw], CAR)
        kind = E.kind_of(spec)
        m = 4 if kind == 'camera_box' else 3
        tmpl = Detection(kind, spec.name, 0, 1.0, np.zeros(m), np.eye(m), pose, ego, {})
        h, _, _ = measurement(tmpl, CAR, CAMERAS, H)
        z = h(x[k:k+1])[0]+0.3
        R = np.diag([0.04, 0.03, 0.05, 0.02][:m]) if kind != 'camera_box' else 4.0*np.eye(4)
        dets.append(Detection(kind, spec.name, 0, 1.0, z, R, pose, ego, {}))
    return dets


def test_single_pairs_equal_imm_update():
    """One detection per track through association.evaluate / moments / imm_gate and tracker.joint_update equals
    imm_update with models.measurement, for every kind, all pairs in one call; the gate's log-likelihood is
    imm_update's once the padding is removed (ll_pad)."""
    ego = EgoState(1.0, 0.0, 0.0, 0.02, 29.0, 0.1, 0.01)
    x = np.array([[25.0, 3.4, 27.0, 0.2], [-30.0, -3.0, 33.0, 0.0], [12.0, 0.5, 30.0, -0.4]])
    st = init_state(x, np.diag([1.0, 0.2, 2.0, 0.3]), (0.8, 0.1, 0.1))
    st.means += np.random.default_rng(1).normal(0, 0.3, st.means.shape)
    dets = _pair_dets(ego, x)
    tab = A.DetTable.build([([d], SPECS[d.sensor], 0.0) for d in dets], CAR)
    rows, dj = [0, 1, 2, 0], np.arange(4)
    sub = IMMState(st.means[rows], st.covs[rows], st.mu[rows])
    X, wm, wc = sigma_points(sub.means, sub.covs)
    mo = A.moments(tab, dj, A.evaluate(tab, dj, X, CAR, H), wm, wc)
    ll, nis, _ = A.imm_gate(mo, tab.R[dj], sub.mu, tab, dj)
    post = joint_update(sub, X, wm, np.arange(4), mo.nu, mo.Pzz, mo.D, tab.R[dj])
    for n, (d, r) in enumerate(zip(dets, rows)):
        hh, RR, rr = measurement(d, CAR, CAMERAS, H)
        ref, lref, nref = imm_update(IMMState(st.means[r], st.covs[r], st.mu[r]), d.z, RR, hh, rr)
        assert np.allclose(post.means[n], ref.means, atol=1e-9) and np.allclose(post.covs[n], ref.covs, atol=1e-11)
        assert np.allclose(post.mu[n], ref.mu, atol=1e-12)
        assert np.isclose(ll[n]+tab.ll_pad[n], lref)
    # several detections of one track: the information-form fusion is close to applying them one after another
    one = IMMState(st.means[:1], st.covs[:1], st.mu[:1])
    X1, wm, wc = sigma_points(one.means, one.covs)
    dj = np.array([0, 3])
    Xs = np.repeat(X1, 2, axis=0)
    mo = A.moments(tab, dj, A.evaluate(tab, dj, Xs, CAR, H), wm, wc)
    joint = joint_update(one, X1, wm, np.array([0]), mo.nu, mo.Pzz, mo.D, tab.R[dj])
    seq = IMMState(st.means[0], st.covs[0], st.mu[0])
    for j in dj:
        hh, RR, rr = measurement(dets[j], CAR, CAMERAS, H)
        seq = imm_update(seq, dets[j].z, RR, hh, rr)[0]
    sd = np.sqrt(np.diagonal(seq.covs, axis1=-2, axis2=-1))
    assert np.all(np.abs(joint.means[0]-seq.means) < 0.05*sd)
    assert np.allclose(joint.covs[0], seq.covs, rtol=0.05, atol=1e-6) and np.allclose(joint.mu[0], seq.mu, atol=0.02)
    # the per-row radar model equals models.h_radar, also from inside a box
    X = np.random.default_rng(2).normal([20, 1, 28, 0], [5, 2, 2, 0.5], size=(9, 4))
    X[0, :2] = (dets[0].sensor_pose[0]+1.0, dets[0].sensor_pose[1]+0.3)
    d = dets[0]
    assert np.allclose(A.h_radar_rows(X, np.tile(d.sensor_pose[:2], (9, 1)), np.full(9, d.sensor_pose[2]),
                                      np.tile(tab.vs[0], (9, 1)), CAR), h_radar(X, d, CAR))


def test_camera_rows_and_truncation_match_models():
    rng = np.random.default_rng(3)
    egos = [EgoState(0.0, 0.0, 0.0, 0.0, 30.0, pitch=0.01), EgoState(0.0, 5.0, -3.6, 0.03, 28.0, pitch=-0.02)]
    poses, cams, X, ref = [], [], [], []
    for spec in CAMERAS:
        for ego in egos:
            pose = G.sensor_pose(spec.mount, [ego.x, ego.y, ego.yaw], CAR)
            det = Detection('camera_box', spec.name, 0, 0.0, np.zeros(4), np.eye(4), pose, ego, {})
            ahead = np.r_[rng.uniform(-4, 4, 6), rng.uniform(4, 60, 6)][:, None]
            rel = np.array([[np.cos(pose[2]), np.sin(pose[2])]])*ahead
            x = np.column_stack((pose[0]+rel[:, 0]+rng.normal(0, 2, 12), pose[1]+rel[:, 1]+rng.normal(0, 2, 12),
                                 rng.uniform(20, 35, 12), rng.normal(0, 1, 12)))
            ref.append(h_camera(x, det, spec, CAR, H))
            X.append(x)
            poses.append(np.tile(pose, (12, 1)))
            cams.append(np.tile([spec.width_px, spec.height_px, G.focal(spec), ego.pitch, spec.mount.yaw], (12, 1)))
    X, poses, cams, ref = (np.concatenate(v) for v in (X, poses, cams, ref))
    ours = A.h_camera_rows(X, poses, cams, CAR, H)
    assert np.allclose(ours, ref, atol=1e-7)
    # some boxes reach behind the near plane and some lie wholly behind the camera
    corners = G.box_corners_3d(np.column_stack((X[:, :2], np.zeros(len(X)))), CAR, H)
    depth = np.array([G.project(CAMERAS[0], p, c)[2] for p, c in zip(poses, corners)])
    assert ((depth <= 0.1).any(1) & (depth > 0.1).any(1)).sum() > 3 and (depth <= 0.1).all(1).sum() > 3
    spec = CAMERAS[0]
    zs = [np.array([0.5, 300, 100, 400]), np.array([100, 1919, 0.0, 1080]), np.array([100, 300, 100, 400])]
    metas = [{}, {'truncated': True}, {'truncated': False}, {'truncated': np.array([1, 0, 0, 1], bool)}]
    for z in zs:
        for meta in metas:
            d = Detection('camera_box', spec.name, 0, 0.0, z, 4.0*np.eye(4), poses[0], egos[0], meta)
            tab = A.DetTable.build([([d], spec, 0.0)], CAR)
            assert np.array_equal(tab.R[0], camera_R(d, spec))


def test_gnn_and_gate():
    C = np.array([[1.0, 2.0], [1.5, np.inf]])
    r, c = A.gnn(C)
    assert sorted(zip(r, c)) == [(0, 1), (1, 0)]          # both assigned beats the cheapest single pair
    r, c = A.gnn(np.full((2, 3), np.inf))
    assert r.size == 0
    r, c = A.gnn(np.array([[np.inf, 3.0, 1.0]]))
    assert list(zip(r, c)) == [(0, 2)]
    assert A.gate_threshold(0.99, 3) == pytest.approx(chi2.ppf(0.99, 3)) and A.gate_threshold(0.99, 0) == np.inf


def test_expected_pd_and_existence_update():
    radar, lid, cam = RADARS[0], LIDARS[0], CAMERAS[0]
    assert E.radar_pd_range(radar, 100.0) == pytest.approx(radar.pd_ref)
    assert E.radar_pd_range(radar, radar.ref_range) == pytest.approx(radar.pd_ref)
    assert E.radar_pd_range(radar, radar.range_max) == pytest.approx(0.5)
    assert E.radar_pd_range(radar, radar.range_max+1) == 0.0
    pd, cov = E.expected_pd(radar, [30.0, 30.0, 30.0], [0.0, 0.0, np.radians(60)], 0.0, [1.0, 0.0, 1.0], CAR, H)
    assert pd[0] == pytest.approx(radar.pd_ref) and pd[1] == pytest.approx(radar.pd_ref*radar.occluded_pd)
    assert pd[2] == 0 and not cov[2]
    pd, _ = E.expected_pd(lid, [20.0, 20.0, 190.0], 0.0, 0.0, [1.0, 0.0, 1.0], CAR, H)
    assert pd[0] > 0.99 and pd[1] == 0 and 0 < pd[2] < pd[0]
    f = G.focal(cam)
    far = f*H/cam.min_height_px
    pd, _ = E.expected_pd(cam, [30.0, far+10, 30.0], 0.0, 0.0, [1.0, 1.0, 0.1], CAR, H)
    assert pd[0] == cam.pd and pd[1] == 0 and pd[2] == 0
    # existence: a miss with pd 0 changes nothing, misses lower and hits raise it, in any order
    p = np.array([0.6])
    assert np.allclose(E.apply_log_odds(p, E.miss_log_factor(0.0, 0.99)), p)
    lo = E.apply_log_odds(p, E.miss_log_factor(0.9, 0.99))
    hi = E.apply_log_odds(p, E.hit_log_factor(0.9, 0.99, 5.0))
    assert lo < p < hi
    f1, f2 = E.miss_log_factor(0.7, 0.99), E.hit_log_factor(0.5, 0.99, 3.0)
    assert np.isclose(E.apply_log_odds(E.apply_log_odds(p, f1, 1.0), f2, 1.0), E.apply_log_odds(p, f1+f2, 1.0))
    assert E.apply_log_odds(np.array([0.5]), 50.0) == pytest.approx(E.EXIST_MAX)
    assert E.survive(1.0, CFG, 2.0) == pytest.approx(CFG.exist_survive**2)


# ------------------------------------------------------------------------------------------------ scenes
def traffic_mix():
    return [Car(35.0, 1, 27.0), Car(60.0, 0, 26.0), Car(15.0, 2, 33.0), Car(-40.0, 1, 31.0), Car(90.0, 2, 30.0),
            Car(-30.0, 0, 29.0), Car(-55.0, 2, 32.5)]


def test_tracks_born_confirmed_accurate():
    scene = Scene(traffic_mix(), platoon(), seed=3)
    tr = run(scene, make_tracker(), 3.0)
    X = scene.truth(3.0)
    b = tr.belief(3.0)
    idx, conf = match(b, X)
    assert np.all(idx >= 0), 'every car near the team has a confirmed track'
    assert len(conf) == len(X), 'no confirmed false tracks'
    est = conf.mean[idx]
    assert np.all(np.abs(est[:, 0]-X[:, 0]) < 0.6) and np.all(np.abs(est[:, 1]-X[:, 1]) < 0.3)
    assert np.all(np.abs(est[:, 2]-X[:, 2]) < 0.8) and np.all(np.abs(est[:, 3]) < 0.4)
    assert np.all(conf.p_exist > 0.9)
    lane_true = [1, 0, 2, 1, 2, 0, 2]
    assert np.all(conf.lane_prob[idx].argmax(1) == lane_true)
    # honest covariances: position errors within 4 sigma
    err = (est-X)[:, :2]
    nees = np.einsum('ki,ki->k', err, np.linalg.solve(conf.cov[idx][:, :2, :2], err[..., None])[..., 0])
    assert np.all(nees < chi2.ppf(0.9999, 2))
    rec = tr.nis_records()
    assert rec.dtype == NIS_DTYPE and len(rec) > 500 and set(np.unique(rec['kind'])) == {'lidar_box', 'radar',
                                                                                         'camera_box'}
    for kind in ('lidar_box', 'radar'):
        r = rec[rec['kind'] == kind]
        assert 0.5 < np.mean(r['nis']/r['dof']) < 2.0


def test_ids_stable_through_crossing_lane_changes():
    """Two cars ahead swap lanes (left into the middle, middle into the left) at the same time, 9 m apart: each keeps
    its track id, no car gets a second track, and the lane probabilities announce each change before it is half
    done."""
    a = Car(40.0, 2, 29.0, changes=[(1.0, 3.0, -1)])
    b = Car(31.0, 1, 29.0, changes=[(1.0, 3.0, +1)])
    scene = Scene([a, b, Car(70.0, 0, 28.0)], platoon(29.0), seed=4)
    ids, lanes = [], []

    def observe(t, tr):
        if t >= 0.6:
            idx, conf = match(tr.belief(t), scene.truth(t), 2.0)
            assert np.all(idx >= 0) and len(conf) == 3, t
            ids.append(conf.id[idx])
            lanes.append((t, conf.lane_prob[idx[:2]]))
    run(scene, make_tracker(), 5.5, observe=observe)
    ids = np.array(ids)
    assert np.all(ids == ids[0])
    mid = [lp for t, lp in lanes if 2.2 <= t <= 2.5]          # the changes are half done at 2.5 s
    assert all(lp[0].argmax() == 1 and lp[1].argmax() == 2 for lp in mid)
    end = lanes[-1][1]
    assert end[0].argmax() == 1 and end[1].argmax() == 2 and end.max(1).min() > 0.9


def test_lane_changes_do_not_split_tracks():
    """Seven cars plus two that change lanes (a 3 s quintic, as the traffic model's) next to the team: an IMM track
    lags a lane change for a while, and a detection outside its gate could start a second track on the same car.
    Every car keeps exactly one confirmed track, with one id, and accurate."""
    traffic = traffic_mix()+[Car(25.0, 0, 28.0, changes=[(1.0, 3.0, +1)]), Car(50.0, 2, 30.0, changes=[(2.0, 3.0, -1)])]
    scene = Scene(traffic, platoon(), seed=103)
    ids, err = {}, []

    def observe(t, tr):
        if t >= 1.0:
            X = scene.truth(t)
            idx, conf = match(tr.belief(t), X, 3.0)
            assert np.all(idx >= 0) and len(conf) == len(X), t
            for k, i in enumerate(conf.id[idx]):
                ids.setdefault(k, set()).add(int(i))
            err.append(np.hypot(*(conf.mean[idx, :2]-X[:, :2]).T))
    run(scene, make_tracker(), 5.5, observe=observe)
    assert all(len(v) == 1 for v in ids.values())
    assert np.percentile(np.concatenate(err), 99) < 0.5


def test_occluded_car_keeps_its_track():
    """The team drives single file in the middle lane; a car changes into the middle lane between the team and a
    car ahead and straight back out (3 s changes, as the traffic model's), hiding it for about 2 s. The radars here see
    nothing past an occluder (occluded_pd 0, in the scene and the tracker alike). The hidden car keeps its id, its
    covariance grows while hidden, and it is updated again afterwards."""
    specs = {**SPECS, **{r.name: replace(r, occluded_pd=0.0) for r in RADARS}}
    team = [Car(0.0, 1, 30.0), Car(-14.0, 1, 30.0), Car(-28.0, 1, 30.0)]
    target = Car(45.0, 1, 30.0)
    occluder = Car(18.0, 0, 30.0, changes=[(0.0, 3.0, +1), (3.1, 3.0, -1)])
    scene = Scene([target, occluder], team, seed=5, specs=specs)
    log = []

    def observe(t, tr):
        idx, conf = match(tr.belief(t), scene.truth(t), 4.0)
        k = idx[0]
        log.append((t, conf.id[k] if k >= 0 else -1, conf.cov[k, 0, 0] if k >= 0 else np.inf,
                    conf.last_seen[k] if k >= 0 else -1, conf.p_exist[k] if k >= 0 else 0.0))
    run(scene, make_tracker(specs), 6.0, observe=observe)
    t, ids, var, seen, p = (np.array(v) for v in zip(*log))
    assert np.all(ids[t >= 0.3] == ids[t >= 0.3][0])
    stale = t-seen
    hidden = (t > 2.0) & (t < 4.0)
    assert stale[hidden].max() > 1.8, 'the target was really hidden'
    assert np.all(p[hidden] > 0.5)
    before = var[np.argmin(np.abs(t-1.0))]
    assert var[hidden].max() > 20*before and np.all(np.diff(var[hidden]) > 0)
    assert stale[-1] < 0.15 and var[-1] < 2*before


def test_clutter_tracks_die():
    scene = Scene([], platoon(), seed=6)
    births, alive = [], []
    tr = make_tracker()
    run(scene, tr, 3.0, observe=lambda t, tr: alive.append(len(tr)))
    assert tr.stats['births'] > 20
    b = tr.belief(3.0)
    assert not b.confirmed.any()
    assert max(alive) <= 12 and np.mean(alive) < 5


def test_out_of_order_delivery_equals_in_order():
    traffic = traffic_mix()[:5]+[Car(25.0, 0, 28.0, changes=[(0.4, 3.0, +1)])]
    ref = run(Scene(traffic, platoon(), seed=7), make_tracker(), 2.0)
    late = run(Scene(traffic, platoon(), seed=7), make_tracker(), 2.0, latency=lambda r: r.uniform(0.0, 0.25),
               rng=np.random.default_rng(8))
    assert late.stats['replays'] > 10 and late.stats['late_dropped'] == 0
    a, b = ref.belief(2.1), late.belief(2.1)
    assert np.array_equal(a.id, b.id) and np.array_equal(a.confirmed, b.confirmed)
    for k in ('mean', 'cov', 'p_exist', 'lane_prob', 'mode_prob', 'last_seen'):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=1e-9, atol=1e-9), k
    assert len(ref.nis_records()) == len(late.nis_records())
    # with a hold of the largest latency, each cycle runs once, complete, and gives the same belief
    held = run(Scene(traffic, platoon(), seed=7), make_tracker(hold=0.25), 2.0, latency=lambda r: r.uniform(0.0, 0.25),
               rng=np.random.default_rng(8))
    held.process([], 2.3, Scene(traffic, platoon()).team_states(2.3))
    c = held.belief(2.1)
    assert held.stats['replays'] == 0
    assert np.array_equal(a.id, c.id) and np.allclose(a.mean, c.mean, atol=1e-9)


def test_duplicate_tracks_merge():
    """The front lidar reports one car as two boxes 0.1 m apart, every sweep: two tracks are born from the first
    sweep (one scan's boxes are never grouped) and both keep being updated, then merge into the older one."""
    only = {LIDARS[0].name: LIDARS[0]}
    scene = Scene([Car(30.0, 1, 30.0)], platoon(), seed=9, clutter=False, specs=only, split=0.05)
    counts = []
    tr = run(scene, make_tracker(specs=only), 2.0, observe=lambda t, tr: counts.append(len(tr)))
    assert max(counts) >= 2 and counts[-1] == 1 and tr.stats['merges'] >= 1
    b = tr.belief(2.0)
    assert b.confirmed[0] and b.id[0] == 1


def test_empty_scans_and_team_window():
    """A car the team leaves behind beyond roi_behind is deleted; a scan with no detections still misses the tracks
    it covers (ScanInfo)."""
    cfg = TrackerConfig(roi_behind=30.0)
    tr = TeamTracker(cfg, ROAD, CAR, SPECS, H)
    slow = Car(10.0, 2, 20.0)                   # behind the last team car by more than 30 m from t = 6.4 s
    scene = Scene([slow], platoon(30.0), seed=10, clutter=False)
    gone = []
    run(scene, tr, 7.5, observe=lambda t, tr: gone.append((t, len(tr))))
    t = np.array([g[0] for g in gone])
    n = np.array([g[1] for g in gone])
    assert n[(t > 1) & (t < 3)].min() == 1
    assert n[-1] == 0 and tr.stats['deleted'] >= 1
    # announced scans that see nothing where a track is lower its existence (by at most 1 - MISS_CAP per cycle)
    # until it is deleted
    tr2 = make_tracker()
    scene2 = Scene([Car(40.0, 1, 30.0)], platoon(), seed=11, clutter=False)
    run(scene2, tr2, 1.0)
    p = [tr2.belief(1.0).p_exist[0]]
    for k in range(1, 8):
        t = 1.0+0.05*k
        team = scene2.team_states(t)
        infos = [ScanInfo(s.name, c, t, G.sensor_pose(s.mount, team[c], CAR)) for c in range(3)
                 for s in (*RADARS, *CAMERAS)]
        tr2.process([], t, team, infos)
        b = tr2.belief(t)
        p.append(b.p_exist[0] if len(b) else 0.0)
    assert np.all(np.diff(p[:5]) < 0) and p[1] > 0.9 and p[-1] == 0.0


def test_cycle_time_45_cars():
    """45 traffic cars around the team (43 after keeping the team clear), about 140 detections per cycle with the
    radars' false alarms at full rate: the mean cost of one fusion cycle."""
    rng = np.random.default_rng(12)
    cars = []
    for l in range(3):
        s = -140.0+rng.uniform(0, 10)
        for _ in range(15):
            s += rng.uniform(12, 25)
            cars.append(Car(s, l, rng.uniform(25, 34)))
    team = [Car(-2.0, 1, 30.0), Car(-8.0, 0, 30.0), Car(-5.0, 2, 30.0)]
    cars = [c for c in cars if all(abs(c.s0-m.s0) > 7 or abs(c.d0-m.d0) > 2 for m in team)]
    scene = Scene(cars, team, seed=13, clutter_scale=2.0)
    frames = []
    for k in range(40):
        t = round(k*DT, 10)
        frames.append((t, *scene.step(t, lidar=k % 2 == 0), scene.team_states(t)))
    tr = make_tracker(record_nis=False)
    per = []
    for k, (t, dets, inf, team_) in enumerate(frames):
        t0 = time.perf_counter()
        tr.process(dets, t, team_, inf)
        if k >= 20:
            per.append(time.perf_counter()-t0)
    counts = [len(f[1]) for f in frames[20:]]
    ms = 1e3*np.mean(per)
    print(f'\ntracker cycle: {ms:.1f} ms mean, {1e3*np.max(per):.1f} ms max, {np.mean(counts):.0f} detections, '
          f'{len(tr)} tracks')
    b = tr.belief(frames[-1][0])
    assert np.mean(counts) > 120 and b.confirmed.sum() >= 40
    assert ms < 40.0
