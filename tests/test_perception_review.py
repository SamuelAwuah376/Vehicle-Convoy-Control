"""Regressions for the perception review's fixes: the camera tilt and pitch noise, the coarse gate's and the deletion
test's process noise, the likelihood-ratio assignment cost, the side radar's Doppler blind spot in the expected pd,
the per-beam lidar sweep, the lidar box's velocity model, and the NIS log kept only when recording."""
from dataclasses import replace
from math import radians
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import geometry as G
from nohesi.perception.config import (Perception, TrackerConfig, default_cameras, default_lidars, default_radars)
from nohesi.perception.pipeline import PerceptionSystem
from nohesi.perception.processing.camera import pitch_jacobian, process_camera
from nohesi.perception.processing.lidar import fit_box, process_lidar
from nohesi.perception.sim.camera import simulate_camera
from nohesi.perception.sim.lidar import cast_sweep, simulate_lidar
from nohesi.perception.tracking import association as A
from nohesi.perception.tracking import existence as E
from nohesi.perception.tracking.imm import IMMState, imm_combine
from nohesi.perception.tracking.models import camera_R, h_camera, h_lidar, h_radar
from nohesi.perception.tracking.tracker import MAX_SIGMA_S, ScanInfo, TeamTracker
from nohesi.perception.types import Detection, EgoState, TruthHistory
from nohesi.road import Road
from nohesi.world import Snapshot, World

SC = scenario_v9()
CAR = SC.vehicle
ROAD = Road(SC.road)
CFG = TrackerConfig()
H = 1.45
LIDARS, RADARS, CAMERAS = default_lidars(), default_radars(), default_cameras()
SPECS = {s.name: s for s in (*LIDARS, *RADARS, *CAMERAS)}
LIDAR_R = np.diag([0.08**2, 0.08**2, np.radians(1.0)**2])
LANE = ROAD.lane_center(1)


def snapshot(t, team, traffic, ids=None):
    """A true Snapshot: team rows [s, d, yaw, v], traffic rows [s, d, v] (lane keeping)."""
    team, traffic = np.asarray(team, float), np.asarray(traffic, float).reshape(-1, 3)
    return Snapshot(time=float(t), team=team, team_steer=np.zeros(3), team_s=team[:, 0].copy(),
                    team_d=team[:, 1].copy(), traffic_s=traffic[:, 0].copy(), traffic_d=traffic[:, 1].copy(),
                    traffic_v=traffic[:, 2].copy(),
                    traffic_id=np.arange(len(traffic))+7 if ids is None else np.asarray(ids), collisions=(), done=False,
                    traffic_vd=np.zeros(len(traffic)))


# ------------------------------------------------------------------------------------------------ camera
def physical_projection(spec, pose, P, pitch):
    """Pixels of world points P (3,) for a camera whose body pitches by `pitch` about the body's left axis, from an
    explicit rotation of the level camera frame (independent of geometry.camera_tilt)."""
    c, s = np.cos(pose[2]), np.sin(pose[2])
    dx, dy = P[0]-pose[0], P[1]-pose[1]
    p = np.array([c*dx+s*dy, -s*dx+c*dy, P[2]-pose[3]])
    a = np.array([np.sin(spec.mount.yaw), np.cos(spec.mount.yaw), 0.0])       # the body's y axis, camera level frame
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    q = (np.eye(3)+np.sin(pitch)*K+(1-np.cos(pitch))*K@K).T @ p
    f = G.focal(spec)
    return spec.width_px/2-f*q[1]/q[0], spec.height_px/2-f*q[2]/q[0]


def test_body_pitch_tilts_each_camera_by_its_mount():
    """Nose-down pitch: the front camera's scene moves up, the rear camera's down, a side camera's rolls (a point on
    the boresight stays, one off it moves vertically)."""
    pitch = radians(1.0)
    for spec in CAMERAS:
        pose = G.sensor_pose(spec.mount, (0.0, 0.0, 0.0), CAR)
        for side in (0.0, 8.0):
            b = np.array([np.cos(pose[2]), np.sin(pose[2])])
            P = np.array([*(np.array(pose[:2])+30*b+side*np.array([-b[1], b[0]])), 0.0])
            u0, v0, _ = G.project(spec, pose, P)
            u1, v1, _ = G.project(spec, pose, P, pitch=pitch)
            assert (u1, v1) == pytest.approx(physical_projection(spec, pose, P, pitch), abs=1e-6)
        if spec.name == 'camera_front':
            assert v1 < v0-20
        if spec.name == 'camera_rear':
            assert v1 > v0+5
    R = G.camera_tilt(np.array([0.0, np.pi]), 0.01)
    assert np.allclose(R[0].T @ R[0], np.eye(3)) and R[0, 0, 2] == pytest.approx(np.sin(0.01))
    assert R[1, 0, 2] == pytest.approx(-np.sin(0.01))


def test_pitch_jacobian_matches_the_projection():
    rng = np.random.default_rng(0)
    for spec in CAMERAS:
        pose = G.sensor_pose(spec.mount, (0.0, LANE, 0.0), CAR)
        b = np.array([np.cos(pose[2]), np.sin(pose[2])])
        for _ in range(5):
            xy = np.array(pose[:2])+rng.uniform(8, 25)*b+rng.uniform(-4, 4)*np.array([-b[1], b[0]])
            corners = G.box_corners_3d(np.array([[*xy, 0.0]]), CAR, H)[0]
            box = lambda th: np.array([f(g(G.project(spec, pose, corners, th)[k])) for k, g, f in
                                       ((0, lambda x: x, np.min), (0, lambda x: x, np.max), (1, lambda x: x, np.min),
                                        (1, lambda x: x, np.max))])
            numeric = (box(1e-5)-box(-1e-5))/2e-5
            j = pitch_jacobian(box(0.0)[None], spec)[0]
            tol = 0.02*np.abs(numeric).max() if spec.mount.yaw == 0.0 or abs(spec.mount.yaw) == np.pi else \
                0.3*np.abs(numeric).max()                   # a roll's box: extreme corners switch at 0 (cue_init's rho)
            assert np.allclose(j, numeric, atol=tol), (spec.name, j, numeric)


def frames_nis(spec, car, n=300, ax=0.0, **kw):
    """NIS per dof of simulated boxes of one car against the true state, with the processing's R (kw: process_camera's
    pitch_noise)."""
    tm = np.array([[0.0, LANE, 0.0, 30.0], [-3000.0, LANE, 0.0, 30.0], [-4000.0, LANE, 0.0, 30.0]])
    hist = TruthHistory()
    hist.push(snapshot(0.0, tm, [[*car, 30.0]]), (ax, 0.0, 0.0))
    ego = EgoState(0.0, 0.0, LANE, 0.0, 30.0, ax=ax, pitch=spec.pitch_per_accel*(-ax))
    nis, dof = 0.0, 0
    X = np.array([[car[0], car[1], 30.0, 0.0]])
    for seed in range(n):
        fr = simulate_camera(replace(spec, false_rate=0.0, pd=1.0), 0, hist, 0.0, CAR, np.random.default_rng(seed), H)
        for d in process_camera(fr, spec, ego, None, CAR, H, **kw):
            keep = ~np.asarray(d.meta['truncated'])
            r = (d.z-h_camera(X, d, spec, CAR)[0])[keep]
            nis += r @ np.linalg.solve(camera_R(d, spec)[np.ix_(keep, keep)], r)
            dof += keep.sum()
    return nis/dof


def test_camera_R_carries_the_frame_pitch_noise():
    """Simulated front and rear boxes with the spec's 0.1 deg per-frame pitch noise are consistent with R (NIS per dof
    1 within sampling); R of pixel noise alone gives ~1.3 at the front camera's focal length."""
    front, rear = CAMERAS[0], CAMERAS[1]
    for spec, car, ax in ((front, (60.0, LANE), 0.0), (front, (25.0, LANE-3.6), -6.0), (rear, (-30.0, LANE), 0.0)):
        assert frames_nis(spec, car, ax=ax) == pytest.approx(1.0, abs=0.12), (spec.name, car)
    assert frames_nis(front, (60.0, LANE), pitch_noise=0.0) > 1.2


# ------------------------------------------------------------------------------------------------ tracker
def team_state(t, v=30.0):
    return np.array([[v*t, 0.0, 0.0, v], [v*t-300, 0.0, 0.0, v], [v*t-310, 0.0, 0.0, v]])


def scans(car_x, t, rng, hidden=False):
    """Front lidar box and front radar detection of one car from team car 0 (model-generated); hidden: the sensors
    point away (scans run, the car is not covered)."""
    team = team_state(t)
    ego = EgoState(t, team[0, 0], team[0, 1], 0.0, 30.0)
    dets, infos = [], []
    for spec in (LIDARS[0], RADARS[0]):
        pose = G.sensor_pose(spec.mount, team[0], CAR)
        if hidden:
            pose = (pose[0], pose[1], pose[2]+np.pi, pose[3])
        infos.append(ScanInfo(spec.name, 0, t, pose))
        if hidden:
            continue
        if spec is LIDARS[0]:
            z = h_lidar(car_x[None], CAR)[0]+rng.normal(size=3)*np.sqrt(np.diag(LIDAR_R))
            dets.append(Detection('lidar_box', spec.name, 0, t, z, LIDAR_R, pose, ego, {'n_points': 30, 'scan': t}))
        else:
            R = np.diag([spec.sigma_range, spec.sigma_azimuth, spec.sigma_range_rate])**2
            tmpl = Detection('radar', spec.name, 0, t, np.zeros(3), R, pose, ego, {'scan': t})
            z = h_radar(car_x[None], tmpl, CAR)[0]+rng.normal(size=3)*np.sqrt(np.diag(R))
            dets.append(Detection('radar', spec.name, 0, t, z, R, pose, ego, {'n_points': 2, 'scan': t}))
    return dets, infos, team


def tracked_then_hidden(hide_steps, seed=1, v=28.0):
    rng = np.random.default_rng(seed)
    tr = TeamTracker(CFG, ROAD, CAR, SPECS, cycle=0.05)
    car = lambda t: np.array([40.0+v*t, 0.0, v, 0.0])
    for k in range(80):
        t = 0.05*k
        d, i, team = scans(car(t), t, rng)
        tr.process(d, t, team, i)
    for k in range(80, 80+hide_steps):
        t = 0.05*k
        d, i, team = scans(None, t, rng, hidden=True)
        tr.process(d, t, team, i)
    return tr, car, rng


@pytest.mark.parametrize('offset', [3.5, 6.0])
def test_coarse_gate_keeps_a_car_found_again_after_hiding(offset):
    """Tracked 4 s, unseen 3 s (sigma_s ~6 m), found again `offset` m behind its prediction: the coarse gate passes the
    pair (it adds the process noise the full gate adds) and the track keeps its id."""
    tr, car, rng = tracked_then_hidden(60)
    t = 7.0
    x = car(t)
    x[0] -= offset
    det = scans(x, t, rng)[0][0]
    S = tr._tr
    mc, Pc = imm_combine(IMMState(S.means, S.covs, S.mu))
    tab = A.DetTable.build([([det], LIDARS[0], 0.0)], CAR)
    assert A.coarse_candidates(tab, mc, Pc, S.t, CFG)[0].size == 1
    assert A.coarse_candidates(tab, mc, Pc, S.t)[0].size == 0           # the stored posterior alone: pruned
    ids = tr.belief(t).id.tolist()
    for k in range(140, 150):
        t = 0.05*k
        x = car(t)
        x[0] -= offset
        d, i, team = scans(x, t, rng)
        tr.process(d, t, team, i)
    assert tr.belief(t).id.tolist() == ids and tr.stats['births'] == 1


def test_hidden_track_is_deleted_when_its_belief_sigma_passes_the_limit():
    """A confirmed car hidden from every sensor (no misses) coasts until the belief's own sigma_s, process noise
    included, reaches MAX_SIGMA_S (the deletion test once saw only the stored covariance carried forward)."""
    rng = np.random.default_rng(2)
    tr = TeamTracker(CFG, ROAD, CAR, SPECS, cycle=0.05)
    car = lambda t: np.array([40.0+30.0*t, 0.0, 30.0, 0.0])
    for k in range(80):
        t = 0.05*k
        d, i, team = scans(car(t), t, rng)
        tr.process(d, t, team, i)
    last = None
    for k in range(80, 600):
        t = 0.05*k
        d, i, team = scans(None, t, rng, hidden=True)
        tr.process(d, t, team, i)
        b = tr.belief(t)
        if len(b) == 0:
            break
        last = np.sqrt(b.cov[0, 0, 0])
    assert len(tr.belief(t)) == 0 and 0.9*MAX_SIGMA_S < last <= MAX_SIGMA_S and t < 25.0


def test_assignment_cost_is_the_negative_log_likelihood_ratio():
    """One track, two detections of one radar scan: a clean single point that fits less well (NIS 4.2) and a vague
    cluster that fits better (NIS 0.2, R several times wider). The likelihood prefers the clean point, and so must the
    assignment (NIS + log det S - log det R preferred the vague one)."""
    tr = TeamTracker(CFG, ROAD, CAR, SPECS, cycle=0.05)
    rng = np.random.default_rng(4)
    car = lambda t: np.array([60.0+28.0*t, 0.0, 28.0, 0.0])
    for k in range(40):
        t = 0.05*k
        d, i, team = scans(car(t), t, rng)
        tr.process(d, t, team, i)
    t = 2.0
    team = team_state(t)
    spec = RADARS[0]
    pose = G.sensor_pose(spec.mount, team[0], CAR)
    ego = EgoState(t, team[0, 0], team[0, 1], 0.0, 30.0)
    R_a = np.diag([spec.sigma_range, spec.sigma_azimuth, spec.sigma_range_rate])**2
    R_b = R_a/3+np.diag([0.6, np.radians(1.5), 0.6])**2
    zhat = h_radar(car(t)[None], Detection('radar', spec.name, 0, t, np.zeros(3), R_a, pose, ego, {}), CAR)[0]
    S = tr._tr
    tab = A.DetTable.build([([Detection('radar', spec.name, 0, t, zhat+np.array([0.55, 0.0, 0.0]), R_a, pose, ego,
                                        {'n_points': 1, 'scan': t}),
                              Detection('radar', spec.name, 0, t, zhat+np.array([0.3, 0.0, 0.0]), R_b, pose, ego,
                                        {'n_points': 3, 'scan': t})], spec, 0.0)], CAR)
    g = tr._gate(np.zeros(2, int), np.arange(2), tab, t)
    assert g.gated.all() and g.nis[0] > g.nis[1]
    assert g.ll[0] > g.ll[1] and g.cost[0] < g.cost[1]
    chosen = tr._assign(np.zeros(2, int), np.arange(2), g.gated, g.cost, tab)
    assert chosen.tolist() == [0]
    assert len(S) == 1


def test_radar_doppler_pass():
    """A point with no range rate relative to the world's fixed points fails the stationary test; a car closing fast
    passes; the boundary follows the test's 3 sigma."""
    spec = RADARS[2]
    assert E.doppler_pass(spec, 0.0, 30.0) < 3e-3                       # the test's two 3 sigma tails
    assert E.doppler_pass(spec, 25.0, 30.0) == pytest.approx(1.0, abs=1e-6)
    sigma = np.hypot(spec.sigma_range_rate, 30.0*spec.sigma_azimuth)
    assert E.doppler_pass(spec, 3*sigma, 30.0) == pytest.approx(0.5, abs=0.01)


def alongside_view(gap, steps=100):
    """A traffic car brought alongside team car 1 (decelerating to its speed, 0.3 m behind it, body gap `gap`) and
    held there; the other team cars 262 m away. Share of the alongside steps with the car in the perceived snapshot."""
    sc = replace(SC, perception=Perception(enabled=True))
    world = World(sc, 1000)
    ego_d = world.road.lane_center(1)+1.45

    def snap(t):
        tt = min(max(t, 0.0), 2.0)
        s = -6.3+36*t if t < 0 else -6.3+36*tt-1.5*tt*tt+30*(t-tt)
        team = [[30*t+262, LANE, 0, 30.0], [30*t, ego_d, 0, 30.0], [30*t-262, LANE, 0, 30.0]]
        return snapshot(t, team, [[s, ego_d+CAR.width+gap, 36.0 if t < 0 else 36-3*tt]])
    system = PerceptionSystem(sc.perception, sc, world.road, world.plant.params, 1000)
    system.reset(snap(0.0))
    seen = []
    for k in range(1, steps+1):
        truth = snap(0.05*k)
        view = system.step(truth, np.zeros(3))
        seen.append(bool(len(view.traffic_s)) and np.min(np.hypot(view.traffic_s-truth.traffic_s[0],
                                                                  view.traffic_d-truth.traffic_d[0])) < 1.5)
    return np.mean(seen[40:])


@pytest.mark.parametrize('gap', [0.25, 0.7])
def test_car_alongside_a_team_car_stays_tracked(gap):
    """The side radar's point nearest a car abeam (the one its model predicts) has no range rate and the processing's
    stationary test removes it: expected as a detection, every such scan was a miss and deleted the car within a few
    cycles (the planner then drove into it)."""
    assert alongside_view(gap) == 1.0


# ------------------------------------------------------------------------------------------------ lidar
def moving_truth(rel=40.0):
    """Ego at 30 m/s with a yaw rate, a car ahead `rel` m/s faster and one ahead in the next lane at the ego's speed."""
    h = TruthHistory()
    for t in np.round(np.arange(0.0, 0.501, 0.05), 3):
        team = [[30*t, 0.0, 0.2*t, 30.0], [30*t-300, 0.0, 0.0, 30.0], [30*t-320, 0.0, 0.0, 30.0]]
        h.push(snapshot(t, team, [[12+(30+rel)*t, 0.0, 30+rel], [20+30*t, 3.7, 30.0]]))
    return h


def per_beam(spec, truth, az, times):
    """Each beam cast alone with geometry.raycast at its own fire time."""
    out = np.full(len(az), np.inf)
    for i, (a, t) in enumerate(zip(az, times)):
        s = truth.at(t)
        pose = G.sensor_pose(spec.mount, s.team[0], CAR)
        boxes = G.car_boxes(np.concatenate((s.traffic[:, :3], s.team[1:, :3])), CAR)
        out[i] = G.raycast(pose[:2], np.array([pose[2]+a]), boxes, spec.range_max)[0][0]
    return out


def test_lidar_sweep_casts_every_beam_at_its_own_time():
    spec = replace(LIDARS[0], range_sigma=0.0, dropout=0.0, far_dropout=0.0)
    truth = moving_truth()
    az, times, r, label = cast_sweep(spec, 0, truth, 0.43, CAR)
    want = per_beam(spec, truth, az, times)
    assert np.array_equal(np.isfinite(r), np.isfinite(want)) and np.isfinite(r).sum() > 100
    assert np.max(np.abs(r[np.isfinite(r)]-want[np.isfinite(r)])) < 1e-4   # corners, not poses, interpolated


def test_lidar_box_of_a_slower_car_ahead_is_unbiased_in_the_tracker_model():
    """A car 35-20 m ahead, 10 m/s slower: the box fitted with the ego-speed prior is tilted by the skew across its rear
    face (a degree), and the tracker's model, h_lidar + dz_dv (v_prior - v), explains it: its residual against the true
    state is consistent with R_fit, while against h_lidar alone the heading is biased by several sigma."""
    spec = LIDARS[0]
    ego_at = lambda t: EgoState(float(t), 30*t, 0.0, 0.0, 30.0)
    rng = np.random.default_rng(5)
    plain, model = [], []
    for k in range(16):
        t_end = 1.0+0.1*k
        h = TruthHistory()                                 # the last 0.3 s of snapshots (it keeps 12)
        for t in np.round(t_end+np.arange(-6, 1)*0.05, 3):
            h.push(snapshot(t, team_state(t), [[45.0+20.0*t, 0.0, 20.0]]))
        scan = simulate_lidar(spec, 0, h, t_end, CAR, rng)
        (d,) = [d for d in process_lidar(scan, spec, ego_at, None, CAR, ROAD) if d.meta['n_points'] > 10]
        x = np.array([45.0+20.0*d.time, 0.0, 20.0, 0.0])
        zh = h_lidar(x[None], CAR)[0]
        m = d.meta
        plain.append(d.z-zh)
        r = d.z-(zh+m['dz_dv'] @ (m['v_prior']-x[2:4]))
        model.append(r @ np.linalg.solve(m['R_fit'], r))
    plain = np.array(plain)
    assert abs(np.degrees(plain[:, 2].mean())) > 0.5
    assert np.mean(model) == pytest.approx(3.0, abs=1.0)


# ------------------------------------------------------------------------------------------------ pipeline
def test_nis_log_is_kept_only_when_recording():
    sc = replace(SC, perception=Perception(enabled=True), timing=replace(SC.timing, duration=0.5))
    world = World(sc, 1000)
    for record in (False, True):
        system = PerceptionSystem(sc.perception, sc, world.road, world.plant.params, 1000, record=record)
        system.reset(world.snapshot)
        assert bool(system.tracker.nis_log) == record and system.tracker.record_nis == record


def test_recorded_gap_keys_are_true_ids():
    """Under perception the intentions carry track ids; the recorded gaps map each to its matched true car (or None)
    so the viewer resolves them against Episode.traffic_id, and Episode.views keeps the view's tracks and matches."""
    from nohesi.config import Planner
    from nohesi.evaluation.runner import EpisodeRun, _matched_ids, _true_key
    truth = snapshot(0.0, team_state(0.0), [[50.0, 0.0, 28.0], [80.0, 3.6, 30.0]], ids=[3, 4])
    view = snapshot(0.0, team_state(0.0), [[80.4, 3.5, 30.0], [50.2, 0.1, 28.0], [120.0, 0.0, 30.0]], ids=[1, 2, 5])
    match = _matched_ids(view, truth)
    assert match.tolist() == [4, 3, -1]
    assert _true_key(('traffic', 2), view, match) == ('traffic', 3) and _true_key(('traffic', 5), view, match) is None
    assert _true_key(('team', 1), view, match) == ('team', 1)
    assert _true_key(('traffic', 9), view, None) == ('traffic', 9)
    sc = replace(SC, timing=replace(SC.timing, duration=4.0), perception=Perception(enabled=True))
    run = EpisodeRun(sc, 1000, 'overtaker', Planner(), record_gaps=True)
    while not run.done:
        run.step()
    ep = run.episode()
    assert len(ep.views) == len(ep.gaps) == len(ep.time)-1
    for k, state in enumerate(ep.gaps):
        for key in (x for row in state for x in (row[2], row[3], row[5])):
            if key is not None and key[0] == 'traffic':
                assert key[1] in ep.traffic_id[k]


def test_batched_camera_and_radar_processing_equal_one_by_one():
    """The pipeline processes a step's twelve camera frames, and its twelve radar scans, in one batch each: the same
    detections as frame by frame and scan by scan."""
    from nohesi.perception.processing.camera import process_cameras
    from nohesi.perception.processing.radar import process_radar, process_radars
    from nohesi.perception.sim.camera import simulate_cameras
    from nohesi.perception.sim.radar import simulate_radars
    rng = np.random.default_rng(8)
    s0 = rng.uniform(-60, 120, 30)
    lanes = ROAD.lane_center(rng.integers(0, 3, 30))
    hist = TruthHistory()
    team = np.array([[0.0, LANE, 0.0, 30.0], [-15.0, LANE-3.6, 0.0, 30.0], [12.0, LANE+3.6, 0.02, 30.0]])
    hist.push(snapshot(0.0, team, np.column_stack((s0, lanes, np.full(30, 28.0)))), (-2.0, 0.0, 1.0))
    mates = [np.delete(team[:, :3], c, 0) for c in range(3)]
    egos = [EgoState(0.0, *team[c, :3], 30.0, ax=a, pitch=CAMERAS[0].pitch_per_accel*(-a)) for c, a in
            zip(range(3), (-2.0, 0.0, 1.0))]
    frames = [fr for c in range(3) for fr in simulate_cameras(CAMERAS, c, hist, 0.0, CAR, rng, H)]
    jobs = [(c, s) for c in range(3) for s in CAMERAS]
    batch = process_cameras(frames, [s for _, s in jobs], [egos[c] for c, _ in jobs], [mates[c] for c, _ in jobs],
                            CAR, H)
    assert sum(len(b) for b in batch) > 10
    for fr, (c, s), got in zip(frames, jobs, batch):
        want = process_camera(fr, s, egos[c], mates[c], CAR, H)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert np.allclose(g.z, w.z) and np.allclose(g.R, w.R) and np.allclose(g.meta['init'][0], w.meta['init'][0])
            assert np.allclose(g.meta['init'][1], w.meta['init'][1]) and g.sensor_pose == w.sensor_pose
    scans = simulate_radars(RADARS, [0, 1, 2], hist, 0.0, CAR, rng)
    jobs = [(c, s) for c in range(3) for s in RADARS]
    batch = process_radars(scans, [s for _, s in jobs], [egos[c] for c, _ in jobs], [mates[c] for c, _ in jobs], CAR)
    assert sum(len(b) for b in batch) > 10
    for sc, (c, s), got in zip(scans, jobs, batch):
        want = process_radar(sc, s, egos[c], mates[c], CAR)
        assert [d.meta['n_points'] for d in got] == [d.meta['n_points'] for d in want]
        assert all(np.allclose(g.z, w.z) and np.allclose(g.R, w.R) for g, w in zip(got, want))


def test_far_teammate_radar_points_are_dropped():
    """A teammate 150 m ahead: its radar points scatter across the line of sight by about 0.8 m (0.3 deg), so a fixed
    1 m margin passed one scan in eight on as a car; the margin now covers the points' own noise."""
    from nohesi.perception.processing.radar import process_radar
    from nohesi.perception.sim.radar import simulate_radars
    team = np.array([[0.0, LANE, 0.0, 30.0], [150.0, LANE+1.0, 0.0, 30.0], [-90.0, LANE-1.0, 0.0, 30.0]])
    hist = TruthHistory()
    hist.push(snapshot(0.0, team, np.zeros((0, 3))))
    rng = np.random.default_rng(1)
    ego = EgoState(0.0, 0.0, LANE, 0.0, 30.0)
    leaks = 0
    for _ in range(200):
        for scan, spec in zip(simulate_radars(RADARS[:2], [0], hist, 0.0, CAR, rng), RADARS[:2]):
            for d in process_radar(scan, spec, ego, team[1:, :3], CAR):
                at = np.array(d.sensor_pose[:2])+d.z[0]*np.array([np.cos(d.sensor_pose[2]+d.z[1]),
                                                                  np.sin(d.sensor_pose[2]+d.z[1])])
                leaks += np.min(np.hypot(*(team[1:, :2]-at).T)) < 8.0
    assert leaks/400 < 0.02
