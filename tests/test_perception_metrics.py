"""Perception evaluation metrics (docs/perception_design.md section 12): GOSPA and its decomposition, NEES and NIS
consistency tests, identity switches, per-sensor detection probability, the pooled summary, and an ideal-mode
pipeline run evaluated end to end."""
import itertools
import json
from dataclasses import replace
import numpy as np
import pytest
from scipy.stats import chi2
from nohesi.config import scenario_v9
from nohesi.geometry import body_centre
from nohesi.perception import metrics as M
from nohesi.perception.config import Perception, default_cameras, default_lidars, default_radars
from nohesi.perception.geometry import box_corners_3d, project, sensor_pose, to_sensor
from nohesi.perception.pipeline import PerceptionSystem
from nohesi.perception.tracking.tracker import NIS_DTYPE
from nohesi.perception.types import Belief, Detection, EgoState, TruthState
from nohesi.world import Snapshot, World

VEH = scenario_v9().vehicle
C2 = M.GOSPA_C**2/2


# ---------------------------------------------------------------------------------------------------------- GOSPA
def brute_gospa(X, Y, c=M.GOSPA_C):
    """GOSPA^2 (alpha 2) by enumerating every partial assignment."""
    best = np.inf
    n, m = len(X), len(Y)
    for k in range(min(n, m)+1):
        for xs in itertools.combinations(range(n), k):
            for ys in itertools.permutations(range(m), k):
                d = sum(min(np.linalg.norm(X[i]-Y[j]), c)**2 for i, j in zip(xs, ys))
                best = min(best, d+C2*(n-k)+C2*(m-k))
    return best


def test_gospa_parts_on_hand_cases():
    g = M.gospa(np.array([[0.0, 0.0], [10.0, 0.0]]), np.array([[1.0, 0.0]]))
    assert (g.localization, g.missed, g.false) == (1.0, 0.0, C2)
    assert g.distance == pytest.approx(np.sqrt(1.0+C2))
    np.testing.assert_array_equal(g.est_index, [0])
    far = M.gospa(np.array([[0.0, 0.0]]), np.array([[0.0, 6.0]]))   # beyond c: one missed car and one false track
    assert (far.localization, far.missed, far.false) == (0.0, C2, C2)
    assert far.distance == pytest.approx(M.GOSPA_C)
    assert M.gospa(np.zeros((0, 2)), np.zeros((0, 2))).distance == 0.0
    only = M.gospa(np.zeros((0, 2)), np.ones((3, 2)))
    assert only.missed == 3*C2 and only.false == 0.0
    with pytest.raises(ValueError):
        M.gospa(np.zeros((1, 2)), np.zeros((1, 2)), alpha=1.0)


def test_gospa_equals_the_brute_force_minimum():
    rng = np.random.default_rng(3)
    for _ in range(30):
        X, Y = rng.uniform(0, 12, (rng.integers(0, 5), 2)), rng.uniform(0, 12, (rng.integers(0, 5), 2))
        g = M.gospa(X, Y)
        assert g.distance**2 == pytest.approx(brute_gospa(X, Y))
        assert g.distance**2 == pytest.approx(g.localization+g.missed+g.false)


# ---------------------------------------------------------------------------------------------------------- NEES
def random_spd(rng, k):
    A = rng.normal(size=(k, k))
    return A@A.T+0.5*np.eye(k)


def test_nees_and_its_chi_square_test():
    assert M.nees(np.array([[1.0, 0.0]]), np.array([np.diag([4.0, 1.0])]))[0] == pytest.approx(0.25)
    rng = np.random.default_rng(5)
    P = np.stack([random_spd(rng, 4) for _ in range(3000)])
    err = np.einsum('nij,nj->ni', np.linalg.cholesky(P), rng.normal(size=(3000, 4)))
    ok = M.consistency(M.nees(err, P), 4)
    assert ok['within'] and abs(ok['inside']-0.95) < 0.015 and ok['bounds'][0] < 4 < ok['bounds'][1]
    assert ok['median'] == pytest.approx(chi2.median(4), rel=0.05)
    over = M.consistency(M.nees(err, 4*P), 4)          # a covariance 4x too large: NEES about 1
    assert not over['within'] and over['mean'] == pytest.approx(1.0, rel=0.1)
    assert M.consistency(np.zeros(0), 4) == {'n': 0, 'dof': 4}


# ---------------------------------------------------------------------------------------------------------- frames
def snapshot(t, team_s, s, d, v, ids, vd=None):
    team = np.column_stack((team_s, np.zeros(3), np.zeros(3), np.full(3, 30.0)))
    return Snapshot(t, team, np.zeros(3), np.asarray(team_s, float), np.zeros(3), np.asarray(s, float),
                    np.asarray(d, float), np.asarray(v, float), np.asarray(ids), (), False,
                    traffic_vd=None if vd is None else np.asarray(vd, float))


def belief(t, ids, mean, cov, confirmed, last_seen=None):
    n = len(ids)
    return Belief(t, np.asarray(ids), np.asarray(mean, float), np.asarray(cov, float), np.full(n, 0.9),
                  np.full((n, 3), 1/3), np.tile([0.9, 0.05, 0.05], (n, 1)),
                  np.full(n, t) if last_seen is None else np.asarray(last_seen, float), np.asarray(confirmed, bool))


def synthetic_episode(frames=300, seed=11):
    """Ten cars in the window tracked with exactly their covariance, one car beyond it, one false confirmed track
    and one unconfirmed track."""
    rng = np.random.default_rng(seed)
    s0 = np.array([-70.0, -30.0, -5.0, 20.0, 45.0, 80.0, 120.0, 160.0, 200.0, 240.0, 320.0])
    d0 = np.tile([0.0, 3.7, -3.7], 4)[:len(s0)]
    P = np.stack([random_spd(rng, 4)*0.05 for _ in range(10)])
    L = np.linalg.cholesky(P)
    snaps, beliefs = [], []
    for k in range(frames):
        t = 0.05*k
        team_s = np.array([5.0, 0.0, -5.0])+30.0*t
        s = s0+30.0*t
        v = np.full(len(s0), 30.0)
        ids = np.arange(100, 100+len(s0))
        snaps.append(snapshot(t, team_s, s, d0, v, ids, vd=np.zeros(len(s0))))
        noise = np.einsum('nij,nj->ni', L, rng.normal(size=(10, 4)))
        x = np.column_stack((s[:10], d0[:10], v[:10], np.zeros(10)))+noise
        extra = np.array([[30.0*t+100.0, 7.0, 30.0, 0.0], [30.0*t+60.0, 0.0, 30.0, 0.0]])
        beliefs.append((t, belief(t, np.r_[np.arange(1, 11), 50, 51], np.vstack((x, extra)),
                                  np.concatenate((P, np.stack([np.eye(4)]*2))), [True]*11+[False])))
    return beliefs, snaps


def test_match_frames_counts_gospa_nees_and_the_window():
    beliefs, snaps = synthetic_episode()
    frames, pairs, bins, false, spans = M.match_frames(beliefs, snaps)
    assert len(frames) == len(snaps)
    assert len(false) == len(snaps) and np.all(false['track'] == 50) and not false['ever'].any()
    np.testing.assert_allclose(false['nearest'], np.hypot(120.0-100.0, 7.0), rtol=1e-5)   # car 6: +120 m, d = 0
    assert set(spans['track'].tolist()) == set(range(1, 11)) | {50}       # the unconfirmed track 51 never counts
    np.testing.assert_allclose(spans['last']-spans['first'], snaps[-1].time)
    assert np.all(frames['truth'] == 10) and np.all(frames['tracks'] == 11)       # the car at +315 m is outside
    assert np.all(frames['assigned'] == 10) and np.all(frames['missed'] == 0) and np.all(frames['false'] == 1)
    np.testing.assert_allclose(frames['gospa']**2, frames['localization']+C2, rtol=1e-12)
    np.testing.assert_array_equal(np.sort(np.unique(pairs['truth'])), np.arange(100, 110))
    assert np.all(pairs['track']-pairs['truth'] == -99)
    assert M.consistency(pairs['nees4'], 4)['within']
    assert abs(M.consistency(pairs['nees2'], 2)['inside']-0.95) < 0.02
    assert bins['truth'].sum() == 10*len(snaps) and bins['false'].sum() == len(snaps)
    s = M.summarize([M.evaluate_episode(beliefs, snaps)])
    assert s['gospa']['coverage'] == 1.0 and s['gospa']['false_per_frame'] == 1.0
    assert s['gospa']['false'] == pytest.approx(C2)
    assert s['id_switches']['switches'] == 0 and s['id_switches']['tracks_per_car'] == 1.0
    ft = s['false_tracks']
    assert ft['ghost'] == 1.0 and ft['never_matched'] == 1.0 and ft['never_matched_ids'] == 1
    assert ft['never_matched_life_max_s'] == pytest.approx(0.05*len(snaps))
    assert [b['changing'] for b in s['nees']['state']['by_lane_change']] == [False, True]
    assert abs(s['nees']['state']['per_time']['inside']-0.95) < 0.05
    for c in s['nees']['components'].values():
        assert abs(c['z_mean']) < 0.1 and abs(c['z_std']-1) < 0.1
    json.dumps(s, allow_nan=False)


def test_match_frames_skips_times_without_truth_and_pre_roll():
    beliefs, snaps = synthetic_episode(20)
    early = [(-0.05, beliefs[0][1])]
    frames = M.match_frames(early+beliefs[:10], snaps[2:])[0]
    np.testing.assert_allclose(frames['time'], [b[0] for b in beliefs[2:10]])


def test_id_switches_and_track_swaps():
    pairs = np.zeros(6, M.PAIR_DTYPE)
    pairs['time'] = [0.0, 0.05, 0.15, 0.2, 0.25, 0.3]
    pairs['truth'] = [7, 7, 7, 7, 8, 8]
    pairs['track'] = [1, 1, 1, 2, 1, 1]      # truth 7 keeps track 1 over a gap, then takes track 2; track 1 moves to 8
    ids = M.id_switches(pairs)
    assert ids == {'switches': 1, 'takeovers': None, 'rebirths': None, 'track_swaps': 1, 'pairs': 6, 'truths': 2,
                   'tracks': 2}
    alive = np.array([(1, 0.0, 0.3), (2, 0.2, 0.2)], M.SPAN_DTYPE)     # track 1 still confirmed at the switch
    assert (M.id_switches(pairs, alive)['takeovers'], M.id_switches(pairs, alive)['rebirths']) == (1, 0)
    gone = np.array([(1, 0.0, 0.15), (2, 0.2, 0.2)], M.SPAN_DTYPE)
    assert (M.id_switches(pairs[:4], gone)['takeovers'], M.id_switches(pairs[:4], gone)['rebirths']) == (0, 1)
    assert M.id_switches(np.zeros(0, M.PAIR_DTYPE))['switches'] == 0


# ---------------------------------------------------------------------------------------------------------- NIS
def nis_records(scale, n=20000, gate=0.99, seed=2):
    rng = np.random.default_rng(seed)
    out = []
    for sensor, kind, dof in (('radar_front', 'radar', 3), ('camera_front', 'camera_box', 4)):
        v = scale*rng.chisquare(dof, n)
        v = v[v <= chi2.ppf(gate, dof)]
        r = np.zeros(len(v), NIS_DTYPE)
        r['time'], r['sensor'], r['kind'], r['dof'], r['nis'] = rng.uniform(-1, 30, len(v)), sensor, kind, dof, v
        out.append(r)
    return np.concatenate(out)


def test_nis_against_the_gate_truncated_chi_square():
    rec = nis_records(1.0)
    stats = M.nis_stats(rec, 0.99)
    assert set(stats) == {'radar_front:3', 'camera_front:4'}
    assert sum(g['n'] for g in stats.values()) == int(np.sum(rec['time'] >= 0))
    s = M.nis_summary(stats, 0.99)
    for name in ('all', 'kind:radar', 'sensor:camera_front'):
        assert s[name]['within'] and abs(s[name]['ratio']-1) < 0.02 and abs(s[name]['inside']-0.95) < 0.01
        assert np.allclose(s[name]['pit'], 0.1, atol=0.012)
    m3, _ = M._truncated(3, 0.99)
    assert m3 < 3 and s['kind:radar']['expected'] == pytest.approx(m3)
    pooled = M.nis_summary(M._merge_nis([M.nis_stats(rec[:5000]), M.nis_stats(rec[5000:])]))
    assert pooled['all']['n'] == s['all']['n'] and pooled['all']['mean'] == pytest.approx(s['all']['mean'])
    big = M.nis_summary(M.nis_stats(nis_records(1.5)))['all']
    assert not big['within'] and big['ratio'] > 1.1 and big['pit'][-1] > 0.15
    assert M.nis_stats(np.zeros(0, NIS_DTYPE)) == {}


# ---------------------------------------------------------------------------------------------------------- detection
def scene():
    """Team car 0 at the origin heading +x, its mates behind; traffic: A ahead, B hidden behind A, C ahead left,
    E ahead right, F beyond every range."""
    team = np.array([[0.0, 0.0, 0.0, 30.0], [-50.0, 0.0, 0.0, 30.0], [-60.0, 3.7, 0.0, 30.0]])
    traffic = np.array([[30.0, 0.0, 0.0, 25.0, 0.0], [60.0, 0.0, 0.0, 25.0, 0.0], [40.0, 3.7, 0.0, 26.0, 0.0],
                        [20.0, -3.7, 0.0, 24.0, 0.0], [400.0, 0.0, 0.0, 25.0, 0.0]])
    return TruthState(1.0, traffic, np.arange(5), team, np.zeros(3), np.zeros(3), np.zeros(3))


def det(kind, spec, truth, z, time=None, R=None):
    pose = sensor_pose(spec.mount, truth.team[0], VEH)
    t = truth.time if time is None else time
    return Detection(kind, spec.name, 0, t, np.asarray(z, float), np.eye(len(z)) if R is None else R, pose,
                     EgoState(t, 0.0, 0.0, 0.0, 30.0), {'scan': truth.time})


def test_detection_counter_lidar_hits_misses_occlusion_false_and_teammates():
    spec = default_lidars()[0]
    truth = scene()
    c = body_centre(np.vstack((truth.traffic[:, :3], truth.team[:, :3])), VEH)
    earlier = c[0]-0.05*np.array([25.0, 0.0])          # A at its mean beam time 50 ms before the sweep end
    dets = [det('lidar_box', spec, truth, [*earlier, 0.0], time=truth.time-0.05),
            det('lidar_box', spec, truth, [*c[2], 0.0]), det('lidar_box', spec, truth, [100.0, 10.0, 0.0]),
            det('lidar_box', spec, truth, [*c[6], 0.0])]           # c[5:] the team: c[6] teammate 1
    counter = M.DetectionCounter([spec], VEH)
    counter.add(spec.name, 0, truth.time, dets, truth)
    r = counter.result()[spec.name]
    assert r['opp'].sum() == 4 and r['hit'].sum() == 2            # A, B, C, E in view; A and C seen
    assert r['opp_visible'].sum() == 3 and r['hit_visible'].sum() == 2   # B is hidden behind A
    assert (r['false'], r['mate'], r['scans'], r['detections']) == (1, 1, 1, 4)
    bin_of = lambda r_: np.searchsorted(M.RANGE_EDGES, r_, side='right')-1   # noqa: E731
    assert r['opp'][bin_of(27.7)] == 2 and r['hit'][bin_of(27.7)] == 2     # A (27.7 m) and C (37.8 m)
    assert r['opp'][bin_of(57.7)] == 1 and r['hit'][bin_of(57.7)] == 0     # B
    assert r['opp'][bin_of(18.0)] == 1 and r['opp_visible'][bin_of(18.0)] == 1 and r['hit'][bin_of(18.0)] == 0  # E
    counter.add(spec.name, 0, truth.time, [], truth)
    assert counter.result()[spec.name]['opp'].sum() == 8 and counter.result()[spec.name]['hit'].sum() == 2


def test_detection_counter_radar_and_camera():
    truth = scene()
    radar = default_radars()[0]
    pose = sensor_pose(radar.mount, truth.team[0], VEH)
    rear_a = np.array([[30.0-2.4665+0.05, 0.3]])          # a point on A's rear face
    loc = to_sensor(rear_a, pose)[0]
    counter = M.DetectionCounter([radar, default_cameras()[0]], VEH)
    counter.add(radar.name, 0, truth.time, [det('radar', radar, truth, [np.hypot(*loc), np.arctan2(loc[1], loc[0]),
                                                                          -5.0]),
                                            det('radar', radar, truth, [80.0, 0.5, 0.0])], truth)
    r = counter.result()[radar.name]
    assert r['hit'].sum() == 1 and r['false'] == 1 and r['opp'].sum() == 4
    cam = default_cameras()[0]
    cpose = sensor_pose(cam.mount, truth.team[0], VEH)
    u, v, depth = project(cam, cpose, box_corners_3d(truth.traffic[2:3, :3], VEH, 1.45))
    box = [u.min(), u.max(), v.min(), v.max()]
    counter.add(cam.name, 0, truth.time, [det('camera_box', cam, truth, box),
                                          det('camera_box', cam, truth, [box[0], box[1], box[2]-400, box[3]-400])],
                truth)
    r = counter.result()[cam.name]
    assert r['hit'].sum() == 1 and r['false'] == 1
    s = M.summarize([M.evaluate_episode([], [], None, counter.result())])['detection']
    assert s[radar.name]['pd_all'] == 0.25 and s[cam.name]['false_per_scan'] == 1.0
    json.dumps(s, allow_nan=False)


# ---------------------------------------------------------------------------------------------------------- pipeline
class Counting(PerceptionSystem):
    """The pipeline counting every scan's detections (as experiments/perception_shadow.py does)."""
    def reset_state(self):
        super().reset_state()
        self.counter = M.DetectionCounter(self.sensors, self.params, self.cfg.car_height)

    def _queue_scan(self, spec, car, time, dets, raw, true):
        super()._queue_scan(spec, car, time, dets, raw, true)
        if time >= 0:
            self.counter.add(spec.name, car, time, dets, self.truth.at(time))


def test_ideal_pipeline_tracks_every_car_in_the_window():
    """Experiment E0's sensors on 1 s of v9 driven straight (no planner): every car in the window is matched, with
    centimetre errors, and the detection counter sees every car with the all-seeing ideal lidar."""
    sc = replace(scenario_v9(), perception=Perception(enabled=True, mode='ideal'))
    world = World(sc, 1000)
    system = Counting(sc.perception, sc, world.road, world.plant.params, 1000, record=True)
    snaps = [world.snapshot]
    system.reset(snaps[0])
    for _ in range(20):
        snap, applied = world.step(np.zeros((3, 2)))
        system.step(snap, applied[:, 0].copy())
        snaps.append(snap)
    ev = M.evaluate_episode(system.beliefs, snaps, system.tracker.nis_records(), system.counter.result())
    s = M.summarize([ev])
    assert s['frames'] == 21 and s['gospa']['coverage'] == 1.0 and s['gospa']['false_per_frame'] == 0.0
    assert s['gospa']['rms_position_error'] < 0.05
    assert s['nis']['all']['n'] > 0
    assert s['detection']['lidar_ideal']['pd_all'] == 1.0
