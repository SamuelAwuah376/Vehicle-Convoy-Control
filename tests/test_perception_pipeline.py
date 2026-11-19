"""The end-to-end perception pipeline (docs/perception_design.md sections 3, 5-8): Tier A sensors, processing and the
team tracker behind the seam, the belief adapter (perceived snapshot, belief prediction), the pre-roll, the modes
('sensors', 'shadow', 'ideal') and sharing ('team', 'own')."""
import hashlib
from dataclasses import replace
import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from nohesi.config import Planner, scenario_v9
from nohesi.evaluation.runner import EpisodeRun
from nohesi.perception.belief import belief_prediction, perceived_snapshot
from nohesi.perception.config import Perception, TrackerConfig
from nohesi.perception.pipeline import IDEAL_LIDAR, PerceptionSystem
from nohesi.perception.system import make_perception
from nohesi.perception.tracking.models import process_noise
from nohesi.perception.types import Belief, TruthHistory
from nohesi.road import Road
from nohesi.world import World

SEED = 1000
GATE = 2.0          # m: a confirmed track within this of a true car (Hungarian) is that car's track
NEAR = 100.0        # m from the nearest team car: where the planner's decisions are made
ARRAYS = ('time', 'team', 'team_s', 'team_d', 'traffic_s', 'traffic_d', 'traffic_v', 'traffic_id', 'controls',
          'pair_clearance', 'traffic_clearance')


def scenario(duration, **perception):
    s = scenario_v9()
    return replace(s, timing=replace(s.timing, duration=duration), perception=Perception(enabled=True, **perception))


def recording(sc, world, seed):
    return PerceptionSystem(sc.perception, sc, world.road, world.plant.params, seed, record=True)


def episode_run(duration, perception=recording, **mode):
    r = EpisodeRun(scenario(duration, **mode), SEED, 'overtaker', Planner(), perception=perception)
    while not r.done:
        r.step()
    return r


def match(belief, snap, gate=GATE):
    """Confirmed tracks matched to true cars by position (Hungarian within gate): (confirmed belief, track rows,
    car columns, distances (tracks, cars))."""
    b = belief.select(belief.confirmed)
    D = np.hypot(b.mean[:, None, 0]-snap.traffic_s[None], b.mean[:, None, 1]-snap.traffic_d[None])
    if len(b) == 0:
        return b, np.zeros(0, int), np.zeros(0, int), D
    rows, cols = linear_sum_assignment(np.where(D < gate, D, 1e6))
    ok = D[rows, cols] < gate
    return b, rows[ok], cols[ok], D


def near_team(s, snap, reach=NEAR):
    return np.min(np.abs(np.asarray(s)[:, None]-snap.team_s[None]), axis=1) <= reach


def inside_window(s, snap, cfg, behind=10.0, ahead=60.0):
    """Within the tracker's team window, less `behind` at its rear edge (where cars leave it) and `ahead` at its front
    (respawns land 250-300 m ahead, inside the 300 m window, and are unconfirmed for their first cycles)."""
    s = np.asarray(s)
    return (s >= snap.team_s.min()-cfg.roi_behind+behind) & (s <= snap.team_s.max()+cfg.roi_ahead-ahead)


def truth_at(run_):
    return {round(float(s.time), 6): s for s in run_.snaps}


def digest(ep):
    m = hashlib.sha256()
    for k in ARRAYS:
        m.update(np.ascontiguousarray(getattr(ep, k)).tobytes())
    for t in sorted(ep.score):
        m.update(np.ascontiguousarray(ep.score[t]).tobytes())
    return m.hexdigest()


def drive(system, steps, seed=SEED):
    """system on the truth of a v9 world driven straight at constant speed (no planner): its perceived snapshots."""
    world = World(system.scenario, seed)
    views = [system.reset(world.snapshot)]
    for _ in range(steps):
        snap, applied = world.step(np.zeros((3, 2)))
        views.append(system.step(snap, applied[:, 0].copy()))
    return world, views


def system_for(sc, seed=SEED, **kw):
    world = World(sc, seed)
    return PerceptionSystem(sc.perception, sc, world.road, world.plant.params, seed, **kw)


@pytest.fixture(scope='module')
def shadow():
    """10 s of v9 with the scripted overtaker and the default planner, perception in shadow mode."""
    return episode_run(10.0, mode='shadow')


# ---------------------------------------------------------------------------------------------- the belief adapter
def toy_belief():
    cov = np.stack([np.diag([0.04, 0.01, 0.25, 0.09])*(k+1) for k in range(3)])
    lane_prob = np.array([[0.1, 0.8, 0.1], [0.0, 0.3, 0.7], [1.0, 0.0, 0.0]])
    return Belief(2.0, np.array([7, 3, 9]), np.array([[50.0, 0.1, 30.0, 0.0], [80.0, 1.0, 28.0, 0.8],
                                                      [-20.0, -3.6, 33.0, 0.0]]), cov, np.array([0.9, 0.6, 0.2]),
                  lane_prob, np.full((3, 3), 1/3), np.array([1.9, 1.95, 1.8]), np.array([True, True, False]))


def test_perceived_snapshot_replaces_the_traffic_with_the_confirmed_tracks():
    world = World(scenario_v9(), SEED)
    truth = world.snapshot
    before = {k: np.copy(getattr(truth, k)) for k in ('traffic_s', 'traffic_d', 'traffic_v', 'traffic_id', 'team')}
    view = perceived_snapshot(truth, toy_belief(), world.road)
    np.testing.assert_array_equal(view.traffic_id, [7, 3])
    np.testing.assert_array_equal(view.traffic_s, [50.0, 80.0])
    np.testing.assert_array_equal(view.traffic_d, [0.1, 1.0])
    np.testing.assert_array_equal(view.traffic_v, [30.0, 28.0])
    np.testing.assert_array_equal(view.traffic_vd, [0.0, 0.8])
    np.testing.assert_array_equal(view.traffic_target_lane, [1, 2])
    np.testing.assert_allclose(view.traffic_sig_s, [0.2, np.sqrt(0.08)])
    np.testing.assert_allclose(view.traffic_sig_d, [0.1, np.sqrt(0.02)])
    np.testing.assert_allclose(view.traffic_sig_v, [0.5, np.sqrt(0.5)])
    np.testing.assert_array_equal(view.traffic_p_exist, [0.9, 0.6])
    assert view.traffic_id.dtype.kind == 'i' and view.traffic_target_lane.dtype.kind == 'i'
    for k in ('time', 'team', 'team_steer', 'team_s', 'team_d', 'collisions', 'done', 'team_vy', 'team_r'):
        assert getattr(view, k) is getattr(truth, k)
    for k, v in before.items():
        np.testing.assert_array_equal(getattr(truth, k), v)
    everything = perceived_snapshot(truth, toy_belief(), world.road, confirmed_only=False)
    np.testing.assert_array_equal(everything.traffic_id, [7, 3, 9])
    empty = perceived_snapshot(truth, Belief.empty(0.0), world.road)
    assert len(empty.traffic_s) == 0 and len(empty.traffic_target_lane) == 0 and len(empty.traffic_p_exist) == 0


def test_belief_prediction_is_the_exact_linear_propagation_in_lane():
    """Lane keep at the lane centre is linear, so the unscented propagation is exact: s = s0 + v t and
    var_s = P_ss + 2 t P_sv + t^2 P_vv + q^2 t^3 / 3, whether the horizon is taken in one step or many."""
    cfg, road = TrackerConfig(), Road(scenario_v9().road)
    P = np.diag([0.25, 1e-4, 1.0, 1e-4])
    P[0, 2] = P[2, 0] = 0.1
    centre = road.lane_center(1)
    b = Belief(0.0, np.array([1]), np.array([[100.0, centre, 30.0, 0.0]]), P[None], np.ones(1), np.eye(3)[[1]],
               np.array([[1.0, 0, 0]]), np.zeros(1), np.ones(1, bool))
    times = np.arange(0, 3.01, 0.25)
    s, d, v, psi, sig_s, sig_d = belief_prediction(b, road, times, cfg)
    assert s.shape == (len(times), 1)
    np.testing.assert_allclose(s[:, 0], 100.0+30.0*times, atol=1e-9)
    np.testing.assert_allclose(d[:, 0], centre, atol=1e-9)
    np.testing.assert_allclose(v[:, 0], 30.0)
    np.testing.assert_allclose(psi[:, 0], 0.0, atol=1e-12)
    var = 0.25+2*times*0.1+times**2*1.0+cfg.q_long**2*times**3/3
    np.testing.assert_allclose(sig_s[:, 0], np.sqrt(var), rtol=1e-9)
    assert np.all(np.diff(sig_d[:, 0]) > 0)
    one = belief_prediction(b, road, [3.0], cfg)
    np.testing.assert_allclose([one[0][0, 0], one[4][0, 0]], [s[-1, 0], sig_s[-1, 0]], rtol=1e-9)
    shuffled = belief_prediction(b, road, times[::-1], cfg)
    np.testing.assert_allclose(shuffled[4][::-1], sig_s, rtol=1e-12)
    assert np.allclose(process_noise('keep', cfg, 3.0)[0, 0], cfg.q_long**2*9)
    with pytest.raises(ValueError):
        belief_prediction(b, road, [-0.1], cfg)
    assert all(a.shape == (2, 0) for a in belief_prediction(Belief.empty(0.0), road, [0.0, 1.0]))


def test_belief_prediction_models_and_heading():
    road = Road(scenario_v9().road)
    b = toy_belief()
    keep = belief_prediction(b, road, [0.0, 2.0])
    left = belief_prediction(b, road, [0.0, 2.0], model='left')
    np.testing.assert_allclose(keep[0][0], b.mean[:, 0])
    np.testing.assert_allclose(keep[3][0], np.arctan2(b.mean[:, 3], b.mean[:, 2]))
    assert left[1][1, 0] > keep[1][1, 0]+0.5          # the change model carries car 7 toward the next lane
    assert abs(keep[1][1, 0]-road.lane_center(1)) < 0.1


# ---------------------------------------------------------------------------------------------- construction
def test_make_perception_builds_the_pipeline_with_its_own_rng():
    s = scenario(1.0)
    world = World(s, 7)
    system = make_perception(s, world, 7)
    assert isinstance(system, PerceptionSystem)
    assert system.cfg is s.perception and system.road is world.road and system.params is world.plant.params
    assert system.rng.random() == np.random.default_rng([7, s.perception.seed_tag]).random()
    assert system.hold == pytest.approx(0.08+0.1) and len(system.trackers) == 1
    assert not system.record and system.beliefs == []
    assert system_for(scenario(1.0, mode='ideal')).hold == 0.0
    assert system_for(scenario(1.0, sharing='own')).hold == pytest.approx(0.08)


def test_team_history_matches_the_truth_history():
    """Processing's ego and teammate states interpolate the team exactly as the simulators' TruthHistory does."""
    s = scenario(1.0)
    system = system_for(s)
    snap = World(s, SEED).snapshot
    truth = TruthHistory()
    yaw = np.array([3.1, -3.1, 0.2])
    for k in range(4):
        team = snap.team.copy()
        team[:, 0] += 1.5*k
        team[:, 2] = np.where(k % 2, -yaw, yaw)
        sn = replace(snap, time=0.05*k, team=team, team_vy=np.full(3, 0.1*k), team_r=np.full(3, -0.2*k))
        accel = np.array([1.0, -2.0, 0.5])*k
        truth.push(sn, accel)
        system.team.push(sn, accel)
    for t in (-0.1, 0.0, 0.01, 0.07, 0.124, 0.15, 0.3):
        ts = truth.at(t)
        for car in range(3):
            ego = system.team.ego(car, t, pitch_per_accel=0.5)
            np.testing.assert_allclose([ego.x, ego.y], ts.team[car, :2], atol=1e-12)
            assert abs(np.angle(np.exp(1j*(ego.yaw-ts.team[car, 2])))) < 1e-12
            assert ego.vx == pytest.approx(np.sqrt(ts.team[car, 3]**2-ts.team_vy[car]**2))
            assert (ego.vy, ego.r, ego.ax) == pytest.approx((ts.team_vy[car], ts.team_r[car], ts.team_ax[car]))
            assert ego.pitch == pytest.approx(-0.5*ts.team_ax[car]) and ego.time == t
        np.testing.assert_allclose(system.team.mates(1, t)[:, :2], ts.team[[0, 2], :2], atol=1e-12)


def test_preroll_runs_the_cars_backwards_without_the_world():
    s = scenario(1.0, mode='ideal')
    system = system_for(s, record=True)
    world = World(s, SEED)
    snap = world.snapshot
    traffic = world.traffic.s.copy(), world.traffic.d.copy()
    view = system.reset(snap)
    np.testing.assert_array_equal(world.traffic.s, traffic[0])
    np.testing.assert_array_equal(world.traffic.d, traffic[1])
    assert world.steps == 0
    times = [t for t, _ in system.beliefs]
    np.testing.assert_allclose(times, np.arange(-1.0, 0.001, 0.05), atol=1e-9)
    assert view.time == snap.time and view.team is snap.team
    back = system._backward(snap, 0.4)
    assert back.time == pytest.approx(-0.4) and back.collisions == () and not back.done
    np.testing.assert_allclose(back.traffic_s, snap.traffic_s-0.4*snap.traffic_v)
    np.testing.assert_allclose(back.team[:, 0], snap.team[:, 0]-0.4*snap.team[:, 3])
    np.testing.assert_array_equal(back.team[:, 1:], snap.team[:, 1:])
    np.testing.assert_allclose(back.team_s, back.team[:, 0])
    # The tracks are confirmed by the end of the pre-roll, before the first decision.
    b, rows, cols, D = match(system.belief, snap)
    inside = inside_window(snap.traffic_s, snap, s.perception.tracker)
    assert set(np.flatnonzero(inside)) <= set(cols) and inside.sum() > 30


# ---------------------------------------------------------------------------------------------- ideal (E0)
def test_ideal_mode_tracks_the_truth():
    r = episode_run(2.0, mode='ideal')
    system = r.perception
    truth = truth_at(r)
    cfg = system.cfg.tracker
    assert system.sensors == (IDEAL_LIDAR,) and system.hold == 0.0
    worst, speed, checked = 0.0, 0.0, 0
    for t, belief in system.beliefs:
        if t < -1e-9:
            continue
        snap = truth[round(t, 6)]
        b, rows, cols, D = match(belief, snap, gate=1.0)
        assert set(np.flatnonzero(inside_window(snap.traffic_s, snap, cfg))) <= set(cols), t
        unmatched = np.ones(len(b), bool)
        unmatched[rows] = False
        assert not np.any(unmatched & inside_window(b.mean[:, 0], snap, cfg)), t     # no false tracks
        worst = max(worst, float(D[rows, cols].max()))
        speed = max(speed, float(np.abs(b.mean[rows, 2]-snap.traffic_v[cols]).max()))
        checked += 1
    assert checked == len(r.snaps)
    assert worst < 0.1 and speed < 0.5
    view = r.view
    assert view is not r.snap and len(view.traffic_s) > 30 and np.all(view.traffic_sig_s < 0.1)


# ---------------------------------------------------------------------------------------------- shadow
def test_shadow_runs_the_pipeline_but_returns_the_truth(shadow):
    system = shadow.perception
    assert shadow.episode().end_reason == 'time' and len(shadow.controls) == 200
    assert len(system.beliefs) == 21+200 and shadow.view is shadow.snap
    assert system.tracker.stats['replays'] == 0 and system.tracker.stats['late_dropped'] == 0
    # The world is untouched: the same episode as with perception off.
    off = EpisodeRun(replace(shadow.scenario, perception=Perception()), SEED, 'overtaker', Planner())
    while not off.done:
        off.step()
    assert off.perception is None and digest(off.episode()) == digest(shadow.episode())
    small = system_for(scenario(1.0, mode='shadow'))
    world = World(small.scenario, SEED)
    assert small.reset(world.snapshot) is world.snapshot
    snap, applied = world.step(np.zeros((3, 2)))
    assert small.step(snap, applied[:, 0]) is snap and len(small.belief) > 0


def test_shadow_beliefs_cover_the_traffic_near_the_team(shadow):
    system, truth = shadow.perception, truth_at(shadow)
    seen = total = false = steps = 0
    err, dv, ids = [], [], {}
    switches = 0
    for t, belief in system.beliefs:
        if t < -1e-9:
            continue
        snap = truth[round(t, 6)]
        b, rows, cols, D = match(belief, snap)
        near = near_team(snap.traffic_s, snap)
        matched = np.zeros(len(near), bool)
        matched[cols] = True
        total += near.sum()
        seen += (near & matched).sum()
        unmatched = np.ones(len(b), bool)
        unmatched[rows] = False
        false += (unmatched & near_team(b.mean[:, 0], snap)).sum()
        steps += 1
        keep = near[cols]
        err += list(D[rows, cols][keep])
        dv += list(np.abs(b.mean[rows, 2]-snap.traffic_v[cols])[keep])
        for a, j in zip(rows[keep], cols[keep]):
            tid = int(snap.traffic_id[j])
            switches += tid in ids and ids[tid] != int(b.id[a])
            ids[tid] = int(b.id[a])
        assert np.all(belief.last_seen <= t-system.hold+1e-6)      # nothing is used before it can arrive
    err, dv = np.array(err), np.array(dv)
    print(f'\nshadow, within {NEAR:.0f} m: coverage {seen/total:.4f} of {total}, false confirmed {false/steps:.2f} per '
          f'step, position error mean {err.mean():.3f} p95 {np.percentile(err, 95):.3f} max {err.max():.3f} m, '
          f'speed error p95 {np.percentile(dv, 95):.3f} m/s, id switches {switches} over {len(ids)} cars')
    print('timing ms per call:', {k: round(v, 2) for k, v in system.timing().items()})
    assert seen/total >= 0.99
    assert err.mean() < 0.15 and np.percentile(err, 95) < 0.4
    assert np.percentile(dv, 95) < 1.5
    assert false/steps < 2.0 and switches <= 3


def test_shadow_scan_schedule_and_profile(shadow):
    system = shadow.perception
    scans = system.scan_records()
    rates = {s.name: s.rate for s in system.sensors}
    for car in range(3):
        for name, rate in rates.items():
            mine = scans[(scans['car'] == car) & (scans['sensor'] == name)]
            first = -1.0+(1/rate if name.startswith('lidar') else 0.0)
            np.testing.assert_allclose(mine['time'], first+np.arange(len(mine))/rate, atol=1e-9)
            assert mine['time'][-1] == pytest.approx(10.0)
    assert np.all(scans['true'] <= scans['raw'])
    front = scans[scans['sensor'] == 'lidar_front']
    assert np.all(front['true'] == -1) and front['detections'].mean() > 3
    for name in ('radar_front', 'camera_front', 'radar_rear', 'camera_rear'):
        assert scans[scans['sensor'] == name]['detections'].mean() > 1
    timing = system.timing()
    assert all(timing[k] > 0 for k in ('sim_lidar', 'sim_radar', 'sim_camera', 'proc_lidar', 'proc_radar',
                                       'proc_camera', 'tracker'))
    assert timing['ideal'] == 0 and timing['total'] == pytest.approx(sum(v for k, v in timing.items() if k != 'total'))
    assert system.calls == 221


# ---------------------------------------------------------------------------------------------- sensors
def test_sensors_mode_the_policy_plans_on_the_belief(monkeypatch):
    import nohesi.coordination.centralized as centralized
    seen = []
    original = centralized.traffic_prediction
    monkeypatch.setattr(centralized, 'traffic_prediction', lambda snap, *a: (seen.append(snap), original(snap, *a))[1])
    r = EpisodeRun(scenario(3.0), SEED, 'overtaker', Planner(), perception=recording)
    first = r.view
    assert first is not r.snap and first.time == r.snap.time == 0.0 and first.team is r.snap.team
    assert first.traffic_sig_s is not None and first.traffic_p_exist is not None
    # Confirmed tracks after the pre-roll already cover the traffic near the team.
    b, rows, cols, D = match(r.perception.belief, r.snap)
    near = near_team(r.snap.traffic_s, r.snap)
    assert np.isin(np.flatnonzero(near), cols).mean() >= 0.9
    views = []
    while not r.done:
        view, belief = r.view, r.perception.belief
        assert view.time == r.snap.time == belief.time and view is not r.snap
        np.testing.assert_array_equal(view.traffic_id, belief.id[belief.confirmed])
        np.testing.assert_array_equal(view.traffic_s, belief.mean[belief.confirmed, 0])
        views.append(view)
        r.step()
    assert r.episode().end_reason == 'time' and len(r.controls) == 60
    assert seen and all(any(s is v for v in views) for s in seen)      # the MPC obstacles come from the belief


def test_same_seed_same_beliefs():
    def beliefs(seed):
        system = system_for(scenario(1.0, preroll=0.5), seed, record=True)
        drive(system, 6)
        return system.beliefs
    a, b, c = beliefs(SEED), beliefs(SEED), beliefs(SEED+1)
    assert len(a) == len(b) == 11+6
    fields = ('id', 'mean', 'cov', 'p_exist', 'lane_prob', 'mode_prob', 'last_seen', 'confirmed')
    for (ta, x), (tb, y) in zip(a, b):
        assert ta == tb
        for k in fields:
            np.testing.assert_array_equal(getattr(x, k), getattr(y, k))
    assert any(len(x) != len(y) or not np.array_equal(x.mean, y.mean) for (_, x), (_, y) in zip(a, c))


def test_own_sharing_tracks_each_car_on_its_own_sensors():
    system = system_for(scenario(1.0, sharing='own'), record=True)
    world, views = drive(system, 4)
    assert len(system.trackers) == 3 and len(system.car_beliefs) == len(system.beliefs) == 21+4
    for i, tracker in enumerate(system.trackers):
        records = tracker.nis_records()
        assert len(records) and np.all(records['car'] == i)
        assert tracker.belief(world.snapshot.time).confirmed.sum() > 10
    car0 = system.trackers[0].belief(world.snapshot.time)
    np.testing.assert_array_equal(views[-1].traffic_id, car0.id[car0.confirmed])
    t, beliefs = system.car_beliefs[-1]
    assert t == world.snapshot.time and beliefs[0] is system.beliefs[-1][1]


def test_rl_env_observes_the_belief():
    pytest.importorskip('gymnasium')
    from nohesi.intentions import encode
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.env import NoHesiEnv
    cfg = EnvConfig(version='e5', scenario='v9', score='v10', overtakes=True, follow=True, crash_penalty=-10000.0,
                    aux_stuck=-20.0, stuck_grace=0.5, stuck_escape_only=True, stuck_pass_lanes=True, obs_clock=False)
    env = NoHesiEnv(cfg)
    env.scenario = replace(env.scenario, perception=Perception(enabled=True))
    obs, _ = env.reset(seed=SEED)
    assert isinstance(env.run.perception, PerceptionSystem)
    assert env.snapshot is not env.true_snapshot and env.snapshot.traffic_sig_s is not None
    masks = env.action_masks()
    obs2, reward, terminated, truncated, _ = env.step([encode(0, 0, 0)]*3)
    assert obs.shape == obs2.shape and np.isfinite(obs2).all() and masks.any()
    assert not terminated and not truncated and np.isfinite(reward)
