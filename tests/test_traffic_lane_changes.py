from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, Planner, Road as RoadConfig, Timing, Traffic as TrafficConfig, scenario_v1, scenario_v2
from nohesi.evaluation.runner import make_controller, run
from nohesi.geometry import footprint, overlap
from nohesi.road import Road
from nohesi.traffic import Traffic
from nohesi.world import World, traffic_states

ROAD = Road(RoadConfig())
FAR = (np.full(3, -1e4), np.ones(3, int), np.full(3, 28.0))   # team out of the way
L = 4.5


def traffic(cars, **cfg):
    """cars: list of (s, lane, v, v0). All cars evaluate MOBIL from t=0."""
    t = Traffic(TrafficConfig(count=len(cars), lane_changes=True, **cfg), ROAD, np.random.default_rng(0),
                np.array([-2e4]), 1)
    t.s, t.lane, t.v, t.v0 = (np.array(c, float) for c in zip(*cars))
    t.lane = t.lane.astype(int); t.target = t.lane.copy(); t.next_eval[:] = 0
    return t


def drive(t, seconds, dt=0.05):
    log = []
    for _ in range(int(round(seconds/dt))):
        t.step(dt, *FAR, L)
        log.append((t.d.copy(), t.vd.copy(), t.lane.copy(), t.target_lane.copy()))
    return log


def test_config_and_scenarios():
    assert not TrafficConfig().lane_changes
    TrafficConfig(lane_changes=True, mobil_politeness=0, mobil_threshold=0, keep_right_bias=-0.1)
    with pytest.raises(ValueError):
        TrafficConfig(lane_change_time=0)
    with pytest.raises(ValueError):
        TrafficConfig(mobil_politeness=-1)
    v3, v4 = SCENARIOS['v3'](), SCENARIOS['v4']()
    assert (v3.version, v4.version) == ('v3', 'v4') and v3.traffic.lane_changes and v4.traffic.lane_changes
    assert v3 == replace(scenario_v1(), version='v3', traffic=replace(scenario_v1().traffic, lane_changes=True))
    assert v4 == replace(scenario_v2(), version='v4', traffic=replace(scenario_v2().traffic, lane_changes=True))


def test_slow_leader_triggers_smooth_change():
    t = traffic([(0, 1, 25, 25), (40, 1, 15, 15)])
    cfg = t.cfg
    log = drive(t, 6.0)
    d = np.array([x[0][0] for x in log]); vd = np.array([x[1][0] for x in log])
    start = next(k for k, x in enumerate(log) if x[3][0] != 1)
    new = log[start][3][0]
    assert new in (0, 2) and t.lane[0] == new and not t.changing[0]
    assert d[-1] == pytest.approx(ROAD.lane_center(new))
    end = next(k for k in range(start, len(log)) if not log[k][1][0])
    assert (end-start)*0.05 == pytest.approx(cfg.lane_change_time, abs=0.06)
    assert np.max(np.abs(vd)) <= 1.875*3.6/cfg.lane_change_time+1e-9
    assert np.all(np.abs(np.diff(d)) <= 2.25*0.05+1e-9)
    # Heading follows atan2(vd, v) during the change.
    t2 = traffic([(0, 1, 25, 25), (40, 1, 15, 15)]); drive(t2, 1.5)
    assert t2.changing[0] and t2.heading[0] == pytest.approx(np.arctan2(t2.vd[0], t2.v[0])) and t2.heading[0] != 0


def test_no_change_when_new_follower_would_brake_hard():
    # Fast cars just behind in both adjacent lanes: entering would force them past mobil_safe_brake.
    t = traffic([(0, 1, 25, 25), (40, 1, 15, 15), (-8, 0, 32, 32), (-8, 2, 32, 32)])
    log = drive(t, 0.5)
    assert all(x[3][0] == 1 for x in log)
    # Same leader, no blockers: it changes at the first evaluation.
    t = traffic([(0, 1, 25, 25), (40, 1, 15, 15)])
    assert drive(t, 0.05)[0][3][0] != 1


def test_no_change_without_room_beside():
    t = traffic([(0, 1, 25, 25), (40, 1, 15, 15), (2, 0, 25, 25), (-3, 2, 25, 25)])
    assert all(x[3][0] == 1 for x in drive(t, 0.5))


def test_changing_car_leads_in_both_lanes():
    t = traffic([(0, 1, 25, 25), (40, 1, 15, 15), (-20, 0, 25, 25)], lane_change_interval=100.0)
    t.next_eval[1:] = np.inf
    t.target[0], t.lc_t[0] = 0, 0.0       # car 0 is moving into lane 0 ahead of car 2
    free = Traffic.idm(traffic([(-20, 0, 25, 25)]), *FAR, L)[0]
    assert t.idm(*FAR, L)[2] < free


def test_snapshot_fields():
    w = World(scenario_v1(), 0)
    assert np.all(w.snapshot.traffic_vd == 0)
    np.testing.assert_array_equal(w.snapshot.traffic_target_lane, ROAD.lane_of(w.snapshot.traffic_d))
    w = World(SCENARIOS['v3'](), 0)
    w.traffic.target[0], w.traffic.lc_t[0] = (2 if w.traffic.lane[0] == 1 else 1), 1.0
    snap = w._snapshot()
    assert snap.traffic_vd.shape == snap.traffic_s.shape and snap.traffic_vd[0] != 0
    assert snap.traffic_target_lane[0] == w.traffic.target[0]
    assert traffic_states(snap, w.road)[0, 2] == pytest.approx(np.arctan2(snap.traffic_vd[0], snap.traffic_v[0]))


def test_no_lane_changes_keeps_lane_centres():
    ep = run(replace(scenario_v1(), timing=Timing(duration=20.0)), 0, 'lane_keeping')
    centres = np.array([ROAD.lane_center(l) for l in range(3)])
    assert np.all(np.min(np.abs(ep.traffic_d[..., None]-centres), axis=-1) == 0)


def test_lane_changes_deterministic():
    sc = replace(SCENARIOS['v3'](), timing=Timing(duration=20.0))
    a, b = run(sc, 1, 'lane_keeping'), run(sc, 1, 'lane_keeping')
    for f in ('traffic_s', 'traffic_d', 'traffic_v', 'traffic_id'):
        np.testing.assert_array_equal(getattr(a, f), getattr(b, f))


def test_v3_run_has_lane_changes_and_no_traffic_overlap():
    sc = SCENARIOS['v3']()
    world = World(sc, 0)
    policy = make_controller('lane_keeping', sc, world, Planner())
    cfg, snap, changes, prev = sc.vehicle, world.snapshot, 0, None
    for k in range(int(round(sc.timing.duration/sc.timing.dt))):
        snap, _ = world.step(policy(snap))
        lane = world.road.lane_of(snap.traffic_d)
        if prev is not None:
            changes += int(np.sum((lane != prev[0]) & (snap.traffic_id == prev[1])))
        prev = lane, snap.traffic_id
        if k % 5 == 0:
            states = traffic_states(snap, world.road)
            for i in range(len(states)):
                for j in range(i+1, len(states)):
                    if abs(states[i, 0]-states[j, 0]) < cfg.length+cfg.width:
                        assert not overlap(footprint(states[i], cfg), footprint(states[j], cfg)), (snap.time, i, j)
        if snap.done:
            break
    assert changes >= 3


def test_traffic_lane_change_cooldown():
    """No traffic car starts a lane change within lane_change_cooldown of finishing its previous one."""
    import numpy as np
    from nohesi.config import SCENARIOS
    from nohesi.rl.scripted import LaneKeeping
    from nohesi.world import World
    sc = SCENARIOS['v4']()
    world = World(sc, 0)
    policy = LaneKeeping(sc, world.road, world.plant)
    snap, last_end, was, gaps, changes = world.snapshot, {}, world.traffic.changing.copy(), [], 0
    ids = world.traffic.id.copy()
    for _ in range(int(60/sc.timing.dt)):
        snap, _ = world.step(policy(snap))
        tr = world.traffic
        for j in range(len(tr.s)):
            key = int(tr.id[j])
            if tr.changing[j] and not was[j]:
                changes += 1
                if key in last_end:
                    gaps.append(tr.time-last_end[key])
            if was[j] and not tr.changing[j] and ids[j] == key:   # a finished change, not a respawn
                last_end[key] = tr.time
        was, ids = tr.changing.copy(), tr.id.copy()
        if snap.done:
            break
    assert changes > 0
    assert not gaps or min(gaps) >= sc.traffic.lane_change_cooldown-1e-6
