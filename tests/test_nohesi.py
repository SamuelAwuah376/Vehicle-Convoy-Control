import numpy as np
import pytest
from nohesi.config import Scenario, Score, Road as RoadConfig, Timing, Vehicle
from nohesi.dynamics import KinematicPlant
from nohesi.evaluation.metrics import metrics
from nohesi.evaluation.runner import run
from nohesi.geometry import clearance, footprint
from nohesi.road import Road
from nohesi.scoring.events import EventDetector, Measurements
from nohesi.scoring.score import Scorer, close_pass_points
from nohesi.world import Snapshot, World

VEHICLE = Vehicle()
ROAD = Road(RoadConfig())


def snapshot(time, team, traffic):
    """team: three (s, d, v); traffic: list of (id, s, d, v). Straight road, zero yaw."""
    team = np.array(team, float); traffic = np.array(traffic, float).reshape(-1, 4)
    states = np.column_stack((team[:, :2], np.zeros(3), team[:, 2]))
    return Snapshot(time, states, np.zeros(3), team[:, 0], team[:, 1], traffic[:, 1], traffic[:, 2],
                    traffic[:, 3], traffic[:, 0].astype(int), (), False)


def drive_past(lateral, start=-20.0, end=20.0, speed=28.0, dt=0.05):
    """Car 0 at d=0 passes a stationary-frame traffic car at d=lateral; teammates are far away."""
    detector = EventDetector(VEHICLE, ROAD)
    overtakes = []
    for k, ds in enumerate(np.arange(start, end, speed*dt*0.25)):
        snap = snapshot(k*dt, [(500+ds, 0, speed), (-500, 0, speed), (-1000, 0, speed)], [(7, 500, lateral, 20)])
        overtakes += detector.update(snap, dt).overtakes
    return overtakes


def test_road_frenet_round_trip_and_lanes():
    s, d = np.array([0., 12.5, 900.]), np.array([-3.6, 0., 3.6])
    np.testing.assert_allclose(ROAD.to_frenet(ROAD.to_xy(s, d)), (s, d))
    assert [ROAD.lane_center(i) for i in range(3)] == pytest.approx([-3.6, 0, 3.6])
    assert list(ROAD.lane_of([-5.0, -1.79, 0.0, 1.81, 5.0])) == [0, 1, 1, 2, 2]
    xy, yaw = ROAD.lane_route(2).at(10.0)
    np.testing.assert_allclose(xy, [10, 3.6]); assert yaw == 0


def test_clearance_known_rectangles():
    a = footprint(np.zeros(4), VEHICLE)
    assert clearance(a, footprint(np.array([0., 3., 0, 0]), VEHICLE)) == pytest.approx(1.2)
    assert clearance(a, footprint(np.array([0., 1., 0, 0]), VEHICLE)) == 0
    corner = footprint(np.array([VEHICLE.length+1, VEHICLE.width+1, 0, 0]), VEHICLE)
    assert clearance(a, corner) == pytest.approx(np.sqrt(2))


def test_plant_speed_dependent_limits():
    plant = KinematicPlant(VEHICLE, 0.05)
    assert plant.accel_limit(10) == VEHICLE.accel_max
    assert plant.accel_limit(40) == pytest.approx(VEHICLE.accel_max*20/40)
    _, a, _ = plant.step(np.array([0., 0, 0, 40]), 10, 0, 0)
    assert a == pytest.approx(1.5)
    # Grip cap: v^2 tan(delta)/L <= lateral limit at 30 m/s.
    cap = plant.steer_limit(30)
    assert 30**2*np.tan(cap)/VEHICLE.wheelbase == pytest.approx(VEHICLE.lateral_accel_max)


def test_traffic_seeded_and_non_overlapping():
    a, b, c = World(Scenario(), 3), World(Scenario(), 3), World(Scenario(), 4)
    np.testing.assert_array_equal(a.traffic.s, b.traffic.s)
    assert not np.array_equal(a.traffic.s, c.traffic.s)
    for world in (a, c):
        t = world.traffic
        assert len(t.s) == Scenario().traffic.count and not world.snapshot.done
        for lane in range(3):
            spacing = np.diff(np.sort(t.s[t.lane == lane]))
            assert np.all(spacing >= Scenario().traffic.min_gap)


def test_traffic_recycles_behind_team_with_new_id():
    world = World(Scenario(), 0)
    t = world.traffic
    t.s[0] = world.snapshot.team_s.min()-200
    old_id = t.id[0]
    t.recycle(world.snapshot.team_s)
    assert t.id[0] != old_id and t.id[0] == t.next_id-1
    assert t.s[0] > world.snapshot.team_s.max()+Scenario().traffic.respawn_ahead


def test_close_pass_detected_and_scored_exactly():
    overtakes = drive_past(lateral=2.8)  # 2.8 m centre offset -> 1.0 m clearance
    assert len(overtakes) == 1
    o = overtakes[0]
    assert (o.vehicle, o.traffic_id) == (0, 7)
    assert o.min_clearance == pytest.approx(1.0)
    cfg = Score()
    assert close_pass_points(o.min_clearance, o.speed, cfg) == pytest.approx(100*(1+0.5/1.5))


def test_wide_pass_not_scored_and_no_pass_without_overtake():
    wide = drive_past(lateral=3.6)
    assert len(wide) == 1 and wide[0].min_clearance == pytest.approx(1.8)
    assert close_pass_points(wide[0].min_clearance, 28, Score()) == 0
    assert drive_past(lateral=2.8, start=-20, end=0) == []   # pulled alongside, never got ahead
    assert drive_past(lateral=2.8, start=2, end=20) == []    # never came from behind


def test_proximity_and_speed_rates():
    scorer = Scorer(Score())
    m = Measurements(0.1, np.array([25., 25., 15.]), np.array([10., 30., 30.]), np.full(3, 5.),
                     np.full(3, np.inf), (), ())
    inc = scorer.update(m)
    np.testing.assert_allclose(inc['proximity'], [0.5, 0.5, 0])     # 10 pts/s split, pair 0-1 only
    np.testing.assert_allclose(inc['speed'], [0.5, 0.5, 0])         # 1 pt/s per m/s above 20
    # Car 2 is too slow, so pair 0-2 earns nothing even when close.
    m2 = Measurements(0.1, np.array([25., 25., 15.]), np.array([30., 5., 30.]), np.full(3, 5.),
                      np.full(3, np.inf), (), ())
    np.testing.assert_allclose(scorer.update(m2)['proximity'], 0)


def test_collision_penalises_and_ends_episode():
    world = World(Scenario(), 0)
    t = world.traffic
    # Put car 0 directly onto traffic car 0, matching its speed.
    world.team[0] = [t.s[0], world.road.lane_center(t.lane[0]), 0, t.v[0]]
    snap, _ = world.step(np.zeros((3, 2)))
    assert snap.done and any(c.vehicle == 0 and c.kind == 'traffic' for c in snap.collisions)
    scorer = Scorer(Score())
    inc = scorer.update(EventDetector(world.plant.params, world.road).update(snap, 0.05))
    assert inc['collision'][0] == -500 and inc['collision'][1:].tolist() == [0, 0]
    with pytest.raises(RuntimeError):
        world.step(np.zeros((3, 2)))


def test_teammate_collision_penalises_both():
    from nohesi.world import Collision
    m = Measurements(0.05, np.full(3, 25.), np.full(3, 30.), np.zeros(3), np.full(3, np.inf),
                     (), (Collision(1, 'team', 2),))
    assert Scorer(Score()).update(m)['collision'].tolist() == [0, -500, -500]


@pytest.mark.parametrize('seed', [0, 1, 2, 3, 4])
def test_scripted_lane_keeping_v0(seed):
    ep = run(Scenario(), seed)
    m = metrics(ep)
    assert m['end_reason'] == 'time' and m['collisions'] == []
    assert m['duration_s'] == pytest.approx(60)
    assert np.all(np.isfinite(ep.team)) and np.all(np.isfinite(ep.traffic_s))
    assert np.max(np.abs(ep.team_d)) < 0.05                          # stays in the middle lane
    assert m['min_clearance_teammates_m'] > 2.0
    assert np.min(ep.controls[:, :, 0]) >= -Vehicle().brake_max
    assert m['team_score'] == pytest.approx(sum(m['individual_scores']))


def test_same_seed_reproduces_metrics():
    strip = lambda m: {k: v for k, v in m.items() if k != 'policy_compute_ms'}
    short = Scenario(timing=Timing(duration=20))
    assert strip(metrics(run(short, 9))) == strip(metrics(run(short, 9)))


@pytest.mark.parametrize('kwargs', [{'team_lane': 3}, {'team_speed': -1}])
def test_invalid_scenario(kwargs):
    with pytest.raises(ValueError):
        Scenario(**kwargs)
