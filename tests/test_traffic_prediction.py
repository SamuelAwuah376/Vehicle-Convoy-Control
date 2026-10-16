from dataclasses import fields, replace
import numpy as np
import pytest
from nohesi.config import Planner, Road as RoadConfig, Scenario, Vehicle
from nohesi.coordination.centralized import CentralizedTeam
from nohesi.dynamics import KinematicPlant
from nohesi.intentions import IntentionManager
from nohesi.prediction import traffic_prediction, vehicle_ahead_incl_merging
from nohesi.rl.scripted import Overtaker
from nohesi.road import Road
from nohesi.world import Snapshot, World
from tests.test_nohesi import snapshot

ROAD, CFG = Road(RoadConfig()), Planner()
TIMES = np.arange(31)*0.1
FAR = [(200, 0, 28), (0, 0, 28), (-300, 0, 28)]


def changing(snap, vd, target):
    if 'traffic_vd' not in {f.name for f in fields(Snapshot)}:
        pytest.skip('Snapshot has no traffic_vd yet')
    return replace(snap, traffic_vd=np.array(vd, float), traffic_target_lane=np.array(target, int))


def test_constant_d_without_lateral_motion():
    snap = snapshot(0.0, FAR, [(5, 50, 1.0, 25), (6, 80, -3.6, 30)])
    s, d, v, psi = traffic_prediction(snap, ROAD, TIMES)
    np.testing.assert_array_equal(s, snap.traffic_s[None, :]+snap.traffic_v[None, :]*TIMES[:, None])
    np.testing.assert_array_equal(d, np.repeat(snap.traffic_d[None, :], len(TIMES), axis=0))
    assert not psi.any() and (v == snap.traffic_v).all()
    s2, d2, _, psi2 = traffic_prediction(changing(snap, [0, 0], [1, 0]), ROAD, TIMES)
    np.testing.assert_array_equal(d2, d); np.testing.assert_array_equal(s2, s); assert not psi2.any()


def test_lane_change_stops_at_target_centre():
    snap = changing(snapshot(0.0, FAR, [(5, 50, 1.0, 25)]), [1.2], [2])
    _, d, _, psi = traffic_prediction(snap, ROAD, TIMES)
    arrive = (3.6-1.0)/1.2   # 2.17 s
    moving = TIMES < arrive
    np.testing.assert_allclose(d[moving, 0], 1.0+1.2*TIMES[moving])
    np.testing.assert_allclose(d[~moving, 0], 3.6)
    np.testing.assert_allclose(psi[moving, 0], np.arctan2(1.2, 25)); assert not psi[~moving].any()


def test_gap_rule_counts_car_merging_into_target_lane():
    # Ego (car 1) asks for lane 2 (end 84 m at the lookahead). A car in lane 1 10 m behind at 31 m/s ends 1 m behind
    # the slot: harmless while it keeps its lane, a too-short gap once it merges into lane 2.
    im = IntentionManager(CFG, ROAD, KinematicPlant(Vehicle(), 0.05).params, lane=1, speed=28.0)
    snap = snapshot(0.0, FAR, [(5, -10, 1.0, 31)])
    assert im.gap_rule(snap, [None]*3, 1, 2, 28.0) == ((None, None), '')
    assert im.gap_rule(changing(snap, [1.2], [2]), [None]*3, 1, 2, 28.0) == (None, 'gap_too_short')
    assert im.gap_rule(changing(snap, [0.0], [1]), [None]*3, 1, 2, 28.0) == ((None, None), '')


def test_vehicle_ahead_includes_merging_traffic():
    snap = snapshot(0.0, [(200, 3.6, 28), (0, 3.6, 28), (-300, 0, 28)], [(5, 30, 1.0, 25)])
    assert vehicle_ahead_incl_merging(snap, ROAD, 0.0, 2, skip=1) == (200, 28)
    assert vehicle_ahead_incl_merging(changing(snap, [1.2], [2]), ROAD, 0.0, 2, skip=1) == (30, 25)


def test_obstacles_follow_predicted_d():
    world = World(Scenario(), 0)
    team = CentralizedTeam(Scenario(), world, Overtaker(world.road, world.plant.params), CFG)
    snap = changing(snapshot(0.0, FAR, [(5, 50, 1.0, 25), (6, 80, -3.6, 30)]), [1.2, 0], [2, 0])
    obs = team._obstacles(snap, 1, {})
    times = np.arange(CFG.horizon+1)*CFG.step
    s, d, v, _ = traffic_prediction(snap, team.road, times)
    for j, key in enumerate([('traffic', 5), ('traffic', 6)]):
        k = obs.index(key)
        np.testing.assert_allclose(obs.d[:, k], d[:, j]); np.testing.assert_allclose(obs.s[:, k], s[:, j])
    assert obs.d[-1, obs.index(('traffic', 5))] == pytest.approx(min(3.6, 1.0+1.2*times[-1]))
    assert (obs.d[:, obs.index(('traffic', 6))] == -3.6).all()
