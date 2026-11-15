"""Haul-road scenarios h1/h2: the traffic is static obstacles (Traffic.static), and h2 runs the team in a mining
envelope (Scenario.team_speed_min/max, gravel friction)."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, Planner, Scenario, Traffic as TrafficCfg
from nohesi.evaluation.runner import EpisodeRun
from nohesi.road import Road
from nohesi.traffic import Traffic
from nohesi.world import World

H1, H2 = SCENARIOS['h1'](), SCENARIOS['h2']()


def test_settings():
    t = H1.traffic
    assert t.static and not t.lane_changes and t.no_blockades and t.count == 12 and t.blockade_gap == 45.0
    assert H1.vehicle == SCENARIOS['v9']().vehicle and H1.team_speed_max == 60.0 and H1.team_speed_min == 0.0
    assert (H2.team_speed, H2.team_speed_min, H2.team_speed_max, H2.vehicle.friction) == (12.5, 5.0, 16.7, 0.6)
    assert H2.traffic == t
    with pytest.raises(ValueError):
        TrafficCfg(static=True, lane_changes=True)
    with pytest.raises(ValueError):
        replace(Scenario(), team_speed_min=Scenario().team_speed+1)


def test_obstacles_stay_put_and_never_seal_the_road():
    for seed in range(1000, 1020):
        w = World(H1, seed)
        assert len(w.traffic.s) == 12 and np.all(w.traffic.v == 0) and w.traffic.blockades() == []
        s0, d0 = w.traffic.s.copy(), w.traffic.d.copy()
        w.traffic.step(0.05, w.snapshot.team_s, Road(H1.road).lane_of(w.snapshot.team_d), w.snapshot.team[:, 3],
                       H1.vehicle.length)
        assert np.array_equal(w.traffic.s, s0) and np.array_equal(w.traffic.d, d0)


def test_passed_obstacles_respawn_ahead_static():
    road = Road(H1.road)
    tr = Traffic(H1.traffic, road, np.random.default_rng(0), np.array([100.0, 80.0, 60.0]), H1.team_lane)
    team = np.full(3, tr.s.max()+400.0)   # the team is far past every obstacle
    ids = tr.id.copy()
    tr.recycle(team)
    moved = tr.id != ids
    assert moved.any() and np.all(tr.s[moved] > team[0]+H1.traffic.respawn_ahead-1e-9) and np.all(tr.v == 0)
    assert tr.blockades() == []


def test_h2_planner_speed_range():
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.env import planner_for
    r = EpisodeRun(replace(H2, timing=replace(H2.timing, duration=0.1)), 0, 'lane_keeping', Planner())
    assert (r.planner.speed_min, r.planner.speed_max) == (5.0, 16.7)
    p = planner_for(EnvConfig(scenario='h2'), H2)
    assert (p.speed_min, p.speed_max) == (5.0, 16.7)
    assert planner_for(EnvConfig(scenario='h1'), H1).speed_min == Planner().speed_min


def test_merge_feature_is_zero_for_obstacles():
    from nohesi.rl.env import merge_possible
    w = World(H1, 1000)
    assert not merge_possible(w.snapshot, w.road, w.plant.params, H1.traffic).any()
