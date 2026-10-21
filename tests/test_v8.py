"""Scenario v5 (BMW M3s, faster traffic, no traffic walls) and score v8 (fast team play)."""
import numpy as np
import pytest
from nohesi.config import SCENARIOS, SCORES, bmw_m3
from nohesi.road import Road
from nohesi.scoring.score import Scorer, close_pass_points
from test_score_v4 import DT, step, close


def test_bmw_m3_and_scenario_v5():
    car = bmw_m3()
    assert (car.length, car.width, car.wheelbase, car.brake_max, car.lateral_accel_max) == (4.79, 1.90, 2.857, 11.0, 9.5)
    assert car.accel_max*car.power_speed == pytest.approx(210.4, abs=1)    # ~375 kW / 1,780 kg
    sc = SCENARIOS['v5']()
    assert sc.vehicle == car and sc.road.lane_width == 3.6 and sc.team_speed == 31.3
    assert (sc.traffic.speed_min, sc.traffic.speed_max, sc.traffic.no_walls) == (20.1, 26.8, True)
    assert sc.traffic.wall_window == pytest.approx(2*car.length)


def wall_traffic(no_walls=True):
    from dataclasses import replace
    from nohesi.traffic import Traffic
    sc = SCENARIOS['v5']()
    cfg = replace(sc.traffic, count=3, no_walls=no_walls)
    tr = Traffic(cfg, Road(sc.road), np.random.default_rng(0), np.array([0.0]), sc.team_lane)
    # Hand-placed: one car per lane within 6 m of each other (a wall), all at 24 m/s.
    tr.s, tr.lane = np.array([500., 503., 506.]), np.array([0, 1, 2])
    tr.v, tr.v0 = np.full(3, 24.0), np.full(3, 24.0)
    tr.target, tr.lc_t, tr.lc_end = tr.lane.copy(), np.full(3, np.nan), np.full(3, -np.inf)
    return tr


def test_spawns_never_form_walls():
    from nohesi.world import World
    for seed in range(5):
        assert World(SCENARIOS['v5'](), seed).traffic.walls() == []


def test_wall_detected_and_rear_car_drops_back():
    tr = wall_traffic()
    assert [sorted(w) for w in tr.walls()] == [[0, 1, 2]]
    a = tr.idm(np.empty(0), np.empty(0, int), np.empty(0), 4.79)
    b = wall_traffic(no_walls=False).idm(np.empty(0), np.empty(0, int), np.empty(0), 4.79)
    assert a[0] < b[0]-0.5 and np.allclose(a[1:], b[1:])   # only the rearmost car (s = 500) slows


def test_lane_change_never_closes_the_last_open_lane():
    tr = wall_traffic()
    tr.lane = np.array([0, 1, 1])                  # lane 2 is open; car 2 moving there would close it
    assert tr._forms_wall(506., 2, skip=2) and not tr._forms_wall(506., 0, skip=2)
    assert not tr._free(506., 2, skip=2)


def test_v8_train_needs_80_mph():
    v8 = SCORES['v8']()
    slow = dict(s=(0., 18., 36.), lanes=(0, 1, 2))                       # 13.5 m gaps: a train at 28 m/s
    assert Scorer(v8).update(step(**slow))['draft'].sum() == 0
    fast = dict(s=(0., 22., 44.), lanes=(0, 1, 2), speeds=(40., 40., 40.))  # 17.5 m <= 0.5 s * 40 m/s
    assert Scorer(v8).update(step(**fast))['draft'].tolist() == pytest.approx([25*DT]*3)
    assert Scorer(SCORES['v7']()).update(step(**slow))['draft'].sum() > 0  # v7: 45 mph was enough


def test_v8_top_speed_bonus_and_settings():
    v8 = SCORES['v8']()
    inc = Scorer(v8).update(step(speeds=(55., 50., 54.)))
    assert inc['top_speed'].tolist() == pytest.approx([10*DT, 0, 10*DT])
    assert (v8.spread_penalty_rate, v8.spread_penalty_cap, v8.team_pass_distance, v8.train_min_speed) == (4.0, 400.0, 50.0, 35.76)
    scorer = Scorer(v8)
    for _ in range(4):
        scorer.update(step(overtakes=[close(2)]))
    assert scorer.update(step(speeds=(55., 50., 50.)))['top_speed'][0] == pytest.approx(10*DT*scorer.multiplier)


def test_env_team_speed_max():
    rl_env = pytest.importorskip('nohesi.rl.env')
    from nohesi.rl.config import EnvConfig
    env = rl_env.NoHesiEnv(EnvConfig(scenario='v5', score='v8'))
    env.reset(seed=0)
    assert env.team.cfg.speed_max == 60.0 and env.scenario.vehicle.brake_max == 11.0
    env4 = rl_env.NoHesiEnv(EnvConfig(scenario='v4', score='v7'))
    env4.reset(seed=0)
    assert env4.team.cfg.speed_max == 35.0
