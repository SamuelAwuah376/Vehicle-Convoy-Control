"""Scenario v6: the traffic never seals the road (no blockades), and the training-only stuck penalty (EnvConfig.aux_stuck)."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nohesi.config import SCENARIOS
from nohesi.evaluation.metrics import stuck_mask
from nohesi.road import Road
from nohesi.traffic import Traffic
from nohesi.world import World

V6 = SCENARIOS['v6']()
NONE = (np.empty(0), np.empty(0, int), np.empty(0))   # no team cars


def placed(cars, **cfg):
    """Traffic with hand-placed cars: list of (s, lane, v)."""
    tr = Traffic(replace(V6.traffic, count=len(cars), **cfg), Road(V6.road), np.random.default_rng(0), np.array([0.0]),
                 V6.team_lane)
    tr.s, tr.lane, tr.v = (np.array(c, float) for c in zip(*cars))
    tr.lane = tr.lane.astype(int); tr.v0 = tr.v.copy()
    tr.target, tr.lc_t, tr.lc_end = tr.lane.copy(), np.full(len(cars), np.nan), np.full(len(cars), -np.inf)
    return tr


def test_scenario_v6():
    v5 = SCENARIOS['v5']()
    assert V6.version == 'v6' and V6.traffic.no_blockades and not V6.traffic.no_walls
    assert V6.traffic.blockade_gap == pytest.approx(2*(V6.vehicle.length+0.25)+0.5)
    assert replace(V6, version='v5', traffic=v5.traffic) == v5   # only the traffic rule differs


def test_blockade_is_a_connected_group_across_all_lanes():
    g = V6.traffic.blockade_gap
    stair = placed([(500., 0, 24.), (500.+0.9*g, 1, 24.), (500.+1.8*g, 2, 24.)])
    assert [sorted(b) for b in stair.blockades()] == [[0, 1, 2]]   # diagonal: no wall, but no way through
    assert stair.walls() == []
    open_ = placed([(500., 0, 24.), (500.+1.1*g, 1, 24.), (500.+2.2*g, 2, 24.)])
    assert open_.blockades() == []                                  # every diagonal slot fits a car
    sides = placed([(500., 0, 24.), (500., 2, 24.)])                # lanes 0 and 2 alongside: lane 1 is open
    assert sides.blockades() == []


def test_spawns_and_lane_changes_never_seal():
    for seed in range(5):
        assert World(V6, seed).traffic.blockades() == []
    tr = placed([(500., 0, 24.), (500., 2, 24.), (600., 0, 24.)])
    assert not tr._free(505., 1) and tr._free(530., 1)              # the gap between the side cars stays open
    assert tr._seals(600., {0, 1}, skip=2) is False and tr._seals(505., {0, 1}, skip=2)


def test_closing_car_is_held_back_by_a_virtual_leader():
    g = V6.traffic.blockade_gap
    tr = placed([(500., 0, 22.), (500., 2, 22.), (470., 1, 27.)])  # lane 1 car closing on the pair from behind
    lead, gap = tr.blockade_leaders()
    assert lead[2] in (0, 1) and gap[2] == pytest.approx(30.-g) and (lead[:2] == -1).all()
    held = tr.idm(*NONE, V6.vehicle.length)
    free = placed([(500., 0, 22.), (500., 2, 22.), (470., 1, 27.)], no_blockades=False).idm(*NONE, V6.vehicle.length)
    assert held[2] < free[2]-0.5 and np.allclose(held[:2], free[:2])
    for _ in range(400):                                            # 20 s: it never seals the road
        tr.step(0.05, *NONE, V6.vehicle.length)
        assert tr.blockades() == []
    assert tr.s[2] < 500.+22.*20 and abs(tr.v[2]-22.) < 1.0        # it settled behind the pair at their speed


def test_v6_traffic_never_blocks_over_an_episode():
    from nohesi.config import Planner
    from nohesi.evaluation.runner import EpisodeRun
    r = EpisodeRun(replace(V6, timing=replace(V6.timing, duration=40.0)), 3, 'overtaker',
                   Planner(speed_max=V6.team_speed_max))
    while not r.done:
        r.step()
        assert r.world.traffic.blockades() == []
    assert r.episode().end_reason == 'time'


def test_stuck_mask():
    road = Road(V6.road)
    lane = [road.lane_center(l) for l in (0, 1, 2)]
    team_s, team_d, team_v = np.array([[0., 0., 300.]]), np.array([[lane[1], lane[1], lane[0]]]), np.array([[22., 30., 22.]])
    # Car 0 sits 40 m behind a 22 m/s car in its lane; car 1 closes on it at 8 m/s; car 2's car ahead is 150 m away.
    traffic_s, traffic_d, traffic_v = np.array([[40., 450.]]), np.array([[lane[1], lane[0]]]), np.array([[22., 22.]])
    assert stuck_mask(road, team_s, team_d, team_v, traffic_s, traffic_d, traffic_v).tolist() == [[True, False, False]]
    # A teammate between the car and the traffic car does not shield it.
    assert stuck_mask(road, np.array([[0., 20., 300.]]), team_d, np.array([[22., 22., 22.]]),
                      traffic_s, traffic_d, traffic_v)[0, :2].tolist() == [True, True]


def test_stuck_penalty_after_the_grace_only():
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.env import NoHesiEnv
    env = NoHesiEnv(replace(EnvConfig(), aux_stuck=-0.25))
    env.reset(seed=0)
    flags = iter([[True, True, False]]*50+[[False, True, False]]*10)
    env.stuck = lambda snap: np.array(next(flags))
    t, total = env._stuck_t, 0.0
    for k in range(60):                                            # 0.05 s steps: 2.5 s stuck, then car 0 drives off
        total += env._stuck_seconds(SimpleNamespace(time=t+0.05*(k+1)))
    assert total == pytest.approx(2*(2.5-2.0)+(3.0-2.5)) and env.stuck_for.tolist() == pytest.approx([0., 3., 0.])
    plain = NoHesiEnv(EnvConfig())                                 # default 0: no stuck term
    plain.reset(seed=0)
    from nohesi.intentions import encode
    assert 'stuck' not in plain.step([encode(0, 0, 0)]*3)[4]['aux']


def test_stuck_term_in_the_reward_and_metrics():
    from nohesi.intentions import encode
    from nohesi.rl.config import EnvConfig
    from nohesi.rl.env import NoHesiEnv
    env = NoHesiEnv(replace(EnvConfig(scenario='v6', score='v8'), aux_stuck=-0.25, stuck_grace=0.0))
    env.reset(seed=0)
    env.stuck = lambda snap: np.ones(3, bool)
    _, reward, _, _, info = env.step([encode(0, 0, 0)]*3)
    seconds = env.team.decide_every*env.scenario.timing.dt
    assert info['aux']['stuck'] == pytest.approx(-0.25*3*seconds)
    assert reward == pytest.approx(sum(info['score_terms'].values())/env.cfg.reward_scale+sum(info['aux'].values()))
    from nohesi.evaluation.metrics import metrics
    from nohesi.evaluation.runner import run
    m = metrics(run(replace(V6, timing=replace(V6.timing, duration=10.0)), 0, 'lane_keeping'))
    assert len(m['stuck_time_s']) == 3 and all(0 <= x <= 10.0 for x in m['stuck_time_s'])


def test_scenario_v7_lane_change_probability():
    from nohesi.config import Traffic as TrafficConfig
    v7 = SCENARIOS['v7']()
    assert v7.version == 'v7' and v7.traffic.lane_change_prob == 0.1
    assert replace(v7, version='v6', traffic=replace(v7.traffic, lane_change_prob=1.0)) == V6
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError):
            TrafficConfig(lane_change_prob=bad)

    def slow_leader(prob):   # a 25 m/s car 40 m behind a 15 m/s car: MOBIL picks a change at the first evaluation
        tr = placed([(0., 1, 25.), (40., 1, 15.)], lane_change_prob=prob, lane_changes=True)
        tr.next_eval = np.zeros(2)
        tr.step(0.05, *NONE, V6.vehicle.length)
        return tr
    go, stay = slow_leader(1.0), slow_leader(1e-9)
    assert go.changing[0] and not stay.changing[0]
    assert stay.lc_end[0] == pytest.approx(0.0)   # decided to stay: cooldown from now
    for _ in range(int(V6.traffic.lane_change_cooldown/0.05)-2):
        stay.step(0.05, *NONE, V6.vehicle.length)
    assert stay.lc_end[0] == pytest.approx(0.0)   # no new evaluation during the cooldown
