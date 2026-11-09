"""Scenario v9: v8 with 45 traffic cars (min_gap 15 m), three lane speeds (55-62 / 62-68 / 70-80 mph), traffic lane
changes 3x as often, and spread respawns (Traffic.respawn_spread): a recycled car tries the emptiest lane near the team
first, at several spots, so the no-blockade rule cannot keep a lane empty."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS
from nohesi.road import Road
from nohesi.traffic import Traffic

V8, V9 = SCENARIOS['v8'](), SCENARIOS['v9']()
MPH = 0.44704
ROAD = Road(V9.road)


def placed(cars, cfg=V9.traffic, seed=0):
    """Traffic with hand-placed cars: list of (s, lane)."""
    tr = Traffic(replace(cfg, count=len(cars)), ROAD, np.random.default_rng(seed), np.array([-1e5]), V9.team_lane)
    tr.s, tr.lane = (np.array(c, float) for c in zip(*cars))
    tr.lane = tr.lane.astype(int)
    tr.v = np.array([tr.cfg.lane_speeds[l][0] for l in tr.lane]); tr.v0 = tr.v.copy()
    tr.target, tr.lc_t, tr.lc_end = tr.lane.copy(), np.full(len(cars), np.nan), np.full(len(cars), -np.inf)
    tr.id = np.arange(len(cars)); tr.next_id = len(cars)
    return tr


def test_v9_settings():
    t = V9.traffic
    assert V9.version == 'v9' and (V8.traffic.count, t.count) == (30, 45) and t.min_gap == 15.0
    assert (V8.traffic.spawn_ahead, t.spawn_ahead) == (350.0, 400.0)
    assert np.allclose(t.lane_speeds, ((55*MPH, 62*MPH), (62*MPH, 68*MPH), (70*MPH, 80*MPH)))
    assert (t.speed_min, t.speed_max) == pytest.approx((55*MPH, 80*MPH)) and t.lane_change_prob == 0.3
    assert t.respawn_spread and not V8.traffic.respawn_spread and t.no_blockades
    assert replace(V9, version='v8', traffic=V8.traffic) == V8   # only the traffic changed


def test_v9_places_all_45_cars():
    # Evaluation seeds and a stretch of training seeds (>= 2000), team at its start positions (World): at 350 m ahead
    # (v8's window) 4.6 % of seeds could not place all 45 cars.
    v = V9.vehicle
    team_s = V9.team_start_s+(v.length+v.standstill_gap+(V9.team_start_headway or v.headway)*V9.team_speed)*np.arange(3)[::-1]
    for seed in list(range(1000, 1050))+list(range(2000, 2300)):
        t = Traffic(V9.traffic, ROAD, np.random.default_rng(seed), team_s, V9.team_lane)
        assert len(t.s) == 45 and t.blockades() == []


def dense_two_lanes():
    """Middle and fast lanes: cars alongside each other every 25 m from 0 to 325 m; right lane empty; car 0 far behind."""
    pairs = [(s, l) for s in np.arange(0.0, 326.0, 25.0) for l in (1, 2)]
    return [(-1000.0, 1)]+pairs


def test_spread_respawn_fills_the_lane_a_blockade_kept_empty():
    # One spot 250-300 m ahead: a right-lane car there would close a blockade, the other lanes are within min_gap of a
    # car, so the car is not respawned (and the right lane stays empty).
    off = placed(dense_two_lanes(), replace(V9.traffic, respawn_spread=False))
    off.recycle(np.zeros(3))
    assert off.s[0] == -1000.0 and off.id[0] == 0
    # Spread: the right lane first (empty), at the nearest spot clear of the dense stretch.
    on = placed(dense_two_lanes())
    on.recycle(np.zeros(3))
    c = V9.traffic
    assert on.lane[0] == 0 and on.id[0] == len(on.s) and on.blockades() == []
    assert 325.0+c.blockade_gap+c.blockade_buffer <= on.s[0] < 325.0+c.blockade_gap+c.blockade_buffer+c.respawn_spot_gap
    assert c.lane_speeds[0][0] <= on.v0[0] <= c.lane_speeds[0][1] and on.v[0] == on.v0[0]


def test_spread_respawn_prefers_the_emptiest_lane():
    c = V9.traffic
    cars = [(-1000.0, 1), (50.0, 0), (120.0, 0), (200.0, 0), (80.0, 1), (160.0, 1), (100.0, 2), (900.0, 2)]
    for seed in range(5):
        t = placed(cars, seed=seed)
        t.recycle(np.zeros(3))
        # Lane 2 has one car in the counting window (the one 900 m ahead is outside it), lane 1 has one besides the
        # recycled car itself, lane 0 three: ties between lanes 1 and 2 go either way, never lane 0.
        assert t.lane[0] in (1, 2) and c.respawn_ahead <= t.s[0] < c.respawn_ahead+c.respawn_spot_gap


def test_respawn_unchanged_without_spread():
    """Off, recycle places cars and draws random numbers exactly as before the spread option."""
    def before(tr, team_s):
        rear, front = np.min(team_s), np.max(team_s)
        for i in np.flatnonzero(tr.s < rear-tr.cfg.recycle_behind):
            s = front+tr.cfg.respawn_ahead+tr.rng.uniform(0, 50)
            lanes = [l for l in tr.rng.permutation(tr.road.cfg.lanes) if tr._free(s, l, skip=i)]
            if lanes:
                v0 = tr._speed(lanes[0])
                tr.s[i], tr.lane[i], tr.v[i], tr.v0[i] = s, lanes[0], v0, v0
                tr.target[i], tr.lc_t[i], tr.lc_end[i] = lanes[0], np.nan, -np.inf
                tr.id[i] = tr.next_id; tr.next_id += 1
    for seed in range(3):
        a, b = (Traffic(V8.traffic, Road(V8.road), np.random.default_rng(seed), np.zeros(3), V8.team_lane) for _ in range(2))
        for t in (a, b):
            t.s[::3] -= 1e4
        a.recycle(np.zeros(3))
        before(b, np.zeros(3))
        for k in ('s', 'lane', 'v', 'v0', 'id', 'target'):
            assert np.array_equal(getattr(a, k), getattr(b, k)), k
        assert a.rng.uniform() == b.rng.uniform()
