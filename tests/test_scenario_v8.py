"""Scenario v8: traffic speeds by lane (right and middle 60-65 mph, left 70-80 mph; a lane change draws a speed from the
new lane's range) and 30 traffic cars instead of 20."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, Traffic as TrafficConfig
from nohesi.road import Road
from nohesi.traffic import Traffic
from test_traffic_lane_changes import FAR, L, ROAD, drive, traffic

V8 = SCENARIOS['v8']()
SLOW, FAST = (26.82, 29.06), (31.29, 35.76)
SPEEDS = (SLOW, SLOW, FAST)


def test_v8_settings():
    v7 = SCENARIOS['v7']()
    assert V8.traffic.lane_speeds == SPEEDS and (v7.traffic.count, V8.traffic.count) == (20, 30)
    assert replace(V8, version='v7', traffic=v7.traffic) == v7   # only the traffic changed
    with pytest.raises(ValueError):
        TrafficConfig(lane_speeds=(SLOW, (30., 29.), FAST))   # min above max
    with pytest.raises(ValueError):
        Traffic(TrafficConfig(count=1, lane_speeds=(SLOW, FAST)), ROAD, np.random.default_rng(0), np.array([0.]), 1)


def in_lane_range(t):
    return all(SPEEDS[l][0] <= v <= SPEEDS[l][1] for l, v in zip(t.lane, t.v0))


def test_v8_spawns_and_respawns_in_the_lane_range():
    t = Traffic(V8.traffic, Road(V8.road), np.random.default_rng(3), np.zeros(3), V8.team_lane)
    assert in_lane_range(t) and t.v.tolist() == t.v0.tolist() and len(set(t.v0.tolist())) == len(t.v0)
    t.s[:] -= 1e4   # every car far behind: all respawn ahead
    t.recycle(np.zeros(3))
    assert in_lane_range(t)


def test_v8_lane_change_takes_the_new_lane_speed():
    # A 60 mph car stuck behind a slow car in the middle lane, with a car alongside on its right, moves left and takes
    # a speed in the left lane's 70-80 mph.
    t = traffic([(0., 1, 26.82, 26.82), (20., 1, 15., 15.), (0., 0, 26.82, 26.82)], lane_speeds=SPEEDS,
                speed_min=26.82, speed_max=35.76)
    log = drive(t, 4.0)
    assert log[0][3][0] == 2 and t.lane[0] == 2 and FAST[0] <= t.v0[0] <= FAST[1]
    assert t.v0[1] == 15.0   # no lane change: keeps its speed
    # Moving right from the left lane: 60-65 mph.
    t = traffic([(0., 2, 31.29, 31.29), (20., 2, 15., 15.)], lane_speeds=SPEEDS, speed_min=26.82, speed_max=35.76)
    drive(t, 4.0)
    assert t.lane[0] == 1 and SLOW[0] <= t.v0[0] <= SLOW[1]
