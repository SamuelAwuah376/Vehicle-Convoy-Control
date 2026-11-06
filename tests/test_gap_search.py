"""Gap search (Planner.gap_search, IntentionManager.lane_gap): a stuck car whose lane change finds no room at its target
speed locks a gap behind or ahead of the blocking car with a lower / higher target speed; the policy's target speed comes
back when the change ends."""
from dataclasses import replace
import numpy as np
from nohesi.config import Planner, Road as RoadConfig, Vehicle
from nohesi.dynamics import KinematicPlant
from nohesi.intentions import IntentionManager, encode
from nohesi.road import Road
from tests.test_nohesi import snapshot

ON = replace(Planner(), gap_search=True)
ROAD = Road(RoadConfig())
TEAM = [(0, 0, 28), (-300, 0, 28), (-600, 0, 28)]
SLOW = (1, 40.0, 0.0, 27.0)   # 40 m ahead in the ego's lane, 1 m/s slower: car 0 is stuck


def manager(cfg=ON):
    return IntentionManager(cfg, ROAD, KinematicPlant(Vehicle(), 0.05).params, lane=1, speed=28.0)


def snap(beside_s, beside_v=28.0, stuck=True):
    """Car 0 in lane 1 at 28 m/s; traffic car 2 in lane 2 at beside_s (CG, relative to car 0)."""
    return snapshot(0.0, TEAM, [(2, beside_s, ROAD.lane_center(2), beside_v)]+([SLOW] if stuck else []))


def test_stuck_car_takes_the_gap_behind_or_ahead():
    plans = [None]*3
    for s, front, rear, faster in ((0.0, ('traffic', 2), None, False),    # level: drop behind it (25 m/s)
                                   (2.0, ('traffic', 2), None, False),    # slightly ahead: behind it
                                   (-2.0, None, ('traffic', 2), True),    # slightly behind: get ahead of it
                                   (-4.0, None, ('traffic', 2), True)):
        im = manager()
        assert im.gap_rule(snap(s), plans, 0, 2, 28.0) == (None, 'gap_too_short')
        gap, reason, v = im.lane_gap(snap(s), plans, 0, 2, 28.0)
        assert gap == (front, rear) and reason == '' and (v > 28.0) == faster and v != 28.0, (s, gap, v)
        assert im.last_gap_margin >= ON.gap_search_margin and abs(v-28.0) <= ON.gap_search_dv
        assert manager(Planner()).lane_gap(snap(s), plans, 0, 2, 28.0) == (None, 'gap_too_short', 28.0)   # off
        assert manager().lane_gap(snap(s, stuck=False), plans, 0, 2, 28.0)[0] is None                       # not stuck
    assert manager().lane_gap(snap(-5.0), plans, 0, 2, 28.0) == ((None, ('traffic', 2)), '', 28.0)   # plain rule ok


def test_no_drop_behind_a_car_entirely_behind():
    # A faster car 8 m behind (CG): the plain rule refuses (rear_closing); letting it pass would mean braking behind
    # a car that is behind the ego now: never its front gap car.
    im = manager()
    s = snap(-8.0, beside_v=32.0)
    assert im.gap_rule(s, [None]*3, 0, 2, 28.0)[0] is None
    gap, _, v = im.lane_gap(s, [None]*3, 0, 2, 28.0)
    assert gap is None or (gap[0] != ('traffic', 2) and v > 28.0)


def test_decide_sets_and_restores_the_target_speed():
    im = manager()
    s = snap(0.0)
    im.decide(s, [encode(1, 0, 0), encode(0, 0, 0), encode(0, 0, 0)], [None]*3, [0, 1, 2])
    it = im.intentions[0]
    assert it.phase == 'pending' and it.lane == 2 and it.front == ('traffic', 2) and it.rear is None
    assert it.speed == 25.0 and it.resume == 28.0
    assert [r.reason for r in im.records if r.kind == 'accepted'] == ['search_behind']
    # While it is pending, the policy's speed commands adjust its own target (resume), not the searched speed.
    im.decide(s, [encode(1, 1, 0), encode(0, 0, 0), encode(0, 0, 0)], [None]*3, [0, 1, 2])
    assert it.speed == 25.0 and it.resume == 28.0+ON.speed_step
    im._keep(0, 2)   # the change ends (completed or cancelled): the policy's target speed back
    assert it.speed == 30.0 and it.resume is None


def test_reach_with_a_gentler_brake():
    im = manager()
    hard, gentle = im.reach(28.0, 20.0, 3.0), im.reach(28.0, 20.0, 3.0, brake=4.0)
    assert gentle[0] > hard[0] and np.isclose(gentle[1], 20.0) and np.isclose(hard[1], 20.0)
