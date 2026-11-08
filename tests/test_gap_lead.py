"""Lead-held gap rule (Planner.gap_lead): the gap rule's end position stays behind the slower car ahead in the ego's
lane until the ego is clear of it sideways. Without it, a car boxed in (a lead 12 m ahead, a car alongside in the
target lane at about its speed) is given the gap ahead of the car alongside, which it cannot reach: it stalls at the
lane line, is cancelled and retries (seed 2, batch 9c, 118-136 s)."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import Planner, Road as RoadConfig, Vehicle
from nohesi.dynamics import KinematicPlant
from nohesi.intentions import IntentionManager
from nohesi.road import Road
from tests.test_nohesi import snapshot

ROAD = Road(RoadConfig())
LEAD = replace(Planner(), gap_lead=True)
FAR = [(-300, 0, 23), (-600, 0, 23)]


def manager(cfg=LEAD):
    return IntentionManager(cfg, ROAD, KinematicPlant(Vehicle(), 0.05).params, lane=1, speed=30.0)


def boxed(beside_s=1.0, ego_d=0.0, lead=True):
    """Car 0 in lane 1 at 23 m/s (policy target 30); traffic car 1 12 m ahead in lane 1 at 23 m/s; traffic car 2 in
    lane 2 at beside_s (CG, relative to car 0) at 23.7 m/s."""
    traffic = [(2, beside_s, ROAD.lane_center(2), 23.7)]+([(1, 12.0, 0.0, 23.0)] if lead else [])
    return snapshot(0.0, [(0.0, ego_d, 23.0)]+FAR, traffic)


def test_boxed_in_car_gets_no_gap_ahead_of_the_car_alongside():
    plans = [None]*3
    for s in (-1.0, 1.0, 3.0):
        # Free-road prediction: the gap ahead of the car alongside (it is the rear gap car), unreachable.
        assert manager(Planner()).gap_rule(boxed(s), plans, 0, 2, 30.0) == ((None, ('traffic', 2)), ''), s
        assert manager().gap_rule(boxed(s), plans, 0, 2, 30.0)[0] is None, s
        # With the search, the gap behind it at a lower target speed (one the search reaches: at most gap_search_dv
        # below the policy's target).
        gap, reason, v = manager(replace(LEAD, gap_search=True)).lane_gap(boxed(s), plans, 0, 2, 28.0)
        assert gap == (('traffic', 2), None) and reason == '' and v < 23.7, (s, gap, v)
        # The search alone never runs: the free-road rule accepts first.
        assert manager(replace(Planner(), gap_search=True)).lane_gap(boxed(s), plans, 0, 2, 30.0) == \
            ((None, ('traffic', 2)), '', 30.0)


def test_unchanged_without_a_lead_in_the_way():
    plans = [None]*3
    for s in (-20.0, -8.0, 1.0, 10.0, 30.0):
        assert manager().gap_rule(boxed(s, lead=False), plans, 0, 2, 30.0) == \
            manager(Planner()).gap_rule(boxed(s, lead=False), plans, 0, 2, 30.0)
    im = manager()
    assert im._lead_cap(boxed(lead=False), 0, 2) is None
    assert manager(Planner())._lead_cap(boxed(), 0, 2) is None   # off


def test_cap_from_the_lead():
    im = manager()
    p, c = im.params, im.cfg
    side_gap = p.width+c.margin_lat
    room, v, until = im._lead_cap(boxed(), 0, 2)
    assert room == pytest.approx(max(12.0-p.length-c.margin_long-c.headway*23.0, 0.0)) and v == 23.0
    assert until == pytest.approx(c.gap_lead_time)
    # Part-way across: the lead still overlaps, for the remaining side distance only.
    assert im._lead_cap(boxed(ego_d=1.5), 0, 2)[2] == pytest.approx(c.gap_lead_time*(side_gap-1.5)/side_gap)
    assert im._lead_cap(boxed(ego_d=side_gap+0.05), 0, 2) is None   # clear of it sideways
    # A car in the target lane is a gap car, never the lead.
    s = snapshot(0.0, [(0.0, 2.0, 23.0)]+FAR, [(2, 5.0, ROAD.lane_center(2), 23.0)])
    assert im._lead_cap(s, 0, 2) is None


def test_reach_held_behind_the_lead():
    im = manager()
    free = im.reach(23.0, 30.0, 3.0)
    held = im.reach(23.0, 30.0, 3.0, cap=(0.0, 23.0, 1.2))
    s_12 = im.reach(23.0, 30.0, 1.2, cap=(0.0, 23.0, 1.2))
    assert s_12 == pytest.approx((23.0*1.2, 23.0))
    assert held[0] < free[0] and held[1] > 23.0          # free to speed up once clear
    assert im.reach(23.0, 30.0, 3.0, cap=(500.0, 23.0, 3.0)) == free   # room enough: no effect
    assert im.reach(23.0, 20.0, 3.0, cap=(0.0, 23.0, 3.0)) == im.reach(23.0, 20.0, 3.0)   # slowing: no effect
    assert np.isfinite(held[0])
