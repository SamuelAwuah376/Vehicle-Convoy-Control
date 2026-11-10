"""Fixes from the milestone 8 code review: teammate plans read at the snapshot's time (Planner.mate_plan_time), a
fallback out of the passing lane ends the overtake (Planner.fallback_ends_overtake), a teammate's pending change is
cancelled instead of the lane-keeping car yielding (Planner.yield_pending), the lane route continues past the road's
10 km and yaw_margin_scope is validated."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, Planner, Road as RoadConfig
from nohesi.evaluation.runner import EpisodeRun
from nohesi.intentions import Overtake
from nohesi.planning.base import Plan
from nohesi.road import Road


def team(planner):
    """The overtaker team at the start of a v8 episode: all three in lane 1, car 0 in front, order [0, 1, 2]."""
    r = EpisodeRun(SCENARIOS['v8'](), 0, 'overtaker', planner)
    return r.policy, r.snap


def plan(start, time, v=30.0, step=0.1, n=30):
    k = np.arange(n+1)
    z = np.c_[start+v*step*k, np.zeros(n+1), np.zeros(n+1), np.full(n+1, v)]
    return Plan(time, step, z, np.zeros((n, 2)), 'qp')


def test_teammate_plan_read_at_the_snapshot_time():
    for flag, step in ((False, 20), (True, 21)):
        t, snap = team(replace(Planner(), mate_plan_time=flag))
        p = plan(100.0, snap.time-0.1)                      # the previous replan's plan: one step before the snapshot
        assert t.intentions._plan_step(snap, p, 2.0) == step
        assert t.intentions._predicted_s(snap, [None, p, None], ('team', 1), 2.0) == pytest.approx(100.0+3.0*step)
    t, snap = team(replace(Planner(), mate_plan_time=True))
    assert t.intentions._plan_step(snap, plan(0.0, snap.time), 2.0) == 20   # a plan made at the snapshot: as before


def passing(t, stage='pass'):
    it = t.intentions.intentions[0]
    it.overtake = Overtake(('traffic', 0), +1, 1, 2.0, stage, 0.0, 0.0)   # home lane 1, passing lane 2
    it.lane, it.phase = 2, 'keep'
    return it


def test_fallback_out_of_the_passing_lane_ends_the_overtake():
    assert Planner().fallback_ends_overtake                # on by default since m4.4
    t, snap = team(replace(Planner(), fallback_ends_overtake=False))
    it = passing(t)
    t._keep_lane(snap, 0, 1, 'emergency')                  # off: moved back home, still 'pass'
    assert it.lane == 1 and it.overtake is not None and it.overtake.stage == 'pass'
    t, snap = team(replace(Planner(), fallback_ends_overtake=True))
    it = passing(t)
    t._keep_lane(snap, 0, 2, 'emergency')                  # stays in the passing lane: the overtake goes on
    assert it.overtake is not None
    t._keep_lane(snap, 0, 1, 'emergency')
    assert it.lane == 1 and it.overtake is None
    assert [(r.kind, r.reason) for r in t.intentions.records][-1:] == [('overtake_ended', 'fallback_emergency')]
    it = passing(t, 'return')                              # a return cancelled into the home lane: _overtake_step's
    it.phase, it.origin = 'pending', 2                     # 'returned'
    t._keep_lane(snap, 0, 1, 'emergency')
    assert it.overtake is not None and it.phase == 'keep' and it.lane == 1


def scripted(t, conflicts):
    """Replace the planner's solves and checks: _plan_vehicle returns a marker plan, _conflict looks up `conflicts`
    (car -> hit or None, re-evaluated per call so a test can change it), and records which cars were re-planned."""
    replanned = []
    t._plan_vehicle = lambda snap, i, fresh: replanned.append(i) or plan(0.0, snap.time)
    t._conflict = lambda snap, i, fresh, check: conflicts(i)
    t._emergency = lambda snap, i, plans, check: replanned.append(('emergency', i)) or plan(0.0, snap.time)
    t.emergency_mpc = True
    return replanned


def test_pending_teammate_change_into_the_lane_is_cancelled():
    t, snap = team(replace(Planner(), yield_pending=True))
    jt = t.intentions.intentions[0]
    jt.origin, jt.lane, jt.phase = 0, 1, 'pending'         # car 0 pending from lane 0 into car 2's lane 1
    replanned = scripted(t, lambda i: None)
    fresh = {i: plan(0.0, snap.time) for i in range(3)}
    hit = (('team', 0), 7)
    assert t._cancel_pending(snap, 2, hit, fresh, [0, 1, 2], 2) is None
    assert (jt.phase, jt.lane) == ('keep', 0) and t.intentions.records[-1].reason == 'yield_pending'
    assert replanned == [0, 2] and t.log.conflicts[-1] == (snap.time, 2, ('team', 0), 7, 'mate_cancelled')
    # The car between them in the order (1) conflicts with car 0's new plan: re-planned, then emergency.
    t, snap = team(replace(Planner(), yield_pending=True))
    jt = t.intentions.intentions[0]
    jt.origin, jt.lane, jt.phase = 0, 1, 'pending'
    replanned = scripted(t, lambda i: (('team', 0), 3) if i == 1 else None)
    assert t._cancel_pending(snap, 2, hit, fresh, [0, 1, 2], 2) is None
    assert replanned == [0, 1, ('emergency', 1), 2]
    # Not pending (committed), or not into the keeping car's lane: the rule does not apply.
    for phase, lane in (('committed', 1), ('pending', 2)):
        t, snap = team(replace(Planner(), yield_pending=True))
        jt = t.intentions.intentions[0]
        jt.origin, jt.lane, jt.phase = 0 if lane == 1 else 1, lane, phase
        replanned = scripted(t, lambda i: None)
        assert t._cancel_pending(snap, 2, hit, fresh, [0, 1, 2], 2) is hit and replanned == []


def test_lane_route_continues_past_the_road_end():
    road = Road(RoadConfig())
    xy, heading = road.lane_route(0).at(np.array([9990.0, 10500.0]))
    assert xy[:, 0].tolist() == [9990.0, 10500.0] and np.all(xy[:, 1] == road.lane_center(0)) and np.all(heading == 0)


def test_yaw_margin_scope_validated():
    assert Planner(yaw_margin_scope='gap').yaw_margin_scope == 'gap'
    with pytest.raises(ValueError):
        Planner(yaw_margin_scope='gaps')
