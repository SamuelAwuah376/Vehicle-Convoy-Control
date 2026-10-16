import pytest
from nohesi.config import Planner, Road as RoadConfig, Vehicle
from nohesi.dynamics import KinematicPlant
from nohesi.intentions import IntentionManager
from nohesi.road import Road
from tests.test_nohesi import snapshot

CFG = Planner()


def manager():
    return IntentionManager(CFG, Road(RoadConfig()), KinematicPlant(Vehicle(), 0.05).params, lane=1, speed=28.0)


# Car 1 at s=0, 28 m/s, holding 28: predicted end 84. A rear car at s_r closing at 31 m/s is at s_r+93 after 3 s,
# leaving -13.5-s_r m of bumper room; the rule needs margin_long + 3 m/s * rear_closing_time.
FAR = [(200, 0, 28), (0, 0, 28), (-300, 0, 28)]
NEED = CFG.margin_long+3*CFG.rear_closing_time


def rear_at(room):
    return -13.5-room


def test_fast_rear_traffic_rejected_slow_accepted():
    im = manager()
    assert im.gap_rule(snapshot(0.0, FAR, [(5, rear_at(NEED-0.2), 3.6, 31)]), [None]*3, 1, 2, 28.0) == (None, 'rear_closing')
    assert im.gap_rule(snapshot(0.0, FAR, [(5, rear_at(NEED+0.2), 3.6, 31)]), [None]*3, 1, 2, 28.0) == ((None, ('traffic', 5)), '')
    slow = snapshot(0.0, FAR, [(5, rear_at(NEED-0.2), 3.6, 27)])          # not closing: plain margin only
    assert im.gap_rule(slow, [None]*3, 1, 2, 28.0) == ((None, ('traffic', 5)), '')
    assert im.last_gap_margin == pytest.approx(84-(rear_at(NEED-0.2)+81)-4.5)


def test_fast_rear_teammate_rejected_slow_accepted():
    im = manager()
    fast = snapshot(0.0, [(200, 0, 28), (0, 0, 28), (rear_at(NEED-0.2), 3.6, 31)], [(5, -500, 0, 25)])
    assert im.gap_rule(fast, [None]*3, 1, 2, 28.0) == (None, 'rear_closing')
    slow = snapshot(0.0, [(200, 0, 28), (0, 0, 28), (rear_at(NEED-0.2), 3.6, 27)], [(5, -500, 0, 25)])
    assert im.gap_rule(slow, [None]*3, 1, 2, 28.0) == ((None, ('team', 2)), '')


def test_pass_and_cut_in_allowed_when_lookahead_gap_holds():
    # A slower car beside the ego now ends 17.5 m behind the predicted slot: locking in front of it is allowed
    # (the MPC stays beside it until the reference is ahead). A faster car beside now ends 6.5 m ahead of the
    # slot and becomes the front gap car instead: merge in behind it.
    im = manager()
    assert im.gap_rule(snapshot(0.0, FAR, [(5, 2, 3.6, 20)]), [None]*3, 1, 2, 28.0) == ((None, ('traffic', 5)), '')
    assert im.gap_rule(snapshot(0.0, FAR, [(5, 2, 3.6, 31)]), [None]*3, 1, 2, 28.0) == ((('traffic', 5), None), '')


def test_reachable_end_shorter_than_average_speed():
    im = manager()
    run, v = im.reach(20.0, 35.0, 3.0)
    assert run < 0.5*(20+35)*3 and v < 35.0
    assert im.reach(28.0, 28.0, 3.0) == pytest.approx((84.0, 28.0))
    assert im.reach(28.0, 22.0, 3.0)[0] == pytest.approx(22*3+6*6/(2*6), abs=0.5)   # brake 1 s then hold


def locked(im, rear):
    it = im.intentions[1]
    it.origin, it.lane, it.phase, it.rear = 1, 2, 'pending', rear
    return it


def test_stale_rear_relocked():
    im = manager()
    it = locked(im, ('team', 2))
    im.update_phases(snapshot(0.0, [(200, 0, 28), (0, 0.3, 28), (10, 3.6, 28)], [(5, -500, 0, 25)]))
    assert it.phase == 'pending' and (it.front, it.rear) == (('team', 2), None)
    assert im.records[-1].kind == 'relocked'


def test_stale_rear_pending_cancelled():
    im = manager()
    it = locked(im, ('team', 2))
    im.update_phases(snapshot(0.0, [(200, 0, 28), (0, 0.3, 28), (10, 3.6, 28)], [(5, 1, 3.6, 28)]))
    assert it.phase == 'keep' and it.lane == 1
    assert (im.records[-1].kind, im.records[-1].reason) == ('cancelled', 'stale_gap')


def test_stale_rear_committed_key_cleared():
    im = manager()
    it = locked(im, ('team', 2))
    it.front = ('traffic', 7)
    im.update_phases(snapshot(0.0, [(200, 0, 28), (0, 1.5, 28), (10, 3.6, 28)], [(5, 1, 3.6, 28), (7, 60, 3.6, 28)]))
    assert it.phase == 'committed' and it.lane == 2
    assert (it.front, it.rear) == (('traffic', 7), None)


def test_recycled_gap_car_unlocked():
    im = manager()
    it = locked(im, ('traffic', 99))
    im.update_phases(snapshot(0.0, FAR, [(5, -500, 0, 25)]))
    assert it.phase == 'pending' and it.rear is None
