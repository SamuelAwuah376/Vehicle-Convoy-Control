import numpy as np
import pytest
from dataclasses import replace
from nohesi.config import Planner, Timing, scenario_v1
from nohesi.coordination.centralized import CentralizedTeam
from nohesi.intentions import ACTIONS, IntentionManager, encode
from nohesi.planning.base import Target
from tests.test_nohesi import snapshot
from tests.test_nohesi_m2 import CFG, P, ROAD, SC, drive

MATES = [(200, -3.6, 28), (0, -3.6, 28), (-300, -3.6, 28)]   # car 1 at s=0 in lane 0; teammates far away
FAR = [(5, -500, 0, 25)]


def manager(lane=0, cfg=CFG):
    return IntentionManager(cfg, ROAD, P, lane=lane, speed=28.0)


def act(cmd):
    return np.array([encode(0, 0), encode(cmd, 0), encode(0, 0)])


def test_actions_and_encode_round_trip():
    assert len(ACTIONS) == 45 and all(ACTIONS[encode(*a)] == a for a in ACTIONS)
    assert encode(-2, -1, -1) == 0 and encode(2, 1, 1) == 44 and encode(0, 0) == 22


def test_two_lanes_from_middle_lane_has_no_lane():
    im = manager(lane=1)
    snap = snapshot(0.0, [(200, 0, 28), (0, 0, 28), (-300, 0, 28)], FAR)
    for cmd, lane in ((2, 3), (-2, -1)):
        im.decide(snap, act(cmd), [None]*3, order=[0, 1, 2])
        r = im.records[-1]
        assert (r.kind, r.reason, r.lane) == ('rejected', 'no_lane', lane) and im.intentions[1].phase == 'keep'


def test_empty_road_accepted_to_far_lane():
    im = manager()
    im.decide(snapshot(0.0, MATES, FAR), act(2), [None]*3, order=[0, 1, 2])
    it = im.intentions[1]
    assert (it.phase, it.origin, it.lane) == ('pending', 0, 2) and im.records[-1].kind == 'accepted'
    assert im.target(1).d == pytest.approx(3.6)


def test_middle_lane_corridor():
    im = manager()
    # Alongside throughout; a faster car closing from 30 m behind overlaps at 2.1-2.9 s (inside the window).
    for car in ((7, 0, 0, 28), (7, -30, 0, 40)):
        assert im.gap_rule(snapshot(0.0, MATES, [car]), [None]*3, 1, 2, 28.0) == (None, 'middle_blocked')
    # 15 m behind at equal speed, or closing from 50 m (overlap after 3.8 s): clear.
    for car in ((7, -15, 0, 28), (7, -50, 0, 40)):
        assert im.gap_rule(snapshot(0.0, MATES, [car]), [None]*3, 1, 2, 28.0) == ((None, None), '')
    # A teammate in the middle lane alongside blocks too.
    mates = [(200, -3.6, 28), (0, -3.6, 28), (2, 0, 28)]
    assert im.gap_rule(snapshot(0.0, mates, FAR), [None]*3, 1, 2, 28.0) == (None, 'middle_blocked')


def test_destination_gap_uses_double_lookahead():
    # Lane 2 car at -50, 40 m/s: 14 m behind the ego's end after 3 s, but 2 m behind after 4 s.
    snap = snapshot(0.0, MATES, [(7, -50, 3.6, 40)])
    assert manager().gap_rule(snap, [None]*3, 1, 2, 28.0) == (None, 'gap_too_short')
    im = manager(cfg=replace(CFG, double_gap_lookahead=CFG.gap_lookahead))
    assert im.gap_rule(snap, [None]*3, 1, 2, 28.0) == ((None, ('traffic', 7)), '')


def test_teammate_crossing_middle_from_other_side_rejected():
    im = manager()
    other = im.intentions[0]
    other.origin, other.lane, other.phase = 2, 1, 'pending'    # teammate 0 moving right into the middle lane
    snap = snapshot(0.0, [(10, 3.6, 28), (0, -3.6, 28), (-300, -3.6, 28)], FAR)
    assert im.gap_rule(snap, [None]*3, 1, 2, 28.0) == (None, 'teammate_merging')
    other.origin, other.lane = 1, 2                             # same direction ahead in lane 2: an ordinary gap car
    assert im.gap_rule(snap, [None]*3, 1, 2, 28.0)[1] != 'teammate_merging'


def test_commit_at_first_line_complete_at_destination():
    im = manager()
    snap = lambda t, d: snapshot(t, [(200, -3.6, 28), (3, d, 28), (-300, -3.6, 28)], FAR)
    im.decide(snap(0.0, -3.6), act(2), [None]*3, order=[0, 1, 2])
    im.decide(snap(0.0, -3.6), act(1), [None]*3, order=[0, 1, 2])          # different command aborts while pending
    assert im.intentions[1].phase == 'keep' and im.records[-1].kind == 'aborted'
    im.decide(snap(0.0, -3.6), act(2), [None]*3, order=[0, 1, 2])
    it = im.intentions[1]
    im.update_phases(snap(0.1, -3.0))
    assert it.phase == 'pending'
    im.update_phases(snap(0.2, -2.6))                                     # footprint edge at the 0/1 line (-1.8)
    assert it.phase == 'committed'
    im.decide(snap(0.2, -2.6), act(2), [None]*3, order=[0, 1, 2])         # same change: not masked
    im.decide(snap(0.2, -2.6), act(1), [None]*3, order=[0, 1, 2])
    assert it.phase == 'committed' and it.lane == 2 and im.records[-1].kind == 'masked'
    im.update_phases(snap(1.0, 0.0))
    assert it.phase == 'committed'                                        # middle lane centre is not the target
    im.update_phases(snap(2.0, 3.5))
    assert (it.phase, it.lane) == ('keep', 2)
    assert [r.kind for r in im.records if r.vehicle == 1] == ['accepted', 'aborted', 'accepted', 'committed',
                                                               'masked', 'completed']


def test_stale_relock_uses_double_lookahead():
    im = manager()
    it = im.intentions[1]
    it.origin, it.lane, it.phase, it.rear = 0, 2, 'pending', ('team', 2)
    im.update_phases(snapshot(0.0, [(200, -3.6, 28), (0, -3.6, 28), (10, 3.6, 30)], FAR))
    assert (it.phase, it.front, it.rear) == ('pending', ('team', 2), None) and im.records[-1].kind == 'relocked'
    assert im.records[-1].gap_margin == pytest.approx(10+30*4-28*4-P.length)   # 4 s, not 3 s


def test_mpc_two_lane_change_within_limits():
    out = drive(Target(d=ROAD.lane_center(2), speed=28.0), state=(0., ROAD.lane_center(0), 0., 28.), seconds=7)
    st, t = out['state'], np.arange(1, len(out['state'])+1)*SC.timing.dt
    assert np.all(np.abs(st[t > 5, 1]-ROAD.lane_center(2)) < 0.1)
    lateral = st[:, 3]**2*np.tan(st[:, 4])/P.wheelbase
    assert np.max(np.abs(lateral)) < 1.05*P.lateral_accel_max
    assert np.max(np.abs(st[:, 1])) < ROAD.width/2-P.width/2


class DoubleOnce:
    """Vehicle `who` asks once (repeating while rejected or in progress) for two lanes left; everyone else holds."""
    name = 'double_once'

    def __init__(self, who=0):
        self.who, self.done = who, False

    def __call__(self, snap, intentions):
        it = intentions[self.who]
        self.done |= it.phase == 'keep' and it.lane == 2
        actions = np.full(3, encode(0, 0))
        actions[self.who] = encode(0 if self.done else 2, 0)
        return actions


def test_run_two_lane_change_completes():
    sc = replace(scenario_v1(), team_lane=0, timing=Timing(duration=30.0))
    ep = run_episode(sc, 0)
    assert ep.collisions == [] and ep.end_reason == 'time'
    kinds = [(r.kind, r.lane) for r in ep.intention_records if r.vehicle == 0 and r.kind in ('accepted', 'completed')]
    assert ('accepted', 2) in kinds and kinds[kinds.index(('accepted', 2))+1] == ('completed', 2)
    assert abs(ep.team_d[-1, 0]-ROAD.lane_center(2)) < 0.5


def run_episode(sc, seed):
    from nohesi.evaluation.runner import run
    return run(sc, seed, lambda world: CentralizedTeam(sc, world, DoubleOnce(), Planner()))
