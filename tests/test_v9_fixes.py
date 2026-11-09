"""Fixes for batch 10's crashes on scenario v9, both off by default: a lane-keeping car leaves the teammates behind it
in its lane out of its MPC obstacles (Planner.ignore_rear_mates), and a committed follower whose leader turns back
finishes its own lane change (Planner.follow_finish_committed)."""
from dataclasses import replace
from nohesi.config import SCENARIOS, Planner
from nohesi.evaluation.runner import EpisodeRun
from tests.test_follow import ON, kinds, leader_changing, manager
from tests.test_nohesi import snapshot


def mates(planner, i, phase='keep'):
    """Teammate keys in car i's MPC obstacles at the start of a v8 episode (the team in one lane, car 0 in front)."""
    r = EpisodeRun(SCENARIOS['v8'](), 0, 'overtaker', planner)
    team = r.policy
    team.intentions.intentions[i].phase = phase
    return [k for k in team._obstacles(r.snap, i, {}).keys if k[0] == 'team']


def test_lane_keeping_car_leaves_teammates_behind_it_out():
    on = replace(Planner(), ignore_rear_mates=True)
    assert mates(Planner(), 0) == [('team', 1), ('team', 2)]        # off: every teammate
    assert mates(on, 0) == []                                       # the front car: both are behind it in its lane
    assert mates(on, 1) == [('team', 0)]                            # the ones ahead stay
    assert mates(on, 2) == [('team', 0), ('team', 1)]
    assert mates(on, 0, 'pending') == [('team', 1), ('team', 2)]    # changing lanes: all of them


def test_committed_follower_finishes_its_change_when_the_leader_turns_back():
    for cfg, expect in ((ON, ('keep', 1, 0)), (replace(ON, follow_finish_committed=True), ('committed', 2, None))):
        im = manager(cfg)
        leader_changing(im)
        im.update_follow(snapshot(0.0, [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], []), [None]*3)
        one = im.intentions[1]
        one.phase = 'committed'
        im.intentions[0].lane, im.intentions[0].phase = 1, 'keep'      # the leader turns back
        im.update_follow(snapshot(0.5, [(0, 0, 28), (-15, 1.0, 28), (-300, 0, 28)], []), [None]*3)   # not across yet
        assert (one.phase, one.lane, one.follow) == expect, cfg.follow_finish_committed
    assert kinds(im, 'follow_ended') == ['leader_turned_back'] and kinds(im, 'aborted') == []
    assert (one.origin, one.front) == (1, ('team', 0))             # its own change and locked gap go on
    # Already across the line: as before, it keeps that lane and still follows.
    im = manager(replace(ON, follow_finish_committed=True))
    leader_changing(im)
    im.update_follow(snapshot(0.0, [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], []), [None]*3)
    one = im.intentions[1]
    one.phase = 'committed'
    im.intentions[0].lane, im.intentions[0].phase = 1, 'keep'
    im.update_follow(snapshot(0.5, [(0, 0, 28), (-15, 2.5, 28), (-300, 0, 28)], []), [None]*3)
    assert (one.phase, one.lane, one.follow) == ('keep', 2, 0)
