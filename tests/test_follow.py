"""Follow mode (Planner.follow, intentions.FOLLOW): the follow check, the leader's train-length gap check, joining the
leader's lane changes or breaking off (the path check), and the follower tracking the leader's path through a lane
change on the nonlinear MPC."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import Planner, Road as RoadConfig, Vehicle
from nohesi.dynamics import KinematicPlant
from nohesi.intentions import ACTIONS, FOLLOW, IntentionManager, encode, encode_overtake
from nohesi.road import Road
from tests.test_nohesi import snapshot

ON = replace(Planner(), follow=True)
L = Vehicle().length


def manager(cfg=ON):
    return IntentionManager(cfg, Road(RoadConfig()), KinematicPlant(Vehicle(), 0.05).params, lane=1, speed=28.0)


def kinds(im, kind):
    return [r.reason for r in im.records if r.kind == kind]


def test_follow_action_and_check():
    assert FOLLOW == encode_overtake(-1, -1) == len(ACTIONS) and not Planner().follow
    im = manager()
    snap = snapshot(0.0, [(20, 0, 28), (0, 0, 28), (-100, -3.6, 28)], [])
    assert im.follow_check(snap, 1) == (0, '')                       # 15.5 m bumper gap at 28 m/s: within 1 s
    assert im.follow_check(snap, 0) == (None, 'no_leader')
    im.intentions[2].lane = 0
    assert im.follow_check(snap, 2) == (None, 'no_leader')           # nobody ahead in lane 0
    far = snapshot(0.0, [(40, 0, 28), (0, 0, 28), (-100, -3.6, 28)], [])
    assert im.follow_check(far, 1) == (None, 'too_far')              # 35.5 m > 28 m
    assert manager(Planner()).follow_check(snap, 1) == (None, 'off')
    keep = [encode(0, 0, 0)]*3
    im.decide(snap, [keep[0], FOLLOW, keep[2]], [None]*3, [0, 1, 2])
    assert im.intentions[1].follow == 0 and kinds(im, 'follow') == ['0']
    im.decide(snap, [keep[0], FOLLOW, keep[2]], [None]*3, [0, 1, 2])  # repeated: continues
    assert im.intentions[1].follow == 0 and len(kinds(im, 'follow')) == 1
    im.decide(snap, keep, [None]*3, [0, 1, 2])                        # any other action ends it
    assert im.intentions[1].follow is None and kinds(im, 'follow_ended') == ['policy']
    off = manager(Planner())
    off.decide(snap, [keep[0], FOLLOW, keep[2]], [None]*3, [0, 1, 2])  # follow off: the slot is an overtake again
    assert off.intentions[1].follow is None and 'follow' not in [r.kind for r in off.records]


def test_followed_leader_takes_a_gap_for_itself():
    # Car 0 changes into lane 2 at 28 m/s (end 84 m on); a 28 m/s rear car there ends 10 m behind: enough for car 0
    # alone (L + margin). By default a followed car needs no more (its follower joins later or breaks off).
    team, rear = [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], [(5, -10, 3.6, 28)]
    im = manager()
    im.intentions[1].follow = 0
    assert im.gap_rule(snapshot(0.0, team, rear), [None]*3, 0, 2, 28.0) == ((None, ('traffic', 5)), '')


def test_followed_leader_needs_room_for_the_train_with_train_room():
    # follow_train_room: not with car 1 following it (one more length and follow_gap * 28 m/s).
    team, rear = [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], [(5, -10, 3.6, 28)]
    im = manager(replace(ON, follow_train_room=True))
    assert im.gap_rule(snapshot(0.0, team, rear), [None]*3, 0, 2, 28.0) == ((None, ('traffic', 5)), '')
    im.intentions[1].follow = 0
    assert im.followers(0) == [1]
    assert im.gap_rule(snapshot(0.0, team, rear), [None]*3, 0, 2, 28.0) == (None, 'gap_too_short')
    room = 10+L+ON.follow_gap*28+0.5
    assert im.gap_rule(snapshot(0.0, team, [(5, -room, 3.6, 28)]), [None]*3, 0, 2, 28.0) == ((None, ('traffic', 5)), '')
    im.intentions[2].follow = 1                                       # the chain counts too
    assert im.followers(0) == [1, 2]
    assert im.gap_rule(snapshot(0.0, team, [(5, -room, 3.6, 28)]), [None]*3, 0, 2, 28.0) == (None, 'gap_too_short')
    # A follower already crossing into the lane is not the leader's rear gap car.
    im.intentions[2].follow = None
    one = im.intentions[1]
    one.origin, one.lane, one.phase = 1, 2, 'pending'
    assert im.gap_rule(snapshot(0.0, team, [(5, -30, 3.6, 28)]), [None]*3, 0, 2, 28.0) == ((None, ('traffic', 5)), '')
    plain = manager()
    two = plain.intentions[1]
    two.origin, two.lane, two.phase = 1, 2, 'pending'
    assert plain.gap_rule(snapshot(0.0, team, [(5, -30, 3.6, 28)]), [None]*3, 0, 2, 28.0)[0] == (None, ('team', 1))


def leader_changing(im, t=0.0):
    lead = im.intentions[0]
    lead.origin, lead.lane, lead.phase, lead.since = 1, 2, 'pending', t
    im.intentions[1].follow = 0


def test_follower_joins_the_leaders_lane_change():
    im = manager()
    leader_changing(im)
    snap = snapshot(0.0, [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], [])
    im.update_follow(snap, [None]*3)
    one = im.intentions[1]
    assert (one.phase, one.origin, one.lane, one.front) == ('pending', 1, 2, ('team', 0))
    assert kinds(im, 'accepted') == ['follow']
    im.intentions[0].lane, im.intentions[0].phase = 1, 'keep'      # the leader backs out: so does the follower
    im.update_follow(snap, [None]*3)
    assert (one.phase, one.lane, one.follow) == ('keep', 1, 0) and kinds(im, 'aborted') == ['follow']


def test_follower_breaks_off_when_the_path_closes():
    im = manager()
    leader_changing(im)
    blocked = [(5, -15, 3.6, 28)]                                   # alongside the follower in lane 2: no room
    for t in (0.0, 0.5, 1.0):
        im.update_follow(snapshot(t, [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], blocked), [None]*3)
        assert im.intentions[1].follow == 0 and im.intentions[1].phase == 'keep'   # waiting within follow_join_time
    im.update_follow(snapshot(1.1, [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], blocked), [None]*3)
    assert im.intentions[1].follow is None and kinds(im, 'follow_ended') == ['join_gap_too_short']
    # A cancelled join (slot closing, a conflict, ...) ends the follow; so does a leader beyond follow_break.
    im = manager()
    leader_changing(im)
    im.update_follow(snapshot(0.0, [(0, 0, 28), (-15, 0, 28), (-300, 0, 28)], []), [None]*3)
    im.cancel(1, 0.2, 'slot_margin')
    assert im.intentions[1].follow is None and kinds(im, 'follow_ended') == ['cancelled_slot_margin']
    im = manager()
    im.intentions[1].follow = 0
    im.update_follow(snapshot(0.0, [(60, 0, 28), (0, 0, 28), (-300, 0, 28)], []), [None]*3)   # 55.5 m > 1.5 s
    assert im.intentions[1].follow is None and kinds(im, 'follow_ended') == ['too_far']


def test_follower_tracks_the_leaders_path_through_a_lane_change():
    """Nonlinear MPC on the dynamic plant, no traffic nearby: car 0 changes into lane 2 at 2 s, cars 1 and 2 follow
    it (a chain) from the start. Both followers drive the leader's lateral path where it drove it and keep the train."""
    from nohesi.config import SCENARIOS, SCORES
    from nohesi.coordination.centralized import CentralizedTeam
    from nohesi.evaluation.runner import EpisodeRun
    s = SCENARIOS['v8']()
    scenario = replace(s, score=SCORES['v10'](), vehicle=replace(s.vehicle, model='dynamic'),
                       traffic=replace(s.traffic, count=1), team_start_headway=0.4,
                       timing=replace(s.timing, duration=10.0))
    planner = replace(Planner(mpc='nonlinear', speed_max=scenario.team_speed_max), follow=True)

    def policy(snap, intents):   # the leader asks for lane 2 from 2 s and holds that command until it is there
        it = intents[0]
        lane = 0 if snap.time < 2.0 else it.lane-it.origin if it.phase != 'keep' else 2-it.lane
        return [encode(lane, 0, 0), FOLLOW, FOLLOW]
    r = EpisodeRun(scenario, 0, lambda world: CentralizedTeam(scenario, world, policy, planner), planner)
    r.world.traffic.s[:] = 1e5                                   # the one traffic car far out of the way
    r.snap = r.world.snapshot
    while not r.done:
        r.step()
    ep = r.episode()
    assert ep.end_reason == 'time' and [it.follow for it in r.policy.intentions.intentions] == [None, 0, 1]
    assert list(Road(scenario.road).lane_of(ep.team_d[-1])) == [2, 2, 2]
    for f in (1, 2):   # the follower's lateral position where it was along the road vs the leader's there (the MPC
        # previews the path, so it moves a little early or late: within a fifth of the lane width)
        ahead = ep.team_s[:, 0] > np.maximum.accumulate(np.r_[-np.inf, ep.team_s[:-1, 0]])
        lead_d = np.interp(ep.team_s[:, f], ep.team_s[ahead, 0], ep.team_d[ahead, 0])
        moving = (ep.time > 2.0) & (ep.team_s[:, f] > ep.team_s[0, 0])
        assert np.abs(ep.team_d[moving, f]-lead_d[moving]).max() < 0.7
    gap = -np.diff(ep.team_s[-1])-L
    assert np.all(gap < 0.5*ep.team[-1, 1:, 3]) and np.all(gap > 0.2*ep.team[-1, 1:, 3])
    assert ep.train_time > 5.0
