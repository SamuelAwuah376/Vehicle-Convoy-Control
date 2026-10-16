from tests.test_intentions_gap import CFG, manager
from tests.test_nohesi import snapshot

DT = 0.05
FAR = [(5, -500, 0, 25)]


def changing(im, origin, lane, phase='committed'):
    it = im.intentions[1]
    it.origin, it.lane, it.phase = origin, lane, phase
    return it


def drive(im, d_of_t, until, traffic=FAR):
    """Car 1 at s=28t, d=d_of_t(t); teammates far. Returns the step time of the first record of car 1 kind
    'cancelled' (or None)."""
    for k in range(int(round(until/DT))+1):
        t = k*DT
        im.update_phases(snapshot(t, [(300+28*t, 0, 28), (28*t, d_of_t(t), 28), (-300+28*t, 0, 28)], traffic))
        if im.intentions[1].phase == 'keep':
            return t
    return None


def last(im):
    r = im.records[-1]
    return r.kind, r.reason


def test_stuck_committed_change_cancelled_stalled_into_current_lane():
    im = manager()
    it = changing(im, 0, 2)
    t = drive(im, lambda t: -2.05, 5.0)               # pinned on the lane 0/1 line (lane_of = 0)
    assert t is not None and abs(t-CFG.commit_stall_time) <= DT+1e-9
    assert last(im) == ('cancelled', 'stalled') and it.lane == 0 and it.phase == 'keep'


def test_steady_progress_not_cancelled():
    im = manager()
    it = changing(im, 0, 2)
    assert drive(im, lambda t: -2.05+0.25*t, 5.0) is None
    assert it.phase == 'committed' and not any(r.kind == 'cancelled' for r in im.records)


def test_committed_too_long_timeout_into_middle_lane():
    im = manager()
    it = changing(im, 0, 2)
    t = drive(im, lambda t: -3.0+0.5*t, 8.0)          # progressing, but at t = 6 still in the middle lane
    assert t is not None and abs(t-CFG.commit_max_time) <= DT+1e-9
    assert last(im) == ('cancelled', 'timeout') and it.lane == 1


def test_pending_too_long_timeout_back_to_origin():
    im = manager()
    it = changing(im, 1, 2, 'pending')
    t = drive(im, lambda t: 0.0, 6.0)                 # never reaches the lane line
    assert t is not None and abs(t-CFG.pending_max_time) <= DT+1e-9
    assert last(im) == ('cancelled', 'timeout') and it.lane == 1


def test_rear_car_ahead_but_overtaken_at_lookahead_not_relocked_every_step():
    # Bug geometry (v2 seed 2): ego committed 0 -> 2 at 18.7 m/s targeting 32; the locked rear car (20.7 m/s) is
    # 4.8 m ahead now, behind the predicted end at the 4 s lookahead. It used to be relocked onto the same gap
    # every step.
    im = manager()
    it = changing(im, 0, 2)
    it.front, it.rear, it.speed = ('traffic', 28), ('traffic', 20), 32.0
    end = im.reach(18.7, 32.0, CFG.double_gap_lookahead)[0]
    assert 4.8+20.72*CFG.double_gap_lookahead < end < 59.6+20.81*CFG.double_gap_lookahead
    for k in range(30):
        t = k*DT
        im.update_phases(snapshot(t, [(-1+18.7*t, 0, 18.7), (18.7*t, -2.05, 18.7), (-300, 0, 18.7)],
                                  [(20, 4.8+20.72*t, 3.6, 20.72), (28, 59.6+20.81*t, 3.6, 20.81)]))
    assert sum(r.kind == 'relocked' for r in im.records) <= 1
    assert (it.front, it.rear) == (('traffic', 28), ('traffic', 20))


def test_relock_to_identical_gap_not_recorded():
    # Rear teammate 10 m ahead now and at the lookahead: stale, relocked once as the front car, then left alone.
    im = manager()
    it = changing(im, 1, 2, 'pending')
    it.rear = ('team', 2)
    for k in range(10):
        t = k*DT
        im.update_phases(snapshot(t, [(200+28*t, 0, 28), (28*t, 0.3, 28), (10+28*t, 3.6, 28)], FAR))
    assert [r.kind for r in im.records] == ['relocked'] and (it.front, it.rear) == (('team', 2), None)


def test_forced_cancel_blocks_retry_for_cooldown():
    from nohesi.config import Planner
    im = manager()
    snap = snapshot(0.0, [(200, 0, 28), (0, 0, 28), (-300, 0, 28)], [(5, -500, 0, 25)])
    assert im.gap_rule(snap, [None]*3, 1, 2, 28.0)[0] is not None
    it = im.intentions[1]
    it.origin, it.lane, it.phase = 1, 2, 'committed'
    im.cancel(1, 0.0, 'stalled', lane=1)
    assert im.records[-1].lane == 1                                   # record carries the kept lane
    assert im.gap_rule(snap, [None]*3, 1, 2, 28.0) == (None, 'cooldown')
    later = snapshot(Planner().retry_cooldown+0.1, [(200, 0, 28), (0, 0, 28), (-300, 0, 28)], [(5, -500, 0, 25)])
    assert im.gap_rule(later, [None]*3, 1, 2, 28.0)[0] is not None
