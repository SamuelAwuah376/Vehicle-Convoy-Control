import numpy as np
import pytest
from nohesi.config import Scenario, Timing, score_v2, score_v3
from nohesi.evaluation.metrics import metrics
from nohesi.evaluation.runner import run
from nohesi.scoring.events import LaneChange, Measurements, Overtake
from nohesi.scoring.score import Scorer, TERMS
from nohesi.world import Collision

DT = 0.1



def V3():
    """v3 with the 4 s window and 20 m traffic distance these timing tests were written for (logic, not defaults)."""
    from dataclasses import replace
    return replace(score_v3(), lc_chain_window=4.0, lc_traffic_distance=20.0)

def lc(start, end, size=2, vehicle=0, ds=10.0, speed=28.0, from_lane=0):
    return LaneChange(end, vehicle, start, from_lane, from_lane+size, size, ds, speed)


def close(t, vehicle=0):
    return Overtake(t, vehicle, 7, 0.5, 28.0)


def play(cfg, until, passes=(), changes=(), collisions=()):
    """Step a Scorer at DT to `until`; events fire at the step whose time matches their time."""
    scorer, at = Scorer(cfg), lambda t, k: round(t/DT) == k
    combos = []
    for k in range(1, int(round(until/DT))+1):
        scorer.update(Measurements(DT, np.full(3, 28.), np.full(3, 100.), np.full(3, 5.), np.full(3, np.inf),
                                   tuple(o for o in passes if at(o.time, k)),
                                   tuple(Collision(0, 'traffic', 1) for t in collisions if at(t, k)),
                                   tuple(c for c in changes if at(c.time, k))))
        combos.append(scorer.combo)
    return scorer, combos


def kinds(scorer):
    return [(a['kind'], a['points']) for a in scorer.maneuvers]


def test_chain_double_and_single():
    for size, base in ((2, 300.), (1, 150.)):
        s, _ = play(V3(), 5.0, [close(1.5)], [lc(1.0, 2.0, size), lc(3.0, 4.0, 1, from_lane=size)])
        kind = 'chain_double' if size == 2 else 'chain_single'
        assert kinds(s) == [(kind, pytest.approx(base*1.5))]   # one pass: multiplier 1.5
        assert s.maneuvers[0]['time'] == pytest.approx(4.0) and s.maneuvers[0]['follow_up_time'] == 4.0


def test_pass_only_after_window():
    s, _ = play(V3(), 7.0, [close(3.0)], [lc(1.0, 2.0)])
    assert not s.maneuvers
    s, _ = play(V3(), 7.2, [close(3.0)], [lc(1.0, 2.0)])
    assert kinds(s) == [('pass_double', pytest.approx(90*1.5))]
    s, _ = play(V3(), 7.2, [close(3.0)], [lc(1.0, 2.0, 1)])
    assert kinds(s) == [('pass_single', pytest.approx(60*1.5))]


def test_no_pass_base_any_size():
    for size in (1, 2):
        s, _ = play(V3(), 6.0, changes=[lc(1.0, 2.0, size)])
        assert not s.maneuvers
        s, _ = play(V3(), 6.2, changes=[lc(1.0, 2.0, size)])
        assert kinds(s) == [('base', 25.0)]
        assert s.total['maneuver'].tolist() == [25., 0, 0]


def test_ineligible_never_scores():
    for change in (lc(1.0, 2.0, ds=25.0), lc(1.0, 2.0, speed=24.0), lc(1.0, 2.0, speed=25.0)):
        s, _ = play(V3(), 10.0, [close(1.5)], [change, lc(3.0, 4.0, 1, ds=30.0)])
        assert not s.maneuvers and s.total['maneuver'].sum() == 0


def test_ineligible_change_is_not_a_follow_up():
    s, _ = play(V3(), 7.0, [close(1.5)], [lc(1.0, 2.0), lc(3.0, 4.0, 1, speed=20.0)])
    assert kinds(s) == [('pass_double', pytest.approx(135.))]


def test_pass_before_start_does_not_link():
    s, _ = play(V3(), 6.5, [close(0.5)], [lc(1.0, 2.0)])
    assert kinds(s) == [('base', 25.0)]


def test_late_follow_up_gets_pass_award_and_starts_own_maneuver():
    s, _ = play(V3(), 12.0, [close(2.5)], [lc(1.0, 2.0), lc(6.0, 7.0, 1, from_lane=2)])
    assert [a['kind'] for a in s.maneuvers] == ['pass_double', 'base']
    assert s.maneuvers[0]['time'] == pytest.approx(6.6) and s.maneuvers[1]['lane_change_time'] == 7.0


def test_follow_up_continues_chain():
    passes = [close(1.5), close(4.5)]
    s, _ = play(V3(), 7.0, passes, [lc(1.0, 2.0), lc(3.0, 4.0, 1, from_lane=2), lc(5.0, 6.0, 2, from_lane=3)])
    assert [a['kind'] for a in s.maneuvers] == ['chain_double', 'chain_single']
    assert s.maneuvers[1]['points'] == pytest.approx(150*2.0)


def test_per_car_and_multiplier_without_combo_change():
    s, combos = play(V3(), 5.0, [close(1.5, 1), close(1.6, 0)],
                     [lc(1.0, 2.0, vehicle=0), lc(3.0, 4.0, 1, vehicle=0, from_lane=2), lc(1.0, 2.0, vehicle=2)])
    assert kinds(s) == [('chain_double', pytest.approx(300*2.0))]   # car 2 has no pass of its own
    assert combos[-1] == 2 and s.max_combo == 2


def test_collision_drops_pending():
    s, _ = play(V3(), 8.0, [close(1.5)], [lc(1.0, 2.0)], collisions=[3.0])
    assert not s.maneuvers


def test_v2_unchanged_and_maneuver_zero():
    events = dict(passes=[close(1.5), close(4.5, 1)], changes=[lc(1.0, 2.0), lc(3.0, 4.0, 1, from_lane=2)])
    v2, _ = play(score_v2(), 12.0, **events)
    plain, _ = play(score_v2(), 12.0, passes=events['passes'])
    v3, _ = play(V3(), 12.0, **events)
    assert v2.total['maneuver'].sum() == 0 and all(a['points'] == 0 for a in v2.maneuvers)
    for term in TERMS:
        np.testing.assert_array_equal(v2.total[term], plain.total[term])
        if term != 'maneuver':
            np.testing.assert_array_equal(v3.total[term], v2.total[term])
    assert v3.total['maneuver'].sum() > 0


def test_run_records_maneuvers():
    ep = run(Scenario(timing=Timing(duration=5.0)), 0, 'lane_keeping', score=V3())
    m = metrics(ep)
    assert 'maneuver' in ep.score and m['maneuvers'] == {'count': {}, 'points': 0.0} and ep.maneuvers == []


def test_v3_defaults_and_traffic_distance():
    cfg = score_v3()
    assert (cfg.lc_chain_window, cfg.lc_traffic_distance, cfg.lc_min_speed) == (3.0, 10.0, 25.0)
    for ds, scores in ((10.0, True), (10.5, False)):
        s, _ = play(cfg, 8.0, [], [lc(1.0, 2.0, ds=ds)])
        assert bool(s.maneuvers) == scores
