"""Score v9: v8 reshaped for team play (10x train points from 75 mph, 10x draft and train passes, a flat penalty for a
lane change without a close pass, 40x speed points)."""
from dataclasses import replace
import pytest
from nohesi.config import SCORES, Score
from nohesi.scoring.events import Overtake
from nohesi.scoring.score import Scorer, close_pass_points
from test_score_v3 import lc, play
from test_score_v3 import close as close_at
from test_score_v4 import DT, step, close


def test_v9_settings():
    v8, v9 = SCORES['v8'](), SCORES['v9']()
    assert (v9.train_rate, v9.train_min_speed, v9.draft_pass_gain, v9.train_pass_gain, v9.lc_base, v9.speed_rate) == \
        (250.0, 33.53, 15.0, 30.0, -300.0, 10.0)
    assert replace(v9, version='v8', train_rate=v8.train_rate, train_min_speed=v8.train_min_speed,
                   draft_pass_gain=v8.draft_pass_gain, train_pass_gain=v8.train_pass_gain, lc_base=v8.lc_base,
                   speed_rate=v8.speed_rate) == v8   # nothing else changed
    with pytest.raises(ValueError):
        Score(lc_pass_single=-1.0)   # only lc_base may be negative


def test_v9_train_from_75_mph():
    v9 = SCORES['v9']()
    train = dict(s=(0., 21., 42.), lanes=(0, 1, 2))   # 16.5 m gaps: a train above 33 m/s
    assert Scorer(v9).update(step(**train, speeds=(34., 34., 34.)))['draft'].tolist() == pytest.approx([250*DT]*3)
    assert Scorer(SCORES['v8']()).update(step(**train, speeds=(34., 34., 34.)))['draft'].sum() == 0   # v8: 80 mph
    assert Scorer(v9).update(step(**train, speeds=(33., 34., 34.)))['draft'].sum() == 0


def test_v9_draft_and_train_passes_10x_solo_pass_unchanged():
    v8, v9 = SCORES['v8'](), SCORES['v9']()
    solo = [Scorer(c).update(step(overtakes=[close()]))['close_pass'][0] for c in (v8, v9)]
    assert solo[1] == pytest.approx(solo[0])
    draft = dict(s=(0., 20., 200.), lanes=(0, 0, 2))   # car 0 drafts car 1 (15.5 m <= 1 s * 28 m/s)
    pair = [Scorer(c).update(step(**draft, overtakes=[close()]))['close_pass'][0] for c in (v8, v9)]
    assert pair[1] == pytest.approx(10*pair[0])
    train = dict(s=(0., 21., 42.), lanes=(0, 1, 2), speeds=(40., 40., 40.))
    trains = [Scorer(c).update(step(**train, overtakes=[close()]))['close_pass'][0] for c in (v8, v9)]
    assert trains[1] == pytest.approx(10*trains[0])


def test_v9_lane_change_without_pass_is_a_flat_penalty():
    v9 = replace(SCORES['v9'](), lc_chain_window=4.0, lc_traffic_distance=20.0)
    s, _ = play(v9, 7.2, [], [lc(1.0, 2.0)])
    assert [(a['kind'], a['points']) for a in s.maneuvers] == [('base', -300.0)]
    passes = [Overtake(0.2, 0, 7, 0.5, 28.0), Overtake(0.4, 0, 8, 0.5, 28.0)]   # before the change: not linked
    s, combos = play(v9, 5.1, passes, [lc(0.6, 1.0)])   # resolves at 5.1 s with the combo still 2x: still flat
    assert combos[-1] == 2 and [(a['kind'], a['points']) for a in s.maneuvers] == [('base', -300.0)]
    s, _ = play(v9, 7.2, [close_at(3.0)], [lc(1.0, 2.0)])   # with a close pass: rewarded as in v8
    assert s.maneuvers[0]['kind'] == 'pass_double' and s.maneuvers[0]['points'] > 0


def test_v9_speed_points():
    inc = Scorer(SCORES['v9']()).update(step(speeds=(30., 25., 20.)))
    assert inc['speed'].tolist() == pytest.approx([10*5*DT, 0, 0])
    assert close_pass_points(0.5, 28.0, SCORES['v9']()) == close_pass_points(0.5, 28.0, SCORES['v8']())
