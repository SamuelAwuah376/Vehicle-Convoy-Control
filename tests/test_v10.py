"""Score v10: v9 with a speed ramp for the draft train (40 % at 60 mph, 50 % at 65, 70 % at 70, 100 % from 80 mph)."""
from dataclasses import replace
import pytest
from nohesi.config import SCORES, Score
from nohesi.scoring.score import Scorer
from test_score_v4 import DT, step, close

TRAIN = dict(s=(0., 14., 28.), lanes=(0, 1, 2))   # 9.5 m bumper gaps: a train from 19 m/s (0.5 s headway)


def test_v10_settings():
    v9, v10 = SCORES['v9'](), SCORES['v10']()
    assert replace(v10, version='v9', train_min_speed=v9.train_min_speed, train_ramp=()) == v9   # nothing else changed
    with pytest.raises(ValueError):
        Score(train_ramp=((30., 0.5), (25., 1.0)))   # speeds must increase
    with pytest.raises(ValueError):
        Score(train_ramp=((25., 0.), (30., 1.0)))    # shares in (0, 1]


@pytest.mark.parametrize('mph, share', [(59., 0.), (61., 0.4+0.1*1/5), (65., 0.5), (70., 0.7), (75., 0.85), (80., 1.0),
                                        (90., 1.0)])
def test_v10_train_points_ramp(mph, share):
    v = mph*0.44704
    inc = Scorer(SCORES['v10']()).update(step(**TRAIN, speeds=(v+1, v, v+2)))   # the slowest car sets the share
    assert inc['draft'].tolist() == pytest.approx([250*share*DT]*3, rel=2e-3, abs=1e-9)


def test_v10_train_pass_ramp():
    points = lambda cfg, v: Scorer(cfg).update(step(**TRAIN, speeds=(v, v, v), overtakes=[close()]))['close_pass'][0]
    v9, v10 = SCORES['v9'](), SCORES['v10']()
    assert points(v10, 31.29)/points(v9, 31.29) == pytest.approx(1+29*0.7)   # v9: no train below 75 mph
    assert points(v10, 40.0) == pytest.approx(points(v9, 40.0))              # full above 80 mph
    drafting = dict(s=(0., 14., 200.), lanes=(0, 0, 2))   # car 0 drafts car 1, no train
    assert Scorer(v10).update(step(**drafting, speeds=(27., 27., 27.), overtakes=[close()]))['close_pass'][0] == \
        pytest.approx(Scorer(v9).update(step(**drafting, speeds=(27., 27., 27.), overtakes=[close()]))['close_pass'][0])


def test_v10_slow_train_pass_keeps_the_draft_pass_bonus():
    line = dict(s=(0., 14., 28.), lanes=(0, 0, 0), speeds=(27., 27., 27.))   # a train at 60.4 mph, car 0 drafting car 1
    solo = Scorer(SCORES['v10']()).update(step(overtakes=[close()]))['close_pass'][0]
    gain = Scorer(SCORES['v10']()).update(step(**line, overtakes=[close()]))['close_pass'][0]/solo
    assert gain == pytest.approx(15.0)   # 1 + 29*0.4 = 12.6 < 15
