import numpy as np
import pytest
from nohesi.config import Scenario, Score, Timing, score_v1
from nohesi.evaluation.metrics import metrics
from nohesi.evaluation.runner import run
from nohesi.scoring.events import Measurements, Overtake
from nohesi.scoring.score import Scorer, close_pass_points
from nohesi.world import Collision


def step(dt=0.1, speeds=(25., 25., 25.), distance=(30., 30., 30.), overtakes=(), collisions=()):
    return Measurements(dt, np.array(speeds), np.array(distance), np.full(3, 5.), np.full(3, np.inf),
                        tuple(overtakes), tuple(collisions))


def close(t=0.0, vehicle=0):
    return Overtake(t, vehicle, 7, 0.5, 28.0)


def test_combo_multiplier_sequence_and_cap():
    cfg, scorer = score_v1(), Scorer(score_v1())
    base = close_pass_points(0.5, 28.0, cfg)
    for k in range(12):
        scorer.update(step(overtakes=[close()]))
        assert scorer.combo == k+1
    assert [p/base for _, p in scorer.awarded] == pytest.approx([min(5., 1+0.5*k) for k in range(12)])
    assert scorer.multiplier == 5.0


def test_two_passes_in_one_step_chain():
    scorer = Scorer(score_v1())
    inc = scorer.update(step(overtakes=[close(vehicle=0), close(vehicle=1)]))
    base = close_pass_points(0.5, 28.0, score_v1())
    assert inc['close_pass'].tolist() == pytest.approx([base, 1.5*base, 0])


def test_proximity_multiplied_speed_not():
    cfg = score_v1()
    plain, combo = Scorer(cfg), Scorer(cfg)
    combo.update(step(overtakes=[close()])); combo.update(step(overtakes=[close()]))
    assert combo.multiplier == 2.0
    m = step(speeds=(30., 30., 30.), distance=(10., 30., 30.))
    a, b = plain.update(m), combo.update(m)
    assert b['proximity'].tolist() == pytest.approx((2*a['proximity']).tolist())
    assert a['proximity'][0] > 0
    assert b['speed'].tolist() == pytest.approx(a['speed'].tolist())
    assert a['speed'].tolist() == pytest.approx([0.25*(30-25)*0.1]*3)


def test_combo_resets_after_window():
    scorer = Scorer(score_v1())
    scorer.update(step(overtakes=[close()]))
    for _ in range(50):   # exactly 5 s later: still alive
        scorer.update(step())
    assert scorer.combo == 1
    scorer.update(step())
    assert scorer.combo == 0 and scorer.multiplier == 1.0


def test_pass_restarts_window():
    scorer = Scorer(score_v1())
    for _ in range(3):
        scorer.update(step(overtakes=[close()]))
        for _ in range(40):
            scorer.update(step())
    assert scorer.combo == 3


def test_combo_resets_on_collision():
    scorer = Scorer(score_v1())
    scorer.update(step(overtakes=[close()])); scorer.update(step(overtakes=[close()]))
    inc = scorer.update(step(collisions=[Collision(1, 'traffic', 4)]))
    assert scorer.combo == 0 and scorer.multiplier == 1.0
    assert inc['collision'].tolist() == [0, -500, 0]


def test_unscored_pass_does_not_count():
    scorer = Scorer(score_v1())
    scorer.update(step(overtakes=[Overtake(0., 0, 3, 2.0, 28.)]))
    assert scorer.combo == 0 and not scorer.awarded


def test_v0_has_no_combo():
    cfg, scorer = Score(), Scorer(Score())
    base = close_pass_points(0.5, 28.0, cfg)
    for _ in range(4):
        inc = scorer.update(step(speeds=(25., 25., 15.), distance=(10., 30., 30.), overtakes=[close()]))
        assert inc['close_pass'][0] == base
        assert inc['proximity'].tolist() == [0.5, 0.5, 0]
        assert inc['speed'].tolist() == pytest.approx([0.5, 0.5, 0])
    assert scorer.multiplier == 1.0
    assert [p for _, p in scorer.awarded] == [base]*4


def test_proximity_distance_v1_vs_v0():
    m = step(speeds=(30., 30., 30.), distance=(15., 30., 30.))
    assert Scorer(Score()).update(m)['proximity'].sum() == 0
    assert Scorer(score_v1()).update(m)['proximity'].tolist() == pytest.approx([0.5, 0.5, 0])


def test_run_with_score_v1():
    scenario = Scenario(timing=Timing(duration=10.0))
    ep = run(scenario, 0, 'lane_keeping', score=score_v1())
    m = metrics(ep)
    assert m['score_version'] == 'v1' and ep.scenario.score.version == 'v1'
    assert ep.combo_multiplier.shape == (len(ep.time)-1,)
    assert set(m['combo']) == {'max_multiplier', 'time_above_1_s', 'max_count'}
    assert metrics(run(scenario, 0, 'lane_keeping'))['score_version'] == 'v0'


def test_score_v2_proximity_rises_as_cars_get_closer():
    from nohesi.config import score_v2
    rates = []
    for distance in (5.0, 10.0, 19.0, 20.0, 25.0):
        scorer = Scorer(score_v2())
        m = Measurements(0.1, np.array([30., 30., 30.]), np.array([distance, 100., 100.]), np.full(3, 5.),
                         np.full(3, np.inf), (), ())
        rates.append(scorer.update(m)['proximity'][0]/0.1)
    # Per car: half of 20 pts/s scaled by (1 - d/20).
    np.testing.assert_allclose(rates, [7.5, 5.0, 0.5, 0.0, 0.0])
