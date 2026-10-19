from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, Planner, Score, Timing, score_v3, score_v4
from nohesi.evaluation.metrics import metrics
from nohesi.evaluation.runner import run
from nohesi.scoring.events import Measurements, Overtake, draft_leaders
from nohesi.scoring.score import Scorer, TERMS, close_pass_points

DT, LENGTH = 0.1, 4.5


def step(distance=(100., 100., 100.), speeds=(28., 28., 28.), s=(0., 100., 200.), lanes=(0, 1, 2), overtakes=()):
    leader, gap = draft_leaders(np.array(s), np.array(lanes), LENGTH)
    return Measurements(DT, np.array(speeds), np.array(distance), np.full(3, 5.), np.full(3, np.inf),
                        tuple(overtakes), (), (), np.array(s), np.zeros(3), np.array(lanes), leader, gap)


def close(vehicle=0):
    return Overtake(0.0, vehicle, 7, 0.5, 28.0)


def pass_points(cfg, distance, vehicle=0, combo=0):
    scorer = Scorer(cfg)
    for _ in range(combo):
        scorer.update(step(overtakes=[close(2)]))
    return scorer.update(step(distance, overtakes=[close(vehicle)]))['close_pass'][vehicle], scorer


@pytest.mark.parametrize('distance, factor', [((100., 100., 100.), 1.0), ((10., 100., 100.), 1.25),
                                              ((0., 0., 100.), 2.0), ((20., 25., 0.), 1.0), ((10., 10., 0.), 1.5)])
def test_team_pass_multiplier(distance, factor):   # PAIRS (0,1), (0,2), (1,2); car 0 passes
    cfg = score_v4()
    points, scorer = pass_points(cfg, distance)
    assert points == pytest.approx(factor*close_pass_points(0.5, 28.0, cfg))
    assert scorer.team_pass == [pytest.approx(factor)]


def test_team_pass_uses_passers_pairs_and_combo_on_top():
    cfg, base = score_v4(), close_pass_points(0.5, 28.0, score_v4())
    points, _ = pass_points(cfg, (100., 100., 10.), vehicle=0)    # 1-2 close, passer 0 far: no bonus
    assert points == pytest.approx(base)
    points, scorer = pass_points(cfg, (100., 100., 10.), vehicle=2, combo=2)
    assert scorer.multiplier == 2.5 and points == pytest.approx(1.25*2.0*base)   # combo 2 before the pass


def draft(cfg=None, combo=0, **kw):
    scorer = Scorer(cfg or score_v4())
    for _ in range(combo):
        scorer.update(step(overtakes=[close(2)]))
    return scorer.update(step(**kw))['draft'], scorer


def test_draft_same_lane_within_headway():
    inc, _ = draft(s=(0., LENGTH+28., 200.), lanes=(1, 1, 2))   # gap 28 m = 1.0 s at 28 m/s
    assert inc.tolist() == pytest.approx([5*DT, 0, 0])
    inc, scorer = draft(combo=2, s=(0., LENGTH+10., LENGTH+20.), lanes=(1, 1, 1))   # 0 follows 1, 1 follows 2
    assert inc.tolist() == pytest.approx([5*DT*scorer.multiplier]*2+[0]) and scorer.multiplier == 2.0


def test_draft_nearest_teammate_ahead():
    m = step(s=(0., 30., 10.), lanes=(1, 1, 1))
    assert m.draft_leader.tolist() == [2, -1, 1] and m.draft_gap.tolist() == pytest.approx([5.5, np.inf, 15.5])
    m = step(s=(0., 3., 50.), lanes=(1, 1, 1))   # overlapping teammate (gap < 0) is not a leader
    assert m.draft_leader.tolist() == [2, 2, -1]


@pytest.mark.parametrize('kw', [dict(s=(0., 20., 200.), lanes=(1, 2, 2)),                  # different lane
                                dict(s=(0., LENGTH+28.5, 200.), lanes=(1, 1, 2)),          # > 1 s headway
                                dict(s=(0., 20., 200.), lanes=(1, 1, 2), speeds=(20., 28., 28.)),
                                dict(s=(0., 20., 200.), lanes=(1, 1, 2), speeds=(28., 20., 28.))])
def test_no_draft(kw):
    assert draft(**kw)[0].sum() == 0


def test_v4_no_proximity_and_v3_no_draft():
    m = dict(distance=(5., 5., 5.), s=(0., 10., 20.), lanes=(1, 1, 1))
    v4, _ = draft(**m)
    s4, s3 = Scorer(score_v4()), Scorer(score_v3())
    a, b = s4.update(step(**m)), s3.update(step(**m))
    assert a['proximity'].sum() == 0 and v4.sum() > 0
    assert b['draft'].sum() == 0 and b['proximity'].sum() > 0
    points, scorer = pass_points(score_v3(), (0., 0., 0.))
    assert points == close_pass_points(0.5, 28.0, score_v3()) and scorer.team_pass == [1.0]


def test_v4_config():
    cfg = score_v4()
    assert (cfg.version, cfg.proximity_rate, cfg.team_pass_gain, cfg.team_pass_distance) == ('v4', 0, 0.5, 20)
    assert (cfg.draft_rate, cfg.draft_headway, cfg.draft_min_speed) == (5, 1.0, 20)
    assert replace(cfg, version='v3', proximity_rate=20., team_pass_gain=0., draft_rate=0.) == score_v3()
    for k in ('team_pass_gain', 'draft_rate', 'proximity_rate'):
        with pytest.raises(ValueError):
            Score(**{k: -1.0})
    for k in ('team_pass_distance', 'draft_headway', 'draft_min_speed'):
        with pytest.raises(ValueError):
            Score(**{k: 0.0})


def test_v3_episode_unchanged():
    """Per-car term totals recorded with the pre-v4 code (scenario v4, seed 0, overtaker, 20 s, score v3)."""
    # Golden values were recorded with the setup-per-solve QP path.
    ep = run(replace(SCENARIOS['v4'](), timing=Timing(duration=20.0)), 0, 'overtaker', Planner(qp_reuse=False), score=score_v3())
    recorded = {'close_pass': [792.4908288505974, 715.0234787505149, 1312.491191483923],
                'proximity': [0.336337730622279, 0.7283486357173736, 0.3920109050950946],
                'speed': [21.629356174065794, 27.007946816801706, 32.79250335165954],
                'collision': [0., 0., 0.], 'maneuver': [25., 0., 0.], 'draft': [0., 0., 0.],
                'spread': [0., 0., 0.]}
    assert set(ep.score) == set(TERMS) == set(recorded)
    for t, v in recorded.items():
        assert ep.score[t].sum(axis=0).tolist() == pytest.approx(v, rel=1e-12, abs=1e-12)
    m = metrics(ep)
    assert m['draft_time_s'] == [0., 0., 0.] and m['team_pass']['count'] == len(ep.awarded)
    assert m['team_pass']['mean_multiplier'] in (None, 1.0)


def test_no_draft_with_traffic_in_between():
    s, lanes = np.array([30.0, 0.0, -500.0]), np.array([1, 1, 0])
    leader, gap = draft_leaders(s, lanes, LENGTH)
    assert leader[1] == 0
    leader, gap = draft_leaders(s, lanes, LENGTH, traffic_s=[15.0], traffic_lane=[1])   # traffic car between car 1 and car 0
    assert leader[1] == -1 and gap[1] == np.inf
    leader, gap = draft_leaders(s, lanes, LENGTH, traffic_s=[15.0], traffic_lane=[2])   # traffic in another lane
    assert leader[1] == 0


def test_score_v5_is_v4_with_5x_drafting():
    from dataclasses import replace
    from nohesi.config import SCORES
    v4, v5 = SCORES['v4'](), SCORES['v5']()
    assert v5.version == 'v5' and v5.draft_rate == 5*v4.draft_rate
    assert replace(v5, version='v4', draft_rate=v4.draft_rate) == v4


def test_draft_pass_scores_3x_while_drafting_or_just_after():
    from nohesi.config import SCORES
    cfg = SCORES['v6']()
    base = close_pass_points(0.5, 28.0, cfg)
    drafting = dict(s=(0., LENGTH+10., 200.), lanes=(1, 1, 2))   # car 0 drafts car 1 (both far from 2)
    alone = dict(s=(0., 100., 200.), lanes=(1, 1, 2))
    far = (100., 100., 100.)                                      # no team-pass bonus
    scorer = Scorer(cfg)                                          # pass while drafting
    assert scorer.update(step(far, overtakes=[close(0)], **drafting))['close_pass'][0] == pytest.approx(3*base)
    scorer = Scorer(cfg)                                          # slingshot: left the draft 1 s before the pass
    scorer.update(step(far, **drafting))
    for _ in range(9):
        scorer.update(step(far, **alone))
    assert scorer.update(step(far, overtakes=[close(0)], **alone))['close_pass'][0] == pytest.approx(3*base)
    assert scorer.draft_pass == [True]
    scorer = Scorer(cfg)                                          # 1.1 s after the draft: plain pass
    scorer.update(step(far, **drafting))
    for _ in range(10):
        scorer.update(step(far, **alone))
    assert scorer.update(step(far, overtakes=[close(0)], **alone))['close_pass'][0] == pytest.approx(base)
    assert scorer.draft_pass == [False]
    scorer = Scorer(SCORES['v5']())                               # v5: no draft-pass bonus
    assert scorer.update(step(far, overtakes=[close(0)], **drafting))['close_pass'][0] == pytest.approx(base)


def test_score_v6_is_v5_with_draft_passes():
    from nohesi.config import SCORES
    v5, v6 = SCORES['v5'](), SCORES['v6']()
    assert (v6.version, v6.draft_pass_gain, v6.draft_pass_window) == ('v6', 3.0, 1.0)
    assert replace(v6, version='v5', draft_pass_gain=v5.draft_pass_gain) == v5


@pytest.mark.parametrize('distance, per_s', [((30., 20., 10.), (0., 0., 0.)),           # all within 30 m: free
                                             ((40., 100., 60.), (1+49, 1+9, 49+9))])    # 2*((d-30)/10)^2, split per pair
def test_spread_penalty_v7(distance, per_s):
    from nohesi.config import SCORES
    inc = Scorer(SCORES['v7']()).update(step(distance))
    assert inc['spread'].tolist() == pytest.approx([-p*DT for p in per_s])
    assert Scorer(SCORES['v6']()).update(step(distance))['spread'].sum() == 0


def test_spread_penalty_ignores_combo():
    from nohesi.config import SCORES
    scorer = Scorer(SCORES['v7']())
    for _ in range(4):
        scorer.update(step(overtakes=[close(2)]))
    assert scorer.multiplier > 1
    assert scorer.update(step((40., 30., 30.)))['spread'].tolist() == pytest.approx([-1*DT, -1*DT, 0])


@pytest.mark.parametrize('distance, factor', [((10., 100., 100.), 1.0),     # only one teammate near: no bonus in v7
                                              ((10., 10., 25.), 1.0),       # passer's mates near, but they are 25 m apart
                                              ((10., 10., 15.), 1.5),       # whole team within 20 m: graded as v4
                                              ((0., 0., 0.), 2.0)])
def test_v7_team_pass_needs_whole_team(distance, factor):
    from nohesi.config import SCORES
    cfg = SCORES['v7']()
    points, scorer = pass_points(cfg, distance)
    assert scorer.team_pass == [pytest.approx(factor)]
    assert points == pytest.approx(factor*close_pass_points(0.5, 28.0, cfg))
    assert pass_points(SCORES['v6'](), (10., 100., 100.))[1].team_pass == [pytest.approx(1.25)]   # v6 unchanged
