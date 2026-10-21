"""Planner m3.6 safety layer: sub-sampled footprint checks, slot-margin back-outs, on/off switches."""
import numpy as np
from nohesi.config import SCENARIOS
from nohesi.planning.fallback import dense, first_conflict

VEH = SCENARIOS['v4']().vehicle


def test_dense_interpolates_between_samples():
    assert dense([0., 1., 3.], 2).tolist() == [0., 0.5, 1., 2., 3.]
    assert dense(np.array([[0., 2.], [10., 20.]]), 2).tolist() == [[0., 1., 2.], [10., 15., 20.]]
    assert dense([1., 2.], 1).tolist() == [1., 2.]


def test_substeps_catch_a_fast_pass_between_plan_nodes():
    ego = np.zeros((4, 4))                              # parked at s = 0, d = 0
    s = np.array([-30., -10., 10., 30.])                # passes straight through the ego between nodes 1 and 2
    other = [(('team', 1), s, np.zeros(4), np.zeros(4))]
    assert first_conflict(ego, other, VEH, 0.1, substeps=1) is None
    assert first_conflict(ego, other, VEH, 0.1, substeps=5) == (('team', 1), 2)



def test_planner_switches_can_be_off():
    from dataclasses import replace
    from nohesi.config import Planner
    assert replace(Planner(), slot_check=False).slot_check is False
    assert replace(Planner(), qp_reuse=False).qp_reuse is False


def test_slot_check_backs_out_only_when_on():
    from collections import Counter
    from dataclasses import replace
    from nohesi.config import Planner, SCORES, Timing
    from nohesi.evaluation.runner import run
    sc = replace(SCENARIOS['v4'](), timing=Timing(duration=40.0))
    count = lambda planner: Counter(r.reason for r in run(sc, 2, 'overtaker', planner, score=SCORES['v3']()).intention_records
                                    if r.kind == 'cancelled')['slot_margin']
    assert count(replace(Planner(), slot_check=False)) == 0
    assert count(Planner()) >= 1    # this episode has at least one closing slot at the default 0.5 s margin
