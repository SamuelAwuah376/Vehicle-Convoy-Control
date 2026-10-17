from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, SCORES, Planner, Timing
from nohesi.dynamics import KinematicPlant
from nohesi.evaluation.runner import run
from nohesi.planning import single
from nohesi.planning.base import Obstacles, Target, constant_speed_plan
from nohesi.planning.single import LaneChangeMPC
from nohesi.road import Road

SC = SCENARIOS['v4']()
ROAD, PLANT = Road(SC.road), KinematicPlant(SC.vehicle, SC.timing.dt)
SETUP, REUSE = Planner(qp_reuse=False), Planner()


@pytest.fixture(scope='module')
def captured():
    """Solve inputs of v4 seed 0 overtaker, first 3 s (90 solves, all three cars through one MPC)."""
    cap, orig = [], single.LaneChangeMPC.solve

    def rec(self, *args, **kw):
        cap.append(args)
        return orig(self, *args, **kw)
    single.LaneChangeMPC.solve = rec
    try:
        run(replace(SC, timing=Timing(duration=3.0)), 0, 'overtaker', SETUP, score=SCORES['v4']())
    finally:
        single.LaneChangeMPC.solve = orig
    return cap


def compare(cap, cfg):
    old, new = LaneChangeMPC(SETUP, PLANT, ROAD), LaneChangeMPC(cfg, PLANT, ROAD)
    pairs = [(old.solve(*a), new.solve(*a)) for a in cap]
    assert [a.status for a, _ in pairs] == [b.status for _, b in pairs]
    dz = max(np.abs(a.z-b.z).max() for a, b in pairs if a.status == 'solved')
    du = np.max([np.abs(a.u-b.u).max(0) for a, b in pairs if a.status == 'solved'], axis=0)
    return new, dz, du


def test_reuse_matches_setup_path(captured):
    # Same QP (hard rows intersected per step, soft rows in slots): solutions agree within OSQP's eps 1e-4
    # accuracy (both paths are ~0.1 m / 0.5 m/s^2 from a tight-tolerance optimum on v4).
    new, dz, du = compare(captured, REUSE)
    assert new.fallbacks == 0 and new._solver is not None
    assert dz < 0.1 and du[0] < 0.3 and du[1] < 5e-3


def test_reused_solver_has_no_memory(captured):
    # Scaling recomputed, rho reset, y zeroed per solve: a solve after others equals one on a fresh instance,
    # so one MPC can serve all cars in any order.
    shared = LaneChangeMPC(REUSE, PLANT, ROAD)
    for a in captured[:-5]:
        shared.solve(*a)
    for a in captured[-5:]:
        p, q = shared.solve(*a), LaneChangeMPC(REUSE, PLANT, ROAD).solve(*a)
        assert p.status == q.status and p.iterations == q.iterations and np.allclose(p.z, q.z, atol=1e-8)


def test_too_few_slots_fall_back_to_setup(captured):
    new, dz, du = compare(captured, Planner(qp_soft_slots=1))
    assert new.fallbacks > 0 and dz < 0.1 and du[0] < 0.3


def test_empty_hard_interval_falls_back_and_reports_infeasible():
    # In-lane cars 3 m ahead and 3 m behind: s <= 3 - gap and s >= -3 + gap cannot both hold.
    N, T = REUSE.horizon, np.arange(REUSE.horizon+1)*REUSE.step
    ref = constant_speed_plan(0.0, REUSE.step, N, 0.0, 0.0, 28.0)
    obs = Obstacles((('traffic', 0), ('traffic', 1)), np.c_[3.0+28.0*T, -3.0+28.0*T], np.zeros((N+1, 2)),
                    np.full((N+1, 2), 28.0))
    args = (0.0, np.array([0.0, 0.0, 0.0, 28.0]), 0.0, 0.0, Target(0.0, 28.0), obs, ref)
    mpc = LaneChangeMPC(REUSE, PLANT, ROAD)
    assert mpc.solve(*args).status == LaneChangeMPC(SETUP, PLANT, ROAD).solve(*args).status == 'infeasible'
    assert mpc.fallbacks == 1
