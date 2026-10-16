import collections
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import SCENARIOS, SCORES, Planner, Scenario, Timing
from nohesi.coordination.centralized import make_mpc
from nohesi.dynamics import KinematicPlant
from nohesi.evaluation.runner import run
from nohesi.planning import single
from nohesi.planning.base import Obstacles, Target, constant_speed_plan
from nohesi.planning.constraints import obstacle_constraints, state_bounds
from nohesi.planning.single import LaneChangeMPC, _Rows
from nohesi.road import Road

SC, CFG = Scenario(), Planner()
ROAD, PLANT = Road(SC.road), KinematicPlant(SC.vehicle, SC.timing.dt)
P, N = PLANT.params, CFG.horizon
T, KS = np.arange(N+1)*CFG.step, np.arange(1, N+1)
LAT_GAP, LONG_GAP, W = P.width+CFG.margin_lat, P.length+CFG.margin_long, ROAD.cfg.lane_width
ZR = constant_speed_plan(0.0, CFG.step, N, 0.0, 0.0, 28.0).z


def spec(obs_s, obs_d, v, key, target=Target(0.0, 28.0), zr=ZR, s0=0.0):
    obs = Obstacles((key,), np.asarray(obs_s, float)[:, None], np.asarray(obs_d, float)[:, None],
                    np.full((N+1, 1), float(v)))
    return obstacle_constraints(CFG, P, ROAD, zr, obs, target, s0)


def test_in_lane_car_ahead_hard_behind_and_soft_headway():
    os_ = 20.0+25.0*T
    c = spec(os_, np.zeros(N+1), 25.0, ('traffic', 3))
    hard, soft = c.select(~c.soft), c.select(c.soft)
    assert set(hard.kind) == {'behind'} and set(soft.kind) == {'headway_behind'}
    rel = np.abs(ZR[1:, 0]-os_[1:]) <= CFG.relevance
    assert np.array_equal(hard.k, KS[rel]) and np.array_equal(soft.k, KS[rel])
    assert np.allclose(hard.hi, os_[1:][rel]-LONG_GAP) and np.all(np.isneginf(hard.lo)) and np.all(hard.cs == 1)
    assert np.all(soft.cv == CFG.headway) and np.all(soft.slack_weight == CFG.w_slack) and np.all(hard.slack_weight == 0)
    assert np.all(c.j == 0) and c.per_stage(N)[0].max() == 1


def test_adjacent_lane_car_beside_side_rows_only():
    c = spec(1.0+28.0*T, np.full(N+1, -W), 28.0, ('traffic', 0))
    assert set(c.kind) == {'left'} and len(c) == N and not c.soft.any()
    assert np.allclose(c.lo, -W+LAT_GAP) and np.all(c.cd == 1) and np.all(c.cs == 0) and np.all(np.isposinf(c.hi))


def test_teammate_ahead_in_lane_uses_teammate_headway_and_ahead_branch():
    c = spec(-10.0+28.0*T, np.zeros(N+1), 28.0, ('team', 1))   # ego ahead of the teammate
    soft = c.select(c.soft)
    assert set(c.select(~c.soft).kind) == {'ahead'} and set(soft.kind) == {'headway_ahead'}
    assert np.allclose(soft.lo, (-10.0+28.0*T[1:])+LONG_GAP+CFG.teammate_headway*28.0) and np.all(soft.cv == 0)


def test_yield_rows_and_s0_convention():
    s0, os_ = 500.0, -3.0+28.0*T   # teammate level/just ahead, changing into the lane: ego drops back
    target = Target(0.0, 28.0, yield_to=(('team', 2),))
    rel = spec(os_, np.full(N+1, 2.0), 28.0, ('team', 2), target)
    zabs = ZR.copy(); zabs[:, 0] += s0
    ab = spec(os_+s0, np.full(N+1, 2.0), 28.0, ('team', 2), target, zabs, s0)
    for f in ('k', 'cs', 'cd', 'cv', 'slack_weight', 'kind', 'j'):
        assert np.array_equal(getattr(rel, f), getattr(ab, f)), f
    assert np.allclose(rel.lo, ab.lo, atol=1e-9) and np.allclose(rel.hi, ab.hi, atol=1e-9)   # s relative to s0
    y = rel.select(rel.kind == 'yield')
    clear = LONG_GAP+CFG.side_zone+CFG.yield_margin
    assert np.array_equal(y.k, KS) and np.all(y.slack_weight == CFG.w_yield)
    assert np.allclose(y.lo, os_[1:]+clear) and np.all(np.isposinf(y.hi))   # ego ahead of the changer: pull ahead
    y = spec(3.0+28.0*T, np.full(N+1, 2.0), 28.0, ('team', 2), target)
    y = y.select(y.kind == 'yield')                                        # ego level/behind: drop back
    assert np.allclose(y.hi, 3.0+28.0*T[1:]-clear) and np.all(np.isneginf(y.lo)) and np.all(y.cs == 1)


def test_gap_cars_beside_until_gap_and_stale_rear_is_ordinary():
    front = spec(3.0+28.0*T, np.full(N+1, W), 28.0, ('team', 1), Target(W, 28.0, front=('team', 1)))
    assert set(front.kind) == {'right'}           # ds > -long_gap: stay beside (to its right) until behind it
    stale = spec(20.0+29.0*T, np.linspace(-2.0, 2.0, N+1), 29.0, ('team', 1), Target(0.0, 28.0, rear=('team', 1)))
    assert set(stale.select(~stale.soft).kind) == {'behind'}   # past the ego: ordinary car ahead


def test_state_bounds():
    c = state_bounds(CFG, P, ROAD)
    edge = ROAD.width/2-P.width/2-CFG.edge_margin
    assert list(c.kind) == ['edge']*N+['speed']*N and np.allclose(c.hi[:N], edge) and np.all(c.lo[N:] == 0)


def test_osqp_rows_follow_description_on_captured_case():
    cap, orig = [], single.LaneChangeMPC.solve

    def rec(self, *args, **kw):
        cap.append(args)
        return orig(self, *args, **kw)
    single.LaneChangeMPC.solve = rec
    try:
        run(replace(SCENARIOS['v4'](), timing=Timing(duration=3.0)), 0, 'overtaker', Planner(), score=SCORES['v3']())
    finally:
        single.LaneChangeMPC.solve = orig
    mpc, kinds = LaneChangeMPC(CFG, PLANT, Road(SCENARIOS['v4']().road)), collections.Counter()
    for time, z0, steer, accel, target, obs, ref, *_ in cap:
        c = obstacle_constraints(CFG, P, mpc.road, ref.z, obs, target, z0[0])
        kinds.update(c.kind.tolist())
        zr = ref.z.copy(); zr[:, 0] -= z0[0]
        rows = _Rows()
        w = mpc._obstacle_rows(rows, zr, obs.s-z0[0], obs.d, obs.v, obs.keys, target)
        A, lo, hi = rows.matrix(mpc.nz+mpc.nu+len(w))
        A = A.toarray()
        r = np.flatnonzero(np.abs(A[:, :mpc.nz]).sum(1) > 0)   # drop the sigma >= 0 rows
        order = np.r_[np.flatnonzero(~c.soft), np.flatnonzero(c.soft)]
        assert len(r) == len(c) and np.array_equal(lo[r], c.lo[order]) and np.array_equal(hi[r], c.hi[order])
        for col, coef in ((0, c.cs), (1, c.cd), (3, c.cv)):
            assert np.array_equal(A[r, mpc.zi(c.k[order], col)], coef[order])
        soft = r[c.soft[order]]   # each soft row has its own slack, relaxing its finite bound
        sig = A[soft, mpc.nz+mpc.nu:]
        assert np.array_equal(sig, np.diag(np.where(np.isfinite(lo[soft]), 1.0, -1.0)))
        assert np.array_equal(w, c.slack_weight[c.soft])
        plan = orig(mpc, time, z0, steer, accel, target, obs, ref)
        if plan.status == 'solved':   # hard rows hold for the solution (OSQP tolerance)
            x = plan.z[c.k, 0]-z0[0], plan.z[c.k, 1], plan.z[c.k, 3]
            val = c.cs*x[0]+c.cd*x[1]+c.cv*x[2]
            assert np.all((val >= c.lo-1e-2)[~c.soft]) and np.all((val <= c.hi+1e-2)[~c.soft])
    # Golden selection counts over the 90 solves of v4 seed 0, first 3 s (pins the selection logic).
    assert len(cap) == 90
    assert kinds == {'left': 617, 'right': 900, 'behind': 6927, 'ahead': 5692, 'headway_ahead': 2370,
                     'headway_behind': 1821}


def test_mpc_factory():
    assert isinstance(make_mpc(CFG, PLANT, ROAD), LaneChangeMPC)   # mpc='nonlinear': tests/test_nonlinear_mpc.py
    with pytest.raises(ValueError):
        Planner(mpc='quadratic')
