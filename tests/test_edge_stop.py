"""Planner.nmpc_edge_stop: the NMPC's plans reach the road-edge bound only with room to stop their outward lateral motion
there (soft rows d +- max(+-d', 0)^2/(2 edge_stop_accel) within the bound); Planner.lean_edge_clamp: a lane target
never aims past the CG's edge bound. Both on by default since planner m4.4. The NMPC tests are skipped without acados."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import Planner, Scenario, bmw_m3
from nohesi.intentions import IntentionManager
from nohesi.road import Road

SC = Scenario()
ROAD, P = Road(SC.road), bmw_m3('dynamic')
EDGE = ROAD.width/2-P.width/2-Planner().edge_margin   # the CG's straight-heading edge bound (4.15 m)


def target_d(lane, lean, **flags):
    m = IntentionManager(replace(Planner(), **flags), ROAD, P, 1, 30.0)
    m.intentions[0].lane, m.intentions[0].lean = lane, lean
    return m.target_d(0)


def test_lean_clamp_keeps_outer_lane_targets_inside_the_bound():
    off = dict(lean_edge_clamp=False)
    assert target_d(0, -1, **off) == pytest.approx(-3.6-0.6) and target_d(2, 1, **off) == pytest.approx(3.6+0.6)   # past 4.15
    limit = EDGE-Planner().lean_edge_clear
    assert target_d(0, -1, lean_edge_clamp=True) == pytest.approx(-limit)
    assert target_d(2, 1, lean_edge_clamp=True) == pytest.approx(limit)
    assert 3.6 < limit < EDGE
    for lane, lean in ((0, 1), (0, 0), (1, -1), (1, 1), (2, -1), (2, 0)):   # every other target unchanged
        assert target_d(lane, lean, lean_edge_clamp=True) == pytest.approx(target_d(lane, lean, **off))
    assert Planner().lean_edge_clamp and Planner().nmpc_edge_stop   # the defaults since m4.4


def test_off_keeps_the_compiled_solver():
    nl = pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)
    base = Planner(mpc='nonlinear', nmpc_edge_stop=False, lean_edge_clamp=False)
    assert nl.config_hash(replace(base, lean_edge_clamp=True, nmpc_edge_yaw=True), P) == nl.config_hash(base, P)
    assert nl.config_hash(replace(base, nmpc_edge_stop=True), P) != nl.config_hash(base, P)


def dodge(gap, v_car, seconds=5.0, **flags):
    """Closed loop on the dynamic plant: 30 m/s in lane 1, a slower car `gap` m ahead in lane 1, target the outer lane
    2, training lateral weight (w_lateral 10). Returns the largest stopping overshoot d + max(d', 0)^2/(2 a) - EDGE and
    the largest body-corner |d| over the solved plans' states, the number of unsolved plans, and the plans."""
    nl = pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)
    from nohesi.dynamics import DynamicPlant
    from nohesi.geometry import footprint
    from nohesi.planning.base import Obstacles, Target, constant_speed_plan
    plant = DynamicPlant(P, SC.timing.dt)
    cfg = replace(Planner(mpc='nonlinear'), w_lateral=10.0, **flags)
    mpc, N = nl.NonlinearMPC(cfg, plant, ROAD), cfg.horizon
    state, steer, plan = np.array([0., 0., 0., 30.0, 0., 0.]), 0.0, None
    over, corner, unsolved, plans = -np.inf, 0.0, 0, []
    for k in range(int(round(seconds/SC.timing.dt))):
        t = k*SC.timing.dt
        if k % 2 == 0:
            times = t+np.arange(N+1)*cfg.step
            cars = Obstacles((('traffic', 0),), (gap+v_car*times)[:, None], np.zeros((N+1, 1)), np.full((N+1, 1), v_car))
            ref = constant_speed_plan(t, cfg.step, N, *state[[0, 1, 3]]) if plan is None else plan.shifted(1)
            plan = mpc.solve(t, state[:4].copy(), steer, 0.0, Target(d=ROAD.lane_center(2), speed=30.0), cars, ref,
                             tuple(state[4:6]))
            plans.append(plan.z.copy())
            if plan.status == 'solved':
                X = mpc.solver.get_flat('x').reshape(N+1, nl.NX)[1:]
                rate = X[:, nl.VX]*np.sin(X[:, nl.PSI])+X[:, nl.VY]*np.cos(X[:, nl.PSI])
                over = max(over, float(np.max(X[:, nl.D]+np.maximum(rate, 0)**2/(2*cfg.edge_stop_accel)-EDGE)))
                corner = max(corner, max(np.abs(ROAD.to_frenet(footprint(z, P))[1]).max() for z in plan.z[1:]))
            else:
                unsolved += 1
        state, _, steer = plant.step(state, plan.u[0, 0], plan.u[0, 1], steer)
    return over, corner, unsolved, plans


def test_hard_dodge_leaves_room_to_stop_at_the_edge():
    off_over, off_corner, off_unsolved, _ = dodge(30.0, 10.0, nmpc_edge_stop=False)   # to the bound at up to ~5 m/s outward
    on_over, on_corner, on_unsolved, _ = dodge(30.0, 10.0, nmpc_edge_stop=True)
    assert off_over > 1.0 and on_over < 0.05
    assert off_corner > ROAD.width/2 > on_corner
    assert on_unsolved <= off_unsolved


def test_no_effect_on_a_normal_lane_change():
    """An outer-lane change that slows its lateral motion before the lane centre never reaches the stopping rows: the
    same plans up to the interior-point solver's response to inactive rows (largest difference ~1e-5 m)."""
    (_, _, _, a), (_, _, _, b) = dodge(40.0, 10.0, nmpc_edge_stop=False), dodge(40.0, 10.0, nmpc_edge_stop=True)
    assert all(np.allclose(x, y, rtol=0, atol=1e-4) for x, y in zip(a, b))
