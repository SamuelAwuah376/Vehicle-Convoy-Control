"""Planner.nmpc_edge_yaw: the NMPC's road-edge bound keeps the body's outermost corner, not only the CG, inside the
edge at the plan's heading (off by default). Skipped without acados."""
from dataclasses import replace
import numpy as np
import pytest

nl = pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)

from nohesi.config import Planner, Scenario, bmw_m3   # noqa: E402
from nohesi.dynamics import DynamicPlant   # noqa: E402
from nohesi.geometry import footprint   # noqa: E402
from nohesi.planning.base import Obstacles, Target, constant_speed_plan   # noqa: E402
from nohesi.road import Road   # noqa: E402

SC = Scenario()
ROAD, PLANT = Road(SC.road), DynamicPlant(bmw_m3('dynamic'), SC.timing.dt)
P, MARGIN = PLANT.params, Planner().edge_margin


def test_bounds_put_the_outermost_corner_at_the_limit():
    limit = ROAD.width/2-MARGIN
    straight = limit-P.width/2
    psi = np.linspace(-0.6, 0.6, 121)
    lo, hi = nl.edge_bounds(psi, P, ROAD.width, MARGIN)
    assert lo[60] == -straight and hi[60] == straight                  # straight ahead: the bound as before
    assert np.all(hi <= straight) and np.all(lo >= -straight) and np.all(lo < hi)
    for a, d_lo, d_hi in zip(psi, lo, hi):
        top = ROAD.to_frenet(footprint(np.array([0.0, d_hi, a, 0.0]), P))[1]
        bottom = ROAD.to_frenet(footprint(np.array([0.0, d_lo, a, 0.0]), P))[1]
        if d_hi < straight:                                            # the corner, not the CG, sets the bound
            assert top.max() == pytest.approx(limit, abs=1e-9) and bottom.min() == pytest.approx(-limit, abs=1e-9)
        assert top.max() <= limit+1e-9 and bottom.min() >= -limit-1e-9
    # At 0.3 rad the bound is ~0.6 m tighter than the straight one (front corner 2.3 m ahead of the CG).
    assert straight-hi[np.argmin(abs(psi-0.3))] == pytest.approx(0.64, abs=0.02)


def dodge(on, gap, v_car, seconds=5.0):
    """Closed loop on the dynamic plant: 30 m/s in lane 1, a slower car `gap` m ahead in lane 1, target the outer lane
    2, training lateral weight (w_lateral 10). Returns the largest body-corner |d| over solved plans and the plans."""
    # planner m4.3's edge handling (no stopping rows, no lean clamp), so the heading-aware bound is tested on its own
    cfg = replace(Planner(mpc='nonlinear'), nmpc_edge_yaw=on, w_lateral=10.0, nmpc_edge_stop=False, lean_edge_clamp=False)
    mpc, N = nl.NonlinearMPC(cfg, PLANT, ROAD), cfg.horizon
    state, steer, accel, plan, worst, plans = np.array([0., 0., 0., 30.0, 0., 0.]), 0.0, 0.0, None, 0.0, []
    for k in range(int(round(seconds/SC.timing.dt))):
        t = k*SC.timing.dt
        if k % 2 == 0:
            times = t+np.arange(N+1)*cfg.step
            cars = Obstacles((('traffic', 0),), (gap+v_car*times)[:, None], np.zeros((N+1, 1)), np.full((N+1, 1), v_car))
            ref = constant_speed_plan(t, cfg.step, N, *state[[0, 1, 3]]) if plan is None else plan.shifted(1)
            plan = mpc.solve(t, state[:4].copy(), steer, accel, Target(d=ROAD.lane_center(2), speed=30.0), cars, ref,
                             tuple(state[4:6]))
            plans.append(plan.z.copy())
            if plan.status == 'solved':
                worst = max(worst, max(np.abs(ROAD.to_frenet(footprint(z, P))[1]).max() for z in plan.z[1:]))
        state, accel, steer = PLANT.step(state, plan.u[0, 0], plan.u[0, 1], steer)
    return worst, plans


def test_hard_dodge_toward_the_edge_keeps_planned_corners_on_the_road():
    off, _ = dodge(False, 30.0, 10.0)   # swerves at up to ~0.5 rad with the CG at its bound
    on, _ = dodge(True, 30.0, 10.0)
    assert off > ROAD.width/2 > on
    assert off-on > 0.3


def test_no_effect_at_small_headings():
    """A normal outer-lane change (headings up to 0.17 rad) never reaches the tighter bound: the same plans up to the
    interior-point solver's response to an inactive bound moving (largest difference 8e-6 m)."""
    (off, a), (on, b) = dodge(False, 40.0, 10.0), dodge(True, 40.0, 10.0)
    assert off == pytest.approx(on, abs=1e-4)
    assert all(np.allclose(x, y, rtol=0, atol=1e-4) for x, y in zip(a, b))
