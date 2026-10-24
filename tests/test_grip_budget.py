"""Planner m3.8 for the dynamic plant: load transfer, brake/drive split and the friction cap in the plant; the braking
cap (grip budget) in the planner; the nonlinear MPC's model matching the plant."""
from dataclasses import replace
import math
import numpy as np
import pytest
from nohesi.config import Planner, SCENARIOS, Vehicle, bmw_m3
from nohesi.dynamics import G, DynamicPlant, KinematicPlant, brake_limit
from nohesi.evaluation.runner import run
from nohesi.rl.scripted import lane_keep_law
from nohesi.road import Road

M3 = bmw_m3('dynamic')
DT = 0.05


def test_default_vehicle_keeps_static_loads_and_even_split():
    plant = DynamicPlant(Vehicle(model='dynamic'), DT)
    fz_f, fz_r, fx_f, fx_r, a = plant.axle_forces(-4.0)
    assert (fz_f, fz_r) == plant.loads
    assert fx_f == fx_r == 0.5*plant.params.mass*-4.0 and a == -4.0


def test_load_transfer_and_split():
    plant, p = DynamicPlant(M3, DT), M3
    for a in (-8.0, 6.0):
        fz_f, fz_r, fx_f, fx_r, ax = plant.axle_forces(a)
        assert fz_f+fz_r == pytest.approx(p.mass*G)
        assert fz_f-plant.loads[0] == pytest.approx(-p.mass*a*p.cg_height/p.wheelbase)   # braking loads the front
        share = p.brake_bias_front if a < 0 else p.drive_bias_front
        assert fx_f/(fx_f+fx_r) == pytest.approx(share) and ax == a


def test_friction_caps_braking_and_front_saturates_first():
    plant, p = DynamicPlant(M3, DT), M3
    *_, a = plant.axle_forces(-p.brake_max)
    assert a == pytest.approx(-p.friction*G, rel=0.02)   # the tyres, not the brakes, set the limit
    # Just below mu g only the front axle is at its limit (the stable way round).
    fz_f, fz_r, fx_f, fx_r, _ = plant.axle_forces(-10.2)
    assert abs(fx_f) == pytest.approx(p.friction*fz_f) and abs(fx_r) < p.friction*fz_r


def test_hard_braking_stays_straight_with_front_bias():
    """Full braking from 35 m/s with a small yaw disturbance: the front-biased M3 stays straight, the same car with a
    50/50 split locks the rear and rotates."""
    def yaw_change(vehicle):
        plant = DynamicPlant(vehicle, DT)
        st, steer = np.array([0.0, 0.0, 0.0, 35.0, 0.0, 0.05]), 0.0
        for _ in range(int(1.5/DT)):
            st, _, steer = plant.step(st, -vehicle.brake_max, 0.0, steer)
        return abs(math.degrees(st[2]))
    assert yaw_change(M3) < 5
    assert yaw_change(replace(M3, brake_bias_front=0.5)) > 3*yaw_change(M3)


def test_brake_limit():
    assert brake_limit(KinematicPlant(bmw_m3(), DT), 0.8) == bmw_m3().brake_max
    assert brake_limit(DynamicPlant(M3, DT), 0.8) == pytest.approx(0.8*M3.friction*G)


def test_lane_keep_law_caps_braking():
    plant = KinematicPlant(bmw_m3(), DT)
    route = Road(SCENARIOS['v5']().road).lane_route(1)
    state = np.array([100.0, 0.0, 0.0, 30.0])
    a, _ = lane_keep_law(state, 100.0, route, plant.params, 30.0, lead=(6.0, 10.0), brake=8.0)
    assert a == -8.0
    a, _ = lane_keep_law(state, 100.0, route, plant.params, 30.0, lead=(6.0, 10.0))
    assert a < -8.0


def test_dynamic_commands_respect_braking_cap():
    """Overtaker on the dynamic M3 for 8 s (linear MPC keeps the test fast): every applied braking command is within
    the grip budget."""
    sc = SCENARIOS['v5']()
    sc = replace(sc, vehicle=replace(sc.vehicle, model='dynamic'), timing=replace(sc.timing, duration=8.0))
    cfg = Planner()
    ep = run(sc, 1010, 'overtaker', cfg)
    a = ep.controls[:, :, 0]
    assert a.min() >= -cfg.grip_brake*M3.friction*G-1e-9


def test_nmpc_model_matches_plant_with_load_transfer():
    """The NMPC's CasADi dynamics equal DynamicPlant.derivatives for the M3 (load transfer, 72/28 brake and 30/70
    drive split) within the braking cap; past mu g the deceleration matches too (capped Fx), while the model keeps its
    5 % lateral-peak floor where the plant has none."""
    nl = pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)
    import casadi as ca
    plant = DynamicPlant(M3, DT)
    x, u, xdot, _, grip = nl.model_dynamics(plant.params)
    f = ca.Function('f', [x, u], [xdot, grip])
    for vx, vy, r, steer, a in ((30.0, 0.2, 0.05, 0.02, -6.0), (25.0, -0.3, -0.1, -0.03, -8.0), (20.0, 0.1, 0.02, 0.01, 5.0)):
        xd, g = f(np.array([0.0, 0.0, 0.0, vx, vy, r, steer, a]), np.zeros(2))
        truth = plant.derivatives((0.0, 0.0, 0.0, vx, vy, r), a, steer)
        np.testing.assert_allclose(np.asarray(xd).ravel()[3:6], truth[3:6], rtol=1e-6, atol=1e-6)
        assert np.all(np.asarray(g) < 1.0)
    xd, _ = f(np.array([0.0, 0.0, 0.0, 25.0, 0.0, 0.0, 0.0, -11.0]), np.zeros(2))
    assert float(xd[3]) == pytest.approx(plant.derivatives((0.0, 0.0, 0.0, 25.0, 0.0, 0.0), -11.0, 0.0)[3])
