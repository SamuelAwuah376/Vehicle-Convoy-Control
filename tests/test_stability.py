"""Stability layer (planner m4.0): the 100 Hz inner loop under every command on the dynamic plant."""
from dataclasses import replace
import math
import numpy as np
from nohesi.config import Planner, SCENARIOS, bmw_m3
from nohesi.dynamics import DynamicPlant
from nohesi.evaluation.runner import EpisodeRun
from nohesi.planning.base import Plan
from nohesi.stability import StabilityLayer

DT = 0.05
PLANT = DynamicPlant(bmw_m3('dynamic'), DT)


def drive(v0, commands, layer):
    """Open-loop (duration, accel, steer) commands from straight running at v0: (max |sideslip| deg, yaw deg, layer)."""
    st, steer, t = np.array([0.0, 0.0, 0.0, v0, 0.0, 0.0]), 0.0, 0.0
    lay = StabilityLayer(PLANT, Planner()) if layer else None
    beta = 0.0
    for dur, a, d in commands:
        for _ in range(int(round(dur/DT))):
            st, _, steer = PLANT.step(st, a, d, steer, lay.inner(0, t) if lay else None)
            t += DT
            beta = max(beta, abs(math.degrees(math.atan2(st[4], max(math.sqrt(max(st[3]**2-st[4]**2, 0.0)), 0.1)))))
    return beta, math.degrees(st[2]), lay


def test_normal_driving_is_left_alone():
    beta, _, lay = drive(30.0, [(3.0, 0.3, 0.005)], True)
    assert lay.interventions['steer'] == 0 and lay.interventions['longitudinal'] == 0
    assert beta < 1


def test_steering_past_grip_does_not_spin():
    """0.12 rad at 30 m/s asks for about 2.5x the grip-limited yaw rate: without the layer the car spins out."""
    beta_off, yaw_off, _ = drive(30.0, [(1.5, 0.0, 0.12)], False)
    beta_on, yaw_on, _ = drive(30.0, [(1.5, 0.0, 0.12)], True)
    assert beta_off > 15 and beta_on < 8 and 10 < yaw_on < yaw_off


def test_full_braking_keeps_steering():
    """Full braking with steer: without the layer the tyres have no cornering force left and the car goes straight;
    with it (ABS-like, lateral first) the car turns while braking hard."""
    _, yaw_off, _ = drive(35.0, [(2.0, -11.0, 0.04)], False)
    _, yaw_on, lay = drive(35.0, [(2.0, -11.0, 0.04)], True)
    assert abs(yaw_off) < 0.5 and yaw_on > 4
    assert lay.interventions['longitudinal'] > 0


def test_plan_lateral_shifts_with_the_plan():
    lat = np.column_stack((np.arange(4.0), 10+np.arange(4.0)))
    plan = Plan(0.0, 0.1, np.zeros((4, 4)), np.zeros((3, 2)), 'qp', lateral=lat).shifted(1)
    np.testing.assert_array_equal(plan.lateral, [[1, 11], [2, 12], [3, 13], [3, 13]])


def test_layer_runs_under_the_coordinator_and_feeds_the_buffer():
    """Overtaker on the dynamic M3 (linear MPC keeps it fast): the coordinator installs the layer, every substep goes
    through it, and the buffer's latest state is the world's state at the end of the step."""
    sc = SCENARIOS['v5']()
    sc = replace(sc, vehicle=replace(sc.vehicle, model='dynamic'), timing=replace(sc.timing, duration=3.0))
    r = EpisodeRun(sc, 1010, 'overtaker', Planner())
    while not r.done:
        r.step()
    layer = r.world.stability
    assert layer is not None and layer is r.policy.stability
    assert layer.interventions['substeps'] == 3*round(3.0/DT)*PLANT.substeps   # 3 cars, every substep
    for i in range(3):
        t, state, _, _ = layer.buffer.latest[i]
        assert t == r.world.time
        np.testing.assert_allclose(state[:4], r.world.team[i])
        assert len(layer.buffer.substeps[i]) == PLANT.substeps+1


def test_kinematic_plant_has_no_layer():
    sc = replace(SCENARIOS['v5'](), timing=replace(SCENARIOS['v5']().timing, duration=0.5))
    r = EpisodeRun(sc, 1010, 'overtaker', Planner())
    assert r.world.stability is None and r.policy.stability is None
