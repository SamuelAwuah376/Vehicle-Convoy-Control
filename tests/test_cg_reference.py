"""Vehicle states are referenced at the centre of gravity (CG): plant, footprint and lane keeping."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from nohesi.config import Road as RoadConfig, Vehicle
from nohesi.dynamics import KinematicPlant, kinematic_step
from nohesi.geometry import body_centre, footprint
from nohesi.road import Road
from nohesi.rl.scripted import lane_keep_law

VEHICLE = Vehicle()
ROAD = Road(RoadConfig())


def params(l_r=1.35, dt=0.01):
    return KinematicPlant(replace(VEHICLE, cg_to_rear_axle=l_r), dt).params


def test_straight_line_motion():
    state = np.array([0., 0.5, 0., 20.])
    for _ in range(10):
        state, a, delta = kinematic_step(state, 0.0, 0.0, 0.0, params(dt=0.1))
    np.testing.assert_allclose(state, [20., 0.5, 0., 20.], atol=1e-12)


@pytest.mark.parametrize('l_r', [1.35, 1.6])
def test_constant_steer_cg_circle(l_r):
    p, delta, v = params(l_r), 0.1, 5.0
    beta = np.arctan(l_r/p.wheelbase*np.tan(delta))
    R = p.wheelbase/(np.cos(beta)*np.tan(delta))
    centre = R*np.array([-np.sin(beta), np.cos(beta)])   # perpendicular to the initial velocity heading beta
    state, xy, rear = np.array([0., 0., 0., v]), [], []
    for _ in range(1500):
        state, _, _ = kinematic_step(state, 0.0, delta, delta, p)
        xy.append(state[:2]); rear.append(state[:2]-l_r*np.array([np.cos(state[2]), np.sin(state[2])]))
    np.testing.assert_allclose(np.linalg.norm(np.array(xy)-centre, axis=1), R, rtol=1e-4)
    # The rear axle traces the classic rear-axle circle L/tan(delta) about the same centre.
    np.testing.assert_allclose(np.linalg.norm(np.array(rear)-centre, axis=1), p.wheelbase/np.tan(delta), rtol=1e-4)
    yaw = 1500*p.dt*v/R   # yaw rate v cos(beta) tan(delta)/L = v/R
    assert np.cos(state[2]-yaw) == pytest.approx(1.0)


@pytest.mark.parametrize('l_r', [1.35, 1.6])
def test_matches_rear_axle_model(l_r):
    """Same steering and the rear-axle speed v cos(beta): the CG model's rear axle follows the rear-axle kinematic
    bicycle (x' = v cos yaw, y' = v sin yaw, yaw' = v tan(delta)/L), integrated finely here as the reference."""
    p, delta = params(l_r), 0.2
    beta = np.arctan(l_r/p.wheelbase*np.tan(delta))
    cg, rear = np.array([l_r, 0., 0., 10.]), np.array([0., 0., 0., 10*np.cos(beta)])
    for _ in range(300):
        cg, _, _ = kinematic_step(cg, 0.0, delta, delta, p)
        for _ in range(10):
            h = p.dt/10
            mid = rear[2]+0.5*h*rear[3]*np.tan(delta)/p.wheelbase
            rear = rear+h*np.array([rear[3]*np.cos(mid), rear[3]*np.sin(mid), rear[3]*np.tan(delta)/p.wheelbase, 0.0])
    np.testing.assert_allclose(cg[:2]-l_r*np.array([np.cos(cg[2]), np.sin(cg[2])]), rear[:2], atol=1e-3)
    assert cg[2] == pytest.approx(rear[2], abs=1e-6)


@pytest.mark.parametrize('l_r, rear, front', [(1.35, -2.25, 2.25), (1.6, -2.5, 2.0)])
def test_footprint_extents_about_cg(l_r, rear, front):
    p = params(l_r)
    poly = footprint(np.array([10., 2., 0., 0.]), p)
    assert poly[:, 0].min() == pytest.approx(10+rear) and poly[:, 0].max() == pytest.approx(10+front)
    assert poly[:, 1].min() == pytest.approx(2-0.9) and poly[:, 1].max() == pytest.approx(2+0.9)
    np.testing.assert_allclose(body_centre(np.array([10., 2., 0., 0.]), p), [10+(rear+front)/2, 2.])
    turned = footprint(np.array([0., 0., np.pi/2, 0.]), p)
    assert turned[:, 1].min() == pytest.approx(rear) and turned[:, 1].max() == pytest.approx(front)
    # Missing cg_to_rear_axle defaults to the wheelbase midpoint.
    bare = SimpleNamespace(wheelbase=2.7, length=4.5, width=1.8)
    np.testing.assert_allclose(footprint(np.zeros(4), bare), footprint(np.zeros(4), params(1.35)))


@pytest.mark.parametrize('l_r', [1.35, 1.6])
def test_pursuit_converges_to_lane_centre(l_r):
    plant = KinematicPlant(replace(VEHICLE, cg_to_rear_axle=l_r), 0.05)
    route, centre = ROAD.lane_route(2), ROAD.lane_center(2)
    state, steer = np.array([0., centre+1.0, 0., 25.]), 0.0
    for _ in range(int(5/0.05)):
        a, delta = lane_keep_law(state, state[0], route, plant.params, 25.0)
        state, _, steer = plant.step(state, a, delta, steer)
    assert abs(state[1]-centre) < 0.05 and abs(state[2]) < 0.01 and state[3] == pytest.approx(25.0, abs=0.1)


def test_limits_and_stop():
    plant = KinematicPlant(VEHICLE, 0.1)
    _, a, _ = plant.step(np.array([0., 0, 0, 40]), 10, 0, 0)
    assert a == pytest.approx(1.5)                                    # power limit
    _, a, _ = plant.step(np.array([0., 0, 0, 10]), -20, 0, 0)
    assert a == -VEHICLE.brake_max
    _, _, delta = plant.step(np.array([0., 0, 0, 10]), 0, 0.5, 0)
    assert delta == pytest.approx(VEHICLE.steer_rate*0.1)             # rate limit
    cap = plant.steer_limit(30)
    assert plant.step(np.array([0., 0, 0, 30]), 0, 0.5, cap)[2] == pytest.approx(cap)   # grip cap
    assert plant.step(np.array([0., 0, 0, 30]), 0, 0.5, 0.5)[2] == pytest.approx(0.44)  # cap, then rate from previous
    state, a, _ = plant.step(np.array([0., 0, 0, 0.3]), -6, 0, 0)     # stops after 0.05 s: no reverse travel
    np.testing.assert_allclose(state, [0.3**2/12, 0, 0, 0], atol=1e-12)
