"""Dynamic bicycle plant (Fiala tyres, kinematic blend at low speed) and the kinematic MPC driving it."""
from dataclasses import replace
import math
import numpy as np
import pytest
from nohesi.config import Planner, Scenario, Vehicle
from nohesi.dynamics import DynamicPlant, KinematicPlant, make_plant
from nohesi.geometry import footprint
from nohesi.planning.base import Obstacles, Target, constant_speed_plan
from nohesi.planning.single import LaneChangeMPC
from nohesi.rl.scripted import lane_keep_law
from nohesi.road import Road
from nohesi.world import World

DYN = Vehicle(model='dynamic')


def lateral_accel(plant, st, a, steer):
    """Inertial CG acceleration normal to the velocity, from the plant derivatives at [x, y, yaw, v, vy, r]."""
    vx = math.sqrt(st[3]**2-st[4]**2)
    d = plant.derivatives((*st[:3], vx, st[4], st[5]), a, steer)
    ax, ay = d[3]-st[5]*st[4], d[4]+st[5]*vx
    return (ay*vx-ax*st[4])/st[3]


def hold(plant, v, steer_cmd, steps, accel=0.0):
    st, steer, ay = np.array([0., 0, 0, v, 0, 0]), 0.0, []
    for _ in range(steps):
        st, a, steer = plant.step(st, accel, steer_cmd, steer)
        ay.append(lateral_accel(plant, st, a, steer))
    return st, steer, np.array(ay)


def test_config_and_factory():
    assert type(make_plant(Vehicle(), 0.05)) is KinematicPlant
    assert type(make_plant(DYN, 0.05)) is DynamicPlant
    with pytest.raises(ValueError):
        Vehicle(model='rigid')
    with pytest.raises(ValueError):
        Vehicle(mass=0.0)
    p = DynamicPlant(DYN, 0.05)
    assert p.substeps == 5 and p.steer_limit(30) == KinematicPlant(DYN, 0.05).steer_limit(30)


def test_low_speed_matches_kinematic():
    kin, dyn = KinematicPlant(Vehicle(), 0.05), DynamicPlant(DYN, 0.05)
    a, b, sa, sb = np.array([0., 0, 0, 5]), np.array([0., 0, 0, 5, 0, 0]), 0.0, 0.0
    for k in range(100):
        cmd = 0.1*math.sin(0.05*k)
        a, _, sa = kin.step(a, 0.0, cmd, sa)
        b, _, sb = dyn.step(b, 0.0, cmd, sb)
    assert np.linalg.norm(a[:2]-b[:2]) < 0.1 and abs(a[2]-b[2]) < 0.01 and abs(a[3]-b[3]) < 0.02
    # Below dynamic_blend_low the plant is exactly kinematic.
    kin_state = kin.step(np.array([0., 0, 0, 2]), 0.5, 0.2, 0.2)[0]
    np.testing.assert_allclose(dyn.step(np.array([0., 0, 0, 2]), 0.5, 0.2, 0.2)[0], kin_state, atol=1e-6)


@pytest.mark.parametrize('mu', [1.0, 0.5])
def test_lateral_accel_saturates_near_mu_g(mu):
    plant = DynamicPlant(replace(DYN, friction=mu), 0.05)
    _, _, ay = hold(plant, 30.0, 0.1, 60)
    # The kinematic bicycle would ask for v^2 tan(0.1)/L ~ 33 m/s^2.
    assert 0.85*mu*9.81 < np.abs(ay).max() <= mu*9.81*1.01


@pytest.mark.parametrize('l_r, c_f, c_r', [(1.35, 8e4, 8e4), (1.5, 8e4, 8e4), (1.2, 6e4, 9e4)])
def test_linear_steady_state_yaw_rate(l_r, c_f, c_r):
    v = replace(DYN, cg_to_rear_axle=l_r, cornering_stiffness_front=c_f, cornering_stiffness_rear=c_r)
    L, l_f = v.wheelbase, v.wheelbase-l_r
    k_us = v.mass*(l_r*c_r-l_f*c_f)/(L*c_f*c_r)   # > 0 understeer (CG ahead / stiffer rear)
    st, steer, _ = hold(DynamicPlant(v, 0.05), 25.0, 0.004, 200)
    vx = math.sqrt(st[3]**2-st[4]**2)
    assert st[5] == pytest.approx(vx*steer/(L+k_us*vx**2), rel=0.02)


def test_no_reverse_and_determinism():
    plant = DynamicPlant(DYN, 0.05)
    st, steer = np.array([0., 0, 0, 8, 0, 0]), 0.0
    for _ in range(60):
        st, a, steer = plant.step(st, -6.0, 0.05, steer)
    assert st[3] == 0 and st[4] == 0 and st[5] == 0
    x = plant.step(st, -6.0, 0.05, steer)[0]
    np.testing.assert_array_equal(x, st)
    runs = [hold(DynamicPlant(DYN, 0.05), 30.0, 0.05, 80)[0] for _ in range(2)]
    np.testing.assert_array_equal(*runs)


def test_world_uses_dynamic_plant():
    sc = Scenario()
    world = World(replace(sc, vehicle=DYN), 0)
    assert isinstance(world.plant, DynamicPlant)
    snap, _ = world.step(np.array([[0.0, 0.02]]*3))
    assert snap.team.shape == (3, 4) and snap.team_vy.shape == (3,) and np.all(snap.team_r != 0)
    kin = World(sc, 0).step(np.array([[0.0, 0.0]]*3))[0]
    assert np.all(kin.team_vy == 0) and np.all(kin.team_r == 0)


LAG_XFAIL = pytest.mark.xfail(strict=True, reason='known limitation: the kinematic MPC saturates its steer cap and '
                              'ignores the ~0.25 s yaw/sideslip lag of the default (soft, 80 kN/rad per axle) tyres: '
                              'heading overshoots, giving a +-3 m limit cycle (inner lane) or an edge excursion (outer)')
STIFF = replace(DYN, cornering_stiffness_front=150000.0, cornering_stiffness_rear=150000.0)


# With the 60/40 front-biased default CG (understeer) the inner-lane change converges; the outer one still fails.
@pytest.mark.parametrize('vehicle, lane0, lane1', [(DYN, 0, 1),
                                                   pytest.param(DYN, 1, 2, marks=LAG_XFAIL), (STIFF, 0, 1)])
def test_kinematic_mpc_drives_dynamic_plant_lane_change(vehicle, lane0, lane1):
    """Closed loop on an empty road (as tests/test_nohesi_m2.drive): plan with the kinematic MPC every 0.1 s and
    execute on the dynamic plant; when the QP fails, lane-keep toward the target lane for that 0.1 s (as
    CentralizedTeam falls back). Must stay on the road and settle in the target lane; next-step plan error is
    larger than the kinematic ~0.02 m but sane."""
    sc, cfg = Scenario(), Planner()
    road, plant = Road(sc.road), DynamicPlant(vehicle, sc.timing.dt)
    mpc, route = LaneChangeMPC(cfg, plant, road), road.lane_route(lane1)
    d0, d1 = road.lane_center(lane0), road.lane_center(lane1)
    state, steer, accel, plan, target = np.array([0., d0, 0, 28, 0, 0]), 0.0, 0.0, None, Target(d=d1, speed=32.0)
    obs = Obstacles((), np.zeros((cfg.horizon+1, 0)), np.zeros((cfg.horizon+1, 0)), np.zeros((cfg.horizon+1, 0)))
    errors, due, ay, failures = [], None, [], 0
    for k in range(int(round(15.0/sc.timing.dt))):
        t = k*sc.timing.dt
        if due is not None and k == due[0]:
            errors.append(np.abs(state[:2]-due[1][:2]))
        if k % 2 == 0:
            fresh = plan is None or due is None
            ref = constant_speed_plan(t, cfg.step, cfg.horizon, *state[[0, 1, 3]]) if fresh else plan.shifted(1)
            new = mpc.solve(t, state[:4].copy(), steer, accel, target, obs, ref)
            if new.status == 'solved':
                plan, due = new, (k+2, new.z[1])
            else:
                failures, due = failures+1, None
        u = plan.u[0] if due is not None else lane_keep_law(state, state[0], route, plant.params, target.speed)
        state, accel, steer = plant.step(state, u[0], u[1], steer)
        ay.append(lateral_accel(plant, state, accel, steer))
        assert np.abs(road.to_frenet(footprint(state[:4], plant.params))[1]).max() < road.width/2, t
    errors = np.array(errors)
    print(f'QP failures {failures}/150; next-step error s mean/max {errors[:, 0].mean():.3f}/{errors[:, 0].max():.3f} '
          f'd mean/max {errors[:, 1].mean():.3f}/{errors[:, 1].max():.3f}  peak |ay| {np.abs(ay).max():.2f}')
    assert abs(state[1]-d1) < 0.1 and abs(state[2]) < 0.02 and abs(state[3]-32.0) < 0.5
    assert errors[:, 1].max() < 0.3 and errors[:, 0].max() < 0.5
