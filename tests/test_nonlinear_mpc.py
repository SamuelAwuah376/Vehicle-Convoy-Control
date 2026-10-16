"""Nonlinear MPC (acados SQP-RTI, dynamic bicycle) driving the dynamic plant. Skipped without acados."""
from dataclasses import replace
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
import pytest

nl = pytest.importorskip('nohesi.planning.nonlinear', exc_type=ImportError)

from nohesi.config import Planner, Scenario, Vehicle   # noqa: E402
from nohesi.coordination.centralized import make_mpc   # noqa: E402
from nohesi.dynamics import DynamicPlant   # noqa: E402
from nohesi.geometry import clearance, footprint   # noqa: E402
from nohesi.planning.base import Obstacles, Target, constant_speed_plan   # noqa: E402
from nohesi.planning.constraints import obstacle_constraints   # noqa: E402
from nohesi.road import Road   # noqa: E402

SC, CFG = Scenario(), Planner(mpc='nonlinear')
ROAD, PLANT = Road(SC.road), DynamicPlant(Vehicle(model='dynamic'), SC.timing.dt)
P, N = PLANT.params, CFG.horizon


@pytest.fixture(scope='module')
def mpc():
    return nl.NonlinearMPC(CFG, PLANT, ROAD)


def obstacles(cars, t):
    """Constant-speed cars (s0, d, v) at time t over the horizon."""
    cars = np.array(cars, float).reshape(-1, 3)
    times = t+np.arange(N+1)[:, None]*CFG.step
    return Obstacles(tuple(('traffic', j) for j in range(len(cars))), cars[:, 0]+cars[:, 2]*times,
                     np.repeat(cars[None, :, 1], N+1, 0), np.repeat(cars[None, :, 2], N+1, 0))


def drive(mpc, target, state, cars=(), seconds=12.0):
    """Closed loop (as tests/test_dynamic_plant.py): plan every 0.1 s, execute plan.u[0] on the dynamic plant."""
    mpc.cache = []
    state, steer, accel, plan, due = np.array(state, float), 0.0, 0.0, None, None
    out = {'state': [], 'errors': [], 'clearance': [], 'status': [], 'ms': []}
    for k in range(int(round(seconds/SC.timing.dt))):
        t = k*SC.timing.dt
        if due is not None and k == due[0]:
            out['errors'].append(np.abs(state[:2]-due[1][:2]))
        if k % 2 == 0:
            ref = constant_speed_plan(t, CFG.step, N, *state[[0, 1, 3]]) if plan is None else plan.shifted(1)
            plan = mpc.solve(t, state[:4].copy(), steer, accel, target, obstacles(cars, t), ref, tuple(state[4:6]))
            out['status'].append(plan.status); out['ms'].append(1e3*plan.solve_time)
            assert plan.status == 'solved', (t, plan.status)
            due = (k+2, plan.z[1])
        state, accel, steer = PLANT.step(state, plan.u[0, 0], plan.u[0, 1], steer)
        out['state'].append(state.copy())
        ego = footprint(state[:4], P)
        out['clearance'].append(min([clearance(ego, footprint(np.array([s+v*(t+SC.timing.dt), d, 0, 0]), P))
                                     for s, d, v in cars], default=np.inf))
        assert np.abs(ROAD.to_frenet(ego)[1]).max() < ROAD.width/2, t
    out['state'], out['errors'] = np.array(out['state']), np.array(out['errors'])
    return out


def test_model_matches_plant_one_step(mpc):
    """One 0.1 s RK4 step of the NMPC's CasADi dynamics vs DynamicPlant.step (dt 0.1, midpoint 0.01 s substeps) at
    constant steer and accel in the linear tyre range."""
    plant = DynamicPlant(Vehicle(model='dynamic'), 0.1)
    for vx, vy, r, steer, a in ((25.0, 0.2, 0.05, 0.02, 0.5), (30.0, -0.3, -0.1, -0.03, -2.0), (20.0, 0.0, 0.0, 0.04, 1.5)):
        x = np.array([0.0, 1.0, 0.03, vx, vy, r, steer, a])
        model = np.asarray(mpc.rk4(x, np.zeros(2))).ravel()
        st = plant.step(np.array([0.0, 1.0, 0.03, math.hypot(vx, vy), vy, r]), a, steer, steer)[0]
        truth = np.array([st[0], st[1], st[2], math.sqrt(st[3]**2-st[4]**2), st[4], st[5]])
        np.testing.assert_allclose(model[:6], truth, atol=2e-3)


def test_factory_and_lazy_import():
    assert type(make_mpc(CFG, PLANT, ROAD)).__name__ == 'NonlinearMPC'
    code = ("import sys; from nohesi.config import Planner, Scenario; from nohesi.road import Road; "
            "from nohesi.dynamics import make_plant; from nohesi.coordination.centralized import make_mpc; "
            "sc = Scenario(); make_mpc(Planner(), make_plant(sc.vehicle, 0.05), Road(sc.road)); "
            "assert 'acados_template' not in sys.modules and 'casadi' not in sys.modules")
    subprocess.run([sys.executable, '-c', code], check=True, cwd=Path(__file__).parents[1])


@pytest.mark.parametrize('lane0, lane1', [(0, 1), (1, 2)])
def test_lane_change_on_dynamic_plant(mpc, lane0, lane1):
    """Inner and outer-lane changes (the kinematic MPC's outer change is a known failure, test_dynamic_plant.py):
    complete, stay on the road, settle in the target lane; next-step plan error stays tiny."""
    d1 = ROAD.lane_center(lane1)
    out = drive(mpc, Target(d=d1, speed=32.0), [0., ROAD.lane_center(lane0), 0, 28, 0, 0])
    st, e = out['state'], out['errors']
    print(f"solve ms mean/p95 {np.mean(out['ms']):.2f}/{np.percentile(out['ms'], 95):.2f}; next-step error "
          f"s max {e[:, 0].max():.4f} d max {e[:, 1].max():.4f}; overshoot {np.max(st[:, 1]-d1):.3f}")
    assert abs(st[-1, 1]-d1) < 0.05 and abs(st[-1, 2]) < 0.01 and abs(st[-1, 3]-32.0) < 0.3
    assert np.max(st[:, 1]-d1) < 0.3
    assert e[:, 1].max() < 0.05 and e[:, 0].max() < 0.05


def test_following_slower_car_keeps_hard_margin(mpc):
    out = drive(mpc, Target(d=0.0, speed=32.0), [0., 0, 0, 28, 0, 0], cars=[(25.0, 0.0, 20.0)], seconds=10.0)
    assert min(out['clearance']) >= CFG.margin_long-0.05
    assert abs(out['state'][-1, 3]-20.0) < 0.5 and abs(out['state'][-1, 1]) < 0.05


def test_more_obstacle_rows_than_slots(mpc):
    """24 cars beside the ego (lanes 0 and 2, within the relevance window): more hard rows per stage than slots.
    The nearest rows are kept, the rest counted, and the plan keeps its lane."""
    cars = [(s, d, 28.0) for s in np.linspace(-55, 55, 12) for d in (-3.6, 3.6)]
    obs, ref = obstacles(cars, 0.0), constant_speed_plan(0.0, CFG.step, N, 0.0, 0.0, 28.0)
    spec = obstacle_constraints(CFG, P, ROAD, ref.z, obs, Target(d=0.0, speed=28.0))
    hard, _ = spec.per_stage(N)
    assert hard.max() > CFG.nmpc_hard_slots
    mpc.cache, before = [], mpc.overflow['hard']
    plan = mpc.solve(0.0, np.array([0.0, 0, 0, 28]), 0.0, 0.0, Target(d=0.0, speed=28.0), obs, ref, (0.0, 0.0))
    assert plan.status == 'solved'
    assert mpc.overflow['hard']-before == np.maximum(hard-CFG.nmpc_hard_slots, 0).sum()
    assert np.abs(plan.z[:, 1]).max() < 0.05
    # Stage 1 keeps the 14 nearest cars (|ds| <= 35 m of +-5, ..., +-55): every hard slot used, s bounds within 40 m.
    C = mpc.solver.constraints_get(1, 'C')[:CFG.nmpc_hard_slots]
    lo, hi = (mpc.solver.constraints_get(1, f)[:CFG.nmpc_hard_slots] for f in ('lg', 'ug'))
    assert np.all(np.abs(C).sum(axis=1) > 0)
    on_s = C[:, 0] != 0
    bounds = np.where(np.abs(lo) < 1e9, lo, hi)[on_s]
    assert on_s.any() and np.all(np.abs(bounds) < 40.0)


def test_build_reused(mpc):
    builds, start = nl.BUILDS, perf_counter()
    again = nl.NonlinearMPC(CFG, PLANT, ROAD)
    assert not again.compiled and nl.BUILDS == builds and perf_counter()-start < 5.0
    assert (nl.build_root()/again.hash/'ready').exists()


def test_concurrent_build_compiles_once():
    """Three processes constructing the same new configuration: one compiles under the lock, the others load it."""
    root = Path('C:/dev/acados_build') if os.name == 'nt' else Path('/tmp/acados_build')
    root = root/f'_test_lock_{os.getpid()}'
    code = ("import sys; from nohesi.config import Planner, Vehicle, Scenario; from nohesi.road import Road; "
            "from nohesi.dynamics import DynamicPlant; from nohesi.planning.nonlinear import NonlinearMPC; "
            "m = NonlinearMPC(Planner(mpc='nonlinear', horizon=10), DynamicPlant(Vehicle(model='dynamic'), 0.05), "
            "Road(Scenario().road)); print('COMPILED', m.compiled)")
    env = dict(os.environ, NOHESI_ACADOS_BUILD=str(root))
    try:
        procs = [subprocess.Popen([sys.executable, '-c', code], cwd=Path(__file__).parents[1], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(3)]
        outs = [p.communicate(timeout=600)[0] for p in procs]
        assert all(p.returncode == 0 for p in procs), outs
        assert sum('COMPILED True' in o for o in outs) == 1 and sum('COMPILED False' in o for o in outs) == 2
    finally:
        shutil.rmtree(root, ignore_errors=True)
