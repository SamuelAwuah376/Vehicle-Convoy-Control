import importlib.util
import json
import subprocess
import sys
import numpy as np
import pytest
from convoy.config import Config
from convoy.controllers import follower_acceleration
from convoy.dynamics import step
from convoy.geometry import footprint, overlap, safety_metrics
from convoy.route import Route
from convoy.simulation import simulate


def test_straight_line_and_exact_stop():
    cfg = Config(dt=0.1)
    state, a, delta = step(np.array([0., 0., 0., 5.]), 2, 0, 0, cfg)
    np.testing.assert_allclose(state, [0.51, 0, 0, 5.2])
    # Stop halfway through the step, integrating only until v=0.
    state, _, _ = step(np.array([0., 0., 0., 0.2]), -4, 0, 0, cfg)
    np.testing.assert_allclose(state, [0.005, 0, 0, 0])


def test_circular_motion_midpoint_accuracy():
    cfg = Config(dt=0.01)
    state = np.array([0., 0., 0., 4.])
    delta = 0.2
    for _ in range(100):
        state, _, _ = step(state, 0, delta, delta, cfg)
    omega = 4*np.tan(delta)/cfg.wheelbase
    radius = cfg.wheelbase/np.tan(delta)
    np.testing.assert_allclose(state[:3], [radius*np.sin(omega), radius*(1-np.cos(omega)), omega], atol=1e-5)


def test_control_limits_and_gap_equilibrium():
    cfg = Config()
    state, a, delta = step(np.array([0., 0., 0., 8.]), 100, 10, 0, cfg)
    assert a == cfg.accel_max
    assert delta == pytest.approx(cfg.steer_rate*cfg.dt)
    speed = np.array([0., 0., 0., 8.])
    assert follower_acceleration(speed, speed, cfg.standstill_gap+cfg.headway*8, cfg) == 0


def test_route_projection_and_geometry():
    route = Route(); cfg = Config()
    for s in [0, 5.123, 100.456, route.total]:
        xy, yaw = route.at(s)
        p, error = route.project(xy)
        assert p == pytest.approx(s, abs=1e-8)
        assert abs(error) < 1e-8
    a = footprint(np.array([0., 0., 0., 0.]), cfg)
    assert overlap(a, a)
    assert not overlap(a, footprint(np.array([10., 0., 0.2, 0.]), cfg))


@pytest.mark.parametrize('scenario,dt', [('nominal', 0.05), ('braking', 0.05), ('offset', 0.05), ('braking', 0.1)])
def test_end_to_end_acceptance(scenario, dt):
    result = simulate(Config(dt=dt), scenario)
    metrics = result.metrics() | safety_metrics(result, Route())
    assert np.all(np.isfinite(result.states))
    assert metrics['collision_pair_samples'] == 0
    assert metrics['lane_boundary_violation_m'] == 0
    assert metrics['max_abs_cross_track_m'] < 0.65
    assert metrics['min_bumper_gap_m'] > 3.0
    assert metrics['gap_error_rms_after_10s_m'] < 1.5
    assert np.max(result.states[-1, :, 3]) < 0.05
    assert abs(metrics['leader_distance_to_goal_m']) < 0.2
    assert np.min(np.diff(result.progress, axis=0)) > -1e-4
    assert np.max(result.controls[:, :, 0]) <= result.cfg.accel_max
    assert np.min(result.controls[:, :, 0]) >= -result.cfg.brake_max
    assert np.max(np.abs(result.controls[:, :, 1])) <= result.cfg.steer_max
    assert np.max(np.abs(np.diff(result.controls[:, :, 1], axis=0))) <= result.cfg.steer_rate*dt+1e-10
    if scenario == 'braking':
        during_braking = (result.time > 25) & (result.time < 27)
        assert np.max(result.states[during_braking, 0, 3]) < 2.1


@pytest.mark.skipif(importlib.util.find_spec('casadi') is None, reason='optional CasADi extra unavailable')
def test_mpc_closed_loop(tmp_path):
    # Native IPOPT Windows output handles conflict with pytest capture.
    # Isolate the native solver with ordinary file handles and a bounded timeout.
    script = '''
import json, sys
from convoy.config import Config
from convoy.simulation import simulate
from convoy.geometry import safety_metrics
from convoy.route import Route
from convoy.mpc import SteeringMPC
from unittest.mock import patch
r = simulate(Config(duration=12), 'offset', 'mpc')
m = safety_metrics(r, Route())
m['failures'] = r.mpc_failures
m['final_error'] = float(abs(r.lateral[-100:]).max())
c = SteeringMPC(Config())
state = r.states[0, 0]
progress = r.progress[0, 0]
with patch.object(c.opti, 'solve', side_effect=RuntimeError('injected failure')):
    assert c.command(state, progress, Route(), 0, 0.123) == 0.123
assert c.failures == 1
open(sys.argv[1], 'w').write(json.dumps(m))
'''
    path = tmp_path/'metrics.json'
    with (tmp_path/'solver.log').open('w') as log:
        subprocess.run([sys.executable, '-c', script, str(path)], stdout=log, stderr=log, check=True, timeout=90)
    metrics = json.loads(path.read_text())
    assert metrics['failures'] == 0
    assert metrics['collision_pair_samples'] == 0
    assert metrics['lane_boundary_violation_m'] == 0
    assert metrics['final_error'] < 0.1


@pytest.mark.parametrize('kwargs', [{'dt': 0}, {'headway': -1}, {'cruise': float('nan')}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)
