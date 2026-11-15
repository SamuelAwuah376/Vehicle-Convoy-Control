"""Closed-loop tests of the yard path tracker (nohesi/yard/mpc.py) on the TruckPlant: dt = Planning.dt (0.1 s), a
command every Planning.mpc_step (0.2 s), held in between."""
import numpy as np
import osqp
import pytest
from nohesi.yard import mpc
from nohesi.yard.config import Planning, Truck
from nohesi.yard.mpc import PathTracker
from nohesi.yard.paths import Path
from nohesi.yard.truck import TruckPlant, wrap

TRUCK, PLAN = Truck(), Planning()
K_MAX = 1/TRUCK.min_radius        # full-lock curvature


def piece(start, kappa, length, direction, ds=0.25):
    """An arc (kappa != 0) or a line from start (x, y, yaw), driven forward (+1) or in reverse (-1), sampled ds apart.
    The heading turns by kappa per metre of signed travel, so a reverse arc with kappa > 0 turns clockwise."""
    n = max(int(np.ceil(length/ds)), 1)
    sig = direction*np.linspace(0.0, length, n+1)
    x0, y0, p0 = start
    if abs(kappa) < 1e-12:
        x, y, yaw = x0+sig*np.cos(p0), y0+sig*np.sin(p0), np.full_like(sig, p0)
    else:
        yaw = p0+kappa*sig
        x, y = x0+(np.sin(yaw)-np.sin(p0))/kappa, y0-(np.cos(yaw)-np.cos(p0))/kappa
    return Path(x, y, wrap(yaw), np.full(n+1, direction))


def chain(start, specs):
    """Path.concat of pieces [(kappa, length, direction), ...] joined end to start."""
    parts = []
    for kappa, length, direction in specs:
        parts.append(piece(start, kappa, length, direction))
        start = (parts[-1].x[-1], parts[-1].y[-1], parts[-1].yaw[-1])
    return Path.concat(parts)


def distance_to(path, x, y):
    """Distance from (x, y) to the path's polyline."""
    ax, ay, ex, ey = path.x[:-1], path.y[:-1], np.diff(path.x), np.diff(path.y)
    u = np.clip(((x-ax)*ex+(y-ay)*ey)/np.maximum(ex*ex+ey*ey, 1e-12), 0.0, 1.0)
    return float(np.min(np.hypot(ax+u*ex-x, ay+u*ey-y)))


def simulate(path, duration=80.0, speed_limit=None, stop_index=None, state0=None, settle=1.0):
    """Closed loop until `settle` s after arrival (or `duration`). speed_limit / stop_index: constants or functions
    of time. Returns the tracker and a record of arrays per plant step (and per command: info, gear, accel_cmd)."""
    tracker, plant = PathTracker(TRUCK, PLAN), TruckPlant(TRUCK, PLAN.dt)
    tracker.set_path(path, 0)
    every = int(round(PLAN.mpc_step/PLAN.dt))
    state = np.array(state0 if state0 is not None else [path.x[0], path.y[0], path.yaw[0], 0.0], float)
    steer = accel = 0.0
    rec = {k: [] for k in ('t', 'state', 'lat', 'status', 'segment', 'progress', 'gear', 'stop')}
    after = None
    for i in range(int(round(duration/PLAN.dt))):
        t = i*PLAN.dt
        if i % every == 0:
            limit = speed_limit(t) if callable(speed_limit) else (TRUCK.v_forward_max if speed_limit is None
                                                                   else speed_limit)
            stop = stop_index(t) if callable(stop_index) else stop_index
            steer_cmd, accel_cmd, gear, info = tracker.command(state, steer, accel, limit, stop)
        state, steer, accel = plant.step(state, steer, accel, steer_cmd, accel_cmd, gear)
        for k, value in (('t', t+PLAN.dt), ('state', state), ('lat', distance_to(path, state[0], state[1])),
                         ('status', info['status']), ('segment', info['segment']), ('progress', info['progress']),
                         ('gear', gear), ('stop', stop)):
            rec[k].append(value)
        if tracker.arrived:
            after = i if after is None else after
            if (i-after)*PLAN.dt >= settle:
                break
    rec = {k: np.array(v) for k, v in rec.items()}
    return tracker, rec


def end_errors(path, rec):
    x, y, yaw, v = rec['state'][-1]
    return float(np.hypot(x-path.x[-1], y-path.y[-1])), float(abs(wrap(yaw-path.yaw[-1]))), float(v)


def assert_parked(tracker, path, rec):
    pos, yaw, v = end_errors(path, rec)
    assert tracker.arrived and rec['status'][-1] == 'arrived'
    assert pos <= PLAN.park_tolerance and yaw <= 0.1 and abs(v) < 0.05, (pos, yaw, v)
    return pos, yaw


def test_forward_straight_parks_within_tolerance():
    path = chain((2.0, -1.0, 0.3), [(0.0, 30.0, 1)])
    tracker, rec = simulate(path)
    pos, yaw = assert_parked(tracker, path, rec)
    v = rec['state'][:, 3]
    assert v.max() <= TRUCK.v_forward_max and v.min() >= 0.0
    assert v.max() > 0.9*TRUCK.v_forward_max            # it does use the yard speed
    assert rec['lat'].max() < 0.05 and pos < 0.1 and yaw < 0.01
    assert set(rec['gear']) == {1} and 'fallback' not in set(rec['status'])


def test_reverse_straight_parks_within_tolerance():
    path = chain((0.0, 0.0, -2.0), [(0.0, 20.0, -1)])
    tracker, rec = simulate(path)
    pos, yaw = assert_parked(tracker, path, rec)
    v = rec['state'][:, 3]
    assert v.max() <= 0.0 and v.min() >= -TRUCK.v_reverse_max
    assert v.min() < -0.9*TRUCK.v_reverse_max
    assert rec['lat'].max() < 0.05 and pos < 0.1 and yaw < 0.01
    assert set(rec['gear']) == {-1}


def test_cusp_stops_shifts_and_reverses_to_the_spot():
    """Forward 15 m, cusp, reverse 10 m along an arc."""
    path = chain((0.0, 0.0, 0.0), [(0.0, 15.0, 1), (1/12, 10.0, -1)])
    cusp = int(path.cusps[0])
    tracker, rec = simulate(path)
    pos, yaw = assert_parked(tracker, path, rec)
    seg, v, xy = rec['segment'], rec['state'][:, 3], rec['state'][:, :2]
    shift = int(np.argmax(seg == 1))
    assert shift > 0 and np.all(seg[:shift] == 0) and np.all(seg[shift:] == 1)
    assert abs(v[shift]) < 0.1                                  # stopped at the cusp ...
    assert np.hypot(*(xy[shift]-(path.x[cusp], path.y[cusp]))) < mpc.CUSP_REACH
    near = np.hypot(xy[:, 0]-path.x[cusp], xy[:, 1]-path.y[cusp]).argmin()
    assert abs(v[near]) < 0.1
    assert v[:shift].min() >= 0.0 and v[shift:].max() <= 0.0   # ... forward before, reverse after
    assert np.all(rec['gear'][:shift] == 1) and np.all(rec['gear'][shift+2:] == -1)
    assert 'cusp' in set(rec['status']) and v.min() >= -TRUCK.v_reverse_max
    assert rec['lat'].max() < 0.2


@pytest.mark.parametrize('arc, direction', [(8.0, 1), (16.0, 1), (8.0, -1)])
def test_s_curve_at_the_turning_radius_limit(arc, direction):
    """Full lock left then full lock right (a lock-to-lock switch takes 3.8 s of steering)."""
    path = chain((0.0, 0.0, 0.0), [(0.0, 5.0, direction), (K_MAX, arc, direction), (-K_MAX, arc, direction),
                                   (0.0, 10.0, direction)])
    tracker, rec = simulate(path, duration=120.0)
    assert_parked(tracker, path, rec)
    assert rec['lat'].max() < 0.5
    assert 'off_path' not in set(rec['status']) and 'fallback' not in set(rec['status'])


def test_stop_index_holds_short_and_resumes():
    path = chain((0.0, 0.0, 0.0), [(0.0, 30.0, 1)])
    k = 60                                                      # s = 15 m
    release = 25.0
    tracker, rec = simulate(path, stop_index=lambda t: k if t < release else None)
    held = rec['t'] <= release
    x, v = rec['state'][:, 0], rec['state'][:, 3]
    assert x[held].max() < path.x[k]                            # the rear axle never reaches the stop sample
    assert x[held].max() > path.x[k]-1.0                        # ... and does not stop far short of it
    assert abs(v[held][-1]) < 0.05 and rec['status'][held][-1] == 'holding'
    assert rec['progress'][held].max() <= k
    assert_parked(tracker, path, rec)                           # resumes once cleared
    assert rec['t'][-1] > release+5.0


def test_stop_index_at_a_cusp_prevents_the_shift():
    path = chain((0.0, 0.0, 0.0), [(0.0, 15.0, 1), (1/12, 10.0, -1)])
    cusp = int(path.cusps[0])
    release = 20.0
    tracker, rec = simulate(path, stop_index=lambda t: cusp if t < release else None)
    held = rec['t'] <= release
    assert np.all(rec['segment'][held] == 0) and np.all(rec['gear'][held] == 1)
    assert rec['state'][held, 0].max() < path.x[cusp]
    assert_parked(tracker, path, rec)


def test_speed_limit_zero_stops_at_brake_comfort_and_resumes():
    path = chain((0.0, 0.0, 0.0), [(0.0, 40.0, 1)])
    tracker, rec = simulate(path, speed_limit=lambda t: 0.0 if 6.0 <= t < 14.0 else TRUCK.v_forward_max)
    t, v = rec['t'], rec['state'][:, 3]
    i0 = int(np.argmax(t >= 6.0))
    stopped = int(np.argmax((t > 6.0) & (v < 0.01)))
    assert t[stopped] < 14.0
    mean_decel = (v[i0]-v[stopped])/(t[stopped]-t[i0])
    assert 0.8*TRUCK.brake_comfort < mean_decel < 1.2*TRUCK.brake_comfort
    assert set(rec['status'][(t > 7.0) & (t < 14.0)]) == {'holding'}
    assert_parked(tracker, path, rec)


class _Failed:
    """An OSQP result whose solve failed (status primal infeasible, no solution)."""
    class info:
        status_val, iter = 3, 0
    x = None


class _NotFinite:
    """An OSQP result reported solved but with non-finite values."""
    class info:
        status_val, iter = 1, 10
    x = np.full(10_000, np.nan)


@pytest.mark.parametrize('result', [_Failed, _NotFinite])
def test_pure_pursuit_fallback_tracks_forward_and_reverse(monkeypatch, result):
    monkeypatch.setattr(osqp.OSQP, 'solve', lambda self, *a, **k: result())
    path = chain((0.0, 0.0, 0.0), [(0.0, 10.0, 1), (0.5*K_MAX, 8.0, 1), (0.0, 4.0, 1), (1/12, 10.0, -1)])
    tracker, rec = simulate(path, duration=120.0)
    statuses = set(rec['status'][:-1])
    assert statuses <= {'fallback', 'arrived'} and 'fallback' in statuses
    assert tracker.stats['fallbacks'] > 50
    assert_parked(tracker, path, rec)
    assert rec['lat'].max() < 0.5
    v = rec['state'][:, 3]
    assert v.max() <= TRUCK.v_forward_max and v.min() >= -TRUCK.v_reverse_max


def test_off_path_commands_a_stop():
    tracker = PathTracker(TRUCK, PLAN)
    tracker.set_path(chain((0.0, 0.0, 0.0), [(0.0, 30.0, 1)]))
    steer_cmd, accel_cmd, gear, info = tracker.command(np.array([3.0, 2.0, 0.0, 2.0]), 0.1, 0.0, 3.0)
    assert info['status'] == 'off_path' and info['lateral_error'] == pytest.approx(2.0)
    assert accel_cmd < 0 and gear == 1 and steer_cmd == 0.1
    assert set(info) == {'progress', 'segment', 'lateral_error', 'heading_error', 'status'}


def test_progress_is_monotone_and_solves_are_cheap():
    path = chain((0.0, 0.0, 0.0), [(0.0, 15.0, 1), (1/12, 10.0, -1)])
    tracker, rec = simulate(path)
    segments = path.segments()
    assert np.all(np.diff(rec['segment']) >= 0)
    for n, (a, b) in enumerate(segments):
        p = rec['progress'][rec['segment'] == n]
        assert np.all(np.diff(p) >= 0) and p.min() >= a and p.max() <= b
    assert tracker.stats['fallbacks'] == 0
    assert tracker.stats['solve_time']/tracker.stats['solves'] < 5e-3
