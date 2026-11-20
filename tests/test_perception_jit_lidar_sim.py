"""The compiled lidar sweep (sim/_lidar_sim_jit.py) against cast_sweep's numpy path, TruthHistory.at against its
original per-car loop, and the truth shared between the sweeps of one firing (sim.lidar._sweep_truth)."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.perception import _jit
from nohesi.perception.types import TruthHistory, TruthState
from nohesi.perception.sim import lidar as SL
from test_perception_lidar_sim import EXACT, EXACT_REAR, P, history


def scene(rng, n_cars=45, inside=False, recycle=False):
    """A random v9-like history: the team at about 30 m/s with yaw rates, traffic in three lanes with lane changes,
    optionally a car on top of car 0's front lidar and a car recycled under a new id mid-history."""
    s0, lane, v = rng.uniform(-150, 300, n_cars), rng.integers(0, 3, n_cars), rng.uniform(24.6, 35.8, n_cars)
    dd, hd = rng.normal(0.0, 0.6, n_cars), rng.normal(0.0, 0.03, n_cars)
    yr = rng.normal(0.0, 0.1, 3)
    x0 = np.array([0.0, -12.0, -24.0])+rng.normal(0.0, 2.0, 3)

    def team(t):
        return [[x0[k]+30*t, 3.7*(k % 2), yr[k]*t, 30.0] for k in range(3)]

    def traffic(t):
        ids = np.arange(n_cars)
        if recycle and t >= 0.3:
            ids = ids.copy()
            ids[3] = 900
        rows = np.column_stack((ids, s0+v*t, 3.7*lane+dd*t, hd, v))
        if inside:
            rows[0, 1:4] = [x0[0]+30*t+1.5, 0.3, 0.2]
        return rows
    return history(team, traffic, times=np.round(np.arange(0.0, 0.501, 0.05), 3))


@pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')
def test_cast_kernel_matches_numpy_path(monkeypatch):
    rng = np.random.default_rng(3)
    beams = hits = 0
    for k in range(12):
        truth = scene(rng, inside=k % 4 == 1, recycle=k % 3 == 2)
        for t_end in (0.2, 0.33, 0.45, 0.6):                     # snapshot-aligned, inside a step, clamped at the end
            for spec in (EXACT, EXACT_REAR):
                for car in range(3):
                    monkeypatch.setattr(_jit, 'ENABLED', True)
                    az, tm, r1, l1 = SL.cast_sweep(spec, car, truth, t_end, P)
                    monkeypatch.setattr(_jit, 'ENABLED', False)
                    _, _, r2, l2 = SL.cast_sweep(spec, car, truth, t_end, P)
                    assert np.array_equal(l1, l2)
                    assert np.array_equal(np.isfinite(r1), np.isfinite(r2))
                    f = np.isfinite(r1)
                    assert np.allclose(r1[f], r2[f], rtol=0, atol=1e-9)
                    beams += len(az)
                    hits += int(f.sum())
    assert hits > 0.1*beams


def test_cast_kernel_single_snapshot_and_no_snapshot_list(monkeypatch):
    rng = np.random.default_rng(5)
    truth = scene(rng)

    class AtOnly:                                   # no snapshot list: knots every CHUNK seconds, read with .at
        def __init__(self, h):
            self.at = h.at
    one = TruthHistory()
    one.push(truth.snaps[4][0])
    for tr in (AtOnly(truth), one):
        for enabled in (True, False):
            monkeypatch.setattr(_jit, 'ENABLED', enabled)
            out = SL.cast_sweep(EXACT, 1, tr, 0.37, P)
            if enabled:
                ref = out
        assert np.array_equal(ref[3], out[3])
        f = np.isfinite(ref[2])
        assert np.array_equal(f, np.isfinite(out[2])) and np.allclose(ref[2][f], out[2][f], rtol=0, atol=1e-9)


def reference_at(h, t):
    """TruthHistory.at as first written: _state per call, traffic matched by a dict per car."""
    snaps = h.snaps
    if t <= snaps[0][0].time:
        return TruthHistory._state(*snaps[0])
    if t >= snaps[-1][0].time:
        return TruthHistory._state(*snaps[-1])
    k = next(k for k in range(1, len(snaps)) if snaps[k][0].time >= t)
    a, b = TruthHistory._state(*snaps[k-1]), TruthHistory._state(*snaps[k])
    w = (t-a.time)/(b.time-a.time)
    traffic = b.traffic.copy()
    index = {int(i): n for n, i in enumerate(a.traffic_id)}
    for n, i in enumerate(b.traffic_id):
        m = index.get(int(i))
        if m is not None:
            traffic[n] = a.traffic[m]+w*(b.traffic[n]-a.traffic[m])
    team = a.team+w*(b.team-a.team)
    team[:, 2] = a.team[:, 2]+w*((b.team[:, 2]-a.team[:, 2]+np.pi) % (2*np.pi)-np.pi)
    lerp = lambda p, q: p+w*(q-p)
    return TruthState(t, traffic, b.traffic_id, team, lerp(a.team_vy, b.team_vy), lerp(a.team_r, b.team_r),
                      lerp(a.team_ax, b.team_ax))


def test_truth_at_matches_reference_and_is_read_only():
    rng = np.random.default_rng(9)
    truth = scene(rng, recycle=True)
    for t in np.r_[-0.1, 0.0, rng.uniform(0.0, 0.5, 60), np.round(np.arange(0.0, 0.501, 0.05), 3), 0.5, 0.7]:
        got, want = truth.at(float(t)), reference_at(truth, float(t))
        for f in ('time', 'traffic', 'traffic_id', 'team', 'team_vy', 'team_r', 'team_ax'):
            assert np.array_equal(getattr(got, f), getattr(want, f)), (t, f)
    end = truth.at(1.0)
    with pytest.raises(ValueError):
        end.traffic[0, 0] = 0.0
    assert truth.snaps[-1][0].team.flags.writeable          # the snapshot's own arrays are left alone


def test_sweeps_of_one_firing_share_the_truth():
    rng = np.random.default_rng(11)
    truth = scene(rng)
    SL._SHARED[:] = None, None
    shared = [SL.cast_sweep(spec, car, truth, 0.4, P) for car in range(3) for spec in (EXACT, EXACT_REAR)]
    first = SL._SHARED[1]
    assert SL._sweep_truth(truth, 0.4-1/EXACT.rate, 0.4, SL.CHUNK, P) is first
    for (car, spec), out in zip([(c, s) for c in range(3) for s in (EXACT, EXACT_REAR)], shared):
        SL._SHARED[:] = None, None                           # each sweep on its own
        alone = SL.cast_sweep(spec, car, truth, 0.4, P)
        assert all(np.array_equal(a, b) for a, b in zip(out, alone))
    nxt = scene(np.random.default_rng(11))                    # the same scene in a new history: not reused
    SL.cast_sweep(EXACT, 0, nxt, 0.4, P)
    assert SL._SHARED[1] is not first
    before = SL._sweep_truth(nxt, 0.3, 0.4, SL.CHUNK, P)
    nxt.push(replace(nxt.snaps[-1][0], time=0.55))           # a new snapshot: recomputed
    after = SL._sweep_truth(nxt, 0.3, 0.4, SL.CHUNK, P)
    assert after is not before and all(np.array_equal(a, b) for a, b in zip(after, before))
    with pytest.raises(ValueError):
        after[4][0, 0, 0, 0] = 0.0                            # read-only: shared by the sweeps
