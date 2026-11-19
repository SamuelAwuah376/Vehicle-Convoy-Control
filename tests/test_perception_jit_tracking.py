"""The compiled team tracker (tracking/_tracking_jit.py) against its numpy path (NOHESI_JIT=0, perception/_jit.py).

The tracker's inputs (detections, scan announcements, team states) are recorded once from 4 s of v9 seed 1000 in
shadow mode (after the 1 s pre-roll; straight constant-speed team, the traffic on its own) and replayed:
- every compiled function next to its numpy reference on the same inputs, call by call (prediction, sigma points,
  gate, joint update, coarse gate, coverage and visibility, existence, births, combine, lane probability): equal to
  rounding (1e-9 relative), the same discrete outcomes (candidate pairs, gated pairs, visibility counts);
- two whole replays, compiled and numpy: the same track ids, confirmations, births, merges, deletions and NIS
  records, states equal to rounding. The filter carries rounding forward, so these differences grow a little over
  the episode (up to ~1e-10 m early, when the young tracks' covariances are large), hence the looser tolerance.
Plus the fallback to numpy for a covariance the plain Cholesky factor cannot take, and the NOHESI_JIT=0 switch."""
import os
import subprocess
import sys
from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import _jit
from nohesi.perception.config import Perception, TrackerConfig
from nohesi.perception.pipeline import PerceptionSystem
from nohesi.perception.tracking import _tracking_jit as K
from nohesi.perception.tracking import association as A
from nohesi.perception.tracking import existence as E
from nohesi.perception.tracking import tracker as TR
from nohesi.perception.tracking.imm import init_state
from nohesi.perception.tracking.models import MODELS, motion
from nohesi.road import Road
from nohesi.world import World

pytestmark = pytest.mark.skipif(not _jit.ENABLED, reason='numba is off (NOHESI_JIT=0 or not installed)')

SEED = 1000
STEPS = 80             # 4 s after the pre-roll
RTOL = 1e-9            # one call, compiled against numpy


@contextmanager
def numpy_path():
    """The numpy path for the duration (perception/_jit.py's switch, read at call time)."""
    old, _jit.ENABLED = _jit.ENABLED, False
    try:
        yield
    finally:
        _jit.ENABLED = old


def close(a, b, rtol=RTOL, what=''):
    """a equals b to rtol relative to b's largest magnitude (states mix metres, m/s and pixels^2)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape, what
    if b.size:
        scale = max(1.0, float(np.nanmax(np.abs(b))) if np.isfinite(b).any() else 1.0)
        np.testing.assert_allclose(a, b, rtol=rtol, atol=rtol*scale, err_msg=what)


@pytest.fixture(scope='module')
def recorded():
    """(constructor arguments, [(detections, now, team, scans)]) of the pipeline's tracker over the episode."""
    s = scenario_v9()
    sc = replace(s, timing=replace(s.timing, duration=10.0), perception=Perception(enabled=True, mode='shadow'))
    world = World(sc, SEED)
    calls = []
    orig = TR.TeamTracker.process

    def record(self, detections, now, team, scans=()):
        calls.append((list(detections), float(now), None if team is None else np.array(team, float), list(scans)))
        return orig(self, detections, now, team, scans)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TR.TeamTracker, 'process', record)
        system = PerceptionSystem(sc.perception, sc, world.road, world.plant.params, SEED)
        system.reset(world.snapshot)
        for _ in range(STEPS):
            snap, applied = world.step(np.zeros((3, 2)))
            system.step(snap, applied[:, 0].copy())
    t = system.tracker
    return (t.cfg, t.road, t.params, t.specs, t.car_height), dict(cycle=t.cycle, hold=t.hold), calls


def replay(recorded):
    """A fresh tracker (NIS log on) fed the recorded calls: (tracker, beliefs after each call)."""
    args, kw, calls = recorded
    tr = TR.TeamTracker(*args, **kw, record_nis=True)
    out = []
    for det, now, team, scans in calls:
        tr.process(det, now, team, scans)
        out.append(tr.belief(now))
    return tr, out


# ---------------------------------------------------------------------------------------------- call by call
def twin(mp, owner, name, compare, counts, copy_args=None):
    """Patch owner.name to run the numpy path first (on copies of its arguments when it mutates them), then the
    compiled one, and compare the two results."""
    raw = owner.__dict__[name]
    static = isinstance(raw, staticmethod)
    f = raw.__func__ if static else raw

    def both(*a, **k):
        ra = copy_args(a) if copy_args else a
        with numpy_path():
            ref = f(*ra, **k)
        out = f(*a, **k)
        compare(out, ref, ra, a)
        counts[name] += 1
        return out
    mp.setattr(owner, name, staticmethod(both) if static else both)


def same_state(out, ref, *_):
    for f in ('means', 'covs', 'mu'):
        close(getattr(out, f), getattr(ref, f), what=f)


def same_gate(out, ref, *_):
    assert np.array_equal(out.gated, ref.gated)
    assert np.array_equal(out.dj, ref.dj) and np.array_equal(out.trk, ref.trk) and np.array_equal(out.row, ref.row)
    for f in ('ll', 'nis', 'cost', 'X', 'wm', 'R'):
        close(getattr(out, f), getattr(ref, f), what=f)
    for f in ('zhat', 'nu', 'Pzz', 'D'):
        close(getattr(out.mo, f), getattr(ref.mo, f), what=f)
    same_state(out.pred, ref.pred)


def same_arrays(out, ref, *_):
    for a, b in zip(out if isinstance(out, tuple) else (out,), ref if isinstance(ref, tuple) else (ref,)):
        if np.asarray(b).dtype == bool or np.issubdtype(np.asarray(b).dtype, np.integer):
            assert np.array_equal(a, b)
        else:
            close(a, b)


def same_existence(out, ref, ra, a):
    close(a[1].p, ra[1].p)        # the tracks' existence after the call, numpy on a copy
    assert np.array_equal(a[1].tp, ra[1].tp)


def copy_tracks(a):
    return (a[0], a[1].copy())+tuple(a[2:])


def test_each_compiled_function_matches_numpy(recorded):
    counts = Counter()
    with pytest.MonkeyPatch.context() as mp:
        for name in ('predict_rows', 'joint_update', '_cv'):
            twin(mp, TR, name, same_state if name != '_cv' else same_arrays, counts)
        twin(mp, A, 'coarse_candidates', same_arrays, counts)
        twin(mp, E, 'visibility', same_arrays, counts)
        twin(mp, TR, 'imm_combine', same_arrays, counts)          # the tracker's own imports of imm's functions
        twin(mp, TR, 'lane_probability', same_arrays, counts)
        twin(mp, TR.TeamTracker, '_gate', same_gate, counts)
        twin(mp, TR.TeamTracker, '_coverage', same_arrays, counts)
        twin(mp, TR.TeamTracker, '_birth_states', same_arrays, counts)
        twin(mp, TR.TeamTracker, '_existence', same_existence,  counts, copy_args=copy_tracks)
        tr, _ = replay(recorded)
    assert tr.stats['births'] > 50 and tr.stats['deleted'] > 10 and tr.stats['merges'] > 0
    for name in ('_gate', '_coverage', '_birth_states', '_existence', 'coarse_candidates', 'predict_rows',
                 'joint_update', 'visibility', '_cv', 'imm_combine', 'lane_probability'):
        assert counts[name] > 50, (name, counts)


# ---------------------------------------------------------------------------------------------- whole episode
def test_replay_matches_numpy_replay(recorded):
    tr, out = replay(recorded)
    with numpy_path():
        ref_tr, ref = replay(recorded)
    assert tr.stats == ref_tr.stats
    for b, r in zip(out, ref):
        assert np.array_equal(b.id, r.id) and np.array_equal(b.confirmed, r.confirmed)
        assert np.array_equal(b.last_seen, r.last_seen)
        for f in ('mean', 'cov', 'p_exist', 'mode_prob', 'lane_prob'):
            np.testing.assert_allclose(getattr(b, f), getattr(r, f), rtol=1e-7, atol=1e-8, err_msg=f)
    nis, rnis = tr.nis_records(), ref_tr.nis_records()
    assert len(nis) > 1000
    for f in ('time', 'track', 'kind', 'sensor', 'car', 'dof'):
        assert np.array_equal(nis[f], rnis[f]), f
    np.testing.assert_allclose(nis['nis'], rnis['nis'], rtol=1e-6, atol=1e-9)


# ---------------------------------------------------------------------------------------------- constants, fallbacks
def test_kernel_constants_are_the_modules_constants():
    assert (K.LIDAR, K.RADAR, K.CAMERA) == (A.LIDAR, A.RADAR, A.CAMERA)
    assert {k: v for k, v in TR.KIND_ORDER.items()} == A.KIND_CODE
    assert np.array_equal(np.array(K.SIGNS), A._SIGNS) and np.array_equal(np.array(K.EDGES), A._EDGES)
    from nohesi.perception.tracking.models import _EDGES
    assert np.array_equal(np.array(K.EDGES), _EDGES)
    cfg, road = TrackerConfig(), Road(scenario_v9().road)
    steps, mp = TR._motion_row(cfg, tuple(motion(m, cfg, road) for m in MODELS))
    assert steps.tolist() == [0, 1, -1] and len(mp) == len(K.MOTION_FIELDS)
    assert dict(zip(K.MOTION_FIELDS, mp))['lanes'] == road.cfg.lanes
    assert len(K.SPEC_FIELDS) == K.S_PD+1 and len(K.COVER_FIELDS) == K.C_HEIGHT+1
    assert len(K.GEOM_FIELDS) == K.G_HEIGHT+1


def test_a_covariance_without_a_plain_factor_takes_the_numpy_path():
    sc = scenario_v9()
    road, cfg = Road(sc.road), TrackerConfig()
    motions = tuple(motion(m, cfg, road) for m in MODELS)
    rng = np.random.default_rng(3)
    A_ = rng.normal(size=(6, 4, 4))
    st = init_state(rng.normal(size=(6, 4))*[50, 3, 5, 0.5]+[0, 0, 30, 0], A_ @ np.swapaxes(A_, 1, 2)+0.1*np.eye(4),
                    cfg.imm_init)
    covs = st.covs.copy()
    covs[2] = np.diag([1.0, -1e-9, 1.0, 1.0])           # not positive definite: ukf.robust_cholesky floors it
    dt = np.linspace(0.05, 0.3, 6)
    steps, mp = TR._motion_row(cfg, motions)
    _, _, _, ok = K.predict_rows(st.means, covs, st.mu, dt, steps, mp, TR._WM, TR._WC, TR._SC)
    assert not ok
    same_state(TR.predict_rows(st.means, covs, st.mu, dt, cfg, motions),
               TR.predict_rows_numpy(st.means, covs, st.mu, dt, cfg, motions))
    same_state(TR.predict_rows(st.means, st.covs, st.mu, dt, cfg, motions),
               TR.predict_rows_numpy(st.means, st.covs, st.mu, dt, cfg, motions))
    _, ok = K.sigma_rows(st.means.reshape(-1, 4), covs.reshape(-1, 4, 4), TR._SC)
    assert not ok


def test_custom_motions_take_the_numpy_path():
    cfg = TrackerConfig()
    st = init_state(np.array([[0.0, 0.0, 30.0, 0.0]]), np.diag([1.0, 0.1, 1.0, 0.1])[None], cfg.imm_init)
    plain = tuple(lambda X, dt: X+0.0 for _ in MODELS)              # no .params: not models.motion's
    assert TR._motion_row(cfg, plain) is None
    same_state(TR.predict_rows(st.means, st.covs, st.mu, [0.1], cfg, plain),
               TR.predict_rows_numpy(st.means, st.covs, st.mu, [0.1], cfg, plain))


def test_nohesi_jit_0_turns_the_kernels_off():
    code = ('from nohesi.perception import _jit; from nohesi.perception.tracking import _tracking_jit as K; '
            'print(_jit.ENABLED, type(K.gate_pairs).__name__)')
    env = dict(os.environ, NOHESI_JIT='0')
    out = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True,
                         cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    assert out.stdout.split() == ['False', 'function']
