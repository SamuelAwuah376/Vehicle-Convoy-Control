"""types.DetectionBatch, the pipeline's processing -> tracker interface: the batches processing builds carry exactly
the arrays DetectionBatch.from_detections takes from their own Detections, so the tracker sees the same numbers
either way, and a tracker fed the pipeline's batches gives the same beliefs, to the bit, as one fed the same
detections as Detection objects (flat, or split across batches and lists). Recorded from 3 s of v9 seed 1000 in
shadow mode (after the 1 s pre-roll), on the path NOHESI_JIT selects."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception.config import Perception
from nohesi.perception.pipeline import PerceptionSystem
from nohesi.perception.tracking import association as A
from nohesi.perception.tracking import tracker as TR
from nohesi.perception.types import DetectionBatch
from nohesi.world import World

SEED = 1000
STEPS = 60
FIELDS = ('time', 'z', 'R', 'pose', 'n_points', 'fitted', 'v_prior', 'dz_dv', 'R_fit', 'trunc_mode', 'truncated',
          'has_init', 'init_xy', 'init_P')


@pytest.fixture(scope='module')
def recorded():
    """(tracker constructor arguments, [(batches, now, team, scans)]) of the pipeline's tracker."""
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


def batches(recorded):
    return [b for dets, *_ in recorded[2] for b in dets]


def same_batch(a, b):
    """Two batches with the same fields, bit for bit, and the same EgoState objects."""
    assert (a.kind, a.sensor, a.car, a.scan) == (b.kind, b.sensor, b.car, b.scan)
    for f in FIELDS:
        x, y = getattr(a, f), getattr(b, f)
        assert (x is None) == (y is None), f
        if x is not None:
            x, y = np.asarray(x), np.asarray(y)
            assert x.shape == y.shape and np.array_equal(x, y), f
            assert x.tobytes() == y.tobytes(), f                 # signed zeros too
    assert len(a.ego) == len(b.ego) and all(e is g for e, g in zip(a.ego, b.ego))


def replay(recorded, convert):
    """A fresh tracker fed the recorded calls with each call's batches passed through convert: the beliefs after
    each call, the stats and the NIS records."""
    args, kw, calls = recorded
    tr = TR.TeamTracker(*args, **kw, record_nis=True)
    out = []
    for dets, now, team, scans in calls:
        tr.process(convert(dets), now, team, scans)
        out.append(tr.belief(now))
    return out, tr.stats, tr.nis_records()


def same_beliefs(a, b):
    assert len(a[0]) == len(b[0])
    for x, y in zip(a[0], b[0]):
        for f in ('id', 'mean', 'cov', 'p_exist', 'lane_prob', 'mode_prob', 'last_seen', 'confirmed'):
            assert getattr(x, f).tobytes() == getattr(y, f).tobytes(), f
    assert a[1] == b[1]
    assert a[2].tobytes() == b[2].tobytes()


def test_the_recording_has_every_kind(recorded):
    kinds = {b.kind for b in batches(recorded) if len(b)}
    assert kinds == {'lidar_box', 'radar', 'camera_box'}
    assert all(isinstance(b, DetectionBatch) for b in batches(recorded))
    assert sum(len(b) for b in batches(recorded)) > 3000


def test_batches_carry_their_detections_arrays(recorded):
    for b in batches(recorded):
        dets = list(b)
        assert len(dets) == len(b) and dets == b.detections() and all(b[k] is d for k, d in enumerate(dets))
        assert all((d.kind, d.sensor, d.car, d.meta['scan']) == (b.kind, b.sensor, b.car, b.scan) for d in dets)
        same_batch(b, DetectionBatch.from_detections(dets, b.kind, b.sensor, b.car, b.scan))


def test_tracker_on_batches_equals_detection_lists(recorded):
    ref = replay(recorded, lambda dets: [d for b in dets for d in b])
    same_beliefs(replay(recorded, list), ref)
    # the same deliveries split: each scan's first detection as a Detection, the rest as a batch of its own
    def split(dets):
        out = []
        for b in dets:
            if len(b) < 2:
                out.append(b)
                continue
            d = list(b)
            out += [d[0], DetectionBatch.from_detections(d[1:], b.kind, b.sensor, b.car, b.scan)]
        return out
    same_beliefs(replay(recorded, split), ref)
    assert ref[1]['births'] > 20 and ref[1]['cycles'] >= STEPS


def test_table_from_batches_equals_table_from_lists(recorded):
    args = recorded[0]
    specs, params = args[3], args[2]
    n = 0
    for dets, *_ in recorded[2][-20:]:
        scans = [(b, specs[b.sensor], 0.0) for b in dets]
        for sort in (False, True):
            a = A.DetTable.build(scans, params, sort=sort)
            b = A.DetTable.build([(list(d), sp, lc) for d, sp, lc in scans], params, sort=sort)
            assert set(a.__dict__) == set(b.__dict__)
            assert all(x is y for x, y in zip(a.dets, b.dets)) and len(a.dets) == len(a)
            for k, v in b.__dict__.items():
                if k not in ('_src', 'spec'):
                    assert getattr(a, k).dtype == v.dtype and getattr(a, k).tobytes() == v.tobytes(), k
            n += 1
    assert n == 40


def test_empty_batch():
    b = DetectionBatch.empty('radar', 'radar_front', 1, 0.25)
    assert len(b) == 0 and list(b) == [] and b.z.shape == (0, 3) and b.R.shape == (0, 3, 3)
    assert DetectionBatch.from_detections([], 'camera_box', 'cam', 0, 0.5).z.shape == (0, 4)
