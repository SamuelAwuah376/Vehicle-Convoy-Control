"""Yard world: the rigid-truck plant (signed speed, gear, lag), footprints, the layout generator and collisions."""
from dataclasses import replace
import numpy as np
import pytest
from nohesi.geometry import clearances
from nohesi.yard.config import Planning, Site, Truck, YardScenario
from nohesi.yard.paths import Path
from nohesi.yard.truck import TruckPlant, corners, disk_centres, disk_offsets
from nohesi.yard.world import YardWorld, follower_starts

T = Truck()


def drive(plant, steps, steer_cmd, accel_cmd, gear, state=None):
    s, st, a = (np.zeros(4) if state is None else state), 0.0, 0.0
    for _ in range(steps):
        s, st, a = plant.step(s, st, a, steer_cmd, accel_cmd, gear)
    return s, st, a


def test_reverse_and_gear():
    plant = TruckPlant(T, 0.1)
    s, _, _ = drive(plant, 30, 0.0, -0.5, -1)
    assert s[3] < 0 and s[0] < 0 and abs(s[1]) < 1e-9          # reversing straight back
    s2, _, _ = drive(plant, 30, 0.0, -0.5, +1)                   # forward gear: braking holds it at 0
    assert s2[3] == 0 and s2[0] == 0
    s3, _, a = drive(plant, 100, 0.0, 1.0, +1)
    assert s3[3] > 0 and a <= T.accel_max+1e-9                   # driving effort capped at accel_max


def test_turning_radius_and_steer_rate():
    plant = TruckPlant(T, 0.05)
    s, st, a = np.array([0, 0, 0, 1.0]), T.steer_max, 0.0
    for _ in range(int(2*np.pi*T.min_radius/1.0/0.05)):         # one full circle at full lock, 1 m/s
        s, st, a = plant.step(s, st, a, T.steer_max, 0.0, +1)
    assert np.hypot(s[0], s[1]) < 0.5                            # back near the start
    _, st, _ = plant.step(np.zeros(4), 0.0, 0.0, T.steer_max, 0.0, +1)
    assert st == pytest.approx(T.steer_rate*0.05)


def test_footprint_and_disks():
    c = corners(T, [[0, 0, 0]])[0]
    assert c[:, 0].min() == pytest.approx(-T.rear_overhang) and c[:, 0].max() == pytest.approx(T.length-T.rear_overhang)
    assert c[:, 1].max() == pytest.approx(T.width/2)
    off, r = disk_offsets(T)
    pts = np.random.default_rng(0).uniform([-T.rear_overhang, -T.width/2], [T.length-T.rear_overhang, T.width/2], (2000, 2))
    d = np.hypot(pts[:, None, 0]-off[None], pts[:, None, 1])
    assert np.all(d.min(axis=1) <= r+1e-9)                       # the disks cover the rectangle
    assert disk_centres(T, [[1, 2, np.pi/2]]).shape == (1, 4, 2)


def test_layouts_are_valid():
    sc = YardScenario()
    for seed in range(40):
        w = YardWorld(sc, seed)
        assert 1 <= w.n <= 3 and w.collisions() == []
        inner = w.obstacles.corners[[k for k, kind in enumerate(w.obstacles.kind) if kind != 'berm']]
        if len(inner):
            assert np.all(np.hypot(inner[..., 0], inner[..., 1]-sc.site.width/2).min(axis=1) >= sc.site.keep_clear)
        feet = w.footprints()
        gaps = clearances(np.repeat(feet, len(w.obstacles), 0), np.tile(w.obstacles.corners, (w.n, 1, 1)))
        assert gaps.min() > 1.0                                  # queued followers stand clear of the berms


def test_collisions_detected():
    w = YardWorld(YardScenario(), 3, n_followers=2)
    w.states[1, :2] = w.states[0, :2]+[2.0, 0.0]
    kinds = {k for _, k, _ in w.collisions()}
    assert 'truck' in kinds
    w.states[1] = [w.leader[0]+1, w.leader[1], 0, 0]
    assert any(k == 'leader' for _, k, _ in w.collisions())


def test_config_checks():
    with pytest.raises(ValueError):
        Truck(wheelbase=8.0, rear_overhang=2.0)                  # no front overhang left
    with pytest.raises(ValueError):
        YardScenario(followers=(0, 3))
    with pytest.raises(ValueError):
        Planning(decision_interval=0.33)
    assert len(follower_starts(T, Site(), 3)) == 3


def test_path_helpers():
    a = Path(np.arange(5.), np.zeros(5), np.zeros(5), [1]*5)
    b = Path(4-np.arange(4.), np.zeros(4), np.zeros(4), [-1]*4)
    p = Path.concat([a, b])
    assert p.cusps.tolist() == [4] and p.segments() == [(0, 4), (4, 7)] and p.length == 7.0
    r = a.retrace(4, 1)
    assert r.x.tolist() == [4, 3, 2, 1] and set(r.direction) == {-1}
    assert p.nearest((3.2, 0.1), 5) == 5 and p.remaining(4) == 3.0
