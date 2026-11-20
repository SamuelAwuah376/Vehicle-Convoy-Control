"""The pipeline's remembered team lookups (pipeline._TeamHistory.at / ego / mates, PerceptionSystem._sensor_pose)
give exactly what the original per-call code gave, and forget on every push."""
from types import SimpleNamespace
import numpy as np
import pytest
from nohesi.config import scenario_v9
from nohesi.perception import pipeline as PL
from nohesi.perception.config import Perception
from nohesi.perception.geometry import sensor_pose, wrap
from nohesi.perception.types import EgoState

VEH = scenario_v9().vehicle


class Reference(PL._TeamHistory):
    """_TeamHistory's at / ego / mates as first written (no memory)."""
    def at(self, t):
        times, states = np.array(self.times), np.array(self.states)
        k = int(np.searchsorted(times, t))
        if k <= 0:
            return states[0].copy()
        if k >= len(times):
            return states[-1].copy()
        w = (t-times[k-1])/(times[k]-times[k-1])
        return states[k-1]+w*(states[k]-states[k-1])

    def ego(self, car, t, pitch_per_accel=0.0):
        x, y, yaw, v, vy, r, ax = (float(a) for a in self.at(t)[car])
        vx = float(np.sqrt(max(v*v-vy*vy, 0.0)))
        return EgoState(float(t), x, y, float(wrap(yaw)), vx, vy, r, ax, pitch_per_accel*(-ax))

    def mates(self, car, t):
        return np.delete(self.at(t)[:, :3], car, axis=0)


def snaps(rng, n=40, dt=0.05):
    yaw = rng.uniform(-np.pi, np.pi, 3)
    for k in range(n):
        yaw = yaw+rng.normal(0.0, 0.3, 3)                 # yaw crossing +-pi: unwrapped along the history
        team = np.column_stack((30*k*dt+rng.normal(0, 1, 3), rng.normal(0, 3, 3), wrap(yaw), rng.uniform(20, 40, 3)))
        yield SimpleNamespace(time=k*dt, team=team, team_vy=rng.normal(0, 0.3, 3), team_r=rng.normal(0, 0.1, 3)), \
            rng.normal(0, 2, 3)


def test_team_lookups_match_the_original():
    rng = np.random.default_rng(4)
    new, ref = PL._TeamHistory(), Reference()
    for snap, accel in snaps(rng):
        new.push(snap, accel)
        ref.push(snap, accel)
        now = snap.time
        times = np.r_[now, now-0.05, now-0.1, rng.uniform(now-0.3, now+0.02, 5), now-10.0]
        for _ in range(2):                                   # the second round comes from memory
            for t in times:
                assert np.array_equal(new.at(t), ref.at(t))
                for car in range(3):
                    for ppa in (0.0, 0.5):
                        assert new.ego(car, t, ppa) == ref.ego(car, t, ppa)
                    assert np.array_equal(new.mates(car, t), ref.mates(car, t))
    with pytest.raises(ValueError):
        new.at(now)[0, 0] = 0.0


def test_lookups_forget_on_push():
    rng = np.random.default_rng(5)
    h = PL._TeamHistory()
    gen = snaps(rng)
    h.push(*next(gen))
    first = h.ego(0, 0.0)
    snap, accel = next(gen)
    h.push(snap, accel)
    assert h.ego(0, 0.05) != first and h.ego(0, 0.05).x == pytest.approx(snap.team[0, 0])


def test_sensor_pose_matches_geometry():
    rng = np.random.default_rng(6)
    system = SimpleNamespace(_offsets={}, params=VEH, team=PL._TeamHistory())
    for snap, accel in snaps(rng, n=10):
        system.team.push(snap, accel)
    cfg = Perception()
    for spec in (*cfg.lidars, *cfg.radars, *cfg.cameras):
        for car in range(3):
            for t in (0.45, 0.3125, 0.0):
                got = PL.PerceptionSystem._sensor_pose(system, spec, car, t)
                assert got == sensor_pose(spec.mount, system.team.at(t)[car], VEH)
