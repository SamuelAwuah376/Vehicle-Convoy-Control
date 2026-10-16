from dataclasses import dataclass, field, replace
import numpy as np


@dataclass(frozen=True)
class Plan:
    """Road-frame trajectory z[k] = [s, d, heading error, v] at time + k*step, with controls u[k] = [a, steer]."""
    time: float
    step: float
    z: np.ndarray       # (N+1, 4)
    u: np.ndarray       # (N, 2)
    source: str         # 'qp', 'shifted' or 'lane_keep'
    status: str = 'solved'
    iterations: int = 0
    solve_time: float = 0.0
    age: int = 0        # steps shifted since it was solved

    def shifted(self, steps=1):
        """Drop the first `steps`, extend the tail at constant speed and heading."""
        z, u = self.z, self.u
        for _ in range(steps):
            s, d, psi, v = z[-1]
            z = np.vstack((z[1:], [s+v*np.cos(psi)*self.step, d+v*np.sin(psi)*self.step, psi, v]))
            u = np.vstack((u[1:], [0.0, u[-1, 1]]))
        return replace(self, time=self.time+steps*self.step, z=z, u=u, age=self.age+steps)

    def at(self, k):
        """State at step k; beyond the horizon, continued at constant speed and heading."""
        if k < len(self.z):
            return self.z[k]
        s, d, psi, v = self.z[-1]
        extra = (k-len(self.z)+1)*self.step
        return np.array([s+v*np.cos(psi)*extra, d+v*np.sin(psi)*extra, psi, v])


def constant_speed_plan(time, step, horizon, s, d, v, source='lane_keep'):
    t = np.arange(horizon+1)*step
    z = np.column_stack((s+v*t, np.full_like(t, d), np.zeros_like(t), np.full_like(t, v)))
    return Plan(time, step, z, np.zeros((horizon, 2)), source)


@dataclass(frozen=True)
class Obstacles:
    """Predicted CG positions of other vehicles at the plan times, shape (N+1, K)."""
    keys: tuple         # ('traffic', id) or ('team', index)
    s: np.ndarray
    d: np.ndarray
    v: np.ndarray

    def index(self, key):
        return self.keys.index(key) if key in self.keys else None

    @staticmethod
    def stack(parts, rows):
        parts = [p for p in parts if p.keys]
        if not parts:
            return Obstacles((), *(np.empty((rows, 0)) for _ in range(3)))
        return Obstacles(sum((p.keys for p in parts), ()), *(np.hstack([getattr(p, n) for p in parts]) for n in 'sdv'))


@dataclass
class Target:
    """What the MPC is asked to do: lane-centre offset, speed, the locked gap's bounding cars, and the
    teammates changing into the vehicle's lane that it should get clear of (soft)."""
    d: float
    speed: float
    front: tuple = None
    rear: tuple = None
    yield_to: tuple = ()
