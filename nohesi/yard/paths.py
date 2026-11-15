"""Planned rear-axle paths: samples at most Planning.path_sample apart, each with the direction of motion (+1 forward,
-1 reverse) from that sample to the next. A cusp is a sample where the direction changes: the truck stops there and
shifts. Progress along a path is a sample index."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Path:
    x: np.ndarray            # (N,) rear-axle samples
    y: np.ndarray
    yaw: np.ndarray          # truck heading at each sample (not the direction of motion when reversing)
    direction: np.ndarray    # (N,) int: motion from sample k to k+1 (+1 forward, -1 reverse); the last repeats N-2's

    def __post_init__(self):
        self.x, self.y, self.yaw = (np.asarray(a, float) for a in (self.x, self.y, self.yaw))
        self.direction = np.asarray(self.direction, int)
        n = len(self.x)
        if not (len(self.y) == len(self.yaw) == len(self.direction) == n) or n < 2:
            raise ValueError('a path needs at least two samples with x, y, yaw and direction each')
        if not np.all(np.isin(self.direction, (-1, 1))):
            raise ValueError('direction must be +1 or -1')
        self.s = np.r_[0.0, np.cumsum(np.hypot(np.diff(self.x), np.diff(self.y)))]

    def __len__(self):
        return len(self.x)

    @property
    def states(self):
        """(N, 3) [x, y, yaw]."""
        return np.column_stack((self.x, self.y, self.yaw))

    @property
    def length(self):
        return float(self.s[-1])

    @property
    def cusps(self):
        """Indices of the samples where the direction of motion changes."""
        return np.flatnonzero(self.direction[1:-1] != self.direction[:-2])+1 if len(self) > 2 else np.zeros(0, int)

    def segments(self):
        """[(i0, i1)]: inclusive index ranges of constant direction; consecutive segments share their cusp sample."""
        bounds = [0, *self.cusps.tolist(), len(self)-1]
        return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]

    def segment_of(self, k):
        """Index into segments() of the segment that sample k moves along (a cusp belongs to the segment it starts)."""
        for n, (a, b) in enumerate(self.segments()):
            if a <= k < b:
                return n
        return len(self.segments())-1

    def nearest(self, xy, lo=0, hi=None):
        """Index of the sample nearest to xy among samples lo .. hi-1."""
        hi = len(self) if hi is None else hi
        lo, hi = max(0, int(lo)), max(int(lo)+1, min(len(self), int(hi)))
        d = np.hypot(self.x[lo:hi]-xy[0], self.y[lo:hi]-xy[1])
        return lo+int(np.argmin(d))

    def remaining(self, k):
        """Path length from sample k to the end (m)."""
        return float(self.s[-1]-self.s[int(k)])

    def sub(self, i0, i1):
        """Samples i0 .. i1 (inclusive) as a new Path."""
        i0, i1 = int(i0), int(i1)
        d = self.direction[i0:i1+1].copy()
        d[-1] = d[-2] if len(d) > 1 else d[-1]
        return Path(self.x[i0:i1+1], self.y[i0:i1+1], self.yaw[i0:i1+1], d)

    def retrace(self, k, i_to=0):
        """Back along the path from sample k to sample i_to < k: the same samples in reverse order with the opposite
        directions of motion (a truck that drove forward from i_to to k reverses back over the same ground)."""
        k, i_to = int(k), int(i_to)
        if k <= i_to:
            raise ValueError('retrace needs k > i_to')
        idx = np.arange(k, i_to-1, -1)
        d = -self.direction[idx-1].copy()          # motion from idx[m] to idx[m+1] = idx[m]-1 reverses direction[idx[m]-1]
        d[-1] = d[-2] if len(d) > 1 else d[-1]
        return Path(self.x[idx], self.y[idx], self.yaw[idx], d)

    @staticmethod
    def concat(parts):
        """Join paths end to start (each part's first sample equals the previous part's last: it is dropped)."""
        x, y, yaw, d = [parts[0].x], [parts[0].y], [parts[0].yaw], [parts[0].direction[:-1]]
        for p in parts[1:]:
            x.append(p.x[1:]); y.append(p.y[1:]); yaw.append(p.yaw[1:]); d.append(p.direction[:-1])
        d.append(parts[-1].direction[-1:])
        return Path(np.concatenate(x), np.concatenate(y), np.concatenate(yaw), np.concatenate(d))
