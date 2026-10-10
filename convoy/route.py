import numpy as np


class Route:
    """Non-self-intersecting sampled route, parameterized by arc length."""
    def __init__(self):
        x = np.linspace(0, 420, 2101)
        y = 12 * np.sin(x / 45) + 4 * np.sin(x / 22)
        self.xy = np.column_stack((x, y))
        self.segments = np.diff(self.xy, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.s = np.r_[0, np.cumsum(self.lengths)]
        self.total = float(self.s[-1])

    def at(self, s):
        s = np.clip(s, 0, self.total)
        i = np.clip(np.searchsorted(self.s, s, side='right') - 1, 0, len(self.lengths)-1)
        fraction = (s - self.s[i]) / self.lengths[i]
        xy = self.xy[i] + np.asarray(fraction)[..., None] * self.segments[i]
        yaw = np.arctan2(self.segments[i, 1], self.segments[i, 0])
        return xy, yaw

    def project(self, xy):
        fraction = np.clip(np.sum((xy-self.xy[:-1])*self.segments, axis=1) / self.lengths**2, 0, 1)
        points = self.xy[:-1] + fraction[:, None]*self.segments
        i = np.argmin(np.sum((points-xy)**2, axis=1))
        error = xy-points[i]
        tangent = self.segments[i]/self.lengths[i]
        lateral = tangent[0]*error[1]-tangent[1]*error[0]
        return float(self.s[i]+fraction[i]*self.lengths[i]), float(lateral)
