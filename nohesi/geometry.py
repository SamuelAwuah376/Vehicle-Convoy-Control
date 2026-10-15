import numpy as np
from convoy.geometry import footprint, overlap

__all__ = ['footprint', 'overlap', 'clearance']


def _point_segment_distance(points, a, b):
    ab = b-a
    t = np.clip((points-a)@ab/(ab@ab), 0, 1)
    return np.linalg.norm(points-(a+t[:, None]*ab), axis=1)


def clearance(a, b):
    """Minimum distance between two convex polygons; 0 when they touch or overlap."""
    if overlap(a, b):
        return 0.0
    best = np.inf
    for poly, other in ((a, b), (b, a)):
        for i in range(len(poly)):
            best = min(best, _point_segment_distance(other, poly[i], poly[(i+1) % len(poly)]).min())
    return float(best)
