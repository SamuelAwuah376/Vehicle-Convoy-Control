import numpy as np


def footprint(state, cfg):
    rear = -(cfg.length-cfg.wheelbase)/2
    front = rear+cfg.length
    local = np.array([[rear, -cfg.width/2], [front, -cfg.width/2],
                      [front, cfg.width/2], [rear, cfg.width/2]])
    c, s = np.cos(state[2]), np.sin(state[2])
    return local @ np.array([[c, s], [-s, c]]) + state[:2]


def overlap(a, b):
    """Separating-axis test for oriented rectangles; touching counts as collision."""
    for poly in (a, b):
        for edge in np.diff(np.vstack((poly, poly[0])), axis=0)[:2]:
            axis = np.array([-edge[1], edge[0]])
            pa, pb = a@axis, b@axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def safety_metrics(result, route):
    worst, collisions = 0.0, 0
    for states in result.states:
        polygons = [footprint(st, result.cfg) for st in states]
        worst = max(worst, max(abs(route.project(p)[1]) for poly in polygons for p in poly))
        collisions += sum(overlap(polygons[i], polygons[j]) for i, j in ((0, 1), (1, 2), (0, 2)))
    return {'max_footprint_lateral_m': float(worst),
            'lane_boundary_violation_m': max(0.0, worst-result.cfg.lane_width/2),
            'collision_pair_samples': int(collisions)}
