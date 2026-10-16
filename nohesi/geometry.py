import numpy as np

__all__ = ['footprint', 'overlap', 'clearance', 'cg_offsets', 'body_centre']


def cg_offsets(cfg):
    """(rear, front) body edges along the heading relative to the CG; the body overhangs both axles equally."""
    l_r = getattr(cfg, 'cg_to_rear_axle', cfg.wheelbase/2)
    o = (cfg.length-cfg.wheelbase)/2
    return -(l_r+o), cfg.wheelbase-l_r+o


def body_centre(states, cfg):
    """Body-centre (x, y) of CG states (..., >=3): CG + (l_f - l_r)/2 along the heading."""
    states = np.asarray(states, float)
    shift = sum(cg_offsets(cfg))/2
    return states[..., :2]+shift*np.stack((np.cos(states[..., 2]), np.sin(states[..., 2])), axis=-1)


def footprint(state, cfg):
    rear, front = cg_offsets(cfg)
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
