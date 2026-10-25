import numpy as np

__all__ = ['footprint', 'overlap', 'clearance', 'footprints', 'overlaps', 'clearances', 'cg_offsets', 'body_centre']


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
        # Every vertex of `other` against every edge of `poly` at once (same arithmetic as _point_segment_distance).
        start, ab = poly, np.roll(poly, -1, axis=0)-poly                      # (E, 2)
        rel = other[None, :, :]-start[:, None, :]                              # (E, P, 2)
        t = np.clip((rel[..., 0]*ab[:, None, 0]+rel[..., 1]*ab[:, None, 1])/(ab[:, 0]*ab[:, 0]+ab[:, 1]*ab[:, 1])[:, None], 0, 1)
        diff = other[None, :, :]-(start[:, None, :]+t[..., None]*ab[:, None, :])
        best = min(best, float(np.sqrt(diff[..., 0]**2+diff[..., 1]**2).min()))
    return float(best)


# Batched versions: many states or polygon pairs at once, bit-identical to the single ones (the same matrix products,
# batched with np.matmul, and the same elementwise arithmetic), for the per-step checks in the hot loop.
def footprints(states, cfg):
    """footprint() of each state in `states` (M, >= 3): (M, 4, 2)."""
    states = np.asarray(states, float)
    rear, front = cg_offsets(cfg)
    local = np.array([[rear, -cfg.width/2], [front, -cfg.width/2], [front, cfg.width/2], [rear, cfg.width/2]])
    c, s = np.cos(states[:, 2]), np.sin(states[:, 2])
    rot = np.stack((np.stack((c, s), -1), np.stack((-s, c), -1)), -2)   # (M, 2, 2) rows [c, s], [-s, c]
    return np.matmul(local[None], rot)+states[:, None, :2]


SMALL = 3   # below this many pairs the per-pair functions are faster (numpy call overhead)


def overlaps(A, B):
    """overlap() of each polygon pair (A[m], B[m]), A and B (M, 4, 2): (M,) bool."""
    if len(A) < SMALL:
        return np.array([overlap(a, b) for a, b in zip(A, B)], bool)
    separated = np.zeros(len(A), bool)
    for poly in (A, B):
        edges = np.diff(np.concatenate((poly, poly[:, :1]), axis=1), axis=1)[:, :2]   # (M, 2, 2)
        for e in range(2):
            axis = np.stack((-edges[:, e, 1], edges[:, e, 0]), -1)[:, :, None]       # (M, 2, 1)
            pa, pb = np.matmul(A, axis)[..., 0], np.matmul(B, axis)[..., 0]
            separated |= (pa.max(axis=1) < pb.min(axis=1)) | (pb.max(axis=1) < pa.min(axis=1))
    return ~separated


def clearances(A, B):
    """clearance() of each polygon pair (A[m], B[m]), A and B (M, 4, 2): (M,), 0 where they touch or overlap."""
    if len(A) < SMALL:
        return np.array([clearance(a, b) for a, b in zip(A, B)], float)
    best = np.full(len(A), np.inf)
    for poly, other in ((A, B), (B, A)):
        start, ab = poly, np.roll(poly, -1, axis=1)-poly                                  # (M, E, 2)
        rel = other[:, None, :, :]-start[:, :, None, :]                                    # (M, E, P, 2)
        t = np.clip((rel[..., 0]*ab[:, :, None, 0]+rel[..., 1]*ab[:, :, None, 1]) /
                    (ab[..., 0]*ab[..., 0]+ab[..., 1]*ab[..., 1])[:, :, None], 0, 1)
        diff = other[:, None, :, :]-(start[:, :, None, :]+t[..., None]*ab[:, :, None, :])
        best = np.minimum(best, np.sqrt(diff[..., 0]**2+diff[..., 1]**2).min(axis=(1, 2)))
    return np.where(overlaps(A, B), 0.0, best)
