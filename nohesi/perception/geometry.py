"""Frames, sensor poses, car boxes, the pinhole camera model and ray casting (shared contract; types.py explains
the frames). Everything is vectorized numpy; nothing here draws random numbers."""
import numpy as np
from ..geometry import cg_offsets, footprints

__all__ = ['wrap', 'rot', 'mount_offset', 'sensor_pose', 'to_sensor', 'to_world', 'car_boxes', 'box_corners_3d',
           'focal', 'camera_tilt', 'project', 'raycast', 'visible_fraction', 'nearest_box_point']


def wrap(a):
    """Angle(s) wrapped to [-pi, pi)."""
    return (np.asarray(a, float)+np.pi) % (2*np.pi)-np.pi


def rot(yaw):
    """(..., 2, 2) rotation matrices: local -> world for a frame at heading yaw."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.stack((np.stack((c, -s), -1), np.stack((s, c), -1)), -2)


def mount_offset(mount, params):
    """(x, y) of a Mount in the car frame (from the CG) for vehicle params (length, wheelbase, cg_to_rear_axle)."""
    rear, front = cg_offsets(params)
    centre = (rear+front)/2
    base = {'': centre, 'front': front, 'rear': rear}[mount.edge]
    return base+mount.x, mount.y


def sensor_pose(mount, car_state, params):
    """(x, y, yaw, z) of a sensor in the world frame, for a car CG state [x, y, yaw, ...]."""
    ox, oy = mount_offset(mount, params)
    x, y, yaw = (float(v) for v in car_state[:3])
    c, s = np.cos(yaw), np.sin(yaw)
    return (x+c*ox-s*oy, y+s*ox+c*oy, float(wrap(yaw+mount.yaw)), float(mount.z))


def to_sensor(points, pose):
    """World (..., 2) points into the sensor frame of pose (x, y, yaw, z)."""
    p = np.asarray(points, float)-np.array(pose[:2])
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((c*p[..., 0]+s*p[..., 1], -s*p[..., 0]+c*p[..., 1]), -1)


def to_world(points, pose):
    """Sensor-frame (..., 2) points into the world frame."""
    p = np.asarray(points, float)
    c, s = np.cos(pose[2]), np.sin(pose[2])
    return np.stack((c*p[..., 0]-s*p[..., 1]+pose[0], s*p[..., 0]+c*p[..., 1]+pose[1]), -1)


def car_boxes(states, params):
    """(M, 4, 2) body rectangles of CG states (M, >= 3) [x, y, heading]: corners rear-right, front-right, front-left,
    rear-left (nohesi.geometry.footprints)."""
    states = np.asarray(states, float)
    if len(states) == 0:
        return np.zeros((0, 4, 2))
    return footprints(states, params)


def box_corners_3d(states, params, height):
    """(M, 8, 3) corners of the cars' 3D boxes: the footprint at z = 0, then at z = height."""
    fp = car_boxes(states, params)
    z0, z1 = np.zeros(fp.shape[:2]+(1,)), np.full(fp.shape[:2]+(1,), float(height))
    return np.concatenate((np.concatenate((fp, z0), -1), np.concatenate((fp, z1), -1)), 1)


def focal(spec):
    """Focal length in pixels of a CameraSpec."""
    return (spec.width_px/2)/np.tan(spec.hfov/2)


def _tilt(yaw, pitch):
    """The nine entries r[i][j] of camera_tilt (lists of arrays, or floats), without stacking them."""
    sa, ca = np.sin(yaw), np.cos(yaw)
    s, k = np.sin(pitch), 1-np.cos(pitch)
    return ((1-k*ca*ca, k*sa*ca, s*ca), (k*sa*ca, 1-k*sa*sa, -s*sa), (-s*ca, s*sa, 1-k))


def camera_tilt(yaw, pitch):
    """(..., 3, 3) orientation R of a camera mounted at yaw `yaw` (Mount.yaw) relative to its level frame (x along
    the boresight, y left, z up) when the body pitches by `pitch` (nose down positive): the rotation by pitch about the
    body's y axis, which in the camera's level frame is (sin yaw, cos yaw, 0) (Rodrigues). So a front camera pitches
    down by pitch, a rear camera up by it, and a side camera rolls. A level-frame point p is R^T p in the camera."""
    yaw, pitch = np.broadcast_arrays(np.asarray(yaw, float), np.asarray(pitch, float))
    return np.stack([np.stack(row, -1) for row in _tilt(yaw, pitch)], -2)


def project(spec, pose, points, pitch=0.0):
    """Pinhole projection of world points (..., 3) through camera `spec` at pose (x, y, yaw, z) with body pitch
    (nose down positive; camera_tilt maps it to the camera by spec.mount.yaw). Returns u, v (pixels; u right, v down,
    principal point at the image centre) and depth (distance along the optical axis; points with depth <= 0 are behind
    the camera, their u, v meaningless)."""
    p = np.asarray(points, float)
    xy = to_sensor(p[..., :2], pose)
    x, y, z = xy[..., 0], xy[..., 1], p[..., 2]-pose[3]
    r = _tilt(float(spec.mount.yaw), np.asarray(pitch, float) if np.ndim(pitch) else float(pitch))
    xc = r[0][0]*x+r[1][0]*y+r[2][0]*z          # R^T p: a front camera's nose-down pitch moves the scene up
    yc = r[0][1]*x+r[1][1]*y+r[2][1]*z
    zc = r[0][2]*x+r[1][2]*y+r[2][2]*z
    f = focal(spec)
    with np.errstate(divide='ignore', invalid='ignore'):
        u = spec.width_px/2-f*yc/xc
        v = spec.height_px/2-f*zc/xc
    return u, v, xc


def raycast(origin, angles, boxes, range_max, skip=()):
    """Nearest hit of rays from world point `origin` (2,) at world angles (n,) against box polygons (M, 4, 2).
    Returns (range (n,), np.inf where nothing within range_max; index (n,) of the box hit, -1 for none).
    Boxes in `skip` (indices) are ignored (the carrying car's own body)."""
    angles = np.asarray(angles, float)
    n = len(angles)
    rng, idx = np.full(n, np.inf), np.full(n, -1)
    keep = np.setdiff1d(np.arange(len(boxes)), np.asarray(skip, int))
    if n == 0 or keep.size == 0:
        return rng, idx
    o = np.asarray(origin, float)
    B = np.asarray(boxes, float)[keep]
    reach = np.linalg.norm(B.mean(1)-o, axis=1)-np.linalg.norm(B[:, 0]-B[:, 2], axis=1)
    keep, B = keep[reach <= range_max], B[reach <= range_max]
    if keep.size == 0:
        return rng, idx
    a = B.reshape(-1, 2)                          # segment starts (4M, 2)
    e = (np.roll(B, -1, axis=1)-B).reshape(-1, 2)  # segment vectors
    d = np.stack((np.cos(angles), np.sin(angles)), -1)   # (n, 2)
    denom = d[:, None, 0]*e[None, :, 1]-d[:, None, 1]*e[None, :, 0]   # cross(d, e)
    w = a-o
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[None, :, 0]*e[None, :, 1]-w[None, :, 1]*e[None, :, 0])/denom   # along the ray
        u = (w[None, :, 0]*d[:, None, 1]-w[None, :, 1]*d[:, None, 0])/denom   # along the segment
    valid = (np.abs(denom) > 1e-12) & (t > 0) & (u >= 0) & (u <= 1)
    t = np.where(valid, t, np.inf)
    j = np.argmin(t, axis=1)
    best = t[np.arange(n), j]
    hit = best <= range_max
    rng[hit] = best[hit]
    idx[hit] = keep[j[hit]//4]
    return rng, idx


def visible_fraction(origin, boxes, target, samples=9, skip=()):
    """Fraction (0..1) of `samples` points along the outline of box `target` facing `origin` that are seen from
    origin without another box (other than those in skip) in the way. The facing outline is the box's corners and
    edge midpoints nearer than its centre, sampled evenly; 0 for an origin inside the box."""
    B = np.asarray(boxes, float)
    box = B[target]
    o = np.asarray(origin, float)
    pts = np.concatenate([box[k][None]+np.linspace(0, 1, samples, endpoint=False)[:, None]*(box[(k+1) % 4]-box[k])
                          for k in range(4)])
    centre = box.mean(0)
    near = np.linalg.norm(pts-o, axis=1) <= np.linalg.norm(centre-o)
    pts = pts[near]
    if len(pts) == 0:
        return 0.0
    dist = np.linalg.norm(pts-o, axis=1)
    ang = np.arctan2(pts[:, 1]-o[1], pts[:, 0]-o[0])
    others = tuple(skip)+(target,)
    r, _ = raycast(o, ang, B, float(dist.max())+1.0, skip=others)
    return float(np.mean(r >= dist-1e-6))


def nearest_box_point(origin, box):
    """The point of box polygon (4, 2) nearest to world point origin (2,) (on its outline)."""
    o, box = np.asarray(origin, float), np.asarray(box, float)
    a, b = box, np.roll(box, -1, axis=0)
    ab = b-a
    t = np.clip(np.einsum('ij,ij->i', o-a, ab)/np.einsum('ij,ij->i', ab, ab), 0, 1)
    p = a+t[:, None]*ab
    return p[np.argmin(np.linalg.norm(p-o, axis=1))]
