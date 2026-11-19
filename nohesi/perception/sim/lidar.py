"""Tier A lidar: one planar sweep ray-cast against every car's box (design section 5).

The beams sweep the field of view from right to left, each fired at its own time (scan skew), and each is cast against
every car and the sensor exactly where they are at that time: the truth (types.TruthHistory) is linear between its
snapshots, so the boxes and the sensor are interpolated linearly between the knots (the sweep's ends and the snapshot
times inside it). Casting all beams against one frozen state per stretch of the sweep would leave a car's relative
motion within the stretch out, a step across its face between stretches that tilts the fitted box: about a degree for
a car 5 m/s off the team's speed 60 m ahead with 5 ms stretches. The rays are tested only against the boxes whose
bearing span over the sweep they cross ((beam, box) pairs), so the cost follows the beams that can hit something.
Road edges return nothing (no guardrail model)."""
import numpy as np
from ..geometry import car_boxes, mount_offset, wrap
from ..types import LidarScan

__all__ = ['CHUNK', 'beam_azimuths', 'cast_sweep', 'simulate_lidar']

CHUNK = 0.005   # s: knot spacing for a truth without a snapshot list (only .at): motion is linear between knots


def beam_azimuths(spec):
    """(n,) sensor-frame azimuths of one sweep in firing order: -fov/2 to +fov/2 every beam_spacing."""
    n = int(np.floor(spec.fov/spec.beam_spacing+1e-9))+1
    return -spec.fov/2+spec.beam_spacing*np.arange(n)


def _matched(p, q):
    """Index into p's traffic of each of q's cars (by id) and the mask of those found."""
    if len(p.traffic_id) == 0:
        return np.zeros(len(q.traffic_id), int), np.zeros(len(q.traffic_id), bool)
    order = np.argsort(p.traffic_id, kind='stable')
    pos = np.minimum(np.searchsorted(p.traffic_id[order], q.traffic_id), len(order)-1)
    return order[pos], p.traffic_id[order][pos] == q.traffic_id


def _knots(truth, t0, t1, chunk):
    """Times the truth is read at: the sweep's ends and the snapshot times inside (exactly linear between them), or a
    grid of `chunk` seconds for a truth without a snapshot list."""
    snaps = getattr(truth, 'snaps', None)
    if snaps is None:
        return np.linspace(t0, t1, max(1, int(np.ceil((t1-t0)/chunk-1e-9)))+1)
    return np.unique(np.r_[t0, [s.time for s, _ in snaps if t0 < s.time < t1], t1])


def _knot_states(truth, knots):
    """(team [x, y, yaw] (K, T, 3) with yaw unwrapped along the knots, traffic [x, y, heading] (K, N, 3), ids (N,)):
    truth.at at every knot, the traffic matched by id to the last knot's cars (a car missing at an earlier knot, new or
    recycled, takes its state at the last one)."""
    states = [truth.at(float(t)) for t in knots]
    last = states[-1]
    traffic = np.repeat(last.traffic[None, :, :3], len(states), 0)
    for k, s in enumerate(states[:-1]):
        m, found = _matched(s, last)
        traffic[k, found] = s.traffic[m[found], :3]
    team = np.array([s.team[:, :3] for s in states])
    team[:, :, 2] = team[0, :, 2]+np.cumsum(np.r_[np.zeros((1, team.shape[1])),
                                                  wrap(np.diff(team[:, :, 2], axis=0))], axis=0)
    return team, traffic, np.asarray(last.traffic_id, int)


def _lerp(knots, values, t):
    """values (K, ...) at the knots, linearly interpolated (clamped) to the times t (n,): (n, ...)."""
    if len(knots) == 1:
        return np.repeat(values[:1], len(t), 0)
    seg = np.clip(np.searchsorted(knots, t, side='right')-1, 0, len(knots)-2)
    w = ((t-knots[seg])/(knots[seg+1]-knots[seg])).reshape((-1,)+(1,)*(values.ndim-1))
    return values[seg]+w*(values[seg+1]-values[seg])


def cast_sweep(spec, car, truth, t_end, params, chunk=CHUNK):
    """Noise-free sweep of lidar `spec` on team car `car` ending at t_end: (azimuth (n,), time (n,), range (n,) with
    np.inf where nothing is hit within range_max, label (n,): the traffic id hit, -2 - team index for a teammate, -1
    for none). The label is for evaluation only. Every traffic car and the other team cars are cast against, each beam
    at its own fire time (module docstring); the carrying car's own body never is. chunk: knot spacing for a truth with
    no snapshot list."""
    az = beam_azimuths(spec)
    n, period = len(az), 1.0/spec.rate
    t_start = t_end-period
    time = np.linspace(t_start, t_end, n)
    knots = _knots(truth, t_start, t_end, chunk)
    team, traffic, ids = _knot_states(truth, knots)
    K, n_team, n_traffic = len(knots), team.shape[1], traffic.shape[1]
    ox, oy = mount_offset(spec.mount, params)
    x, y, yaw = np.moveaxis(team[:, car], -1, 0)
    sensor = np.column_stack((x+np.cos(yaw)*ox-np.sin(yaw)*oy, y+np.sin(yaw)*ox+np.cos(yaw)*oy, yaw+spec.mount.yaw))
    others = np.delete(np.arange(n_team), car)
    states = np.concatenate((traffic, team[:, others]), 1)                      # (K, B, 3)
    labels = np.r_[ids, -2-others]
    B = states.shape[1]
    ranges, label = np.full(n, np.inf), np.full(n, -1)
    if B == 0:
        return az, time, ranges, label
    corners = car_boxes(states.reshape(-1, 3), params).reshape(K, B, 4, 2)

    # Each box's bearing span from the sensor over the knots (relative to the boresight at the first knot), widened by
    # a beam; the beams inside it form its (beam, box) pairs. Seen from outside, a box spans less than pi about its
    # centre's bearing; a sensor inside a box (or a span across +-pi) takes every beam.
    rel = corners-sensor[:, None, None, :2]
    centre = rel.mean(2)
    bearing = np.arctan2(centre[..., 1], centre[..., 0])
    side = wrap(np.arctan2(rel[..., 1], rel[..., 0])-bearing[..., None])
    mid = sensor[0, 2]+wrap(bearing-sensor[0, 2])
    mid = mid[0]+np.r_[np.zeros((1, B)), np.cumsum(wrap(np.diff(mid, axis=0)), axis=0)]   # unwrapped over the knots
    lo = (mid+side.min(-1)).min(0)-sensor[0, 2]-spec.beam_spacing
    hi = (mid+side.max(-1)).max(0)-sensor[0, 2]+spec.beam_spacing
    edges = np.roll(corners, -1, axis=2)-corners
    inside = (edges[..., 1]*rel[..., 0]-edges[..., 0]*rel[..., 1] >= 0).all(-1).any(0)
    reach = (np.hypot(centre[..., 0], centre[..., 1])-np.hypot(params.length, params.width)/2).min(0)
    ok = reach <= spec.range_max
    beam_yaw = _lerp(knots, sensor[:, 2], time)
    beam_rel = az+beam_yaw-sensor[0, 2]                                         # ascending: yaw drifts slowly
    whole = inside | (hi-lo >= 2*np.pi)
    starts, counts = [], []
    for turn in (-2*np.pi, 0.0, 2*np.pi):                                       # a span across the +-pi cut
        i0 = np.where(whole, 0, np.searchsorted(beam_rel, lo+turn))
        i1 = np.where(whole, n, np.searchsorted(beam_rel, hi+turn, side='right'))
        starts.append(i0)
        counts.append(np.where(ok & (~whole | (turn == 0.0)), np.maximum(i1-i0, 0), 0))
    i0, count = np.concatenate(starts), np.concatenate(counts)
    if count.sum() == 0:
        return az, time, ranges, label
    box = np.repeat(np.tile(np.arange(B), 3), count)
    beam = np.arange(count.sum())-np.repeat(np.cumsum(count)-count, count)+np.repeat(i0, count)

    # Each pair's box and sensor at the beam's own time, and geometry.raycast's ray-segment test on its four edges.
    t = time[beam]
    seg = np.clip(np.searchsorted(knots, t, side='right')-1, 0, max(K-2, 0))
    a = corners[seg, box]
    if K > 1:
        w = ((t-knots[seg])/(knots[seg+1]-knots[seg]))[:, None, None]
        a = a+w*(corners[seg+1, box]-a)
    o = _lerp(knots, sensor[:, :2], time)[beam]
    phi = beam_yaw[beam]+az[beam]
    d = np.stack((np.cos(phi), np.sin(phi)), -1)[:, None, :]
    e = np.roll(a, -1, axis=1)-a
    v = a-o[:, None, :]
    denom = d[..., 0]*e[..., 1]-d[..., 1]*e[..., 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        tr = (v[..., 0]*e[..., 1]-v[..., 1]*e[..., 0])/denom
        u = (v[..., 0]*d[..., 1]-v[..., 1]*d[..., 0])/denom
    hit = np.where((np.abs(denom) > 1e-12) & (tr > 0) & (u >= 0) & (u <= 1), tr, np.inf).min(1)
    order = np.lexsort((hit, beam))
    first = order[np.r_[True, beam[order][1:] != beam[order][:-1]]]
    good = first[np.isfinite(hit[first]) & (hit[first] <= spec.range_max)]
    ranges[beam[good]] = hit[good]
    label[beam[good]] = labels[box[good]]
    return az, time, ranges, label


def simulate_lidar(spec, car, truth, t_end, params, rng):
    """One sweep of lidar `spec` on team car `car` from t_end - 1/rate to t_end (types.LidarScan; points None).

    Range noise N(0, range_sigma); a beam returns np.nan with probability dropout + far_dropout * range/range_max
    (reflectivity fall-off), or when nothing is hit within range_max. With layers > 1 each beam is repeated layers
    times in a row (same azimuth and time, still planar), each with its own noise and dropout. Draws a fixed number
    of values from `rng` per sweep, whatever is hit."""
    az, time, true_range, _ = cast_sweep(spec, car, truth, t_end, params)
    layers = max(1, int(spec.layers))
    if layers > 1:
        az, time, true_range = (np.repeat(a, layers) for a in (az, time, true_range))
    noise = spec.range_sigma*rng.standard_normal(len(az))
    u = rng.random(len(az))
    hit = np.isfinite(true_range)
    p_drop = spec.dropout+spec.far_dropout*np.where(hit, true_range, 0.0)/spec.range_max
    measured = np.where(hit & (u >= p_drop), true_range+noise, np.nan)
    return LidarScan(spec.name, car, t_end-1.0/spec.rate, t_end, az, measured, time)
